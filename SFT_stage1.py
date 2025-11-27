import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import traceback
from jinja2 import Template
from huggingface_hub import HfApi, login as hf_login

from transformers import AutoModelForCausalLM, AutoTokenizer
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class Train_dataset(torch.utils.data.Dataset):
    """
    Dataset para entrenamiento SFT con datos médicos en español.
    
    Estructura esperada del dataset:
    - Pregunta: La pregunta médica
    - Razonamiento_Complejo: El proceso de razonamiento paso a paso
    - Respuesta: La respuesta final
    - Archivo_fuente: (opcional) Origen del dato
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Cargar datos
        with open(config.data_path, encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filtrar datos válidos
        newdata = []
        for da in self.data:
            # Verificar que tenga los campos necesarios
            if all(k in da for k in ['Pregunta', 'Razonamiento_Complejo', 'Respuesta']):
                newdata.append(da)
        
        print(f'Datos originales: {len(self.data)}, Datos válidos: {len(newdata)}')
        self.data = newdata

        # Limitar datos para pruebas si se especifica
        if hasattr(config, 'max_samples') and config.max_samples > 0:
            self.data = self.data[:config.max_samples]
            print(f'Limitando a {len(self.data)} muestras para prueba')

        self.max_seq_len = self.config.max_seq_len
        self.debug = 0

        # Template de chat para LLaMA 3 (si el modelo no tiene uno)
        chat_template_llama3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        if not tokenizer.chat_template:
            tokenizer.chat_template = chat_template_llama3
            
        self.template = Template(tokenizer.chat_template)

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self, da):
        """
        Formatea la respuesta con el razonamiento complejo.
        Formato: ## Pensamiento\n\n{razonamiento}\n\n## Respuesta Final\n\n{respuesta}
        """
        # Usar formato bilingüe para compatibilidad
        temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return temp.format(da['Razonamiento_Complejo'], da['Respuesta'])

    def get_prompt(self, da):
        """Construye el prompt de entrenamiento con máscara de labels."""
        q = da['Pregunta']
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'

        # Renderizar el template completo (pregunta + respuesta)
        input_text = self.template.render(
            messages=[
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)

        # Renderizar solo la query (para calcular la máscara de labels)
        query = self.template.render(
            messages=[{"role": "user", "content": q}],
            bos_token=self.tokenizer.bos_token,
            add_generation_prompt=True
        )
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)

        # Labels: -100 para la query (no calcular loss), ids para la respuesta
        labels = [-100] * len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}        

    def collate_fn(self, batch):
        """Función de collate con padding dinámico."""
        data = [self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len, self.max_seq_len)
        
        # Padding con eos_token para input_ids y -100 para labels
        input_ids = [item[:max_len] + [self.tokenizer.eos_token_id] * (max_len - len(item)) for item in input_ids]
        labels = [item[:max_len] + [-100] * (max_len - len(item)) for item in labels]
        
        # Debug: mostrar primeros ejemplos
        if self.debug < 3:
            print(f'\n{"="*50}')
            print(f'DEBUG EJEMPLO {self.debug + 1}:')
            print(f'Input IDs decoded:\n{self.tokenizer.decode(input_ids[-1])}')
            print(f'\nLabels decoded:\n{self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]])}')
            print(f'{"="*50}\n')
            self.debug += 1

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(labels),
        }
    
    def __len__(self):
        return len(self.data)


class SFTMetric:
    """Métricas para el entrenamiento SFT."""
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


def train(args):
    """Función principal de entrenamiento."""
    
    accelerator = Accelerator(
        mixed_precision='bf16', 
        gradient_accumulation_steps=args.gradient_accumulation_steps
    ) 

    # Configurar WandB
    if accelerator.is_main_process:
        wandb_mode = "online" if args.wandb_online else "offline"
        wandb.init(
            project=args.experiment_name, 
            config=vars(args), 
            dir=args.log_dir, 
            mode=wandb_mode,
            name=args.run_name if args.run_name else None
        )
        
        # Login a HuggingFace si se proporciona token
        if args.hf_token:
            hf_login(token=args.hf_token)
            print("✓ Conectado a HuggingFace Hub")
    
    accelerator.print(f'Configuración:\n{args}')

    # Configurar DeepSpeed
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps
    )

    # Cargar tokenizer y modelo
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # Habilitar gradient checkpointing para ahorro de memoria
    model.gradient_checkpointing_enable()

    # Configurar optimizer con weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Crear dataset y dataloader
    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train_bsz_per_gpu, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=train_dataset.collate_fn
    )

    # Scheduler con warmup coseno
    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_rates * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    accelerator.print(f'Configuración de entrenamiento:')
    accelerator.print(f'  - Gradient accumulation steps: {accelerator.gradient_accumulation_steps}')
    accelerator.print(f'  - Data path: {args.data_path}')
    accelerator.print(f'  - Learning rate: {args.learning_rate}')
    accelerator.print(f'  - Número de training steps: {num_training_steps}')
    accelerator.print(f'  - Número de ejemplos: {len(train_dataset)}')
    
    # Preparar modelo con accelerate
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    start_epoch = 0
    start_step = 0
    global_step = 0

    metric = SFTMetric(device=torch.cuda.current_device())

    def save_checkpoint(epoch, step, global_step):
        """Guardar checkpoint del modelo."""
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        
        if accelerator.is_main_process:
            # Limpiar checkpoints antiguos si excedemos el límite
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            
            if args.max_ckpts > 0 and num_checkpoints >= args.max_ckpts:
                checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                oldest_checkpoint = checkpoint_files[0]
                shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))
                
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            
            if accelerator.state.deepspeed_plugin.zero_stage != 3:
                model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
            tokenizer.save_pretrained(output_dir)
            
            # Copiar archivos de configuración del modelo original
            copy_files = []
            for item in os.listdir(args.model_path):
                if os.path.exists(os.path.join(output_dir, item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                s = os.path.join(args.model_path, item)
                if os.path.isfile(s):
                    shutil.copy(s, os.path.join(output_dir, item))
                copy_files.append(item)
            print(f'Modelo guardado en {output_dir}, archivos copiados: {copy_files}')
            
            # Subir a HuggingFace Hub si se especifica
            if args.push_to_hub and args.hf_repo_id:
                try:
                    api = HfApi()
                    api.upload_folder(
                        folder_path=output_dir,
                        repo_id=args.hf_repo_id,
                        commit_message=f"Checkpoint epoch {epoch}, step {global_step}"
                    )
                    print(f'✓ Modelo subido a HuggingFace Hub: {args.hf_repo_id}')
                except Exception as e:
                    print(f'Error subiendo a HuggingFace Hub: {e}')

        if accelerator.state.deepspeed_plugin.zero_stage == 3:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(
                os.path.join(save_dir, 'tfmr'),
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model)
            )
            
        accelerator.wait_for_everyone()
        accelerator.save(
            {"epoch": epoch, "step": step, "global_step": global_step}, 
            os.path.join(save_dir, "training_state.pt")
        )
        accelerator.print(f'✓ Checkpoint checkpoint-{epoch}-{global_step} guardado')

    accelerator.print(f'\nConfiguración DeepSpeed:\n{accelerator.deepspeed_config}')
    model.train()

    # Loop de entrenamiento
    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = (
            tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.n_epochs}") 
            if accelerator.is_main_process 
            else enumerate(train_dataloader)
        )
        
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch == start_epoch and batch_cnt < start_step:
                continue

            # Limpiar caché en el primer batch
            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids = batch['input_ids']
            labels = batch['labels']

            output = model(input_ids=input_ids, labels=labels, return_dict=True, use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            
            accelerator.backward(loss)
            
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(
                    epoch=epoch, 
                    step=batch_cnt, 
                    total=len(train_dataloader), 
                    skip=accelerator.optimizer_step_was_skipped, 
                    loss=round(train_loss, 3), 
                    acc=round(acc, 3), 
                    seq_len=len(input_ids[0]), 
                    lr=lr_scheduler.get_last_lr()[0]
                )

            # Logging a WandB
            if global_step % args.log_every == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': global_step
                }, step=global_step)

        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)
    
    # Cerrar WandB
    if accelerator.is_main_process:
        wandb.finish()
        print("\n✓ Entrenamiento completado exitosamente!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFT Training para Razonamiento Médico Complejo')
    
    # Experiment Args
    parser.add_argument('--experiment_name', type=str, default='medical_sft_spanish')
    parser.add_argument('--run_name', type=str, default=None, help='Nombre del run en WandB')

    # Model Args
    parser.add_argument('--model_path', required=True, type=str, help='Ruta al modelo base')

    # Data Args
    parser.add_argument('--data_path', required=True, type=str, help='Ruta al dataset JSON')
    parser.add_argument('--max_samples', type=int, default=0, help='Limitar número de muestras (0=sin límite)')

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int, help='Máximo número de checkpoints a mantener')
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)
    parser.add_argument('--log_every', default=3, type=int, help='Log cada N steps')

    # WandB Args
    parser.add_argument('--wandb_online', action='store_true', help='Usar WandB online (default: offline)')
    
    # HuggingFace Args
    parser.add_argument('--hf_token', type=str, default=None, help='Token de HuggingFace')
    parser.add_argument('--push_to_hub', action='store_true', help='Subir checkpoints a HuggingFace Hub')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='ID del repositorio en HuggingFace')

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)
