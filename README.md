# 🏥 Medical Reasoning SFT - Entrenamiento de Modelos con Razonamiento Complejo

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-0.15-green.svg)](https://www.deepspeed.ai/)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Tracking-yellow.svg)](https://wandb.ai/)

</div>

## 📋 Descripción

Este repositorio contiene el código para entrenar modelos de lenguaje con **razonamiento médico complejo** mediante Supervised Fine-Tuning (SFT). El modelo aprende a:

1. **Pensar paso a paso** antes de responder (Complex Chain-of-Thought)
2. **Razonar sobre problemas médicos** de forma estructurada
3. **Generar respuestas precisas** basadas en el razonamiento previo

### Formato de Salida del Modelo

```
## Thinking
[Proceso de razonamiento paso a paso]

## Final Response
[Respuesta final basada en el razonamiento]
```

---

## 🗂️ Estructura del Proyecto

```
.
├── SFT_stage1.py                    # 🎯 Script principal de entrenamiento SFT
├── RL_stage2.py                     # Script de Reinforcement Learning (PPO)
├── test_dataset.py                  # Script de verificación del dataset
├── merged_medical_datasets_v2.json  # Dataset de entrenamiento (español)
├── requirements.txt                 # Dependencias
│
├── configs/
│   ├── deepspeed_zero2_local.yaml   # Config para pruebas locales (1 GPU)
│   ├── deepspeed_zero3.yaml         # Config original Zero3
│   └── deepspeed_zero3_8gpu.yaml    # Config optimizada para 8 GPUs
│
├── scripts/
│   ├── run_local_test.sh            # Script de prueba local
│   └── run_8gpu_training.sh         # Script de entrenamiento producción
│
├── data/
│   ├── demo_data.json               # Datos de demostración
│   └── test_sample.json             # Muestra para pruebas (generada)
│
└── evaluation/
    ├── eval.py                      # Script de evaluación
    └── data/eval_data.json          # Datos de evaluación
```

---

## 📊 Estructura del Dataset

El dataset debe ser un archivo JSON con la siguiente estructura:

```json
[
  {
    "Pregunta": "La pregunta médica a responder",
    "Razonamiento_Complejo": "El proceso de pensamiento paso a paso...",
    "Respuesta": "La respuesta final basada en el razonamiento",
    "Archivo_fuente": "Origen del dato (opcional)"
  }
]
```

### Campos Requeridos

| Campo | Descripción |
|-------|-------------|
| `Pregunta` | La pregunta o problema médico |
| `Razonamiento_Complejo` | Cadena de pensamiento detallada |
| `Respuesta` | Respuesta final concisa |

### Campos Opcionales

| Campo | Descripción |
|-------|-------------|
| `Archivo_fuente` | Origen del dato para tracking |

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repositorio>
cd ict
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
.\venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales (opcional)

```bash
# WandB
wandb login

# HuggingFace
huggingface-cli login
```

---

## 🧪 Pruebas del Dataset

Antes de entrenar, verifica que tu dataset esté correcto:

```bash
# Verificar estructura
python test_dataset.py --data_path ./merged_medical_datasets_v2.json

# Verificar estructura + tokenización
python test_dataset.py --data_path ./merged_medical_datasets_v2.json --test_tokenization

# Crear muestra pequeña para pruebas
python test_dataset.py --data_path ./merged_medical_datasets_v2.json --create_sample
```

---

## 🚀 Entrenamiento

### Prueba Local (1 GPU)

Para verificar que todo funciona antes de ejecutar en producción:

```bash
# Dar permisos de ejecución
chmod +x scripts/run_local_test.sh

# Ejecutar prueba local
./scripts/run_local_test.sh
```

O manualmente:

```bash
accelerate launch \
    --config_file ./configs/deepspeed_zero2_local.yaml \
    --num_processes 1 \
    SFT_stage1.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --data_path ./data/test_sample.json \
    --max_samples 10 \
    --max_seq_len 1024 \
    --train_bsz_per_gpu 1 \
    --n_epochs 1 \
    --experiment_name test_local
```

### Entrenamiento Completo (8 GPUs)

```bash
chmod +x scripts/run_8gpu_training.sh

# Entrenamiento básico
./scripts/run_8gpu_training.sh

# Con opciones personalizadas
./scripts/run_8gpu_training.sh \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --experiment_name mi_experimento \
    --n_epochs 3 \
    --wandb_online
```

O manualmente:

```bash
accelerate launch \
    --config_file ./configs/deepspeed_zero3_8gpu.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    SFT_stage1.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --data_path ./merged_medical_datasets_v2.json \
    --max_seq_len 8192 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 8 \
    --n_epochs 3 \
    --learning_rate 5e-6 \
    --experiment_name medical_o1_spanish \
    --wandb_online
```

---

## 📈 Tracking con WandB

### Modo Offline (por defecto)

Los logs se guardan localmente en `./train_logs/` y pueden sincronizarse después:

```bash
wandb sync ./train_logs/<experiment_name>/
```

### Modo Online

Agrega `--wandb_online` al comando de entrenamiento:

```bash
accelerate launch ... SFT_stage1.py ... --wandb_online
```

### Métricas Trackeadas

| Métrica | Descripción |
|---------|-------------|
| `loss` | Loss de entrenamiento |
| `acc` | Accuracy en tokens predichos |
| `lr` | Learning rate actual |
| `skip` | Steps saltados por overflow |

---

## 🤗 Integración con HuggingFace Hub

### Subir Checkpoints Automáticamente

```bash
accelerate launch ... SFT_stage1.py \
    --hf_token YOUR_TOKEN \
    --push_to_hub \
    --hf_repo_id tu-usuario/nombre-modelo
```

### Subir Manualmente

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./ckpts/experiment_name/checkpoint-X-Y/tfmr",
    repo_id="tu-usuario/nombre-modelo",
    commit_message="Upload trained model"
)
```

---

## 📝 Argumentos de Entrenamiento

### Modelo y Datos

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--model_path` | **requerido** | Ruta al modelo base (local o HuggingFace) |
| `--data_path` | **requerido** | Ruta al dataset JSON |
| `--max_samples` | 0 | Limitar ejemplos (0 = sin límite) |

### Entrenamiento

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--max_seq_len` | 8192 | Longitud máxima de secuencia |
| `--train_bsz_per_gpu` | 2 | Batch size por GPU |
| `--gradient_accumulation_steps` | 8 | Steps de acumulación |
| `--n_epochs` | 3 | Número de epochs |
| `--learning_rate` | 5e-6 | Learning rate |
| `--warmup_rates` | 0.05 | Ratio de warmup |
| `--weight_decay` | 0.1 | Weight decay |

### Outputs

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--output_dir` | ./ckpts | Directorio de checkpoints |
| `--log_dir` | ./train_logs | Directorio de logs |
| `--max_ckpts` | 2 | Máximo checkpoints a mantener |
| `--experiment_name` | medical_sft_spanish | Nombre del experimento |

### WandB y HuggingFace

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--wandb_online` | False | Usar WandB en modo online |
| `--hf_token` | None | Token de HuggingFace |
| `--push_to_hub` | False | Subir checkpoints a HF Hub |
| `--hf_repo_id` | None | ID del repo en HuggingFace |

---

## 🧠 Arquitectura del Código

### `SFT_stage1.py`

```
┌─────────────────────────────────────────────────────────────┐
│                    SFT_stage1.py                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Train_dataset                                              │
│  ├── __init__: Carga JSON, valida campos                   │
│  ├── get_response: Formatea Thinking + Response            │
│  ├── get_prompt: Crea input_ids y labels con máscaras      │
│  └── collate_fn: Padding dinámico y batching               │
│                                                             │
│  SFTMetric                                                  │
│  ├── update: Acumula métricas por step                     │
│  └── get_metric: Reduce métricas entre GPUs                │
│                                                             │
│  train()                                                    │
│  ├── Inicializa Accelerator + DeepSpeed                    │
│  ├── Configura WandB + HuggingFace                         │
│  ├── Carga modelo + tokenizer                              │
│  ├── Configura optimizer (AdamW) + scheduler (cosine)      │
│  ├── Loop de entrenamiento con gradient checkpointing      │
│  └── save_checkpoint: Guarda modelo + sube a HF Hub        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Flujo de Datos

```
Dataset JSON
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Pregunta   │───▶│ Chat Template │───▶│  input_ids  │
│             │    │   (Jinja2)    │    │             │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌──────────────┐           │
│ Razonamiento│───▶│ ## Thinking  │           │
│  + Respuesta│    │ ## Response  │           │
└─────────────┘    └──────────────┘           │
                          │                   │
                          ▼                   ▼
                   ┌──────────────────────────────┐
                   │  Labels = [-100] * len(query)│
                   │           + response_ids     │
                   └──────────────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │    Model Forward Pass        │
                   │    CrossEntropyLoss          │
                   └──────────────────────────────┘
```

---

## 💡 Tips y Recomendaciones

### Optimización de Memoria

1. **Reducir `max_seq_len`** si hay OOM (Out of Memory)
2. **Aumentar `gradient_accumulation_steps`** para mantener batch size efectivo
3. **Usar Zero3** para modelos grandes (>7B parámetros)
4. **Activar CPU offload** en la config de DeepSpeed si es necesario

### Velocidad de Entrenamiento

1. **Zero2 es más rápido** que Zero3 para modelos que caben en memoria
2. **Desactivar CPU offload** si tienes suficiente VRAM
3. **Usar flash-attention** si está disponible

### Estabilidad

1. **Warmup del 5-10%** de los steps totales
2. **Learning rate bajo** (1e-6 a 5e-5) para fine-tuning
3. **Gradient clipping** viene habilitado por defecto en DeepSpeed

---

## 🐛 Troubleshooting

### "CUDA out of memory"

```bash
# Reducir batch size
--train_bsz_per_gpu 1

# Reducir longitud de secuencia
--max_seq_len 4096

# Aumentar gradient accumulation
--gradient_accumulation_steps 16
```

### "Tokenizer has no chat_template"

El código asigna automáticamente el template de LLaMA 3 si el modelo no tiene uno.

### WandB no sincroniza

```bash
# Sincronizar manualmente
wandb sync ./train_logs/<experiment_name>/wandb/

# O forzar modo online
--wandb_online
```

---

## 📚 Referencias

- [HuatuoGPT-o1 Paper](https://arxiv.org/pdf/2412.18925)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [TRL Library](https://github.com/huggingface/trl)

---

## 📄 Licencia

Este código está basado en el repositorio [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1) y adaptado para datasets en español con estructura personalizada.

---

<div align="center">

**¡Feliz entrenamiento! 🚀**

</div>
