# 🚀 Comandos para Ejecutar en Google Colab

## Paso 1: Clonar el Repositorio

```bash
# Clonar el repositorio
!git clone https://github.com/TU_USUARIO/HuatuoGPT-o1.git
# O si es un repositorio privado:
# !git clone https://TU_TOKEN@github.com/TU_USUARIO/HuatuoGPT-o1.git

# Cambiar al directorio del proyecto
%cd HuatuoGPT-o1
```

## Paso 2: Instalar Dependencias

```bash
# Instalar las dependencias del proyecto
!pip install -r requirements.txt

# Si hay problemas con deepspeed, instalar por separado:
# !pip install deepspeed==0.15.4

# Verificar la instalación de CUDA (Colab tiene GPU por defecto)
!nvidia-smi
```

## Paso 3: Configurar Credenciales (Opcional)

```bash
# Para WandB (opcional, para tracking)
# !wandb login
# Ingresa tu API key cuando se solicite

# Para HuggingFace (necesario si usas modelos privados)
# !huggingface-cli login
# Ingresa tu token cuando se solicite
```

## Paso 4: Verificar Dataset

```bash
# Verificar que el dataset existe o crearlo
# Si no existe merged_medical_datasets_v2.json, el script lo creará automáticamente
# o puedes usar el dataset de prueba:
!ls -la data/
```

## Paso 5: Dar Permisos de Ejecución al Script

```bash
# Dar permisos de ejecución al script
!chmod +x scripts/run_local_test.sh
```

## Paso 6: Ejecutar el Script de Prueba Local

```bash
# Opción 1: Ejecutar el script directamente
!bash scripts/run_local_test.sh

# Opción 2: Ejecutar con parámetros personalizados
!bash scripts/run_local_test.sh \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --data_path ./data/test_sample.json \
    --max_samples 10 \
    --wandb_online

# Opción 3: Si necesitas token de HuggingFace
!bash scripts/run_local_test.sh \
    --hf_token TU_TOKEN_AQUI
```

## Alternativa: Ejecutar Manualmente sin el Script

Si prefieres ejecutar el comando directamente sin el script bash:

```bash
!accelerate launch \
    --config_file ./configs/deepspeed_zero2_local.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    SFT_stage1.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --data_path ./data/test_sample.json \
    --max_samples 10 \
    --max_seq_len 1024 \
    --train_bsz_per_gpu 1 \
    --gradient_accumulation_steps 2 \
    --n_epochs 1 \
    --experiment_name local_test \
    --learning_rate 1e-5 \
    --warmup_rates 0.1 \
    --log_every 1
```

## Notas Importantes para Colab

1. **GPU**: Asegúrate de activar la GPU en Colab: `Runtime > Change runtime type > GPU`

2. **Memoria**: Si tienes problemas de memoria, reduce:
   - `--max_seq_len` a 512 o 256
   - `--train_bsz_per_gpu` a 1
   - `--max_samples` a 5

3. **Tiempo de Ejecución**: Colab tiene límites de tiempo. Para entrenamientos largos, considera usar Colab Pro o ejecutar en otro entorno.

4. **Persistencia**: Los archivos se perderán al reiniciar el runtime. Considera guardar checkpoints en Google Drive:
   ```bash
   # Montar Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Copiar checkpoints a Drive
   !cp -r ./ckpts/local_test /content/drive/MyDrive/
   ```

## Comandos Completos en una Celda (Todo en Uno)

```python
# Celda 1: Setup completo
%%bash
# Clonar repositorio
git clone https://github.com/TU_USUARIO/HuatuoGPT-o1.git
cd HuatuoGPT-o1

# Instalar dependencias
pip install -r requirements.txt

# Dar permisos
chmod +x scripts/run_local_test.sh

# Verificar GPU
nvidia-smi
```

```python
# Celda 2: Ejecutar entrenamiento
%%bash
cd HuatuoGPT-o1
bash scripts/run_local_test.sh
```

