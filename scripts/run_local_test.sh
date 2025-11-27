#!/bin/bash
# =============================================================================
# Script de prueba local para SFT - 1 GPU
# =============================================================================
# Este script ejecuta un entrenamiento de prueba con un subset pequeño
# del dataset para verificar que todo funciona correctamente.
# =============================================================================

set -e  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  SFT Local Test - Medical Reasoning   ${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuración por defecto
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-1B-Instruct}"  # Modelo pequeño para test
DATA_PATH="${DATA_PATH:-./data/test_sample.json}"
MAX_SAMPLES=10          # Solo 10 muestras para test rápido
MAX_SEQ_LEN=1024        # Longitud reducida para test
TRAIN_BSZ=1             # Batch size mínimo
GRAD_ACCUM=2            # Gradient accumulation
N_EPOCHS=1              # Solo 1 epoch para test
EXPERIMENT_NAME="local_test"

# Parsear argumentos
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --wandb_online)
            WANDB_ONLINE="--wandb_online"
            shift
            ;;
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Argumento desconocido: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuración:${NC}"
echo "  - Modelo: $MODEL_PATH"
echo "  - Dataset: $DATA_PATH"
echo "  - Max samples: $MAX_SAMPLES"
echo "  - Max seq len: $MAX_SEQ_LEN"
echo "  - Batch size: $TRAIN_BSZ"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Epochs: $N_EPOCHS"
echo ""

# Verificar que el dataset existe
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}Error: No se encontró el dataset en $DATA_PATH${NC}"
    echo -e "${YELLOW}Creando dataset de prueba...${NC}"
    
    # Crear dataset de prueba a partir del dataset principal si existe
    if [ -f "./merged_medical_datasets_v2.json" ]; then
        python -c "
import json
with open('./merged_medical_datasets_v2.json', encoding='utf-8') as f:
    data = json.load(f)
sample = data[:20]  # Tomar 20 muestras
with open('$DATA_PATH', 'w', encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)
print(f'Created test sample with {len(sample)} examples')
"
    else
        echo -e "${RED}Error: No se encontró merged_medical_datasets_v2.json${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Iniciando entrenamiento de prueba...${NC}"
echo ""

# Ejecutar entrenamiento con accelerate
accelerate launch \
    --config_file ./configs/deepspeed_zero2_local.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    SFT_stage1.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --max_samples $MAX_SAMPLES \
    --max_seq_len $MAX_SEQ_LEN \
    --train_bsz_per_gpu $TRAIN_BSZ \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --n_epochs $N_EPOCHS \
    --experiment_name "$EXPERIMENT_NAME" \
    --learning_rate 1e-5 \
    --warmup_rates 0.1 \
    --log_every 1 \
    $WANDB_ONLINE \
    ${HF_TOKEN:+--hf_token "$HF_TOKEN"}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✓ Test completado exitosamente!      ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Checkpoints guardados en: ./ckpts/$EXPERIMENT_NAME/"
echo "Logs guardados en: ./train_logs/$EXPERIMENT_NAME/"

