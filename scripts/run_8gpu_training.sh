#!/bin/bash
# =============================================================================
# Script de entrenamiento completo - 8 GPUs
# =============================================================================
# Este script ejecuta el entrenamiento completo del modelo de razonamiento
# médico con el dataset completo usando 8 GPUs.
# =============================================================================

set -e  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  SFT Training - 8 GPU Production      ${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuración por defecto
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DATA_PATH="${DATA_PATH:-./merged_medical_datasets_v2.json}"
MAX_SEQ_LEN=8192
TRAIN_BSZ=2
GRAD_ACCUM=8
N_EPOCHS=3
LEARNING_RATE=5e-6
EXPERIMENT_NAME="${EXPERIMENT_NAME:-medical_o1_spanish}"

# WandB y HuggingFace
WANDB_ONLINE=""
PUSH_TO_HUB=""
HF_TOKEN=""
HF_REPO_ID=""

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
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --n_epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --wandb_online)
            WANDB_ONLINE="--wandb_online"
            shift
            ;;
        --push_to_hub)
            PUSH_TO_HUB="--push_to_hub"
            shift
            ;;
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --hf_repo_id)
            HF_REPO_ID="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Argumento desconocido: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Configuración de entrenamiento:${NC}"
echo "  - Modelo base: $MODEL_PATH"
echo "  - Dataset: $DATA_PATH"
echo "  - Max seq len: $MAX_SEQ_LEN"
echo "  - Batch size por GPU: $TRAIN_BSZ"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $((TRAIN_BSZ * 8 * GRAD_ACCUM))"
echo "  - Epochs: $N_EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Experiment name: $EXPERIMENT_NAME"
echo ""

# Verificar GPUs disponibles
echo -e "${YELLOW}Verificando GPUs...${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Verificar que el dataset existe
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}Error: No se encontró el dataset en $DATA_PATH${NC}"
    exit 1
fi

# Contar ejemplos en el dataset
DATASET_SIZE=$(python -c "import json; print(len(json.load(open('$DATA_PATH'))))")
echo -e "${GREEN}Dataset cargado: $DATASET_SIZE ejemplos${NC}"
echo ""

echo -e "${GREEN}Iniciando entrenamiento...${NC}"
echo -e "${YELLOW}(Logs en ./train_logs/$EXPERIMENT_NAME/)${NC}"
echo ""

# Ejecutar entrenamiento con accelerate
accelerate launch \
    --config_file ./configs/deepspeed_zero3_8gpu.yaml \
    --num_processes 8 \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard \
    SFT_stage1.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --max_seq_len $MAX_SEQ_LEN \
    --train_bsz_per_gpu $TRAIN_BSZ \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --n_epochs $N_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_rates 0.05 \
    --experiment_name "$EXPERIMENT_NAME" \
    --log_every 10 \
    --max_ckpts 3 \
    $WANDB_ONLINE \
    $PUSH_TO_HUB \
    ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
    ${HF_REPO_ID:+--hf_repo_id "$HF_REPO_ID"}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ✓ Entrenamiento completado!          ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Checkpoints guardados en: ./ckpts/$EXPERIMENT_NAME/"
echo "Logs guardados en: ./train_logs/$EXPERIMENT_NAME/"

