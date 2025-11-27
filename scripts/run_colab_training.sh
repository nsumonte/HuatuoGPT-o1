#!/bin/bash
# =============================================================================
# Script de Entrenamiento Optimizado para Google Colab Pro+
# =============================================================================
# GPU: T4 (16GB) o V100 (16GB)
# RAM: ~25GB
# Disco: ~100GB
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}  SFT Training - Optimizado para Google Colab Pro+             ${NC}"
echo -e "${GREEN}================================================================${NC}"

# =============================================================================
# CONFIGURACIÓN OPTIMIZADA PARA COLAB PRO+ (GPU T4 16GB)
# =============================================================================

MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"  # Modelo 1B - perfecto para T4
TRAIN_DATA="./data/train.json"                 # 800 ejemplos
VAL_DATA="./data/val.json"                     # 100 ejemplos
EXPERIMENT_NAME="medical_colab_training"
RUN_NAME="llama3.2-1b-medical-colab-$(date +%m%d-%H%M)"

# Training hyperparameters - OPTIMIZADOS PARA T4 16GB
MAX_SEQ_LEN=2048           # Reducido para caber en 16GB VRAM
TRAIN_BSZ=1                # Batch size 1 por GPU para seguridad
GRAD_ACCUM=8               # Accumulation alto para compensar
LEARNING_RATE=5e-6         # LR estándar para fine-tuning
WEIGHT_DECAY=0.1           # Regularización moderada
WARMUP_RATE=0.1            # 10% warmup
N_EPOCHS=3                 # 3 epochs suficiente para 800 ejemplos

# Logging y evaluación
LOG_EVERY=10               # Log cada 10 steps
EVAL_EVERY=50              # Evaluar cada 50 steps
MAX_CKPTS=2                # Solo 2 checkpoints para ahorrar espacio

# WandB - OFFLINE por defecto en Colab
WANDB_ONLINE=false

# HF Token (pasar como argumento)
HF_TOKEN=""

# =============================================================================
# PARSEAR ARGUMENTOS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --hf_token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --wandb_online)
            WANDB_ONLINE=true
            shift
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Argumento desconocido: $1${NC}"
            exit 1
            ;;
    esac
done

# =============================================================================
# VERIFICAR GPU Y MEMORIA
# =============================================================================

echo -e "\n${BLUE}🔍 Verificando recursos disponibles...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detectada:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    
    # Obtener memoria GPU disponible
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo -e "\n${BLUE}Memoria GPU total: ${GPU_MEM}MB${NC}"
    
    if [ "$GPU_MEM" -lt 15000 ]; then
        echo -e "${RED}⚠️  Warning: GPU con menos de 15GB detectada${NC}"
        echo -e "${YELLOW}   Ajustando configuración para GPU pequeña...${NC}"
        MAX_SEQ_LEN=1024
        TRAIN_BSZ=1
        GRAD_ACCUM=16
        echo -e "${YELLOW}   Nueva config: seq_len=$MAX_SEQ_LEN, batch=$TRAIN_BSZ, grad_accum=$GRAD_ACCUM${NC}"
    fi
else
    echo -e "${RED}❌ No se detectó GPU!${NC}"
    echo -e "${YELLOW}   Asegúrate de estar en modo GPU en Colab${NC}"
    exit 1
fi

# Memoria RAM
echo -e "\n${BLUE}💾 Memoria RAM:${NC}"
free -h | grep Mem

# Espacio en disco
echo -e "\n${BLUE}💿 Espacio en disco:${NC}"
df -h / | tail -1

# =============================================================================
# PREPARAR DATASET
# =============================================================================

echo -e "\n${YELLOW}📊 Preparando dataset...${NC}"

if [ ! -f "merged_medical_datasets_v2.json" ]; then
    echo -e "${RED}❌ Error: No se encontró merged_medical_datasets_v2.json${NC}"
    exit 1
fi

# Crear splits si no existen
if [ ! -f "$TRAIN_DATA" ] || [ ! -f "$VAL_DATA" ]; then
    echo -e "${YELLOW}   Creando splits train/val/test...${NC}"
    python prepare_dataset.py \
        --input merged_medical_datasets_v2.json \
        --train_samples 800 \
        --val_samples 100 \
        --test_samples 100 \
        --output_dir ./data
    echo -e "${GREEN}✓ Dataset dividido${NC}"
else
    echo -e "${GREEN}✓ Dataset ya existe${NC}"
fi

# =============================================================================
# VALIDAR ARCHIVOS
# =============================================================================

echo -e "\n${YELLOW}📋 Validando archivos...${NC}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo -e "${RED}❌ Error: No se encontró $TRAIN_DATA${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Train data: $TRAIN_DATA${NC}"

if [ ! -f "$VAL_DATA" ]; then
    echo -e "${YELLOW}⚠️  Warning: No se encontró $VAL_DATA${NC}"
    echo -e "${YELLOW}   Entrenando sin validation set${NC}"
    VAL_DATA=""
else
    echo -e "${GREEN}✓ Val data: $VAL_DATA${NC}"
fi

# Verificar que SFT_stage1_v2.py existe
if [ ! -f "SFT_stage1_v2.py" ]; then
    echo -e "${RED}❌ Error: No se encontró SFT_stage1_v2.py${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Script de entrenamiento encontrado${NC}"

# =============================================================================
# PREPARAR DIRECTORIOS
# =============================================================================

OUTPUT_DIR="./ckpts/$EXPERIMENT_NAME"
LOG_DIR="./train_logs/$EXPERIMENT_NAME"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo -e "${GREEN}✓ Directorios creados${NC}"

# =============================================================================
# MOSTRAR CONFIGURACIÓN
# =============================================================================

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN}  CONFIGURACIÓN DE ENTRENAMIENTO${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "${BLUE}Modelo:${NC}"
echo "  Path: $MODEL_PATH"
echo ""
echo -e "${BLUE}Datos:${NC}"
echo "  Train: $TRAIN_DATA"
echo "  Val: ${VAL_DATA:-'Sin validación'}"
echo ""
echo -e "${BLUE}Experimento:${NC}"
echo "  Name: $EXPERIMENT_NAME"
echo "  Run: $RUN_NAME"
echo ""
echo -e "${BLUE}Hyperparameters:${NC}"
echo "  Max Seq Length: $MAX_SEQ_LEN tokens"
echo "  Batch Size per GPU: $TRAIN_BSZ"
echo "  Gradient Accumulation: $GRAD_ACCUM steps"
echo "  Effective Batch Size: $((TRAIN_BSZ * GRAD_ACCUM))"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Warmup Rate: $WARMUP_RATE"
echo "  Epochs: $N_EPOCHS"
echo ""
echo -e "${BLUE}Logging:${NC}"
echo "  Log Every: $LOG_EVERY steps"
echo "  Eval Every: ${EVAL_EVERY} steps"
echo "  Max Checkpoints: $MAX_CKPTS"
echo "  WandB Online: $WANDB_ONLINE"
echo ""
echo -e "${BLUE}Estimaciones:${NC}"
# Con 800 ejemplos, batch_size efectivo 8, son ~100 steps por epoch
STEPS_PER_EPOCH=$((800 / (TRAIN_BSZ * GRAD_ACCUM)))
TOTAL_STEPS=$((STEPS_PER_EPOCH * N_EPOCHS))
echo "  Steps por epoch: ~$STEPS_PER_EPOCH"
echo "  Total steps: ~$TOTAL_STEPS"
echo "  Tiempo estimado: ~1.5-2 horas"
echo -e "${GREEN}================================================================${NC}"

# =============================================================================
# CONFIRMAR EJECUCIÓN
# =============================================================================

echo -e "\n${YELLOW}⏸️  Presiona Enter para continuar o Ctrl+C para cancelar...${NC}"
read -t 10 || true

# =============================================================================
# CONSTRUIR COMANDO
# =============================================================================

CMD="accelerate launch \
    --config_file ./configs/deepspeed_zero2_local.yaml \
    --num_processes 1 \
    --num_machines 1 \
    --machine_rank 0 \
    SFT_stage1_v2.py \
    --experiment_name $EXPERIMENT_NAME \
    --run_name $RUN_NAME \
    --model_path $MODEL_PATH \
    --data_path $TRAIN_DATA \
    --max_seq_len $MAX_SEQ_LEN \
    --train_bsz_per_gpu $TRAIN_BSZ \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_rates $WARMUP_RATE \
    --n_epochs $N_EPOCHS \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --max_ckpts $MAX_CKPTS \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR"

# Agregar validation data si existe
if [ -n "$VAL_DATA" ]; then
    CMD="$CMD --val_data_path $VAL_DATA"
fi

# Agregar WandB online si está habilitado
if [ "$WANDB_ONLINE" = true ]; then
    CMD="$CMD --wandb_online"
fi

# Agregar HuggingFace token si se proporcionó
if [ -n "$HF_TOKEN" ]; then
    CMD="$CMD --hf_token $HF_TOKEN"
fi

# =============================================================================
# EJECUTAR ENTRENAMIENTO
# =============================================================================

echo -e "\n${GREEN}🚀 Iniciando entrenamiento...${NC}\n"

# Ejecutar con captura de errores
if eval $CMD; then
    # =============================================================================
    # POST-ENTRENAMIENTO EXITOSO
    # =============================================================================
    
    echo -e "\n${GREEN}================================================================${NC}"
    echo -e "${GREEN}  ✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    echo ""
    echo -e "${BLUE}📁 Outputs:${NC}"
    echo "  Checkpoints: $OUTPUT_DIR"
    echo "  Logs: $LOG_DIR"
    echo ""
    
    # Listar checkpoints
    echo -e "${BLUE}💾 Checkpoints disponibles:${NC}"
    ls -lh "$OUTPUT_DIR" | grep checkpoint || echo "  (ninguno encontrado)"
    echo ""
    
    # Información de WandB
    if [ "$WANDB_ONLINE" = false ]; then
        echo -e "${YELLOW}📊 Para sincronizar métricas con WandB:${NC}"
        WANDB_RUN=$(ls -t $LOG_DIR/wandb/offline-run-* 2>/dev/null | head -1)
        if [ -n "$WANDB_RUN" ]; then
            echo "  wandb sync $WANDB_RUN"
        fi
        echo ""
    fi
    
    # Comando para análisis
    echo -e "${BLUE}📈 Para analizar métricas offline:${NC}"
    WANDB_RUN=$(ls -t $LOG_DIR/wandb/offline-run-* 2>/dev/null | head -1)
    if [ -n "$WANDB_RUN" ]; then
        echo "  python analyze_metrics.py $WANDB_RUN"
    fi
    echo ""
    
    echo -e "${GREEN}¡Listo! 🎉${NC}"
    
else
    # =============================================================================
    # POST-ENTRENAMIENTO CON ERRORES
    # =============================================================================
    
    echo -e "\n${RED}================================================================${NC}"
    echo -e "${RED}  ❌ ENTRENAMIENTO FALLÓ${NC}"
    echo -e "${RED}================================================================${NC}"
    echo ""
    echo -e "${YELLOW}🔍 Posibles causas:${NC}"
    echo "  1. Out of Memory (OOM)"
    echo "     → Reducir MAX_SEQ_LEN o TRAIN_BSZ"
    echo "  2. Archivo de datos corrupto"
    echo "     → Verificar formato JSON"
    echo "  3. Dependencias faltantes"
    echo "     → Ejecutar: pip install -r requirements.txt"
    echo ""
    echo -e "${YELLOW}📋 Ver logs en: $LOG_DIR${NC}"
    echo ""
    exit 1
fi