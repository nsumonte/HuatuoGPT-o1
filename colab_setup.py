"""
Script para ejecutar en Google Colab
Copia y pega cada sección en celdas separadas de Colab
"""

# ============================================================================
# CELDA 1: Clonar repositorio y cambiar directorio
# ============================================================================
"""
!git clone https://github.com/TU_USUARIO/HuatuoGPT-o1.git
%cd HuatuoGPT-o1
"""

# ============================================================================
# CELDA 2: Instalar dependencias
# ============================================================================
"""
!pip install -r requirements.txt
!nvidia-smi  # Verificar GPU
"""

# ============================================================================
# CELDA 3: Configurar credenciales (OPCIONAL)
# ============================================================================
"""
# Descomenta si necesitas WandB o HuggingFace
# !wandb login
# !huggingface-cli login
"""

# ============================================================================
# CELDA 4: Dar permisos y ejecutar script
# ============================================================================
"""
!chmod +x scripts/run_local_test.sh
!bash scripts/run_local_test.sh
"""

# ============================================================================
# ALTERNATIVA: Ejecutar manualmente sin script bash
# ============================================================================
"""
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
"""

