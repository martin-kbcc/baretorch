#!/bin/bash
set -e

# CUDA Memory Management & Distributed Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Hardware Setup (8x H100 SXM5 Cluster)
NUM_GPUS=8

# 2.5B Model Architecture
MODEL_TYPE="baretorch"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"
TOKENIZER_NAME="Qwen/Qwen3.5-9B"
D_MODEL=2048
NUM_HEADS=16
NUM_LAYERS=36
SEQ_LEN=2048
RANK=16

# Optimization (Full Production Batching for 80GB H100 VRAM)
PER_GPU_BATCH_SIZE=8    # 8 seqs per GPU x 8 GPUs = 64 seqs/step
GRAD_ACCUM=1            # Global batch size = 64 seqs/step (~131k tokens/step)
LEARNING_RATE=3e-4
SCHEDULER="wsd"
WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
MAX_STEPS=50000

# Distillation Hyperparameters
ALPHA_CE=0.5
ALPHA_KL=0.5
TEMPERATURE=1.0

# Paths
OUTPUT_DIR="${ROOT_DIR}/checkpoints_distill_2.5B_prod"
DATA_CACHE_DIR="${ROOT_DIR}/teacher_predictions"

LOGGING_STEPS=50
SAVE_STEPS=1000
EVAL_STEPS=1000

mkdir -p "$OUTPUT_DIR"

GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

echo "======================================================================"
echo "🚀 Launching Production BareTorch Distillation Engine (8x H100 SXM5 80GB)..."
echo "  ├─ Hardware Config   : ${NUM_GPUS}x NVIDIA H100 SXM5 (80GB)"
echo "  ├─ Tokenizer Name    : ${TOKENIZER_NAME}"
echo "  ├─ Batch Setup       : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} seqs/step"
echo "  ├─ Step Throughput   : ${TOKENS_PER_STEP} tokens/step (~131k tokens/step)"
echo "  ├─ Model Dimension   : d_model=${D_MODEL}, layers=${NUM_LAYERS}, heads=${NUM_HEADS}"
echo "======================================================================"

torchrun --nproc_per_node=${NUM_GPUS} "${ROOT_DIR}/train_distill.py" \
    --model_type ${MODEL_TYPE} \
    --layer_sequence ${LAYER_SEQUENCE} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --data_cache_dir ${DATA_CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --scheduler ${SCHEDULER} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --d_model ${D_MODEL} \
    --num_heads ${NUM_HEADS} \
    --num_layers ${NUM_LAYERS} \
    --seq_len ${SEQ_LEN} \
    --rank ${RANK} \
    --alpha_ce ${ALPHA_CE} \
    --alpha_kl ${ALPHA_KL} \
    --temperature ${TEMPERATURE} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --r2_sync