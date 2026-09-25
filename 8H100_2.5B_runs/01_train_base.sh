#!/usr/bin/env bash
set -e

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Hardware Setup (8x H100 SXM5 80GB Cluster)
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

# Optimization & Batching for 80GB H100 VRAM
# Global Batch = 8 GPUs * 4 per-device batch * 8 grad accum = 256 sequences/step
# Token Throughput = 256 * 2048 = 524,288 tokens/step (~0.52M tokens/step)
PER_GPU_BATCH_SIZE=4
GRAD_ACCUM=8
LEARNING_RATE=5e-4
SCHEDULER="wsd"
WARMUP_STEPS=2000

# 100 Billion Tokens Target Calculation:
# Total Steps = 100,000,000,000 / 524,288 ≈ 190,735 steps
MAX_STEPS=190735
# 15% High-Density WSD Annealing Phase (~15.0 Billion Tokens)
DECAY_STEPS=28610     
WEIGHT_DECAY=0.1

# Distillation Hyperparameters
ALPHA_CE=0.5
ALPHA_KL=0.5
TEMPERATURE=1.0

# Output Paths & Cloud Persistence
OUTPUT_DIR="${ROOT_DIR}/checkpoints_distill_2.5B_100BT"
DATA_CACHE_DIR="${ROOT_DIR}/teacher_predictions"
R2_PREFIX="checkpoints_2.5B_100BT"

LOGGING_STEPS=250
SAVE_STEPS=5000       # Checkpoint and sync to Cloudflare R2 every ~2.62B tokens
EVAL_STEPS=5000

mkdir -p "$OUTPUT_DIR"

GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

echo "======================================================================"
echo "🚀 Launching Production BareTorch Distillation Engine (8x H100 SXM5 80GB)"
echo "  ├─ Hardware Config   : ${NUM_GPUS}x NVIDIA H100 SXM5 (80GB)"
echo "  ├─ Model Dimension   : d_model=${D_MODEL}, layers=${NUM_LAYERS}, heads=${NUM_HEADS} (2.5B Params)"
echo "  ├─ Batch Setup       : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} seqs/step"
echo "  ├─ Step Throughput   : ${TOKENS_PER_STEP} tokens/step (~524k tokens/step)"
echo "  ├─ Target Dataset    : 100 Billion Tokens"
echo "  ├─ Total Steps       : ${MAX_STEPS} steps"
echo "  ├─ WSD Annealing     : Warmup=${WARMUP_STEPS} steps | Decay=${DECAY_STEPS} steps"
echo "  └─ Cloud Sync        : Active -> Cloudflare R2 (${R2_PREFIX})"
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
    --decay_steps ${DECAY_STEPS} \
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
    --use_qk_norm \
    --tie_embeddings \
    --compile \
    --grad_checkpointing \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --r2_sync \
    --r2_prefix ${R2_PREFIX}