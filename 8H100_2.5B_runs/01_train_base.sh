#!/usr/bin/env bash
set -e

# Increase system open file descriptor limit for multi-worker memmaps
ulimit -n 65536

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Hardware Setup (8x H100 SXM5 80GB Cluster)
NUM_GPUS=8

# BareTorch Model Architecture (Adjusted to 2.5B Untied)
MODEL_TYPE="baretorch"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"
TOKENIZER_NAME="Qwen/Qwen3.5-9B"
D_MODEL=2048
NUM_HEADS=16
NUM_LAYERS=24
SEQ_LEN=2048
RANK=16

# Optimization & Batching for 80GB H100 VRAM + FSDP (ZeRO-2)
# Global Batch = 8 GPUs * 4 per-device batch * 8 grad accum = 256 sequences/step
# Token Throughput = 256 * 2048 = 524,288 tokens/step (~0.524M tokens/step)
PER_GPU_BATCH_SIZE=4
GRAD_ACCUM=8
LEARNING_RATE=5e-4
SCHEDULER="wsd"
WARMUP_STEPS=1000

# ==============================================================================
# 75 Billion Tokens Target Calculation:
# Total Steps = 75,000,000,000 / 524,288 = 143,052 steps
# 10% WSD Annealing Phase = 7.5 Billion Tokens / 524,288 = 14,305 steps
# Stable Phase ends at Step: 143,052 - 14,305 = 128,747 steps
# ==============================================================================
MAX_STEPS=143052
DECAY_STEPS=14305
WEIGHT_DECAY=0.1

# Distillation Hyperparameters (50/50 Multi-Task Objective)
ALPHA_CE=0.5
ALPHA_KL=0.5
TEMPERATURE=1.0

# Output Paths & Cloud Persistence
# Retaining the same directory ensures seamless resumption from checkpoint-80000
OUTPUT_DIR="${ROOT_DIR}/checkpoints_distill_2.5B_50BT"
DATA_CACHE_DIR="${ROOT_DIR}/teacher_predictions"
R2_PREFIX="checkpoints_2.5B_75BT"

LOGGING_STEPS=250
SAVE_STEPS=2000      # Checkpoint and sync to Cloudflare R2 every ~1.04B tokens
EVAL_STEPS=2000

mkdir -p "$OUTPUT_DIR"

GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))
STABLE_END_STEP=$((MAX_STEPS - DECAY_STEPS))

echo "======================================================================"
echo "🚀 Launching Production BareTorch Flagship Engine (8x H100 SXM5 80GB)"
echo "  ├─ Hardware Config   : ${NUM_GPUS}x NVIDIA H100 SXM5 (80GB)"
echo "  ├─ Model Dimension   : d_model=${D_MODEL}, layers=${NUM_LAYERS}, heads=${NUM_HEADS} (~2.53B Untied)"
echo "  ├─ Batch Setup       : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} seqs/step"
echo "  ├─ Step Throughput   : ${TOKENS_PER_STEP} tokens/step (~524k tokens/step)"
echo "  ├─ Target Dataset    : 75 Billion Tokens"
echo "  ├─ Total Steps       : ${MAX_STEPS} steps"
echo "  ├─ Stable Phase Ends : Step ${STABLE_END_STEP} (~67.5B tokens)"
echo "  ├─ WSD Annealing     : Warmup=${WARMUP_STEPS} steps | Decay=${DECAY_STEPS} steps (Last 10% / 7.5BT)"
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
    --compile \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --r2_sync \
    --r2_prefix ${R2_PREFIX}