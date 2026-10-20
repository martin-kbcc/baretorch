#!/usr/bin/env bash
set -e

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Hardware Setup (8x H100 SXM5 80GB Cluster)
NUM_GPUS=8

# Model & Tokenizer Paths
TOKENIZER_NAME="Qwen/Qwen3.5-9B"

# Automatically resolve checkpoint-142000 or fallback to parent directory
if [ -d "${ROOT_DIR}/checkpoints_distill_2.5B_50BT_baretorch/checkpoint-142000" ]; then
    PRETRAINED_MODEL_PATH="${ROOT_DIR}/checkpoints_distill_2.5B_50BT_baretorch/checkpoint-142000"
elif [ -d "${ROOT_DIR}/checkpoint-142000" ]; then
    PRETRAINED_MODEL_PATH="${ROOT_DIR}/checkpoint-142000"
else
    PRETRAINED_MODEL_PATH="${ROOT_DIR}/checkpoints_distill_2.5B_50BT_baretorch"
fi

OUTPUT_DIR="${ROOT_DIR}/checkpoints_2.5B_sft"
R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints_2.5B_sft"

# Optimization & Batching for 80GB H100 VRAM + FSDP (ZeRO-2)
# Global Batch = 8 GPUs * 4 per-device batch * 4 grad accum = 128 sequences/step
# Token Throughput = 128 * 4096 = 524,288 tokens/step (~0.52M tokens/step)
PER_GPU_BATCH_SIZE=4
GRAD_ACCUM=4
SEQ_LEN=4096
ROPE_THETA=10000.0

LEARNING_RATE=2e-5
NUM_EPOCHS=1
WARMUP_STEPS=60
WEIGHT_DECAY=0.01

# 600,000 samples (~450M-500M packed tokens) hits the SFT sweet spot for 2.5B
# and completes ~900 steps in ~1.4 hours on 8x H100s with --compile enabled.
MAX_SAMPLES=600000

LOGGING_STEPS=10
SAVE_STEPS=200
EVAL_STEPS=200

mkdir -p "$OUTPUT_DIR"

GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

echo "======================================================================"
echo "🚀 Launching Production BareTorch Supervised Fine-Tuning (SFT) Engine"
echo "  ├─ Hardware Config   : ${NUM_GPUS}x NVIDIA H100 SXM5 (80GB)"
echo "  ├─ Pretrained Path   : ${PRETRAINED_MODEL_PATH}"
echo "  ├─ Output Directory  : ${OUTPUT_DIR}"
echo "  ├─ Context Window    : ${SEQ_LEN} tokens (RoPE Theta: ${ROPE_THETA})"
echo "  ├─ Dataset Target    : ${MAX_SAMPLES} conversations (~480M tokens)"
echo "  ├─ Batch Setup       : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} seqs/step"
echo "  ├─ Step Throughput   : ${TOKENS_PER_STEP} tokens/step (~524k tokens/step)"
echo "  ├─ Learning Rate     : ${LEARNING_RATE} | Warmup=${WARMUP_STEPS} steps | Epochs=${NUM_EPOCHS}"
echo "  ├─ Torch Compile     : Enabled (Targeted CS-LRAD Sub-Modules)"
echo "  └─ Cloud Sync        : Active -> Cloudflare R2 (${R2_BUCKET}/${R2_PREFIX})"
echo "======================================================================"

torchrun --nproc_per_node=${NUM_GPUS} "${ROOT_DIR}/train_sft.py" \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --seq_len ${SEQ_LEN} \
    --rope_theta ${ROPE_THETA} \
    --max_samples ${MAX_SAMPLES} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --compile \
    --grad_checkpointing \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}"