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

# Hardware Setup (Dual-RTX Workstation)
NUM_GPUS=2

# Checkpoint Paths
PRETRAINED_MODEL_PATH="${SCRIPT_DIR}/checkpoints_300M_2B_baretorch/checkpoint-30518"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints_300M_sft"
TOKENIZER_NAME="Qwen/Qwen3.5-9B"

# SFT Dataset & Batching Parameters
# 300,000 samples (~600M packed tokens) for optimal 300M model alignment
MAX_SAMPLES=300000
NUM_EPOCHS=1
SEQ_LEN=2048

PER_GPU_BATCH_SIZE=2
GRAD_ACCUM=8
LEARNING_RATE=3e-5
WARMUP_STEPS=100
WEIGHT_DECAY=0.01

# Cloud Storage & Sync
R2_PREFIX="checkpoints_local_sft"

GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "🚀 Launching BareTorch 300M Stage 1 SFT Engine (Dual-GPU Workstation)"
echo "  ├─ Foundation Checkpoint : ${PRETRAINED_MODEL_PATH}"
echo "  ├─ Output Directory      : ${OUTPUT_DIR}"
echo "  ├─ Hardware Setup        : ${NUM_GPUS}x NVIDIA GPUs"
echo "  ├─ Batch Setup           : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} seqs/step"
echo "  ├─ Step Throughput       : ${TOKENS_PER_STEP} tokens/step (~65k tokens/step)"
echo "  ├─ SFT Dataset Mix       : 60% Chat / 20% Code / 20% Math"
echo "  ├─ Dataset Sample Limit  : ${MAX_SAMPLES} sequences (~600M tokens)"
echo "  └─ Cloud Sync            : Active -> Cloudflare R2 (${R2_PREFIX})"
echo "======================================================================"

torchrun --nproc_per_node=${NUM_GPUS} "${ROOT_DIR}/train_sft.py" \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --max_samples ${MAX_SAMPLES} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --seq_len ${SEQ_LEN} \
    --compile \
    --r2_sync \
    --r2_prefix "${R2_PREFIX}"