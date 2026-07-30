#!/bin/bash
set -e

# ==============================================================================
#      BareTorch Stage 2.5: Cold-Start CoT SFT Warmup Launcher (Cloud 0.5B)
#              Scale Configuration: ~500M Hybrid on 8x NVIDIA H100
# ==============================================================================

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

# Directory & Model Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAINED_CKPT="${BASE_DIR}/checkpoints_500m_simpo/checkpoint-simpo-final"
OUTPUT_DIR="${BASE_DIR}/checkpoints_500m_sft_cold_start"

# Cloudflare R2 Configurations
R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"
R2_REMOTE_SIMPO_PATH="r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_simpo/checkpoint-simpo-final"

# GPU Hardware Config
NUM_GPUS=8

# Hyperparameters
NUM_EPOCHS=2
BATCH_SIZE=16             # Per-GPU batch size
GRAD_ACCUM=1              # Effective Batch Size = 16 * 1 * 8 GPUs = 128
LEARNING_RATE=2e-5
SEQ_LEN=2048              # Extended context for complex multi-step CoT reasoning traces

echo "================================================================================"
echo "🚀 LAUNCHING COLD-START CoT SFT WARMUP ON 8x NVIDIA H100 SXM (80GB)"
echo "================================================================================"
echo "• Base Checkpoint  : ${PRETRAINED_CKPT}"
echo "• Output Directory : ${OUTPUT_DIR}"
echo "• Active GPUs      : ${NUM_GPUS}x H100 (Distributed DDP)"
echo "• Context Length   : ${SEQ_LEN} tokens"
echo "• Num Epochs       : ${NUM_EPOCHS}"
echo "• Per-GPU Batch    : ${BATCH_SIZE}"
echo "• Global Batch Size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)) sequences/step"
echo "• Learning Rate    : ${LEARNING_RATE}"
echo "• Grad Checkpoint  : ENABLED"
echo "• Cloud Sync       : Cloudflare R2 (${R2_REMOTE_SIMPO_PATH})"
echo "================================================================================"

# Check for SimPO checkpoint locally; if missing, pull down from Cloudflare R2
echo "🔍 Checking for Stage 2 SimPO checkpoint locally at ${PRETRAINED_CKPT}..."
if [ ! -d "${PRETRAINED_CKPT}" ] || [ -z "$(ls -A "${PRETRAINED_CKPT}" 2>/dev/null)" ]; then
    echo "⚠️  SimPO checkpoint missing at ${PRETRAINED_CKPT}."
    if command -v rclone &> /dev/null; then
        echo "📥 Downloading SimPO weights from Cloudflare R2 (${R2_REMOTE_SIMPO_PATH})..."
        mkdir -p "${PRETRAINED_CKPT}"
        rclone copy "${R2_REMOTE_SIMPO_PATH}" "${PRETRAINED_CKPT}" --transfers 8
        echo "✅ Successfully restored SimPO checkpoint from R2."
    else
        echo "❌ Error: rclone is not installed and local SimPO weights were not found!"
        exit 1
    fi
else
    echo "✅ Found local SimPO checkpoint at ${PRETRAINED_CKPT}."
fi

cd "${BASE_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    "${BASE_DIR}/train_sft_cold_start.py" \
    --pretrained_model_path "${PRETRAINED_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --seq_len "${SEQ_LEN}" \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}" \
    --grad_checkpointing

echo "================================================================================"
echo "✅ Cold-Start CoT SFT Warmup Completed Successfully!"
echo "================================================================================"