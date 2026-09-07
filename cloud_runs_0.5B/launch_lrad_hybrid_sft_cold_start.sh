#!/bin/bash
set -e

# ==============================================================================
#      BareTorch Stage 2.5: Cold-Start CoT SFT Warmup Launcher (Local Dual 4090)
#            Scale Configuration: ~500M Hybrid on 2x NVIDIA RTX 4090 (24GB)
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

# ==============================================================================
#                               Hardware Config
# ==============================================================================
NUM_GPUS=2

# ==============================================================================
#                               Directory & Model Paths Config
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRETRAINED_CKPT="/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_sft/checkpoint-2759"
OUTPUT_DIR="${BASE_DIR}/checkpoints_500m_sft_cold_start"
TOKENIZER_NAME="HuggingFaceTB/SmolLM2-360M"

# Cloudflare R2 Configurations
R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"
R2_REMOTE_SFT_PATH="r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_sft/checkpoint-2759"

# ==============================================================================
#                     Hyperparameters (Adjusted for ~320 Steps)
# ==============================================================================
NUM_EPOCHS=3
PER_GPU_BATCH_SIZE=2
GRAD_ACCUM=2               # Global Batch = 2 GPUs * 2 batch * 2 accum = 8 sequences (8,192 tokens/step)
LEARNING_RATE=2e-5         # Slightly higher LR for smaller global batch size
WARMUP_STEPS=30
WEIGHT_DECAY=0.01
SEQ_LEN=1024              # Ideal block length for packed GSM8K ChatML traces

# ==============================================================================
#                               Startup Summary
# ==============================================================================
GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

echo "================================================================================"
echo "🚀 LAUNCHING COLD-START CoT SFT WARMUP ON ${NUM_GPUS}x NVIDIA RTX 4090 (24GB)"
echo "================================================================================"
echo "• Base Checkpoint   : ${PRETRAINED_CKPT}"
echo "• Tokenizer         : ${TOKENIZER_NAME}"
echo "• Output Directory  : ${OUTPUT_DIR}"
echo "• Active GPUs       : ${NUM_GPUS}x RTX 4090 (Distributed DDP)"
echo "• Context Length    : ${SEQ_LEN} tokens"
echo "• Num Epochs        : ${NUM_EPOCHS}"
echo "• Per-GPU Batch     : ${PER_GPU_BATCH_SIZE}"
echo "• Grad Accumulation : ${GRAD_ACCUM}"
echo "• Global Batch Size : ${GLOBAL_BATCH_SEQS} sequences/step (${TOKENS_PER_STEP} tokens/step)"
echo "• Learning Rate     : ${LEARNING_RATE}"
echo "• Sub-Module Compile: ENABLED"
echo "================================================================================"

# Check for base SFT checkpoint locally; if missing, pull down from Cloudflare R2
echo "🔍 Checking for base SFT checkpoint locally at ${PRETRAINED_CKPT}..."
if [ ! -d "${PRETRAINED_CKPT}" ] || [ -z "$(ls -A "${PRETRAINED_CKPT}" 2>/dev/null)" ]; then
    echo "⚠️  Base SFT checkpoint missing at ${PRETRAINED_CKPT}."
    if command -v rclone &> /dev/null; then
        echo "📥 Downloading SFT weights from Cloudflare R2 (${R2_REMOTE_SFT_PATH})..."
        mkdir -p "${PRETRAINED_CKPT}"
        rclone copy "${R2_REMOTE_SFT_PATH}" "${PRETRAINED_CKPT}" --transfers 8 --s3-provider Cloudflare
        echo "✅ Successfully restored SFT checkpoint from R2."
    else
        echo "❌ Error: rclone is not installed and local SFT weights were not found!"
        exit 1
    fi
else
    echo "✅ Found local SFT checkpoint at ${PRETRAINED_CKPT}."
fi

cd "${BASE_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    "${BASE_DIR}/train_sft_cold_start.py" \
    --pretrained_model_path "${PRETRAINED_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${PER_GPU_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --seq_len "${SEQ_LEN}" \
    --compile \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}"

echo "================================================================================"
echo "✅ Cold-Start CoT SFT Warmup Completed Successfully!"
echo "================================================================================"