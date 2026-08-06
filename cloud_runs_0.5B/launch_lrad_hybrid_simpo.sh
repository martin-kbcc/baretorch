#!/bin/bash
set -e

# ==============================================================================
#            BareTorch Stage 2: SimPO Alignment Launcher (Cloud 0.5B)
#              Scale Configuration: ~500M Hybrid on 4x NVIDIA H100
# ==============================================================================

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

# ==============================================================================
#                               Hardware & Cluster Config
# ==============================================================================
NUM_GPUS=4

# ==============================================================================
#                        Directory & Model Paths Config
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${BASE_DIR}/checkpoints_500m_sft"
OUTPUT_DIR="${BASE_DIR}/checkpoints_500m_simpo"
TOKENIZER_NAME="HuggingFaceTB/SmolLM2-360M"

# Cloudflare R2 Configurations
R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"
R2_REMOTE_SFT_PATH="r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_sft"

# ==============================================================================
#                   Training Hyperparameters (Stage 2 SimPO)
# ==============================================================================
NUM_EPOCHS=1
PER_GPU_BATCH_SIZE=16       # Per-GPU batch size (chosen vs. rejected pairs)
GRAD_ACCUM=2              # Global batch size = 4 GPUs * 16 batch * 2 accum = 128 preference pairs/step
LEARNING_RATE=1e-5        # Optimal SimPO learning rate for 500M models
BETA=2.0                  # Implicit reward scale factor
GAMMA=0.8                 # Reward margin penalty
NUM_SAMPLES=0             # 0 = Train on full preference dataset (~60k pairs)
SEQ_LEN=2048              # Matched to native 2K context window

# Validation & Checkpointing Config
VAL_RATIO=0.05            # 5% split for validation
EVAL_STEPS=50             # Run evaluation every 50 steps
SAVE_STEPS=100            # Save checkpoint every 100 steps

# ==============================================================================
#                                Startup Summary
# ==============================================================================
GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))

echo "================================================================================"
echo "🚀 LAUNCHING SIMPO ALIGNMENT ON ${NUM_GPUS}x NVIDIA H100 SXM (80GB)"
echo "================================================================================"
echo "• Base SFT Checkpoint : ${CHECKPOINT_DIR}"
echo "• Tokenizer           : ${TOKENIZER_NAME}"
echo "• Output Directory    : ${OUTPUT_DIR}"
echo "• Active GPUs         : ${NUM_GPUS}x H100 (Distributed DDP)"
echo "• Sequence Length     : ${SEQ_LEN} tokens"
echo "• Epochs / Samples    : ${NUM_EPOCHS} / Full Dataset (${NUM_SAMPLES})"
echo "• Per-GPU Batch       : ${PER_GPU_BATCH_SIZE}"
echo "• Global Batch Size   : ${GLOBAL_BATCH_SEQS} preference pairs/step"
echo "• Learning Rate       : ${LEARNING_RATE}"
echo "• Beta / Gamma        : ${BETA} / ${GAMMA}"
echo "• Val Ratio / Eval    : ${VAL_RATIO} / Every ${EVAL_STEPS} steps"
echo "• Save Interval       : Every ${SAVE_STEPS} steps"
echo "• Grad Checkpointing  : ENABLED"
echo "• Sub-Module Compile  : ENABLED"
echo "• Cloud Sync          : Cloudflare R2 (${R2_REMOTE_SFT_PATH})"
echo "================================================================================"

# Check for SFT checkpoint locally; if missing, pull down from Cloudflare R2
echo "🔍 Checking for SFT checkpoint locally at ${CHECKPOINT_DIR}..."
if [ ! -d "${CHECKPOINT_DIR}" ] || [ -z "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
    echo "⚠️  SFT checkpoint missing at ${CHECKPOINT_DIR}."
    if command -v rclone &> /dev/null; then
        echo "📥 Downloading Stage 1 SFT weights from Cloudflare R2 (${R2_REMOTE_SFT_PATH})..."
        mkdir -p "${CHECKPOINT_DIR}"
        rclone copy "${R2_REMOTE_SFT_PATH}" "${CHECKPOINT_DIR}" --transfers 8
        echo "✅ Successfully restored SFT checkpoint from R2."
    else
        echo "❌ Error: rclone is not installed and local SFT weights were not found!"
        exit 1
    fi
else
    echo "✅ Found local SFT checkpoint at ${CHECKPOINT_DIR}."
fi

cd "${BASE_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    train_simpo.py \
    --pretrained_model_path "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${PER_GPU_BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LEARNING_RATE}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --num_samples "${NUM_SAMPLES}" \
    --val_ratio "${VAL_RATIO}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --seq_len "${SEQ_LEN}" \
    --compile \
    --grad_checkpointing \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}"

echo "================================================================================"
echo "✅ Cloud SimPO Alignment completed successfully!"
echo "================================================================================"