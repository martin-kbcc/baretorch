#!/bin/bash
set -e

# ==============================================================================
#      BareTorch Stage 3: GRPO Reasoning Alignment Launcher (Cloud 0.5B)
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
CHECKPOINT_DIR="${BASE_DIR}/checkpoints_500m_sft_cold_start"
OUTPUT_DIR="${BASE_DIR}/checkpoints_500m_grpo"

# Cloudflare R2 Configurations
R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"
R2_REMOTE_COLD_START_PATH="r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_sft_cold_start"

# Optional Resume Target (Leave empty "" to train from scratch)
RESUME_CHECKPOINT=""

RESUME_ARG=""
if [ -n "${RESUME_CHECKPOINT}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

# GPU Hardware Config
NUM_GPUS=8

# Training Hyperparameters
NUM_EPOCHS=1
BATCH_SIZE=8              # Prompts per GPU per step
NUM_GENERATIONS=8         # Group size (G=8): Candidate rollouts per prompt for stable advantage estimation
GRAD_ACCUM=1              # Effective Batch Size = 8 prompts/GPU * 1 accum * 8 GPUs = 64 prompts (512 total rollouts/step)
LEARNING_RATE=1e-6        # Policy learning rate
BETA=0.04                 # KL divergence penalty weight
CLIP_EPS=0.2              # PPO clipping epsilon
NUM_SAMPLES=0             # 0 = Train on full reasoning dataset (e.g., GSM8K / MATH)

# Sequence Lengths (Expanded context for multi-step CoT mathematical reasoning)
MAX_PROMPT_LEN=1024
MAX_COMPLETION_LEN=1024

# Validation & Checkpointing Config
VAL_RATIO=0.05
EVAL_STEPS=50
SAVE_STEPS=100

# Enable Gradient Checkpointing
GRAD_CKPT="--grad_checkpointing"

echo "================================================================================"
echo "🚀 LAUNCHING RULE-BASED RL (GRPO) REASONING ALIGNMENT ON 8x NVIDIA H100 SXM (80GB)"
echo "================================================================================"
echo "• Base Checkpoint    : ${CHECKPOINT_DIR}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "• Resuming From      : ${RESUME_CHECKPOINT}"
else
    echo "• Resuming From      : DISABLED (Training from scratch)"
fi
echo "• Output Directory   : ${OUTPUT_DIR}"
echo "• Active GPUs        : ${NUM_GPUS}x H100 (Distributed DDP)"
echo "• Group Size (G)     : ${NUM_GENERATIONS} rollouts per prompt"
echo "• Per-GPU Prompts    : ${BATCH_SIZE}"
echo "• Global Rollouts    : $((BATCH_SIZE * NUM_GENERATIONS * GRAD_ACCUM * NUM_GPUS)) rollouts/step"
echo "• Prompt / Comp Cap  : ${MAX_PROMPT_LEN} / ${MAX_COMPLETION_LEN} tokens"
echo "• Beta (KL) / Clip   : ${BETA} / ${CLIP_EPS}"
echo "• Val Ratio / Eval   : ${VAL_RATIO} / Every ${EVAL_STEPS} steps"
echo "• Save Interval      : Every ${SAVE_STEPS} steps"
echo "• Grad Checkpoint    : ENABLED"
echo "• Cloud Sync         : Cloudflare R2 (${R2_REMOTE_COLD_START_PATH})"
echo "================================================================================"

# Check for Stage 2.5 Cold-Start SFT checkpoint locally; if missing, pull down from Cloudflare R2
echo "🔍 Checking for Stage 2.5 Cold-Start SFT checkpoint locally at ${CHECKPOINT_DIR}..."
if [ ! -d "${CHECKPOINT_DIR}" ] || [ -z "$(ls -A "${CHECKPOINT_DIR}" 2>/dev/null)" ]; then
    echo "⚠️  Cold-Start SFT checkpoint missing at ${CHECKPOINT_DIR}."
    if command -v rclone &> /dev/null; then
        echo "📥 Downloading Cold-Start SFT weights from Cloudflare R2 (${R2_REMOTE_COLD_START_PATH})..."
        mkdir -p "${CHECKPOINT_DIR}"
        rclone copy "${R2_REMOTE_COLD_START_PATH}" "${CHECKPOINT_DIR}" --transfers 8
        echo "✅ Successfully restored Cold-Start SFT checkpoint from R2."
    else
        echo "❌ Error: rclone is not installed and local Cold-Start SFT weights were not found!"
        exit 1
    fi
else
    echo "✅ Found local Cold-Start SFT checkpoint at ${CHECKPOINT_DIR}."
fi

cd "${BASE_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    "${BASE_DIR}/train_grpo.py" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_generations "${NUM_GENERATIONS}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LEARNING_RATE}" \
    --beta "${BETA}" \
    --clip_eps "${CLIP_EPS}" \
    --num_samples "${NUM_SAMPLES}" \
    --val_ratio "${VAL_RATIO}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --max_prompt_len "${MAX_PROMPT_LEN}" \
    --max_completion_len "${MAX_COMPLETION_LEN}" \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}" \
    ${GRAD_CKPT} \
    ${RESUME_ARG}

echo "================================================================================"
echo "✅ Cloud GRPO Reasoning Alignment completed successfully!"
echo "================================================================================"