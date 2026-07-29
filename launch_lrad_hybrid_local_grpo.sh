#!/bin/bash
set -e

# ==============================================================================
# BareTorch Stage 3: GRPO Local Multi-GPU Alignment Launcher (Dual RTX 4090s)
# ==============================================================================

# Performance & Environment Optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCH_CPP_MIN_LOG_LEVEL=2

# Directory & Model Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_100m_sft_cold_start/checkpoint-444"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints_100m_grpo"

# Optional Resume Target (Leave empty "" to train from scratch)
# Example to resume: RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint-50"
RESUME_CHECKPOINT="${OUTPUT_DIR}/checkpoint-150"

RESUME_ARG=""
if [ -n "${RESUME_CHECKPOINT}" ]; then
    RESUME_ARG="--resume_from_checkpoint ${RESUME_CHECKPOINT}"
fi

# GPU Hardware Config
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# Training Hyperparameters
NUM_EPOCHS=1
BATCH_SIZE=4           # Prompts per GPU per step (with G=4, runs 16 rollouts per GPU)
NUM_GENERATIONS=4      # Group size G: candidate completions generated per prompt
GRAD_ACCUM=4           # Effective Batch Size = 4 * 4 * 2 GPUs = 32 prompts (128 rollouts)
LEARNING_RATE=1e-6
BETA=0.04              # KL penalty weight
CLIP_EPS=0.2           # PPO clipping epsilon
NUM_SAMPLES=0          # Sub-sample count for local test (set to 0 for full GSM8K)

# Sequence Lengths
MAX_PROMPT_LEN=512
MAX_COMPLETION_LEN=512

# Validation & Checkpointing Config
VAL_RATIO=0.05
EVAL_STEPS=25
SAVE_STEPS=50

# Enable Gradient Checkpointing
GRAD_CKPT="--grad_checkpointing"

echo "================================================================================"
echo "🚀 LAUNCHING RULE-BASED RL (GRPO) REASONING ALIGNMENT ON DUAL-RTX 4090s"
echo "================================================================================"
echo "• Base Checkpoint    : ${CHECKPOINT_DIR}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "• Resuming From      : ${RESUME_CHECKPOINT}"
else
    echo "• Resuming From      : DISABLED (Training from scratch)"
fi
echo "• Output Directory   : ${OUTPUT_DIR}"
echo "• Active GPUs        : ${NUM_GPUS} (Devices: ${CUDA_VISIBLE_DEVICES})"
echo "• Group Size (G)     : ${NUM_GENERATIONS} rollouts per prompt"
echo "• Per-GPU Prompts    : ${BATCH_SIZE} (Effective Prompt Batch: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)))"
echo "• Prompt / Comp Cap  : ${MAX_PROMPT_LEN} / ${MAX_COMPLETION_LEN} tokens"
echo "• Beta (KL) / Clip   : ${BETA} / ${CLIP_EPS}"
echo "• Val Ratio / Eval   : ${VAL_RATIO} / Every ${EVAL_STEPS} steps"
echo "• Save Interval      : Every ${SAVE_STEPS} steps"
echo "• Grad Checkpoint    : ENABLED"
echo "================================================================================"

cd "${SCRIPT_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    train_grpo.py \
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
    ${GRAD_CKPT} \
    ${RESUME_ARG}

echo "================================================================================"
echo "✅ Local GRPO Reasoning Alignment completed successfully!"
echo "================================================================================"