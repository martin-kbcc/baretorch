#!/bin/bash
set -e

# ==============================================================================
# BareTorch Stage 2: SimPO Local Multi-GPU Alignment Launcher (Dual RTX 4090s)
# ==============================================================================

# Performance & Environment Optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCH_CPP_MIN_LOG_LEVEL=2

# Directory & Model Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_100m_sft/checkpoint-1980"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints_100m_simpo"

# GPU Hardware Config
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# Training Hyperparameters
NUM_EPOCHS=1
BATCH_SIZE=8           # Per-GPU batch size
GRAD_ACCUM=2           # Effective Batch Size = 8 * 2 * 2 GPUs = 32
LEARNING_RATE=5e-6
BETA=2.0
GAMMA=0.8
NUM_SAMPLES=10000      # Set to 0 to train on full dataset (~60k)
SEQ_LEN=2048

# Validation & Checkpointing Config
VAL_RATIO=0.05         # 5% split for validation
EVAL_STEPS=50          # Run validation evaluation every N steps (use 500 for full dataset)
SAVE_STEPS=100         # Save step checkpoint every N steps (use 500 for full dataset)

# Enable Gradient Checkpointing
GRAD_CKPT="--grad_checkpointing"

echo "================================================================================"
echo "🚀 LAUNCHING SIMPO ALIGNMENT ON LOCAL DUAL-RTX 4090s"
echo "================================================================================"
echo "• Base Checkpoint  : ${CHECKPOINT_DIR}"
echo "• Output Directory : ${OUTPUT_DIR}"
echo "• Active GPUs      : ${NUM_GPUS} (Devices: ${CUDA_VISIBLE_DEVICES})"
echo "• Epochs / Samples : ${NUM_EPOCHS} / ${NUM_SAMPLES}"
echo "• Per-GPU Batch    : ${BATCH_SIZE} (Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS)))"
echo "• Sequence Length  : ${SEQ_LEN}"
echo "• Beta / Gamma     : ${BETA} / ${GAMMA}"
echo "• Val Ratio / Eval : ${VAL_RATIO} / Every ${EVAL_STEPS} steps"
echo "• Save Interval    : Every ${SAVE_STEPS} steps"
echo "• Grad Checkpoint  : ENABLED"
echo "================================================================================"

cd "${SCRIPT_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    train_simpo.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --lr "${LEARNING_RATE}" \
    --beta "${BETA}" \
    --gamma "${GAMMA}" \
    --num_samples "${NUM_SAMPLES}" \
    --val_ratio "${VAL_RATIO}" \
    --eval_steps "${EVAL_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --seq_len "${SEQ_LEN}" \
    ${GRAD_CKPT}

echo "================================================================================"
echo "✅ Local SimPO Alignment completed successfully!"
echo "================================================================================"