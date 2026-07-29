#!/bin/bash
set -e

# ==============================================================================
#                      BareTorch Stage 1: SFT Launcher
# ==============================================================================

# CUDA Memory Management to prevent VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCH_CPP_MIN_LOG_LEVEL=2

# Checkpoint & Output Configurations
PRETRAINED_CHECKPOINT="./checkpoints_100m_hybrid_baretorch/checkpoint-30000"
OUTPUT_DIR="./checkpoints_100m_sft"

# Hyperparameters
DATASET_NAME="HuggingFaceTB/smoltalk"
DATASET_CONFIG="all"
MAX_SAMPLES=100000        # Sub-sample count (set to 0 for full ~1M dataset)
BATCH_SIZE=12             # Increased per-GPU batch size thanks to gradient checkpointing
GRAD_ACCUM=2              # Effective global batch size = 16 * 2 * 2 GPUs = 64
LEARNING_RATE=5e-5
NUM_EPOCHS=1
SEQ_LEN=2048

echo "======================================================================"
echo "🚀 Launching BareTorch Stage 1: Supervised Fine-Tuning (SFT)..."
echo "  ├─ Pre-trained Model Checkpoint : ${PRETRAINED_CHECKPOINT}"
echo "  ├─ Target Output Directory     : ${OUTPUT_DIR}"
echo "  ├─ Dataset                    : ${DATASET_NAME} (${DATASET_CONFIG})"
echo "  ├─ Per-GPU Batch Size          : ${BATCH_SIZE}"
echo "  └─ Global Batch Size           : $(( BATCH_SIZE * GRAD_ACCUM * 2 ))"
echo "======================================================================"

# Execute DDP via torchrun across 2x RTX 4090 GPUs
torchrun --nproc_per_node=2 train_sft.py \
    --pretrained_model_path "${PRETRAINED_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --max_samples ${MAX_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --seq_len ${SEQ_LEN} \
    --grad_checkpointing