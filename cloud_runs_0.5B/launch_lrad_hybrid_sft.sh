#!/bin/bash
set -e

# ==============================================================================
#                 BareTorch Stage 1: SFT Launcher (Cloud 0.5B)
#             Scale Configuration: ~500M Hybrid on 4x NVIDIA H100
# ==============================================================================

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

# ==============================================================================
#                             Hardware & Cluster Config
# ==============================================================================
NUM_GPUS=4

# ==============================================================================
#                        Checkpoint & Cloud Sync Config
# ==============================================================================
PRETRAINED_CHECKPOINT="./checkpoints_500m_hybrid_baretorch/checkpoint-47683"
OUTPUT_DIR="./checkpoints_500m_sft"

R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"
R2_REMOTE_PATH="r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_hybrid_baretorch/checkpoint-47683"

# ==============================================================================
#                   Dataset & Hyperparameters (Stage 1 SFT)
# ==============================================================================
DATASET_NAME="HuggingFaceTB/smoltalk"
DATASET_CONFIG="all"
MAX_SAMPLES=0             # 0 = Use full dataset (~1M samples)

PER_GPU_BATCH_SIZE=16     # Per-GPU batch size
GRAD_ACCUM=2              # Global batch size = 4 GPUs * 16 batch * 2 accum = 128 sequences
LEARNING_RATE=5e-5        # Optimal SFT learning rate for 500M parameter models
WARMUP_STEPS=100
WEIGHT_DECAY=0.01
NUM_EPOCHS=1
SEQ_LEN=2048              # Matched to native 2K pre-training context window

# ==============================================================================
#                                Startup Summary
# ==============================================================================
GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

echo "======================================================================"
echo "🚀 Launching BareTorch Stage 1: Supervised Fine-Tuning (SFT)..."
echo "  ├─ Pre-trained Checkpoint : ${PRETRAINED_CHECKPOINT}"
echo "  ├─ Target Output Dir      : ${OUTPUT_DIR}"
echo "  ├─ Hardware Config        : ${NUM_GPUS}x NVIDIA H100 SXM (80GB)"
echo "  ├─ Dataset                : ${DATASET_NAME} (${DATASET_CONFIG})"
echo "  ├─ Context Length         : ${SEQ_LEN} tokens"
echo "  ├─ Per-GPU Batch Size     : ${PER_GPU_BATCH_SIZE}"
echo "  └─ Global Batch Size      : ${GLOBAL_BATCH_SEQS} seqs/step (${TOKENS_PER_STEP} tokens/step)"
echo "======================================================================"

# Check for pretrained checkpoint locally; if missing, pull down from Cloudflare R2
echo "🔍 Checking for pre-trained checkpoint locally..."
if [ ! -d "${PRETRAINED_CHECKPOINT}" ] || [ -z "$(ls -A "${PRETRAINED_CHECKPOINT}" 2>/dev/null)" ]; then
    echo "⚠️  Pre-trained checkpoint missing at ${PRETRAINED_CHECKPOINT}."
    if command -v rclone &> /dev/null; then
        echo "📥 Downloading pre-trained weights from Cloudflare R2 (${R2_REMOTE_PATH})..."
        mkdir -p "${PRETRAINED_CHECKPOINT}"
        rclone copy "${R2_REMOTE_PATH}" "${PRETRAINED_CHECKPOINT}" --transfers 8
        echo "✅ Successfully restored pre-trained checkpoint from R2."
    else
        echo "❌ Error: rclone is not installed and local pre-trained weights were not found!"
        exit 1
    fi
else
    echo "✅ Found local pre-trained checkpoint at ${PRETRAINED_CHECKPOINT}."
fi

# Execute DDP via torchrun across 4x H100 GPUs
torchrun --nproc_per_node=${NUM_GPUS} train_sft.py \
    --pretrained_model_path "${PRETRAINED_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --max_samples ${MAX_SAMPLES} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_epochs ${NUM_EPOCHS} \
    --seq_len ${SEQ_LEN} \
    --compile \
    --grad_checkpointing \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}"