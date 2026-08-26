#!/bin/bash
# /home/martinkb/Desktop/BareTorch_F/cloud_runs_0.5B/launch_lrad_hybrid_sft.sh
set -e

# ==============================================================================
#                  BareTorch Stage 1: SFT Launcher (Local Dual RTX 4090)
#          Scale Configuration: ~500M Hybrid on 2x NVIDIA RTX 4090 (24GB)
# ==============================================================================

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

# ==============================================================================
#                                Hardware Config
# ==============================================================================
NUM_GPUS=2

# ==============================================================================
#                        Checkpoint & Cloud Sync Config
# ==============================================================================
PRETRAINED_CHECKPOINT="./checkpoints_500m_hybrid_baretorch/checkpoint-190735"
OUTPUT_DIR="./checkpoints_500m_sft"
TOKENIZER_NAME="HuggingFaceTB/SmolLM2-360M"

R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"
R2_REMOTE_PATH="r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_hybrid_baretorch/checkpoint-190735"

# ==============================================================================
#                   Dataset & Hyperparameters (Stage 1 SFT)
# ==============================================================================
DATASET_NAME="HuggingFaceTB/smol-smoltalk"
DATASET_CONFIG="default"
MAX_SAMPLES=0             # 0 = Use full dataset (~485k samples)

PER_GPU_BATCH_SIZE=2      # Per-GPU batch size (Optimal for 24GB VRAM at 2K seq_len)
GRAD_ACCUM=16             # Global batch size = 2 GPUs * 2 batch * 16 accum = 64 sequences
LEARNING_RATE=1e-5        # Optimal LR for 500M SFT to maximize instruction adherence
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
echo "  ├─ Tokenizer              : ${TOKENIZER_NAME}"
echo "  ├─ Target Output Dir      : ${OUTPUT_DIR}"
echo "  ├─ Hardware Config        : ${NUM_GPUS}x NVIDIA RTX 4090 (24GB)"
echo "  ├─ Dataset                : ${DATASET_NAME} (${DATASET_CONFIG})"
echo "  ├─ Context Length         : ${SEQ_LEN} tokens"
echo "  ├─ Learning Rate          : ${LEARNING_RATE}"
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

# Execute DDP via torchrun across 2x RTX 4090 GPUs
torchrun --nproc_per_node=${NUM_GPUS} train_sft.py \
    --pretrained_model_path "${PRETRAINED_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
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
    --compile