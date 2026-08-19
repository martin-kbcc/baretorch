#!/bin/bash
set -e

# ==============================================================================
#                 BareTorch Foundational Pre-Training Launcher
#           Scale Configuration: ~500M Hybrid on 4x NVIDIA H100
# ==============================================================================

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

# ==============================================================================
#                                Hardware & Cluster Config
# ==============================================================================
NUM_GPUS=4

# ==============================================================================
#                               Model Architecture Config
# ==============================================================================
MODEL_TYPE="baretorch"
LAYER_SEQUENCE="cs_lrad,cs_lrad,cs_lrad,transformer"
TOKENIZER_NAME="HuggingFaceTB/SmolLM2-360M"
D_MODEL=1152
NUM_HEADS=16
NUM_LAYERS=24
CHUNK_SIZE=32
RANK=8
DROPOUT=0.0
SEQ_LEN=2048

# ==============================================================================
#           Optimization & Hyperparameters (3 Epochs / ~756B Token Runway)
# ==============================================================================
PER_GPU_BATCH_SIZE=64
GRAD_ACCUM=4           # Set to 4 so global batch size remains 1024 seqs/step on 4 GPUs
LEARNING_RATE=6e-4     # Optimal peak LR for ~500M params across multi-epoch run
SCHEDULER="cosine"
WARMUP_STEPS=2000      # 2,000 steps warmup (~4.2B tokens) for longer runway stability
WEIGHT_DECAY=0.1
MAX_STEPS=360489       # (252B tokens * 3 epochs) / (4 GPUs * 64 batch * 4 accum * 2048 seq_len)
MAX_VAL_SAMPLES=5000

# ==============================================================================
#                               Paths & Cloud Sync Config
# ==============================================================================
OUTPUT_DIR="./checkpoints_500m_hybrid_baretorch"
DATA_CACHE_DIR="./tokenized_bin"

R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"

LOGGING_STEPS=1000
SAVE_STEPS=10000       # Checkpoint every ~20.97B tokens
EVAL_STEPS=10000

# ==============================================================================
#                                Startup Summary
# ==============================================================================
GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))
TOKENS_PER_STEP=$((GLOBAL_BATCH_SEQS * SEQ_LEN))

echo "======================================================================"
echo "🚀 Launching BareTorch Cloud Pre-Training (500M 3:1 Hybrid)..."
echo "  ├─ Tokenizer        : ${TOKENIZER_NAME}"
echo "  ├─ Target Runway    : 3 Epochs (~756.0 Billion Tokens)"
echo "  ├─ Total Steps      : ${MAX_STEPS} steps"
echo "  ├─ Hardware Config  : ${NUM_GPUS}x NVIDIA H100 SXM (80GB)"
echo "  ├─ Context Length   : ${SEQ_LEN} tokens"
echo "  ├─ Batch Setup      : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} seqs/step"
echo "  ├─ Step Throughput  : ${TOKENS_PER_STEP} tokens/step (~2.10M tokens/step)"
echo "  ├─ Peak Learning Rate: ${LEARNING_RATE}"
echo "  ├─ Architecture     : ${NUM_LAYERS} Layers (d_model=${D_MODEL}, heads=${NUM_HEADS}, rank=${RANK})"
echo "  ├─ Layer Sequence   : ${LAYER_SEQUENCE}"
echo "  ├─ Checkpoint Freq  : Every ${SAVE_STEPS} steps (~20.97B tokens)"
echo "  ├─ Cloud Sync       : Cloudflare R2 (r2:${R2_BUCKET}/${R2_PREFIX})"
echo "======================================================================"

# Check Cloudflare R2 for existing checkpoints to enable auto-resume on fresh nodes
if command -v rclone &> /dev/null; then
    echo "🔍 Checking Cloudflare R2 for existing checkpoints..."
    mkdir -p "$OUTPUT_DIR"
    rclone copy "r2:${R2_BUCKET}/${R2_PREFIX}/$(basename "$OUTPUT_DIR")" "$OUTPUT_DIR" --transfers 8 || true
    
    LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CKPT" ]; then
        echo "✅ Restored existing checkpoint: $(basename "$LATEST_CKPT")"
    else
        echo "ℹ️  No remote checkpoints found. Starting fresh run."
    fi
else
    echo "⚠️  rclone not found. Skipping remote checkpoint restore check."
fi

# Run across 4x H100 GPUs via torchrun (~500M Parameter Foundational Model)
torchrun --nproc_per_node=${NUM_GPUS} train.py \
    --model_type ${MODEL_TYPE} \
    --layer_sequence ${LAYER_SEQUENCE} \
    --tokenizer_name ${TOKENIZER_NAME} \
    --data_cache_dir ${DATA_CACHE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_val_samples ${MAX_VAL_SAMPLES} \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --scheduler ${SCHEDULER} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay ${WEIGHT_DECAY} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --d_model ${D_MODEL} \
    --num_heads ${NUM_HEADS} \
    --num_layers ${NUM_LAYERS} \
    --chunk_size ${CHUNK_SIZE} \
    --rank ${RANK} \
    --dropout ${DROPOUT} \
    --seq_len ${SEQ_LEN} \
    --compile \
    --grad_checkpointing \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --r2_sync \
    --r2_bucket "${R2_BUCKET}" \
    --r2_prefix "${R2_PREFIX}"