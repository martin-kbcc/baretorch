#!/bin/bash
set -e

# ==============================================================================
#                  BareTorch Foundational Pre-Training Launcher
#           Scale Configuration: ~500M Hybrid on 8x NVIDIA H100
# ==============================================================================

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN

OUTPUT_DIR="./checkpoints_500m_hybrid_baretorch"
R2_BUCKET="baretorch-data"
R2_PREFIX="checkpoints"

echo "======================================================================"
echo "🚀 Launching BareTorch Cloud Pre-Training (500M 3:1 Hybrid)..."
echo "  ├─ Target Runway   : ~52.4 Billion Tokens (100,000 steps)"
echo "  ├─ Hardware Config : 8x NVIDIA H100 SXM (80GB)"
echo "  ├─ Context Length  : 4,096 tokens (Native 4K Window)"
echo "  ├─ Batch Setup     : 8 GPUs x 16 batch x 1 accum = 128 seqs/step"
echo "  ├─ Step Throughput : 524,288 tokens/step (~0.52M tokens/step)"
echo "  ├─ Architecture    : 24 Layers (d_model=1536, heads=16, rank=8)"
echo "  ├─ Layer Sequence  : cs_lrad,cs_lrad,cs_lrad,transformer"
echo "  ├─ Checkpoint Freq : Every 2,000 steps (~1.05B tokens)"
echo "  ├─ Cloud Sync      : Cloudflare R2 (r2:${R2_BUCKET}/${R2_PREFIX})"
echo "======================================================================"

# Check Cloudflare R2 for existing checkpoints to enable auto-resume on fresh nodes
if command -v rclone &> /dev/null; then
    echo "🔍 Checking Cloudflare R2 for existing checkpoints..."
    mkdir -p "$OUTPUT_DIR"
    rclone copy "r2:${R2_BUCKET}/${R2_PREFIX}/checkpoints_500m_hybrid_baretorch" "$OUTPUT_DIR" --transfers 8 || true
    
    LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CKPT" ]; then
        echo "✅ Restored existing checkpoint: $(basename $LATEST_CKPT)"
    else
        echo "ℹ️  No remote checkpoints found. Starting fresh run."
    fi
else
    echo "⚠️  rclone not found. Skipping remote checkpoint restore check."
fi

# Run across 8x H100 GPUs via torchrun (~500M Parameter Foundational Model)
torchrun --nproc_per_node=8 train.py \
    --model_type baretorch \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --data_cache_dir "./tokenized_bin" \
    --output_dir "./checkpoints_500m_hybrid" \
    --max_val_samples 5000 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --scheduler "cosine" \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --batch_size 16 \
    --grad_accum 1 \
    --d_model 1536 \
    --num_heads 16 \
    --num_layers 24 \
    --chunk_size 32 \
    --rank 8 \
    --dropout 0.0 \
    --seq_len 4096 \
    --compile \
    --grad_checkpointing \
    --logging_steps 200 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --r2_sync \
    --r2_bucket "$R2_BUCKET" \
    --r2_prefix "$R2_PREFIX"