#!/bin/bash
set -e

# ==============================================================================
#                 BareTorch Foundational Pre-Training Launcher
#                 Scale Configuration: 60 Billion Tokens (100M Hybrid)
# ==============================================================================

# CUDA Memory Management & CPU Thread Optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TORCH_CPP_MIN_LOG_LEVEL=2

echo "======================================================================"
echo "🚀 Launching BareTorch 60B Token Pre-Training (100M Hybrid)..."
echo "  ├─ Target Runway   : 60,000,000,000 Tokens (60B)"
echo "  ├─ Hardware Config : 2x RTX 4090 (Dual DDP)"
echo "  ├─ Batch Setup     : 2 GPUs x 8 batch x 4 accum = 64 seqs/step (131k tokens/step)"
echo "  ├─ Total Steps     : 457,764 steps"
echo "  ├─ Layer Sequence  : cs_lrad,cs_lrad,cs_lrad,transformer"
echo "  ├─ Checkpoint Freq : Every 5,000 steps (~0.65B tokens)"
echo "======================================================================"

# Run on 2x RTX 4090 GPUs (~100M Parameter Foundational Hybrid Model)
torchrun --nproc_per_node=2 train.py \
    --model_type baretorch \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --data_cache_dir "./tokenized_bin" \
    --output_dir "./checkpoints_100m_hybrid" \
    --max_val_samples 5000 \
    --max_steps 457764 \
    --learning_rate 6e-4 \
    --scheduler "cosine" \
    --warmup_steps 5000 \
    --weight_decay 0.1 \
    --batch_size 8 \
    --grad_accum 4 \
    --d_model 512 \
    --num_heads 16 \
    --num_layers 12 \
    --chunk_size 32 \
    --rank 8 \
    --dropout 0.0 \
    --seq_len 2048 \
    --compile \
    --grad_checkpointing \
    --logging_steps 500 \
    --save_steps 5000 \
    --eval_steps 5000