#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Step 2: Launching 600M BareTorch Pre-Training (10B Tokens)"
echo "============================================================"

# Dual RTX 4090 DDP + 8-Bit AdamW + BF16 configuration:
# Global Batch Size = 2 GPUs * 4 per-device batch * 4 grad accum * 2048 sequence length = 65,536 tokens/step
# Total Steps for 10B Tokens = 10,000,000,000 / 65,536 ≈ 152,588 steps

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=2 ../train_distill_ddp.py \
    --model_type baretorch \
    --d_model 1024 \
    --num_heads 16 \
    --num_kv_heads 4 \
    --num_layers 24 \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --tokenizer_name "Qwen/Qwen3.5-9B" \
    --data_cache_dir "./data_10B" \
    --max_steps 152588 \
    --batch_size 2 \
    --grad_accum 8 \
    --learning_rate 1e-3 \
    --scheduler wsd \
    --warmup_steps 2000 \
    --decay_steps 15259 \
    --use_qk_norm \
    --tie_embeddings \
    --compile \
    --logging_steps 200 \
    --save_steps 2500 \
    --eval_steps 2500 \
    --r2_sync \
    --r2_prefix "checkpoints_local_600M" \
    --output_dir "./checkpoints_600M_10B"