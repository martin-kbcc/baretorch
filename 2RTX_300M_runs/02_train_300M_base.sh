#!/usr/bin/env bash
set -e

echo "============================================================"
echo "Step 2: Launching 300M BareTorch Pre-Training (6B Tokens)"
echo "============================================================"

# Dual RTX 4090 DDP configuration:
# Global Batch Size = 2 GPUs * 16 per-device batch * 2048 sequence length = 65,536 tokens/step
# Total Steps for 2B Tokens = 2,000,000,000 / 65,536 ≈ 30,518 steps

torchrun --nproc_per_node=2 ../train_distill.py \
    --model_type baretorch \
    --d_model 768 \
    --num_heads 16 \
    --num_layers 12 \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --tokenizer_name "Qwen/Qwen3.5-9B" \
    --data_cache_dir "./data_6B" \
    --max_steps 30518 \
    --batch_size 2 \
    --grad_accum 8 \
    --learning_rate 1e-3 \
    --scheduler wsd \
    --warmup_steps 1000 \
    --decay_steps 3052 \
    --use_qk_norm \
    --tie_embeddings \
    --compile \
    --logging_steps 200 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --r2_sync \
    --r2_prefix "checkpoints_local" \
    --output_dir "./checkpoints_300M_2B"