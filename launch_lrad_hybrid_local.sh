#!/bin/bash
set -e

# Run on 2 RTX 4090 GPUs (Distributed)
torchrun --nproc_per_node=2 train.py \
    --model_type baretorch \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --max_steps 10000 \
    --learning_rate 3e-4 \
    --scheduler "cosine" \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --d_model 256 \
    --num_heads 4 \
    --num_layers 4 \
    --chunk_size 32 \
    --rank 8 \
    --dropout 0.1 \
    --seq_len 1024 \
    --batch_size 16 \
    --compile \
    --grad_checkpointing \
    --logging_steps 2000 \
    --save_steps 2000 \
    --eval_steps 2000