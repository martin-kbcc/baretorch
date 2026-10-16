#!/bin/bash
set -e

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_CHECKPOINT="/data/baretorch/checkpoints_distill_2.5B_50BT_baretorch/checkpoint-143052"
DATA_DIR="/data/baretorch/teacher_predictions"
OUTPUT_DIR="/data/baretorch/checkpoints_lca_2.5B_32k"

torchrun --nproc_per_node=8 /home/martinkb/Desktop/BareTorch_F/train_lca.py \
    --base_checkpoint "$BASE_CHECKPOINT" \
    --data_cache_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type baretorch \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --seq_len 32768 \
    --rope_theta 1000000.0 \
    --batch_size 1 \
    --grad_accum 4 \
    --learning_rate 8e-5 \
    --scheduler wsd \
    --warmup_steps 200 \
    --decay_steps 1500 \
    --max_steps 9536 \
    --weight_decay 0.1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --r2_sync