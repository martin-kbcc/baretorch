#!/bin/bash
# =====================================================================
#   BareTorch 500M Symmetric Hybrid Cluster-Scale Training Launcher
#   Hardware Target: 8x NVIDIA H100 SXM5 (80GB VRAM)
# =====================================================================
set -e

# Setup distributed master node configurations (standard for multi-GPU nodes)
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"
export OMP_NUM_THREADS=1

# Optimal cluster parameters
MODEL_TYPE="baretorch"
LAYER_SEQ="cs_lrad,cs_lrad,cs_lrad,transformer" # Your 75/25 Champion Sequence

# Scaling Laws Step Configurations
# For Chinchilla-Optimal (10B tokens): Set to 9536 steps, warmup 1000
# For Full DCLM-100BT Pre-training (100B tokens): Set to 95367 steps, warmup 2000
MAX_STEPS=95367
WARMUP_STEPS=2000

# 500M Model Structural Parameters (1280 Hidden, 24 Layers, 20 Heads)
D_MODEL=1280
NUM_LAYERS=24
NUM_HEADS=20
CHUNK_SIZE=32
RANK=8
DROPOUT=0.1
SEQ_LEN=2048

# Optimizer & Hardware Parameters
BATCH_SIZE=64            # 64 sequences per GPU (Total global batch size = 512)
LEARNING_RATE=3e-4
SCHEDULER="cosine"
WEIGHT_DECAY=0.1

# Cluster Monitoring Intervals (Significantly increased to prevent IO throttling)
LOGGING_STEPS=500        # Log training metrics every ~500M tokens
SAVE_STEPS=5000          # Save checkpoints every ~5B tokens
EVAL_STEPS=5000          # Run evaluation split every ~5B tokens

echo "====================================================================="
echo "   Starting Distributed pre-training on 8x H100 SXM5 Node..."
echo "   Model Size: 500M Parameters"
echo "   Global Batch Size: 1.05 Million tokens per step"
echo "   Target Token Count: 100 Billion Tokens (Full DCLM-100BT)"
echo "   Layer Configuration: 18x CS-LRAD | 6x Attention"
echo "====================================================================="

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --model_type $MODEL_TYPE \
    --layer_sequence $LAYER_SEQ \
    --max_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --scheduler $SCHEDULER \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --chunk_size $CHUNK_SIZE \
    --rank $RANK \
    --dropout $DROPOUT \
    --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE \
    --compile \
    --grad_checkpointing \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS