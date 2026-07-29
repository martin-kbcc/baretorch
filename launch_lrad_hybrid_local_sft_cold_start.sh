#!/bin/bash
set -e

# Environment Configuration
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

SCRIPT_DIR="/home/martinkb/Desktop/BareTorch_F"
PRETRAINED_CKPT="${SCRIPT_DIR}/checkpoints_100m_simpo/checkpoint-simpo-final"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints_100m_sft_cold_start"

echo "================================================================================"
echo "🚀 LAUNCHING COLD-START CoT SFT WARMUP ON DUAL-RTX 4090s"
echo "================================================================================"
echo "• Base Checkpoint  : ${PRETRAINED_CKPT}"
echo "• Output Directory : ${OUTPUT_DIR}"
echo "• Active GPUs      : 2 (Devices: 0,1)"
echo "• Num Epochs       : 2"
echo "• Per-GPU Batch    : 8 (Effective Batch Size: 32)"
echo "• Learning Rate    : 2e-5"
echo "• Sequence Length  : 1024"
echo "• Grad Checkpoint  : ENABLED"
echo "================================================================================"

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    "${SCRIPT_DIR}/train_sft_cold_start.py" \
    --pretrained_model_path "${PRETRAINED_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs 2 \
    --batch_size 8 \
    --grad_accum 2 \
    --learning_rate 2e-5 \
    --seq_len 1024 \
    --grad_checkpointing

echo "================================================================================"
echo "✅ Cold-Start CoT SFT Warmup Completed Successfully!"
echo "================================================================================"