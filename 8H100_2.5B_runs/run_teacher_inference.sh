#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${ROOT_DIR}/tokenized_bin"
OUTPUT_DIR="${ROOT_DIR}/teacher_predictions"
MODEL_NAME="Qwen/Qwen3.5-9B"

# PyTorch Memory Management & Distributed Setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

echo "================================================================="
echo "🚀 Launching Production TransformerEngine FP8 Extraction via torchrun (8x H100 SXM5)"
echo "================================================================="

# Utilizing 8x H100s with PyTorch Inductor compilation and full R2 background sync
torchrun --nproc_per_node=8 "${ROOT_DIR}/teacher_inference.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --seq_len 2048 \
  --batch_size 48 \
  --attn_implementation "sdpa" \
  --dtype_input uint32 \
  --logit_chunk_size 1024 \
  --use_fp8 \
  --compile \
  --r2_sync

echo ""
echo "🎉 Production extraction complete! All outputs saved to: ${OUTPUT_DIR} and synced to R2."