#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${ROOT_DIR}/tokenized_bin_sample"
OUTPUT_DIR="${ROOT_DIR}/teacher_predictions_sample"
MODEL_NAME="Qwen/Qwen3.5-9B"

# PyTorch Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================="
echo "🚀 Launching TransformerEngine FP8 Extraction via torchrun (2x H100 SXM5)"
echo "================================================================="

# Utilizing 2x H100s with scaled batch size and FlashAttention-2
torchrun --nproc_per_node=2 "${ROOT_DIR}/teacher_inference.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --seq_len 2048 \
  --batch_size 16 \
  --attn_implementation "sdpa" \
  --dtype_input uint32 \
  --logit_chunk_size 512 \
  --use_fp8 \
  --r2_sync

echo ""
echo "🎉 Extraction run complete! Outputs saved to: ${OUTPUT_DIR}"