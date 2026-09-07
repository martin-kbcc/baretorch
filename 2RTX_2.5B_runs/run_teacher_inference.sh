#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${ROOT_DIR}/tokenized_bin_sample"
OUTPUT_DIR="${ROOT_DIR}/teacher_predictions_sample"
MODEL_NAME="Qwen/Qwen3.5-9B"

# PCIe Multi-GPU & PyTorch Memory Management Overrides
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================================================="
echo "🚀 Launching TransformerEngine FP8 Extraction via torchrun (Dual RTX 4090)"
echo "================================================================="

# Using nproc_per_node=2 to utilize both RTX 4090s via PyTorch DDP
torchrun --nproc_per_node=2 "${ROOT_DIR}/teacher_inference.py" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --seq_len 2048 \
  --batch_size 4 \
  --attn_implementation "sdpa" \
  --dtype_input uint32 \
  --logit_chunk_size 512 \
  --use_fp8

echo ""
echo "🎉 Extraction run complete! Outputs saved to: ${OUTPUT_DIR}"