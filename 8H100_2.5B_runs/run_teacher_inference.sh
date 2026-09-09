#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_BASE_DIR="${ROOT_DIR}/tokenized_bin"
OUTPUT_BASE_DIR="${ROOT_DIR}/teacher_predictions"
MODEL_NAME="Qwen/Qwen3.5-9B"

# PyTorch Memory Management, Offline HF Mode & Distributed Setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1

echo "================================================================="
echo "🚀 Launching Batch Teacher Logit Extraction across All Datasets (8x H100 SXM)"
echo "================================================================="

for DATASET_PATH in "${INPUT_BASE_DIR}"/*/; do
    DATASET_NAME="$(basename "${DATASET_PATH}")"

    DATASET_INPUT="${DATASET_PATH}"
    DATASET_OUTPUT="${OUTPUT_BASE_DIR}/${DATASET_NAME}"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "📂 Processing Dataset: ${DATASET_NAME}"
    echo "📥 Input:  ${DATASET_INPUT}"
    echo "📤 Output: ${DATASET_OUTPUT}"
    echo "-----------------------------------------------------------------"

    mkdir -p "${DATASET_OUTPUT}"

    torchrun --nproc_per_node=8 "${ROOT_DIR}/teacher_inference.py" \
      --input_dir "${DATASET_INPUT}" \
      --output_dir "${DATASET_OUTPUT}" \
      --model_name "${MODEL_NAME}" \
      --seq_len 2048 \
      --batch_size 40 \
      --attn_implementation "sdpa" \
      --dtype_input uint32 \
      --logit_chunk_size 1024 \
      --use_fp8 \
      --compile \
      --r2_sync

    echo "✅ Completed extraction for: ${DATASET_NAME}"
done

echo ""
echo "🎉 Production batch extraction complete across all datasets! Outputs synced to R2."