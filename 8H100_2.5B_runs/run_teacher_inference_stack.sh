#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

DATASET_NAME="stack_dedup"
DATASET_INPUT="${ROOT_DIR}/tokenized_bin/${DATASET_NAME}"
DATASET_OUTPUT="${ROOT_DIR}/teacher_predictions/${DATASET_NAME}"
MODEL_NAME="Qwen/Qwen3.5-9B"

# PyTorch Memory Management & Distributed Setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# Check if model is cached locally; if not, allow online fetch for the first run
if [ ! -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B" ]; then
    echo "🌐 Model not found in local cache. Enabling online download..."
    export HF_HUB_OFFLINE=0
else
    export HF_HUB_OFFLINE=1
fi

echo "================================================================="
echo "🚀 Launching Batch Teacher Logit Extraction for '${DATASET_NAME}'"
echo "================================================================="
echo "📥 Input:  ${DATASET_INPUT}"
echo "📤 Output: ${DATASET_OUTPUT}"
echo "================================================================="

if [ ! -d "${DATASET_INPUT}" ]; then
    echo "❌ Error: Input directory ${DATASET_INPUT} does not exist. Run fetch_data_stack.sh first."
    exit 1
fi

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

echo ""
echo "🎉 Extraction complete for ${DATASET_NAME}! Outputs synced to R2."