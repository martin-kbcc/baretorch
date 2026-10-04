#!/usr/bin/env bash
set -e

# CUDA Memory Management & Distributed NCCL Tuning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export TORCH_CPP_MIN_LOG_LEVEL=2
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Hardware Setup (Dual-RTX Workstation)
NUM_GPUS=2

# Checkpoint Paths
SFT_MODEL_PATH="${SCRIPT_DIR}/checkpoints_300M_sft/checkpoint-3197"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints_300M_grpo"
TOKENIZER_NAME="Qwen/Qwen3.5-9B"

# RL Dataset & Rollout Hyperparameters
DATASET_NAME="openai/gsm8k"
DATASET_CONFIG="main"

NUM_GENERATIONS=4
MAX_PROMPT_LENGTH=512
MAX_COMPLETION_LENGTH=1024

PER_GPU_BATCH_SIZE=2
GRAD_ACCUM=4
LEARNING_RATE=5e-6
BETA=0.04
NUM_EPOCHS=1

# Cloud Storage & Sync
R2_PREFIX="checkpoints_local_grpo"

GLOBAL_BATCH_SEQS=$((NUM_GPUS * PER_GPU_BATCH_SIZE * GRAD_ACCUM))

mkdir -p "$OUTPUT_DIR"

echo "======================================================================"
echo "🧠 Launching BareTorch 300M Stage 2 GRPO Engine (Dual-GPU Workstation)"
echo "  ├─ Foundation SFT Checkpoint : ${SFT_MODEL_PATH}"
echo "  ├─ Output Directory          : ${OUTPUT_DIR}"
echo "  ├─ Hardware Setup            : ${NUM_GPUS}x NVIDIA GPUs (DDP Mode)"
echo "  ├─ Prompt Batching           : ${NUM_GPUS} GPUs x ${PER_GPU_BATCH_SIZE} batch x ${GRAD_ACCUM} accum = ${GLOBAL_BATCH_SEQS} prompts/step"
echo "  ├─ Rollouts per Prompt (G)   : ${NUM_GENERATIONS} completion generations"
echo "  ├─ RL Target Dataset         : ${DATASET_NAME} (${DATASET_CONFIG})"
echo "  ├─ Learning Rate / KL Beta   : LR=${LEARNING_RATE} | Beta=${BETA}"
echo "  └─ Cloud Sync                : Active -> Cloudflare R2 (${R2_PREFIX})"
echo "======================================================================"

torchrun --nproc_per_node=${NUM_GPUS} "${ROOT_DIR}/train_grpo.py" \
    --sft_model_path "${SFT_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --tie_word_embeddings \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --num_generations ${NUM_GENERATIONS} \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --learning_rate ${LEARNING_RATE} \
    --beta ${BETA} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --grad_accum ${GRAD_ACCUM} \
    --num_epochs ${NUM_EPOCHS} \
    --r2_sync \
    --r2_prefix "${R2_PREFIX}"