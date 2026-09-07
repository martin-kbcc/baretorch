#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
REMOTE_BUCKET="r2:baretorch-data/tokenized_bin"
LOCAL_TARGET_DIR="${ROOT_DIR}/tokenized_bin"
RANDOM_SEED=42

echo "================================================================="
echo "📥 Fetching Deterministic 165B Token Dataset Subsets from Cloudflare R2"
echo "================================================================="
echo "Target Root Directory: ${LOCAL_TARGET_DIR}"
echo "Deterministic Seed:    ${RANDOM_SEED}"
echo "================================================================="

# Array of datasets and their target sampling ratios (DatasetName:Ratio)
DATASETS=(
  "openr1_math:1.0"          # ~1B tokens (100%)
  "finemath_4plus:1.0"       # ~10B tokens (100%)
  "cosmopedia_v2:1.0"        # ~28B tokens (100%)
  "fineweb_edu_100bt:0.55"   # ~55B tokens (55%)
  "stack_dedup:0.175"        # ~35B tokens (17.5%)
  "dclm_100bt:0.25"          # ~25B tokens (25%)
  "finepdfs_100bt:0.11"      # ~11B tokens (11%)
)

mkdir -p "${LOCAL_TARGET_DIR}"
TEMP_LIST_DIR=$(mktemp -d)
trap 'rm -rf "${TEMP_LIST_DIR}"' EXIT

for entry in "${DATASETS[@]}"; do
  DATASET_NAME="${entry%%:*}"
  RATIO="${entry#*:}"

  REMOTE_DATASET_PATH="${REMOTE_BUCKET}/${DATASET_NAME}"
  LOCAL_DATASET_DIR="${LOCAL_TARGET_DIR}/${DATASET_NAME}"
  LIST_FILE="${TEMP_LIST_DIR}/${DATASET_NAME}_files.txt"

  mkdir -p "${LOCAL_DATASET_DIR}"

  echo ""
  echo "🔍 Listing shards for '${DATASET_NAME}' (Target Sampling Ratio: ${RATIO})..."

  # 1. Fetch file list from R2 and select shards deterministically via Python
  rclone lsf --recursive "${REMOTE_DATASET_PATH}" | python3 -c "
import random, sys, math

seed = int(sys.argv[1])
ratio = float(sys.argv[2])
raw_lines = sys.stdin.read().strip().splitlines()

# Filter for binary shard files, excluding temporary working files
files = [f.strip() for f in raw_lines if f.strip().endswith('.bin') and not f.strip().endswith('.tmp')]

# Lexicographical sort to guarantee identical baseline ordering across systems
files.sort()

if ratio < 1.0:
    rng = random.Random(seed)
    rng.shuffle(files)
    num_to_select = max(1, math.ceil(len(files) * ratio))
    files = files[:num_to_select]
    files.sort()  # Re-sort selected files for clean sequential transfer

for f in files:
    print(f)
" "${RANDOM_SEED}" "${RATIO}" > "${LIST_FILE}"

  FETCH_COUNT=$(wc -l < "${LIST_FILE}" | tr -d ' ')
  echo "📦 Selected ${FETCH_COUNT} shards deterministically for '${DATASET_NAME}'."

  if [ "${FETCH_COUNT}" -eq 0 ]; then
    echo "⚠️ Warning: No matching binary files found for ${DATASET_NAME}. Skipping download."
    continue
  fi

  # 2. Fetch selected shards using rclone --files-from
  echo "⬇️ Syncing files for '${DATASET_NAME}'..."
  rclone copy "${REMOTE_DATASET_PATH}" "${LOCAL_DATASET_DIR}" \
    --files-from "${LIST_FILE}" \
    --progress \
    --transfers 16 \
    --s3-chunk-size 64M \
    --ignore-checksum \
    --s3-disable-checksum \
    --fast-list

done

echo ""
echo "================================================================="
echo "🎉 Deterministic 165B Dataset Sync Complete!"
echo "Data location: ${LOCAL_TARGET_DIR}"
echo "================================================================="