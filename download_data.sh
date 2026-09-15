#!/usr/bin/env bash
set -e

R2_PREFIX="r2:baretorch-data/teacher_predictions"
LOCAL_SAMPLE_DIR="./teacher_predictions_sample"

DATASETS=(
  "fineweb_edu_100bt"
  "stack_dedup"
  "dclm_100bt"
  "finepdfs_100bt"
  "cosmopedia_v2"
  "finemath_4plus"
  "openr1_math"
)

mkdir -p "${LOCAL_SAMPLE_DIR}"

for ds in "${DATASETS[@]}"; do
  echo "🔍 Fetching 1 sample shard for '${ds}'..."
  
  # Find the first token file in the dataset directory
  FIRST_FILE=$(rclone lsf "${R2_PREFIX}/${ds}/" --include "*_packed_tokens.bin" --recursive | head -n 1)
  
  if [ -z "${FIRST_FILE}" ]; then
    echo "⚠️ No packed_tokens binary found for ${ds}, skipping..."
    continue
  fi
  
  # Extract the base shard prefix (strip _packed_tokens.bin)
  BASE_PREFIX="${FIRST_FILE%_packed_tokens.bin}"
  
  mkdir -p "${LOCAL_SAMPLE_DIR}/${ds}"
  
  # Copy the exact 3 matching binary files for this shard
  rclone copy "${R2_PREFIX}/${ds}" "${LOCAL_SAMPLE_DIR}/${ds}" \
    --include "${BASE_PREFIX}_packed_tokens.bin" \
    --include "${BASE_PREFIX}_teacher_indices.bin" \
    --include "${BASE_PREFIX}_teacher_values.bin" \
    --progress
done

echo "✅ Sample shard acquisition complete!"