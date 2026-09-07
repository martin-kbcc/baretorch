#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
REMOTE_PATH="r2:baretorch-data/tokenized_bin/dclm_100bt"
LOCAL_TARGET_DIR="${ROOT_DIR}/tokenized_bin_sample"

echo "================================================================="
echo "📥 Fetching Sample Dataset from Cloudflare R2 (2x H100 Run)"
echo "================================================================="

mkdir -p "${LOCAL_TARGET_DIR}/train"
mkdir -p "${LOCAL_TARGET_DIR}/val"

# Fetch validation shard
echo "Downloading validation shard..."
rclone copy "${REMOTE_PATH}/val/" "${LOCAL_TARGET_DIR}/val/" \
  --include "shard_00000.bin" \
  --progress \
  --transfers 4 \
  --s3-chunk-size 64M \
  --ignore-checksum \
  --s3-disable-checksum \
  --fast-list

# Fetch 2 sample training shards for validation
echo "Downloading 2 sample training shards..."
rclone copy "${REMOTE_PATH}/train/" "${LOCAL_TARGET_DIR}/train/" \
  --include "shard_00002.bin" \
  --include "shard_00003.bin" \
  --progress \
  --transfers 4 \
  --s3-chunk-size 64M \
  --ignore-checksum \
  --s3-disable-checksum \
  --fast-list

echo ""
echo "🎉 2x H100 sample dataset ready at: ${LOCAL_TARGET_DIR}"