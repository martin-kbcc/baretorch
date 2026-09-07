#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
REMOTE_PATH="r2:baretorch-data/tokenized_bin/dclm_100bt"
LOCAL_TARGET_DIR="${ROOT_DIR}/tokenized_bin_sample"

echo "================================================================="
echo "📥 Fetching Sample Token Test Sample from Cloudflare R2 (Local Run)"
echo "================================================================="

mkdir -p "${LOCAL_TARGET_DIR}/train"
mkdir -p "${LOCAL_TARGET_DIR}/val"

# Fetch validation shards
echo "Downloading validation shards..."
rclone copy "${REMOTE_PATH}/val/" "${LOCAL_TARGET_DIR}/val/" \
  --include "shard_00000.bin" \
  --progress \
  --transfers 4 \
  --s3-chunk-size 64M \
  --ignore-checksum \
  --s3-disable-checksum \
  --fast-list

# Fetch initial training shards
echo "Downloading sample training shards..."
rclone copy "${REMOTE_PATH}/train/" "${LOCAL_TARGET_DIR}/train/" \
  --include "shard_00002.bin" \
  --progress \
  --transfers 4 \
  --s3-chunk-size 64M \
  --ignore-checksum \
  --s3-disable-checksum \
  --fast-list

echo ""
echo "🎉 Local sample dataset ready at: ${LOCAL_TARGET_DIR}"