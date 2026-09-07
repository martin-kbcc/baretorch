#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
REMOTE_PATH="r2:baretorch-data/tokenized_bin/dclm_100bt"
LOCAL_TARGET_DIR="${ROOT_DIR}/tokenized_bin"

echo "================================================================="
echo "📥 Fetching Full Tokenized Dataset from Cloudflare R2 (8x H100 Production)"
echo "================================================================="

mkdir -p "${LOCAL_TARGET_DIR}/train"
mkdir -p "${LOCAL_TARGET_DIR}/val"

# Fetch all validation shards
echo "Downloading all validation shards..."
rclone copy "${REMOTE_PATH}/val/" "${LOCAL_TARGET_DIR}/val/" \
  --progress \
  --transfers 16 \
  --s3-chunk-size 64M \
  --ignore-checksum \
  --s3-disable-checksum \
  --fast-list

# Fetch all training shards
echo "Downloading all training shards..."
rclone copy "${REMOTE_PATH}/train/" "${LOCAL_TARGET_DIR}/train/" \
  --progress \
  --transfers 16 \
  --s3-chunk-size 64M \
  --ignore-checksum \
  --s3-disable-checksum \
  --fast-list

echo ""
echo "🎉 Full dataset sync complete at: ${LOCAL_TARGET_DIR}"