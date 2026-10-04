#!/usr/bin/env bash
set -e

# Make script executable: chmod +x 01_download_10B_data.sh

echo "============================================================"
echo "Step 1: Downloading Proportional 10B Dataset from R2"
echo "============================================================"

python3 sync_10b_dataset.py \
    --remote "r2:baretorch-data/teacher_predictions" \
    --target_dir "./data_10B" \
    --total_tokens 10000000000