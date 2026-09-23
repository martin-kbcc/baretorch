#!/usr/bin/env bash
set -e

# Make script executable: chmod +x 01_download_6B_data.sh

echo "============================================================"
echo "Step 1: Downloading Proportional 6B Dataset from R2"
echo "============================================================"

python3 sync_6b_dataset.py \
    --remote "r2:baretorch-data/teacher_predictions" \
    --target_dir "./data_6B" \
    --total_tokens 6000000000