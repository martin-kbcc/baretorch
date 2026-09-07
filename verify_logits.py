import os
import numpy as np
import torch
from transformers import AutoTokenizer

OUTPUT_DIR = "./teacher_predictions_sample"
MODEL_NAME = "Qwen/Qwen3.5-9B"
SEQ_LEN = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Locate generated output files
indices_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_teacher_indices.bin")]
values_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith("_teacher_values.bin")]

if not indices_files:
    print("❌ No teacher logit files found!")
    exit(1)

sample_indices_path = os.path.join(OUTPUT_DIR, indices_files[0])
sample_values_path = os.path.join(OUTPUT_DIR, values_files[0])

# Load memmaps
raw_indices = np.fromfile(sample_indices_path, dtype=np.uint32)
raw_values = np.fromfile(sample_values_path, dtype=np.float16)

num_seqs = len(raw_indices) // (SEQ_LEN * 4)
indices = raw_indices.reshape(num_seqs, SEQ_LEN, 4)
values = raw_values.reshape(num_seqs, SEQ_LEN, 4)

print("=" * 60)
print(f"🔍 Validating File: {indices_files[0]}")
print(f"📊 Total Sequences: {num_seqs} | Shape: {indices.shape}")
print(f"1. NaN / Inf Check (Values) : {'❌ FAILED' if np.isnan(values).any() or np.isinf(values).any() else '✅ PASSED'}")
print(f"2. Position 0 Masking Check : {'✅ PASSED' if (values[:, 0, :] == 0).all() and (indices[:, 0, :] == 0).all() else '❌ FAILED'}")
print(f"3. Vocab Index Range Check  : {'✅ PASSED' if indices.max() < 248320 else '❌ FAILED (Out of bounds)'}")
print(f"4. Logprob Value Range      : Min = {values.min():.2f}, Max = {values.max():.2f}")

# Print human-readable token preview
sample_seq_idx = 0
top1_tokens = indices[sample_seq_idx, 1:10, 0]  # First 10 tokens
decoded_text = tokenizer.decode(top1_tokens)
print(f"5. Sample Token Top-1 Text  : '{decoded_text}'")
print("=" * 60)