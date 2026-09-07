import os
import numpy as np

datasets = [
    "fineweb_edu_100bt", "stack_dedup", "dclm_100bt", "finepdfs_100bt",
    "cosmopedia_v2", "finemath_4plus", "openr1_math"
]
seq_len = 2048
num_seqs = 50  # Tiny sample count for instant testing

for split in ["train", "val"]:
    for ds in datasets:
        out_dir = f"./mock_teacher_predictions/{split}/{ds}"
        os.makedirs(out_dir, exist_ok=True)
        
        prefix = os.path.join(out_dir, "shard_00000")
        
        # 1. Tokens (uint32)
        tokens = np.random.randint(0, 100000, size=(num_seqs, seq_len), dtype=np.uint32)
        tokens.tofile(f"{prefix}_packed_tokens.bin")
        
        # 2. Teacher Top-4 Indices (uint32)
        indices = np.random.randint(0, 100000, size=(num_seqs, seq_len, 4), dtype=np.uint32)
        indices.tofile(f"{prefix}_teacher_indices.bin")
        
        # 3. Teacher Top-4 Values (float16)
        values = np.random.randn(num_seqs, seq_len, 4).astype(np.float16)
        values.tofile(f"{prefix}_teacher_values.bin")

print("✅ Mock dataset generated at ./mock_teacher_predictions")