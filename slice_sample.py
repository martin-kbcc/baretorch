import glob
import os

SAMPLE_DIR = "./teacher_predictions_sample"
SEQ_LEN = 2048
TARGET_SEQS = 500  # ~1,024,000 tokens per dataset

token_files = glob.glob(os.path.join(SAMPLE_DIR, "**/*_packed_tokens.bin"), recursive=True)

for t_path in token_files:
    prefix = t_path.replace("_packed_tokens.bin", "")
    idx_path = f"{prefix}_teacher_indices.bin"
    val_path = f"{prefix}_teacher_values.bin"

    if not (os.path.exists(idx_path) and os.path.exists(val_path)):
        continue

    # Calculate target byte limits based on fixed strides
    tok_bytes = TARGET_SEQS * SEQ_LEN * 4         # uint32 (4 bytes)
    idx_bytes = TARGET_SEQS * SEQ_LEN * 4 * 4     # uint32 x 4 (16 bytes)
    val_bytes = TARGET_SEQS * SEQ_LEN * 4 * 2     # float16 x 4 (8 bytes)

    # Truncate files in-place
    for path, target_size in [(t_path, tok_bytes), (idx_path, idx_bytes), (val_path, val_bytes)]:
        if os.path.getsize(path) > target_size:
            with open(path, "r+b") as f:
                f.truncate(target_size)

    print(f"✂️ Truncated {os.path.basename(prefix)} to {TARGET_SEQS} sequences (~{TARGET_SEQS * SEQ_LEN / 1e6:.2f}M tokens)")

print("🎉 Micro-sample dataset prepared cleanly!")