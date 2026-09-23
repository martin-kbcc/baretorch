import os
import glob

DATA_DIR = "./data_6B"
SEQ_LEN = 2048

# Block sizes in bytes per sequence block (2048 tokens)
BYTES_PER_SEQ = {
    "_packed_tokens.bin": SEQ_LEN * 4,               # uint32 = 8,192 bytes
    "_teacher_indices.bin": SEQ_LEN * 4 * 4,         # uint32 (seq, top4) = 32,768 bytes
    "_teacher_values.bin": SEQ_LEN * 4 * 2,          # float16 (seq, top4) = 16,384 bytes
}

print("=" * 70)
print("🔍 Auditing downloaded shards in ./data_6B for corruptions/truncations")
print("=" * 70)

corrupted_files = []
all_files = glob.glob(f"{DATA_DIR}/**/*.bin", recursive=True)

if not all_files:
    print(f"❌ No .bin files found in {DATA_DIR}")
    exit(1)

for file_path in sorted(all_files):
    file_size = os.path.getsize(file_path)
    file_type = None
    for suffix in BYTES_PER_SEQ:
        if file_path.endswith(suffix):
            file_type = suffix
            break

    if not file_type:
        continue

    block_size = BYTES_PER_SEQ[file_type]
    remainder = file_size % block_size

    if remainder != 0:
        corrupted_files.append((file_path, file_size, remainder, block_size))
        print(f"❌ CORRUPTED/TRUNCATED: {file_path}")
        print(f"   ├─ Size: {file_size:,} bytes")
        print(f"   └─ Remainder: {remainder} bytes (Must be a multiple of {block_size} bytes)")
    else:
        num_seqs = file_size // block_size
        print(f"✅ OK: {os.path.basename(file_path)} ({num_seqs:,} sequence blocks | {file_size / 1e9:.2f} GB)")

print("=" * 70)
if corrupted_files:
    print(f"⚠️ Found {len(corrupted_files)} corrupted or incomplete file(s):")
    for f_path, _, _, _ in corrupted_files:
        print(f"  -> {f_path}")
    
    print("\nAction: Deleting corrupted files so rclone can re-download them cleanly...")
    for f_path, _, _, _ in corrupted_files:
        os.remove(f_path)
        print(f"  🗑️ Deleted: {f_path}")
    print("\nRun './01_download_6B_data.sh' to re-fetch the clean files from R2.")
else:
    print("🎉 All downloaded binary shards are 100% valid and aligned!")
print("=" * 70)