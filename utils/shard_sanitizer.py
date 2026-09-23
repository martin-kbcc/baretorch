import os
import glob
import numpy as np

VOCAB_SIZE = 248320
EOS_TOKEN_ID = 151643  # Qwen <|endoftext|> ID
MAX_ALLOWED_BAD = 100  # Threshold for auto-repair vs quarantine

bin_files = sorted(glob.glob('/home/ubuntu/baretorch/tokenized_bin/**/*.bin', recursive=True))
print(f"🔍 Validating {len(bin_files)} shards...\n")

repaired, quarantined, pristine = 0, 0, 0

for path in bin_files:
    # 1. Check byte alignment (uint32 requires multiple of 4 bytes)
    file_size = os.path.getsize(path)
    if file_size % 4 != 0:
        print(f"❌ QUARANTINED (Truncated byte length): {path}")
        os.rename(path, path + ".corrupt")
        quarantined += 1
        continue

    # 2. Fast memory-mapped token inspection
    data = np.memmap(path, dtype=np.uint32, mode='r+')
    bad_indices = np.where(data >= VOCAB_SIZE)[0]
    num_bad = len(bad_indices)

    if num_bad == 0:
        pristine += 1
    elif num_bad <= MAX_ALLOWED_BAD:
        # Sanitize isolated artifacts to EOS
        data[bad_indices] = EOS_TOKEN_ID
        data.flush()
        print(f"⚠️ REPAIRED ({num_bad} tokens replaced with EOS): {path}")
        repaired += 1
    else:
        # File is severely corrupted
        print(f"❌ QUARANTINED ({num_bad:,} bad tokens found): {path}")
        os.rename(path, path + ".corrupt")
        quarantined += 1

print("\n" + "=" * 60)
print(f"Pristine Shards  : {pristine}")
print(f"Repaired Shards  : {repaired}")
print(f"Quarantined      : {quarantined}")
print("=" * 60)