#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Dataset Shard Sanitizer & Token Counter")
    parser.add_argument("--data_dir", type=str, required=True, help="Target directory containing .bin shards")
    parser.add_argument("--vocab_size", type=int, default=248320, help="Maximum allowed vocabulary size (Qwen3.5 default: 248320)")
    parser.add_argument("--eos_id", type=int, default=151643, help="Token ID to replace invalid tokens with")
    parser.add_argument("--max_allowed_bad", type=int, default=100, help="Max bad tokens allowed for auto-repair before quarantine")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length for calculating sequence counts")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel CPU worker processes")
    return parser.parse_args()

def quarantine_triplet(path):
    """Quarantines the target file and any existing companion files in the triplet."""
    prefix = path
    for suffix in ["_packed_tokens.bin", "_teacher_indices.bin", "_teacher_values.bin"]:
        if path.endswith(suffix):
            prefix = path[:-len(suffix)]
            break

    for suffix in ["_packed_tokens.bin", "_teacher_indices.bin", "_teacher_values.bin"]:
        companion_path = f"{prefix}{suffix}"
        if os.path.exists(companion_path) and not companion_path.endswith(".corrupt"):
            try:
                os.rename(companion_path, companion_path + ".corrupt")
            except OSError:
                pass

def sanitize_integer_shard(path, vocab_size, eos_id, max_allowed_bad):
    """Sanitizes uint32 token ID or teacher index files and returns token count."""
    file_size = os.path.getsize(path)
    if file_size % 4 != 0:
        print(f"❌ QUARANTINED (Truncated uint32 length {file_size} bytes): {path}")
        quarantine_triplet(path)
        return "quarantined", 0

    data = np.memmap(path, dtype=np.uint32, mode='r+')
    bad_indices = np.where(data >= vocab_size)[0]
    num_bad = len(bad_indices)
    token_count = len(data) if path.endswith("_packed_tokens.bin") else 0

    if num_bad == 0:
        return "pristine", token_count
    elif num_bad <= max_allowed_bad:
        data[bad_indices] = eos_id
        data.flush()
        print(f"⚠️ REPAIRED ({num_bad} invalid indices -> EOS {eos_id}): {os.path.basename(path)}")
        return "repaired", token_count
    else:
        print(f"❌ QUARANTINED ({num_bad:,} invalid indices > threshold {max_allowed_bad}): {path}")
        quarantine_triplet(path)
        return "quarantined", 0

def sanitize_value_shard(path, max_allowed_bad=100):
    """Sanitizes float16 teacher value logit files (repairs isolated NaNs/Infs to 0.0)."""
    file_size = os.path.getsize(path)
    if file_size % 2 != 0:
        print(f"❌ QUARANTINED (Truncated float16 length {file_size} bytes): {path}")
        quarantine_triplet(path)
        return "quarantined", 0

    data = np.memmap(path, dtype=np.float16, mode='r+')
    invalid_mask = np.isnan(data) | np.isinf(data)
    nan_count = invalid_mask.sum()

    if nan_count == 0:
        return "pristine", 0
    elif nan_count <= max_allowed_bad:
        data[invalid_mask] = 0.0
        data.flush()
        print(f"⚠️ REPAIRED ({nan_count} NaN/Inf values replaced with 0.0): {os.path.basename(path)}")
        return "repaired", 0
    else:
        print(f"❌ QUARANTINED ({nan_count:,} NaN/Inf values > threshold {max_allowed_bad}): {path}")
        quarantine_triplet(path)
        return "quarantined", 0

def process_shard(path, vocab_size, eos_id, max_allowed_bad):
    """Worker wrapper to route files based on name."""
    filename = os.path.basename(path)
    if "teacher_values" in filename:
        return sanitize_value_shard(path, max_allowed_bad)
    else:
        return sanitize_integer_shard(path, vocab_size, eos_id, max_allowed_bad)

def main():
    args = parse_args()
    bin_files = sorted(glob.glob(os.path.join(args.data_dir, "**/*.bin"), recursive=True))
    bin_files = [f for f in bin_files if not f.endswith(".corrupt")]
    
    print(f"🔍 Validating {len(bin_files)} .bin shards in '{args.data_dir}'...")
    print(f"⚙️ Config: VOCAB_SIZE={args.vocab_size} | EOS_ID={args.eos_id} | MAX_ALLOWED_BAD={args.max_allowed_bad} | WORKERS={args.workers}\n")

    stats = {"pristine": 0, "repaired": 0, "quarantined": 0}
    total_valid_tokens = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_shard, path, args.vocab_size, args.eos_id, args.max_allowed_bad): path
            for path in bin_files
        }
        for future in as_completed(futures):
            try:
                status, tokens = future.result()
                stats[status] += 1
                total_valid_tokens += tokens
            except Exception as e:
                path = futures[future]
                print(f"❌ ERROR processing {path}: {e}")
                quarantine_triplet(path)
                stats["quarantined"] += 1

    total_seqs = total_valid_tokens // args.seq_len

    print("\n" + "=" * 70)
    print("📊 DATASET SANITIZATION & ACCOUNTING SUMMARY")
    print("=" * 70)
    print(f"Pristine Shards       : {stats['pristine']:,}")
    print(f"Repaired Shards       : {stats['repaired']:,}")
    print(f"Quarantined Shards    : {stats['quarantined']:,}")
    print("-" * 70)
    print(f"Total Valid Tokens    : {total_valid_tokens:,} ({total_valid_tokens / 1e9:.3f} Billion / {total_valid_tokens / 1e6:.1f} Million)")
    print(f"Equivalent Sequences  : {total_seqs:,} (Seq Len = {args.seq_len})")
    print("=" * 70)

if __name__ == "__main__":
    main()