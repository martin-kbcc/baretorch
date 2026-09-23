import os
import subprocess
import json
import argparse

RATIOS = {
    "fineweb_edu_100bt": 0.32,
    "stack_dedup": 0.24,
    "dclm_100bt": 0.16,
    "finemath_4plus": 0.10,
    "cosmopedia_v2": 0.10,
    "finepdfs_100bt": 0.07,
    "openr1_math": 0.01,
}

BYTES_PER_TOKEN_PACKED = 4  # uint32 = 4 bytes per token

def get_remote_shards(remote_path):
    """Query rclone for file list in JSON format."""
    cmd = ["rclone", "lsjson", remote_path, "--recursive"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return []
    try:
        return json.loads(res.stdout)
    except Exception:
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=str, default="r2:baretorch-data/teacher_predictions/train")
    parser.add_argument("--target_dir", type=str, default="./data_6B/train")
    parser.add_argument("--total_tokens", type=float, default=6e9)
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)

    print("=" * 70)
    print(f"🚀 Preparing 6B Token Proportional Dataset Sync from R2")
    print(f"Target Total Tokens: {args.total_tokens / 1e9:.2f} Billion")
    print("=" * 70)

    for ds_name, ratio in RATIOS.items():
        target_tokens = args.total_tokens * ratio
        target_packed_bytes = target_tokens * BYTES_PER_TOKEN_PACKED
        
        print(f"\n📦 Processing '{ds_name}' (Target: {target_tokens/1e6:.1f}M tokens | ~{target_packed_bytes/1e9:.2f} GB packed tokens)")
        
        ds_remote_path = f"{args.remote}/{ds_name}"
        files = get_remote_shards(ds_remote_path)
        
        # Group files by prefix (prefix_packed_tokens.bin, prefix_teacher_indices.bin, prefix_teacher_values.bin)
        packed_files = [f for f in files if f["Path"].endswith("_packed_tokens.bin")]
        
        accumulated_bytes = 0
        selected_prefixes = []

        for p_file in sorted(packed_files, key=lambda x: x["Path"]):
            prefix = p_file["Path"].replace("_packed_tokens.bin", "")
            accumulated_bytes += p_file["Size"]
            selected_prefixes.append(prefix)
            if accumulated_bytes >= target_packed_bytes:
                break

        print(f"  └─ Selected {len(selected_prefixes)} shard triplets ({accumulated_bytes / 1e9:.2f} GB token binary data)")

        local_ds_dir = os.path.join(args.target_dir, ds_name)
        os.makedirs(local_ds_dir, exist_ok=True)

        # Download selected shard triplets via rclone
        for prefix in selected_prefixes:
            for suffix in ["_packed_tokens.bin", "_teacher_indices.bin", "_teacher_values.bin"]:
                rel_file = f"{prefix}{suffix}"
                file_remote = f"{ds_remote_path}/{rel_file}"
                file_local = os.path.join(local_ds_dir, os.path.basename(rel_file))

                if not os.path.exists(file_local):
                    print(f"    ├─ Downloading: {os.path.basename(rel_file)}...")
                    cmd = [
                        "rclone", "copyto",
                        file_remote, file_local,
                        "--transfers", "8",
                        "--s3-chunk-size", "64M"
                    ]
                    subprocess.run(cmd)

    print("\n✅ 6B Proportional Dataset Sync Complete!")

if __name__ == "__main__":
    main()