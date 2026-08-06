import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
import time
import warnings
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Silence environment noise and warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import transformers

transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


def process_shard(args):
  """Worker function to tokenize a Parquet shard using atomic .tmp file writes."""
  shard_path, output_dir, tokenizer_name, shard_idx, dtype_str = args
  shard_name = os.path.basename(shard_path)

  dtype = np.dtype(dtype_str)
  bytes_per_token = dtype.itemsize

  bin_filename = f"shard_{shard_idx:05d}.bin"
  out_bin_path = os.path.join(output_dir, bin_filename)
  tmp_bin_path = out_bin_path + ".tmp"

  # 1. Skip if fully completed .bin file already exists
  if os.path.exists(out_bin_path) and os.path.getsize(out_bin_path) > 0:
    total_tokens = os.path.getsize(out_bin_path) // bytes_per_token
    return shard_name, total_tokens

  # 2. Clean up leftover temporary file from any interrupted run
  if os.path.exists(tmp_bin_path):
    os.remove(tmp_bin_path)

  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_name, local_files_only=False, use_fast=True
  )
  tokenizer.model_max_length = 10**9  # Suppress sequence length warnings

  eos_token_id = (
      tokenizer.eos_token_id
      if tokenizer.eos_token_id is not None
      else tokenizer.pad_token_id
  )
  total_tokens = 0

  try:
    pf = pq.ParquetFile(shard_path)

    # Dynamically select column name ("text" for web/synthetic, "content" for code)
    schema_names = pf.schema.names
    text_column = "text" if "text" in schema_names else "content"

    # Write to .tmp file first
    with open(tmp_bin_path, "wb") as f:
      for batch in pf.iter_batches(batch_size=1000, columns=[text_column]):
        texts = batch[text_column].to_pylist()

        tokenized_batch = tokenizer(
            texts, add_special_tokens=False, truncation=False, padding=False
        )["input_ids"]

        chunk_tokens = []
        for doc_tokens in tokenized_batch:
          chunk_tokens.extend(doc_tokens)
          chunk_tokens.append(eos_token_id)

        if chunk_tokens:
          arr = np.array(chunk_tokens, dtype=dtype)
          f.write(arr.tobytes())
          total_tokens += len(arr)

    # Atomic rename once completely finished
    os.replace(tmp_bin_path, out_bin_path)
    return shard_name, total_tokens

  except Exception as e:
    if os.path.exists(tmp_bin_path):
      os.remove(tmp_bin_path)
    return None, 0


def main():
  parser = argparse.ArgumentParser(
      description="BareTorch Ultra-Fast Parquet-to-Binary Tokenizer"
  )
  parser.add_argument(
      "--raw_dir",
      type=str,
      default="./raw_smollm_parquet",
      help="Directory containing raw Parquet files.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./tokenized_bin",
      help="Output directory for binary files.",
  )
  parser.add_argument(
      "--tokenizer_name",
      type=str,
      default="HuggingFaceTB/SmolLM2-360M",
      help="Hugging Face tokenizer identifier.",
  )
  parser.add_argument(
      "--num_proc",
      type=int,
      default=12,
      help="Number of parallel worker processes.",
  )
  parser.add_argument(
      "--val_shards",
      type=int,
      default=2,
      help="Number of initial shards reserved for validation.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for shuffling shards before split.",
  )

  args = parser.parse_args()

  # 1. Dynamically evaluate vocab size to set uint16 vs uint32 safely
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
  vocab_size = getattr(tokenizer, "vocab_size", len(tokenizer))

  if vocab_size < 65536:
    dtype_str = "uint16"
  else:
    dtype_str = "uint32"

  print("=" * 70)
  print(
      f"Tokenizer: '{args.tokenizer_name}' | Vocab Size: {vocab_size:,} |"
      f" Selected Dtype: {dtype_str}"
  )
  print("=" * 70)

  os.makedirs(args.output_dir, exist_ok=True)
  val_dir = os.path.join(args.output_dir, "val")
  train_dir = os.path.join(args.output_dir, "train")
  os.makedirs(val_dir, exist_ok=True)
  os.makedirs(train_dir, exist_ok=True)

  # 2. Discover recursive Parquet files
  parquet_shards = sorted(
      glob.glob(f"{args.raw_dir}/**/*.parquet", recursive=True)
  )
  if not parquet_shards:
    print(f"❌ Error: No .parquet files found in '{args.raw_dir}'.")
    return

  # 3. Shuffle shards with a fixed seed to get balanced train/val splits across subsets
  random.seed(args.seed)
  random.shuffle(parquet_shards)

  print(
      f"\n🚀 Found {len(parquet_shards)} Parquet shards across sub-datasets."
  )
  print(
      f"Starting tokenization with {args.num_proc} processes (Val shards:"
      f" {args.val_shards})...\n"
  )

  tasks = []
  for idx, shard_path in enumerate(parquet_shards):
    target_dir = val_dir if idx < args.val_shards else train_dir
    tasks.append((shard_path, target_dir, args.tokenizer_name, idx, dtype_str))

  start_time = time.time()
  total_tokens = 0

  with mp.Pool(processes=args.num_proc) as pool:
    pbar = tqdm(
        pool.imap_unordered(process_shard, tasks),
        total=len(tasks),
        desc="Tokenizing Shards",
        unit="shard",
    )
    for result in pbar:
      if result[0] is not None:
        total_tokens += result[1]
        pbar.set_postfix({"Total Tokens": f"{total_tokens:,}"})

  bytes_per_token = 2 if dtype_str == "uint16" else 4
  elapsed_min = (time.time() - start_time) / 60
  total_gb = (total_tokens * bytes_per_token) / (1024**3)

  print("\n" + "=" * 70)
  print(
      f"🎉 SUCCESS: Tokenized {len(parquet_shards)} shards in"
      f" {elapsed_min:.2f} minutes!"
  )
  print(f"📦 Total Tokens Generated: {total_tokens:,}")
  print(f"💾 Total {dtype_str} Dataset Size: {total_gb:.2f} GB")
  print(f"📁 Binary Dataset Path: {os.path.abspath(args.output_dir)}")
  print("=" * 70 + "\n")


if __name__ == "__main__":
  main()