import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Target directory and dataset categories
DATA_DIR = "./synthetic_teacher_predictions"
SEQ_LEN = 2048
SEQS_PER_DATASET = 20  # Total: 140 sequences (~286k tokens)

DATASETS = {
    "train/fineweb_edu_100bt": "The rapid advancement of artificial intelligence and deep learning architectures has revolutionized computational linguistics. Large language models leverage transformer attention mechanisms to model long-range context across massive web text corpora.",
    "train/stack_dedup": "def train_step(model, batch, optimizer):\n    optimizer.zero_grad()\n    outputs = model(batch['input_ids'])\n    loss = outputs.loss\n    loss.backward()\n    optimizer.step()\n    return loss.item()",
    "train/dclm_100bt": "Geopolitics and global economic trends in the early 21st century have shifted trade dynamics significantly. International supply chains rely heavily on semiconductor manufacturing and logistics optimization.",
    "train/finepdfs_100bt": "Abstract: We present an empirical study on the scaling laws of autoregressive neural networks. By analyzing loss trajectories across floating-point compute regimes, we demonstrate optimal parameter-to-token ratios.",
    "train/cosmopedia_v2": "Chapter 4: Introduction to Quantum Mechanics. Imagine an electron trapped in a potential well. Unlike classical particles, its energy states are quantized, defined by the Schrödinger wave equation.",
    "train/finemath_4plus": "Theorem 3.2: Let f be a continuous function on the closed interval [a, b]. By the Mean Value Theorem, there exists at least one point c in (a, b) such that f'(c) equals the average rate of change.",
    "train/openr1_math": "<thought>To solve x^2 - 5x + 6 = 0, I can factor the quadratic equation into (x - 2)(x - 3) = 0. Therefore, the roots are x = 2 and x = 3.</thought>\nFinal Answer: {2, 3}",
}

print("📥 Loading Qwen/Qwen3.5-9B Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", trust_remote_code=True)

for rel_path, sample_text in DATASETS.items():
    out_dir = os.path.join(DATA_DIR, rel_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # Encode text into token IDs
    raw_ids = tokenizer.encode(sample_text)
    
    # Repeat text to fill SEQ_LEN * SEQS_PER_DATASET
    repeated_ids = (raw_ids * ((SEQ_LEN * SEQS_PER_DATASET // len(raw_ids)) + 1))[: SEQ_LEN * SEQS_PER_DATASET]
    tokens = np.array(repeated_ids, dtype=np.uint32).reshape(SEQS_PER_DATASET, SEQ_LEN)
    
    # Generate structured top-4 teacher predictions
    # Top-1 index = ground-truth token, Top-2..4 = random plausible tokens
    indices = np.zeros((SEQS_PER_DATASET, SEQ_LEN, 4), dtype=np.uint32)
    indices[:, :, 0] = tokens  # Top-1 is ground truth
    indices[:, :, 1:] = np.random.randint(0, len(tokenizer), size=(SEQS_PER_DATASET, SEQ_LEN, 3), dtype=np.uint32)
    
    # Generate peaked teacher logits (Top-1 has high probability)
    values = np.zeros((SEQS_PER_DATASET, SEQ_LEN, 4), dtype=np.float16)
    values[:, :, 0] = 5.0  # High logit for ground truth
    values[:, :, 1:] = np.random.randn(SEQS_PER_DATASET, SEQ_LEN, 3) * 0.5
    
    # Write files
    prefix = os.path.join(out_dir, "shard_00000")
    tokens.tofile(f"{prefix}_packed_tokens.bin")
    indices.tofile(f"{prefix}_teacher_indices.bin")
    values.tofile(f"{prefix}_teacher_values.bin")

# Create a small val set copy
for rel_path in DATASETS.keys():
    train_folder = os.path.join(DATA_DIR, rel_path)
    val_folder = train_folder.replace("train/", "val/")
    os.makedirs(val_folder, exist_ok=True)
    for src_file in glob.glob(f"{train_folder}/*.bin"):
        dst_file = os.path.join(val_folder, os.path.basename(src_file))
        with open(src_file, "rb") as f_in, open(dst_file, "wb") as f_out:
            f_out.write(f_in.read())

print(f"✅ Real synthetic dataset created at '{DATA_DIR}'!")