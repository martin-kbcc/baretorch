import os
import math
import argparse
import logging
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm

from baretorch import BareTorchForCausalLM

# ==============================================================================
#                                Logging Configuration
# ==============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

# ==============================================================================
#                             DDP Environment Setup
# ==============================================================================
def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
    return local_rank, global_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def sync_azure_checkpoint(local_ckpt_path: str, output_dir: str):
    """Asynchronously syncs checkpoint to Azure Blob Storage if credentials exist."""
    blob_url = os.environ.get("AZURE_BLOB_URL", "").strip().rstrip("/")
    sas_token = os.environ.get("AZURE_SAS_TOKEN", "").strip().lstrip("?")
    
    if blob_url and sas_token and os.path.exists(local_ckpt_path):
        ckpt_name = os.path.basename(os.path.normpath(local_ckpt_path))
        rel_output_dir = os.path.basename(os.path.normpath(output_dir))
        target_blob_url = f"{blob_url}/{rel_output_dir}/{ckpt_name}?{sas_token}"
        
        logger.info(f"[Azure Sync] Syncing {ckpt_name} to Azure Blob Storage...")
        cmd = [
            "azcopy", "copy",
            local_ckpt_path,
            target_blob_url,
            "--recursive=true",
            "--overwrite=true"
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==============================================================================
#                       Preference Dataset (ChatML Format)
# ==============================================================================
class PreferenceDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048):
        self.data = []
        max_prompt_len = max_length - 128  # Guarantee at least 128 tokens reserved for completion

        for sample in dataset:
            prompt = sample["instruction"] if "instruction" in sample else sample["prompt"]
            chosen = sample["chosen"] if isinstance(sample["chosen"], str) else sample["chosen"][-1]["content"]
            rejected = sample["rejected"] if isinstance(sample["rejected"], str) else sample["rejected"][-1]["content"]

            prompt_str = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            chosen_resp_str = chosen + "<|im_end|>"
            rejected_resp_str = rejected + "<|im_end|>"

            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)[:max_prompt_len]
            
            # Encode responses with remaining context length budget
            avail_resp_len = max_length - len(prompt_ids)
            chosen_resp_ids = tokenizer.encode(chosen_resp_str, add_special_tokens=False)[:avail_resp_len]
            rejected_resp_ids = tokenizer.encode(rejected_resp_str, add_special_tokens=False)[:avail_resp_len]

            if len(chosen_resp_ids) == 0:
                chosen_resp_ids = [tokenizer.eos_token_id]
            if len(rejected_resp_ids) == 0:
                rejected_resp_ids = [tokenizer.eos_token_id]

            # Guaranteed length matching between input_ids and labels
            chosen_ids = prompt_ids + chosen_resp_ids
            chosen_labels = [-100] * len(prompt_ids) + chosen_resp_ids

            rejected_ids = prompt_ids + rejected_resp_ids
            rejected_labels = [-100] * len(prompt_ids) + rejected_resp_ids

            self.data.append({
                "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
                "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
                "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
                "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_collate(batch, pad_id, pad_to_multiple_of=32):
    """
    Dynamic Batching: Pads sequence tensors to the batch maximum rounded up to 
    a multiple of 32 to guarantee CS-LRAD and CS-TTT chunk-size compatibility.
    """
    def pad_tensor(tensors, pad_value):
        max_len = max(len(t) for t in tensors)
        if pad_to_multiple_of > 0:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            
        padded = torch.full((len(tensors), max_len), pad_value, dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, :len(t)] = t
        return padded

    return {
        "chosen_ids": pad_tensor([b["chosen_ids"] for b in batch], pad_id),
        "chosen_labels": pad_tensor([b["chosen_labels"] for b in batch], -100),
        "rejected_ids": pad_tensor([b["rejected_ids"] for b in batch], pad_id),
        "rejected_labels": pad_tensor([b["rejected_labels"] for b in batch], -100),
    }

# ==============================================================================
#                  Length-Normalized Log Probability Computation
# ==============================================================================
def compute_log_probs(model, input_ids, labels):
    """Computes length-normalized per-sequence log probabilities in bf16 autocast context."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, return_dict=True).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        loss_mask = (shift_labels != -100)
        safe_labels = shift_labels.clone()
        safe_labels[~loss_mask] = 0

        per_token_log_probs = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        per_token_log_probs = per_token_log_probs * loss_mask

        response_lengths = loss_mask.sum(dim=-1).clamp(min=1.0)
        avg_log_probs = per_token_log_probs.sum(dim=-1) / response_lengths
        return avg_log_probs

# ==============================================================================
#                      Distributed Validation Evaluation Loop
# ==============================================================================
@torch.no_grad()
def evaluate_simpo(model, val_loader, args, local_rank, world_size):
    """Computes validation SimPO loss and reward margin synchronized across all DDP GPUs."""
    model.eval()
    total_loss = 0.0
    total_margin = 0.0
    num_batches = 0

    for batch in val_loader:
        chosen_ids = batch["chosen_ids"].to(local_rank)
        chosen_labels = batch["chosen_labels"].to(local_rank)
        rejected_ids = batch["rejected_ids"].to(local_rank)
        rejected_labels = batch["rejected_labels"].to(local_rank)

        log_prob_chosen = compute_log_probs(model, chosen_ids, chosen_labels)
        log_prob_rejected = compute_log_probs(model, rejected_ids, rejected_labels)

        logits_diff = args.beta * (log_prob_chosen - log_prob_rejected) - args.gamma
        loss = -F.logsigmoid(logits_diff).mean()
        reward_margin = (args.beta * (log_prob_chosen - log_prob_rejected)).mean()

        total_loss += loss.item()
        total_margin += reward_margin.item()
        num_batches += 1

    metrics = torch.tensor([total_loss, total_margin, num_batches], device=local_rank, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    avg_loss = (metrics[0] / metrics[2]).item() if metrics[2] > 0 else 0.0
    avg_margin = (metrics[1] / metrics[2]).item() if metrics[2] > 0 else 0.0

    model.train()
    return avg_loss, avg_margin

# ==============================================================================
#                           Main Training Routine
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="BareTorch Stage 2: SimPO Preference Alignment Engine")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_100m_sft/checkpoint-1980", help="Path to baseline SFT checkpoint folder.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_100m_simpo", help="Directory to save aligned SimPO weights.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-GPU batch size.")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=5e-6, help="SimPO learning rate.")
    parser.add_argument("--beta", type=float, default=2.0, help="SimPO reward scaling factor beta.")
    parser.add_argument("--gamma", type=float, default=0.8, help="SimPO target reward margin gamma.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Sub-sample N rows from preference dataset. Set to 0 for full dataset.")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation set ratio (e.g. 0.05 = 5%).")
    parser.add_argument("--eval_steps", type=int, default=50, help="Run validation evaluation every N optimization steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save intermediate checkpoint every N optimization steps.")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length cap.")
    parser.add_argument("--grad_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    args = parser.parse_args()

    local_rank, global_rank, world_size = setup_ddp()
    is_main = (global_rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"🚀 Initializing BareTorch SimPO DDP Training across {world_size} GPU(s)...")

    # 1. Tokenizer Setup with ChatML Special Tokens
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = args.seq_len
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>"], "pad_token": "<|im_end|>"}
    tokenizer.add_special_tokens(special_tokens)

    # 2. Model Loading & Config Setup
    if is_main:
        logger.info(f"Loading baseline SFT weights from: {args.checkpoint_dir}")

    model = BareTorchForCausalLM.from_pretrained(args.checkpoint_dir).to(local_rank)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable KV cache during training
    model.config.use_grad_checkpointing = args.grad_checkpointing

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # 3. Preference Dataset Loading & Train/Val Split
    split_str = f"train[:{args.num_samples}]" if args.num_samples > 0 else "train"
    if is_main:
        logger.info(f"Loading UltraFeedback preference dataset (Split: '{split_str}')...")

    raw_dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split=split_str)
    
    if len(raw_dataset) > 100 and args.val_ratio > 0:
        dataset_split = raw_dataset.train_test_split(test_size=args.val_ratio, seed=42)
        train_raw, val_raw = dataset_split["train"], dataset_split["test"]
    else:
        train_raw, val_raw = raw_dataset, raw_dataset

    train_dataset = PreferenceDataset(train_raw, tokenizer, max_length=args.seq_len)
    val_dataset = PreferenceDataset(val_raw, tokenizer, max_length=args.seq_len)
    
    if is_main:
        logger.info(f"Dataset split complete: {len(train_dataset):,} train samples | {len(val_dataset):,} validation samples.")

    # Train & Val DataLoaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id, pad_to_multiple_of=32),
        num_workers=4,
        pin_memory=True
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id, pad_to_multiple_of=32),
        num_workers=4,
        pin_memory=True
    )

    unwrapped_model = model.module if world_size > 1 else model
    optimizer = torch.optim.AdamW(unwrapped_model.parameters(), lr=args.lr, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    unwrapped_model.train()
    optimizer.zero_grad()

    if is_main:
        logger.info("🔥 Starting Stage 2: SimPO Preference Alignment...")
        logger.info(f"• Total Steps: {total_steps} | Epochs: {args.num_epochs} | Beta: {args.beta} | Gamma: {args.gamma}")

    global_step = 0
    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"SimPO Epoch {epoch+1}/{args.num_epochs}", disable=not is_main)
        for step, batch in enumerate(pbar):
            chosen_ids = batch["chosen_ids"].to(local_rank)
            chosen_labels = batch["chosen_labels"].to(local_rank)
            rejected_ids = batch["rejected_ids"].to(local_rank)
            rejected_labels = batch["rejected_labels"].to(local_rank)

            log_prob_chosen = compute_log_probs(model, chosen_ids, chosen_labels)
            log_prob_rejected = compute_log_probs(model, rejected_ids, rejected_labels)

            logits_diff = args.beta * (log_prob_chosen - log_prob_rejected) - args.gamma
            loss = -F.logsigmoid(logits_diff).mean() / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                train_margin = (args.beta * (log_prob_chosen - log_prob_rejected)).detach().mean().item()
                train_loss_val = loss.item() * args.grad_accum

                # Update Progress Bar
                if is_main:
                    pbar.set_postfix({
                        "loss": f"{train_loss_val:.4f}",
                        "margin": f"{train_margin:.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })

                # Periodic Validation Evaluation
                if global_step % args.eval_steps == 0:
                    val_loss, val_margin = evaluate_simpo(model, val_loader, args, local_rank, world_size)
                    if is_main:
                        logger.info(
                            f"\n📊 [Step {global_step}/{total_steps}] EVALUATION | "
                            f"Train Loss: {train_loss_val:.4f} | Train Margin: {train_margin:.3f} | "
                            f"Val Loss: {val_loss:.4f} | Val Margin: {val_margin:.3f}"
                        )

                # Periodic Step Checkpointing
                if global_step % args.save_steps == 0 and is_main:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"💾 Saving intermediate checkpoint to '{ckpt_dir}'...")
                    unwrapped_model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    sync_azure_checkpoint(ckpt_dir, args.output_dir)

    # Save final aligned checkpoint and tokenizer
    if is_main:
        final_path = os.path.join(args.output_dir, "checkpoint-simpo-final")
        logger.info(f"Saving final SimPO checkpoint to '{final_path}'...")
        unwrapped_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        
        sync_azure_checkpoint(final_path, args.output_dir)
        logger.info("✅ Stage 2: SimPO Preference Alignment completed successfully!")

    cleanup_ddp()

if __name__ == "__main__":
    main()