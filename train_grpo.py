import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
from datasets import load_dataset
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from baretorch import BareTorchForCausalLM

# ==============================================================================
#                                Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)


# ==============================================================================
#                                DDP Environment Setup
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


def sync_r2_checkpoint(
    local_ckpt_path: str,
    output_dir: str,
    r2_bucket: str = "baretorch-data",
    r2_prefix: str = "checkpoints",
    sync_async: bool = True,
):
    """Syncs checkpoint to Cloudflare R2 via rclone."""
    if not os.path.exists(local_ckpt_path):
        return

    ckpt_name = os.path.basename(os.path.normpath(local_ckpt_path))
    rel_output_dir = os.path.basename(os.path.normpath(output_dir))
    target_r2_path = (
        f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}/{ckpt_name}"
    )

    logger.info(
        f"[R2 Sync] Syncing {ckpt_name} to Cloudflare R2 ({target_r2_path})..."
    )
    cmd = [
        "rclone",
        "copy",
        local_ckpt_path,
        target_r2_path,
        "--transfers",
        "8",
        "--s3-provider",
        "Cloudflare",
    ]
    if sync_async:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(cmd, check=False)


# ==============================================================================
#                        CSV Metric Logger Utility
# ==============================================================================
def log_csv_metrics(
    csv_path: str,
    step: int,
    epoch: int,
    loss: float,
    train_acc: float,
    train_fmt: float,
    lr: float,
    val_acc: float = None,
    val_fmt: float = None,
):
    """Appends training and validation metrics to a structured CSV file."""
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "step",
                "epoch",
                "loss",
                "train_acc",
                "train_fmt",
                "val_acc",
                "val_fmt",
                "lr",
            ])
        writer.writerow([
            step,
            epoch,
            f"{loss:.6f}",
            f"{train_acc:.4f}",
            f"{train_fmt:.4f}",
            f"{val_acc:.4f}" if val_acc is not None else "",
            f"{val_fmt:.4f}" if val_fmt is not None else "",
            f"{lr:.6e}",
        ])


# ==============================================================================
#                     GSM8K Dataset & Verifiable Reward Logic
# ==============================================================================
def parse_gsm8k_target(answer_str: str) -> str:
    """Extracts ground-truth target answer string from GSM8K answer text."""
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip().replace(",", "")
    return answer_str.strip()


def extract_model_answer(completion_text: str) -> str:
    """Extracts answer inside \boxed{...} or falls back to final trailing number."""
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", completion_text)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion_text)
    if numbers:
        return numbers[-1].strip()
    return ""


def compute_verifiable_rewards(completion_text: str, target_answer: str):
    """Computes deterministic rewards with partial format credit."""
    extracted = extract_model_answer(completion_text)
    acc_reward = 1.0 if extracted == target_answer else 0.0

    fmt_reward = 0.0
    if "<think>" in completion_text:
        fmt_reward += 0.25
    if "</think>" in completion_text:
        fmt_reward += 0.25
    if "\\boxed{" in completion_text:
        fmt_reward += 0.50

    total_reward = acc_reward + (0.2 * fmt_reward)
    return total_reward, acc_reward, fmt_reward


class GSM8KDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_prompt_len=512):
        self.samples = []
        system_prompt = (
            "<|im_start|>system\nYou are a helpful AI assistant that solves math"
            " problems step-by-step. Put your reasoning inside <think>...</think>"
            " tags and write the final answer inside \\boxed{...}.<|im_end|>\n"
        )

        for sample in dataset:
            question = sample["question"]
            answer = sample["answer"]
            target = parse_gsm8k_target(answer)

            formatted_prompt = (
                f"{system_prompt}<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            )
            prompt_ids = tokenizer.encode(
                formatted_prompt, add_special_tokens=False
            )[:max_prompt_len]

            self.samples.append({
                "prompt_str": formatted_prompt,
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "target_answer": target,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate_grpo(batch, pad_id):
    """Collates prompts with LEFT-PADDING for autoregressive GRPO rollouts."""
    max_len = max(len(b["prompt_ids"]) for b in batch)
    batch_size = len(batch)

    padded_prompts = torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    for i, b in enumerate(batch):
        p_ids = b["prompt_ids"]
        padded_prompts[i, max_len - len(p_ids) :] = p_ids

    return {
        "prompt_ids": padded_prompts,
        "prompt_strs": [b["prompt_str"] for b in batch],
        "target_answers": [b["target_answer"] for b in batch],
    }


# ==============================================================================
#                 Per-Token Log Probabilities Computation
# ==============================================================================
def compute_per_token_log_probs(
    model, input_ids, prompt_len, pad_token_id, eos_token_id
):
    """Computes log probabilities with safe torch.where masking and isolated EOS tracking."""
    attn_mask = (input_ids != pad_token_id).long()

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids, attention_mask=attn_mask, return_dict=True
        ).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits.float(), dim=-1)

        seq_len = shift_labels.size(1)
        idx = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        in_completion = idx >= (prompt_len - 1)

        # Restrict EOS counting exclusively to completion tokens to protect left-padded prompts
        is_eos_in_completion = (shift_labels == eos_token_id) & in_completion
        cum_eos = torch.cumsum(is_eos_in_completion.long(), dim=-1)
        valid_completion = (cum_eos == 0) | ((cum_eos == 1) & is_eos_in_completion)

        mask = in_completion & valid_completion

        safe_labels = shift_labels.clone()
        safe_labels[~mask] = 0

        per_token_lp = torch.gather(
            log_probs, dim=2, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Safe Masking: Replaces -inf * 0 = NaN with exact zero assignment
        per_token_lp = torch.where(
            mask, per_token_lp, torch.zeros_like(per_token_lp)
        )
        return per_token_lp, mask


# ==============================================================================
#                      Distributed GRPO Validation Loop
# ==============================================================================
@torch.no_grad()
def evaluate_grpo(model, val_loader, tokenizer, args, local_rank, world_size):
    """Runs evaluation rollouts and computes average accuracy & format scores."""
    model.eval()
    unwrapped = model.module if hasattr(model, "module") else model

    total_acc = 0.0
    total_fmt = 0.0
    total_samples = 0

    for batch in val_loader:
        prompt_ids = batch["prompt_ids"].to(local_rank)
        targets = batch["target_answers"]
        prompt_attn_mask = (prompt_ids != tokenizer.pad_token_id).long()

        unwrapped.config.use_cache = True
        outputs = unwrapped.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_attn_mask,
            max_new_tokens=args.max_completion_len,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        unwrapped.config.use_cache = False

        for i, out in enumerate(outputs):
            comp_text = tokenizer.decode(
                out[prompt_ids.size(1) :], skip_special_tokens=False
            )
            _, acc, fmt = compute_verifiable_rewards(comp_text, targets[i])
            total_acc += acc
            total_fmt += fmt
            total_samples += 1

    metrics = torch.tensor(
        [total_acc, total_fmt, total_samples],
        device=local_rank,
        dtype=torch.float32,
    )
    if world_size > 1:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    avg_acc = (metrics[0] / metrics[2]).item() if metrics[2] > 0 else 0.0
    avg_fmt = (metrics[1] / metrics[2]).item() if metrics[2] > 0 else 0.0

    model.train()
    return avg_acc, avg_fmt


# ==============================================================================
#                            Main Training Routine
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="BareTorch Stage 3: Rule-Based RL (GRPO / RLVR) Engine"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints_500m_sft_cold_start",
        help="Path to baseline Cold-Start CoT SFT checkpoint folder.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to specific GRPO checkpoint directory to resume state from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_500m_grpo",
        help="Directory to save GRPO aligned weights and metrics.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Prompts per GPU per step."
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Group size G: candidate completions sampled per prompt.",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=4, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-6, help="GRPO learning rate."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL penalty coefficient against reference model.",
    )
    parser.add_argument(
        "--clip_eps", type=float, default=0.2, help="PPO ratio clipping epsilon."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help=(
            "Sub-sample N questions from GSM8K dataset. Set to 0 for full"
            " dataset."
        ),
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set split ratio.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=25,
        help="Run evaluation every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save intermediate checkpoint every N steps.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="Prompt context token cap.",
    )
    parser.add_argument(
        "--max_completion_len",
        type=int,
        default=512,
        help="Max new rollout completion tokens.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable targeted torch.compile for CS-LRAD sub-modules.",
    )

    # Cloud Storage / Sync
    parser.add_argument(
        "--r2_sync",
        action="store_true",
        help=(
            "Enable background checkpoint syncing to Cloudflare R2 via rclone."
        ),
    )
    parser.add_argument(
        "--r2_bucket",
        type=str,
        default="baretorch-data",
        help="Cloudflare R2 bucket name.",
    )
    parser.add_argument(
        "--r2_prefix",
        type=str,
        default="checkpoints",
        help="Prefix path inside R2 bucket.",
    )

    args = parser.parse_args()

    local_rank, global_rank, world_size = setup_ddp()
    is_main = global_rank == 0
    # Enable TF32 Tensor Cores for FP32 matmuls on RTX 4090s
    torch.set_float32_matmul_precision("high")

    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(
            "🚀 Initializing BareTorch GRPO DDP Training across"
            f" {world_size} GPU(s)..."
        )

    csv_metrics_path = os.path.join(args.output_dir, "metrics.csv")
    actor_ckpt_path = (
        args.resume_from_checkpoint
        if args.resume_from_checkpoint
        else args.checkpoint_dir
    )

    # 1. Tokenizer Setup (Try checkpoint directory first)
    try:
        tokenizer = AutoTokenizer.from_pretrained(actor_ckpt_path)
        if is_main:
            logger.info(f"Loaded tokenizer from checkpoint: {actor_ckpt_path}")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        if is_main:
            logger.info(f"Loaded base tokenizer: {args.tokenizer_name}")

    tokenizer.padding_side = "left"
    desired_tokens = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    existing_vocab = tokenizer.get_vocab()
    tokens_to_add = [t for t in desired_tokens if t not in existing_vocab]

    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    if "<|im_end|>" in tokenizer.get_vocab():
        tokenizer.pad_token = "<|im_end|>"
        tokenizer.eos_token = "<|im_end|>"

    # 2. Model Loading
    if is_main:
        logger.info(f"Loading Actor Policy from: {actor_ckpt_path}")
        logger.info(
            "Loading Reference Policy (Frozen) from initial checkpoint:"
            f" {args.checkpoint_dir}"
        )

    actor_model = BareTorchForCausalLM.from_pretrained(
        actor_ckpt_path, torch_dtype=torch.bfloat16
    ).to(local_rank)
    if len(tokenizer) > actor_model.get_input_embeddings().weight.shape[0]:
        actor_model.resize_token_embeddings(
            len(tokenizer), pad_to_multiple_of=64, mean_resizing=False
        )
    actor_model.config.pad_token_id = tokenizer.pad_token_id
    actor_model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    actor_model.config.use_cache = False

    ref_model = BareTorchForCausalLM.from_pretrained(
        args.checkpoint_dir, torch_dtype=torch.bfloat16
    ).to(local_rank)
    if len(tokenizer) > ref_model.get_input_embeddings().weight.shape[0]:
        ref_model.resize_token_embeddings(
            len(tokenizer), pad_to_multiple_of=64, mean_resizing=False
        )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Targeted Sub-Module Compilation (ONLY LRADDecoderBlock instances to prevent outer model compilation nesting)
    if args.compile:
        if is_main:
            logger.info(
                "⚡ Applying targeted torch.compile strictly to LRADDecoderBlock sub-modules..."
            )
        compiled_blocks = 0
        for name, module in actor_model.named_modules():
            if module.__class__.__name__ == "LRADDecoderBlock":
                module.forward = torch.compile(module.forward)
                compiled_blocks += 1

        for name, module in ref_model.named_modules():
            if module.__class__.__name__ == "LRADDecoderBlock":
                module.forward = torch.compile(module.forward)

        if is_main:
            logger.info(
                f"Successfully compiled {compiled_blocks} LRADDecoderBlock(s)."
            )

    if world_size > 1:
        actor_model = DDP(
            actor_model, device_ids=[local_rank], find_unused_parameters=False
        )

    # 3. Dataset Loading
    split_str = (
        f"train[:{args.num_samples}]" if args.num_samples > 0 else "train"
    )
    raw_dataset = load_dataset("openai/gsm8k", "main", split=split_str)

    if len(raw_dataset) > 50 and args.val_ratio > 0:
        ds_split = raw_dataset.train_test_split(test_size=args.val_ratio, seed=42)
        train_raw, val_raw = ds_split["train"], ds_split["test"]
    else:
        train_raw, val_raw = raw_dataset, raw_dataset

    train_dataset = GSM8KDataset(
        train_raw, tokenizer, max_prompt_len=args.max_prompt_len
    )
    val_dataset = GSM8KDataset(
        val_raw, tokenizer, max_prompt_len=args.max_prompt_len
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True,
        )
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda b: pad_collate_grpo(b, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
        )
        if world_size > 1
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=lambda b: pad_collate_grpo(b, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True,
    )

    unwrapped_actor = actor_model.module if world_size > 1 else actor_model
    optimizer = torch.optim.AdamW(
        unwrapped_actor.parameters(), lr=args.lr, weight_decay=0.01
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # 4. Resume State
    start_step = 0
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        opt_path = os.path.join(args.resume_from_checkpoint, "optimizer.pt")
        sched_path = os.path.join(args.resume_from_checkpoint, "scheduler.pt")
        state_path = os.path.join(args.resume_from_checkpoint, "training_state.json")

        if os.path.exists(opt_path):
            optimizer.load_state_dict(
                torch.load(opt_path, map_location=f"cuda:{local_rank}")
            )
        if os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                start_step = json.load(f).get("global_step", 0)

    unwrapped_actor.train()
    optimizer.zero_grad()

    if is_main:
        logger.info("🔥 Starting Stage 3: Rule-Based RL (GRPO) Alignment...")

    global_step = start_step
    total_batches_to_skip = start_step * args.grad_accum
    batch_counter = 0

    accum_acc_scores = []
    accum_fmt_scores = []
    accum_losses = []

    for epoch in range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(
            train_loader,
            desc=f"GRPO Epoch {epoch+1}/{args.num_epochs}",
            disable=not is_main,
        )
        for step, batch in enumerate(pbar):
            batch_counter += 1

            if batch_counter <= total_batches_to_skip:
                continue

            prompt_ids = batch["prompt_ids"].to(local_rank)
            targets = batch["target_answers"]
            B, P = prompt_ids.size()

            # Rollout Generation with Prompt Attention Mask
            unwrapped_actor.eval()
            unwrapped_actor.config.use_cache = True
            repeated_prompts = prompt_ids.repeat_interleave(
                args.num_generations, dim=0
            )
            prompt_attn_mask = (repeated_prompts != tokenizer.pad_token_id).long()

            with torch.no_grad():
                rollout_outputs = unwrapped_actor.generate(
                    input_ids=repeated_prompts,
                    attention_mask=prompt_attn_mask,
                    max_new_tokens=args.max_completion_len,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            unwrapped_actor.config.use_cache = False
            unwrapped_actor.train()

            # Reward Calculation & Group Advantage Normalization
            rewards = []
            for i in range(B * args.num_generations):
                prompt_idx = i // args.num_generations
                comp_text = tokenizer.decode(
                    rollout_outputs[i, P:], skip_special_tokens=False
                )
                tot_r, acc_r, fmt_r = compute_verifiable_rewards(
                    comp_text, targets[prompt_idx]
                )
                rewards.append(tot_r)
                accum_acc_scores.append(acc_r)
                accum_fmt_scores.append(fmt_r)

            reward_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=local_rank
            ).view(B, args.num_generations)

            mean_r = reward_tensor.mean(dim=1, keepdim=True)
            std_r = reward_tensor.std(dim=1, keepdim=True)
            advantages = ((reward_tensor - mean_r) / (std_r + 1e-8)).view(-1)

            # Synchronize sequence length across DDP ranks
            max_seq = rollout_outputs.size(1)
            if world_size > 1:
                max_seq_tensor = torch.tensor([max_seq], device=local_rank)
                dist.all_reduce(max_seq_tensor, op=dist.ReduceOp.MAX)
                max_seq = max_seq_tensor.item()

                if rollout_outputs.size(1) < max_seq:
                    pad_len = max_seq - rollout_outputs.size(1)
                    pad_suffix = torch.full(
                        (rollout_outputs.size(0), pad_len),
                        tokenizer.pad_token_id,
                        device=local_rank,
                    )
                    rollout_outputs = torch.cat([rollout_outputs, pad_suffix], dim=1)

            padded_len = ((max_seq + 31) // 32) * 32
            if padded_len > max_seq:
                pad_suffix = torch.full(
                    (B * args.num_generations, padded_len - max_seq),
                    tokenizer.pad_token_id,
                    device=local_rank,
                )
                padded_rollouts = torch.cat([rollout_outputs, pad_suffix], dim=1)
            else:
                padded_rollouts = rollout_outputs

            actor_lp, mask = compute_per_token_log_probs(
                actor_model,
                padded_rollouts,
                P,
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
            )

            with torch.no_grad():
                ref_lp, _ = compute_per_token_log_probs(
                    ref_model,
                    padded_rollouts,
                    P,
                    tokenizer.pad_token_id,
                    tokenizer.eos_token_id,
                )

            log_ratio = actor_lp - actor_lp.detach()
            ratio = torch.exp(log_ratio)

            adv_expanded = advantages.unsqueeze(-1)
            surr1 = ratio * adv_expanded
            surr2 = (
                torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                * adv_expanded
            )

            policy_loss = -torch.min(surr1, surr2)

            # Bounded KL penalty calculation (prevents gradient explosion when policy diverges)
            log_diff = torch.clamp(ref_lp - actor_lp, min=-10.0, max=10.0)
            kl_penalty = torch.exp(log_diff) - log_diff - 1.0
            kl_penalty = torch.clamp(kl_penalty, min=0.0, max=10.0)

            # Safe Loss Masking: Replaces invalid positions with zero prior to reduction
            raw_grpo = policy_loss + args.beta * kl_penalty
            masked_grpo = torch.where(mask, raw_grpo, torch.zeros_like(raw_grpo))

            grpo_loss = masked_grpo.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            total_loss = grpo_loss.mean() / args.grad_accum
            accum_losses.append(total_loss.item() * args.grad_accum)

            total_loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                # Defensive check: verify gradients are finite before stepping optimizer
                valid_grads = True
                for p in unwrapped_actor.parameters():
                    if p.grad is not None and (
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    ):
                        valid_grads = False
                        break

                if valid_grads:
                    torch.nn.utils.clip_grad_norm_(
                        unwrapped_actor.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                else:
                    if is_main:
                        logger.warning(
                            f"⚠️ [Step {global_step}] Skipped optimizer.step() due to"
                            " NaN/Inf in gradients."
                        )

                optimizer.zero_grad()
                global_step += 1

                mean_acc = (
                    sum(accum_acc_scores) / max(len(accum_acc_scores), 1)
                )
                mean_fmt = (
                    sum(accum_fmt_scores) / max(len(accum_fmt_scores), 1)
                )
                loss_val = sum(accum_losses) / max(len(accum_losses), 1)
                current_lr = scheduler.get_last_lr()[0]

                accum_acc_scores.clear()
                accum_fmt_scores.clear()
                accum_losses.clear()

                if is_main:
                    pbar.set_postfix({
                        "step": f"{global_step}/{total_steps}",
                        "loss": f"{loss_val:.4f}",
                        "acc": f"{mean_acc:.2f}",
                        "fmt": f"{mean_fmt:.2f}",
                        "lr": f"{current_lr:.2e}",
                    })

                val_acc_log, val_fmt_log = None, None

                if global_step > 0 and global_step % args.eval_steps == 0:
                    val_acc_log, val_fmt_log = evaluate_grpo(
                        actor_model, val_loader, tokenizer, args, local_rank, world_size
                    )
                    if is_main:
                        logger.info(
                            f"\n📊 [Optimizer Step {global_step}/{total_steps}] GRPO"
                            f" EVALUATION | Train Acc: {mean_acc:.2f} | Train Format:"
                            f" {mean_fmt:.2f} | Val Acc: {val_acc_log:.2f} | Val Format:"
                            f" {val_fmt_log:.2f}"
                        )

                if is_main:
                    log_csv_metrics(
                        csv_metrics_path,
                        step=global_step,
                        epoch=epoch + 1,
                        loss=loss_val,
                        train_acc=mean_acc,
                        train_fmt=mean_fmt,
                        lr=current_lr,
                        val_acc=val_acc_log,
                        val_fmt=val_fmt_log,
                    )

                if (
                    global_step > 0
                    and global_step % args.save_steps == 0
                    and is_main
                ):
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    logger.info(f"💾 Saving complete GRPO checkpoint to '{ckpt_dir}'...")
                    unwrapped_actor.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)

                    torch.save(
                        optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt")
                    )
                    with open(
                        os.path.join(ckpt_dir, "training_state.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            {"global_step": global_step, "epoch": epoch + 1}, f, indent=2
                        )

                    if enable_r2_sync:
                        sync_r2_checkpoint(
                            ckpt_dir,
                            args.output_dir,
                            r2_bucket,
                            r2_prefix,
                            sync_async=True,
                        )

    if is_main:
        final_path = os.path.join(args.output_dir, "checkpoint-grpo-final")
        logger.info(f"Saving final GRPO checkpoint to '{final_path}'...")
        unwrapped_actor.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

        torch.save(
            optimizer.state_dict(), os.path.join(final_path, "optimizer.pt")
        )
        torch.save(
            scheduler.state_dict(), os.path.join(final_path, "scheduler.pt")
        )
        with open(
            os.path.join(final_path, "training_state.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(
                {"global_step": global_step, "epoch": args.num_epochs}, f, indent=2
            )

        if enable_r2_sync:
            sync_r2_checkpoint(
                final_path, args.output_dir, r2_bucket, r2_prefix, sync_async=False
            )
        logger.info("✅ Stage 3: Rule-Based RL (GRPO) Alignment completed successfully!")

    cleanup_ddp()


if __name__ == "__main__":
    main()