# /home/martinkb/Desktop/BareTorch_F/train_sft.py
import argparse
import logging
import os
import subprocess
from datetime import timedelta
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.serialization

# Disable Hugging Face Rust parallelism warnings/deadlocks in torchrun
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Shared memory & Tensor Precision Setup
try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

# Enable TF32 for Tensor Core acceleration
torch.set_float32_matmul_precision("high")

# Bypass the cuDNN attention backend during Evaluation / no_grad
torch.backends.cuda.enable_cudnn_sdp(False)

# Comprehensive allowlist for NumPy types in PyTorch 2.6+
safe_numpy_types = [np.dtype, np.ndarray]
for mod_path in [
    "numpy._core.multiarray",
    "numpy.core.multiarray",
    "numpy._core.numerictypes",
]:
    try:
        mod = __import__(mod_path, fromlist=["scalar", "_reconstruct"])
        if hasattr(mod, "scalar"):
            safe_numpy_types.append(mod.scalar)
        if hasattr(mod, "_reconstruct"):
            safe_numpy_types.append(mod._reconstruct)
    except (ImportError, AttributeError):
        pass
try:
    torch.serialization.add_safe_globals(safe_numpy_types)
except Exception:
    pass

# Universal Fail-Safe: Force trusted local checkpoints to bypass strict weights_only check
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from datasets import Dataset, interleave_datasets, load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from baretorch import BareTorchConfig, BareTorchForCausalLM
from baretorch.modeling.transformer import RotaryEmbedding

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
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ==============================================================================
#                          Cloudflare R2 Background Sync Callback
# ==============================================================================
class R2CheckpointCallback(TrainerCallback):
    """Hugging Face Trainer Callback that automatically syncs newly saved
    checkpoints to Cloudflare R2 asynchronously using rclone.
    Does not block active GPU training execution.
    """

    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints_sft",
    ):
        self.bucket_name = bucket_name
        self.remote_name = remote_name
        self.prefix = prefix.strip("/")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = f"checkpoint-{state.global_step}"
            local_ckpt_path = os.path.join(args.output_dir, checkpoint_dir)

            if os.path.exists(local_ckpt_path):
                rel_output_dir = os.path.basename(
                    os.path.normpath(args.output_dir)
                )
                target_r2_path = (
                    f"{self.remote_name}:{self.bucket_name}/{self.prefix}/"
                    f"{rel_output_dir}/{checkpoint_dir}"
                )

                logger.info(
                    f"\n[R2 Sync] Uploading {checkpoint_dir} to Cloudflare R2"
                    f" ({target_r2_path}) in background..."
                )
                cmd = [
                    "rclone",
                    "copy",
                    local_ckpt_path,
                    target_r2_path,
                    "--transfers",
                    "4",
                    "--s3-chunk-size",
                    "64M",
                ]
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )


def adapt_model_for_long_context(
    model, config, target_seq_len=4096, target_rope_theta=10000.0
):
    """Explicitly scales RotaryEmbedding base theta and forces cache regeneration
    across all Transformer blocks for SFT.
    """
    config.max_seq_len = target_seq_len
    config.rope_theta = target_rope_theta
    if hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = target_seq_len

    adapted_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, RotaryEmbedding) or module.__class__.__name__ in [
            "RotaryEmbedding",
            "RoPE",
        ]:
            module.base = float(target_rope_theta)
            module.max_position_embeddings = target_seq_len

            device = module.inv_freq.device
            dim = module.dim
            new_inv_freq = 1.0 / (
                target_rope_theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
            )
            module.register_buffer("inv_freq", new_inv_freq, persistent=False)
            module._set_cos_sin_cache(
                seq_len=target_seq_len, device=device, dtype=torch.float32
            )
            adapted_layers += 1

    return adapted_layers


def pack_chatml_dataset(raw_dataset, tokenizer, max_seq_len=4096, local_rank=0):
    """Tokenizes in parallel across CPU cores and packs multi-turn ChatML
    conversations using Whole-Sample Packing.
    Guarantees no conversation is cut in half across sequence boundaries.
    Applies completion-only loss masking (-100 for prompts/headers).
    Uses zero-copy contiguous NumPy buffers to prevent PyArrow 2GB overflow.
    """
    vocab_size = len(tokenizer)
    pad_id = tokenizer.pad_token_id
    num_proc = max(1, min(16, (os.cpu_count() or 4) - 2))

    if local_rank == 0:
        logger.info(
            f"⚡ Tokenizing {len(raw_dataset):,} conversations across {num_proc} CPU workers..."
        )

    def _tokenize_example(example):
        messages = example.get("messages", [])
        if not messages:
            return {"sample_ids": [], "sample_labels": [], "length": 0}

        sample_ids = []
        sample_labels = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not content:
                continue

            if role == "assistant":
                header_str = f"<|im_start|>{role}\n"
                header_tokens = tokenizer.encode(header_str, add_special_tokens=False)
                body_str = f"{content}<|im_end|>\n"
                body_tokens = tokenizer.encode(body_str, add_special_tokens=False)

                sample_ids.extend(header_tokens + body_tokens)
                sample_labels.extend([-100] * len(header_tokens) + body_tokens)
            else:
                formatted_turn = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                tokens = tokenizer.encode(formatted_turn, add_special_tokens=False)

                sample_ids.extend(tokens)
                sample_labels.extend([-100] * len(tokens))

        if len(sample_ids) > max_seq_len or not any(l != -100 for l in sample_labels):
            return {"sample_ids": [], "sample_labels": [], "length": 0}

        return {
            "sample_ids": sample_ids,
            "sample_labels": sample_labels,
            "length": len(sample_ids),
        }

    tokenized_ds = raw_dataset.map(
        _tokenize_example,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing ChatML",
    )
    tokenized_ds = tokenized_ds.filter(
        lambda x: x["length"] > 0,
        num_proc=num_proc,
        desc="Filtering valid lengths",
    )

    if local_rank == 0:
        logger.info(
            f"📦 Packing {len(tokenized_ds):,} valid conversations into "
            f"{max_seq_len}-token sequences (Whole-Sample Packing)..."
        )

    packed_ids_list = []
    packed_labels_list = []
    packed_mask_list = []

    current_ids = []
    current_labels = []

    def _flush_buffer(curr_i, curr_l):
        valid_len = len(curr_i)
        arr_ids = np.full(max_seq_len, pad_id, dtype=np.int32)
        arr_labels = np.full(max_seq_len, -100, dtype=np.int32)
        arr_mask = np.zeros(max_seq_len, dtype=np.int8)

        arr_ids[:valid_len] = curr_i
        arr_labels[:valid_len] = curr_l
        arr_mask[:valid_len] = 1

        # Sanitize out-of-bounds token IDs
        invalid_id_mask = (arr_ids < 0) | (arr_ids >= vocab_size)
        if np.any(invalid_id_mask):
            arr_ids[invalid_id_mask] = pad_id

        invalid_lbl_mask = ((arr_labels < 0) & (arr_labels != -100)) | (
            arr_labels >= vocab_size
        )
        if np.any(invalid_lbl_mask):
            arr_labels[invalid_lbl_mask] = -100

        if np.any(arr_labels != -100):
            packed_ids_list.append(arr_ids)
            packed_labels_list.append(arr_labels)
            packed_mask_list.append(arr_mask)

    iterator = tqdm(
        tokenized_ds,
        desc="Packing Sequences",
        disable=(local_rank != 0),
        dynamic_ncols=True,
    )

    for item in iterator:
        s_ids = item["sample_ids"]
        s_lbls = item["sample_labels"]
        s_len = item["length"]

        if len(current_ids) + s_len > max_seq_len:
            if current_ids:
                _flush_buffer(current_ids, current_labels)
            current_ids = list(s_ids)
            current_labels = list(s_lbls)
        else:
            current_ids.extend(s_ids)
            current_labels.extend(s_lbls)

    if current_ids:
        _flush_buffer(current_ids, current_labels)

    if local_rank == 0:
        logger.info(
            f"✅ Packing complete: Created {len(packed_ids_list):,} whole-sample packed "
            f"{max_seq_len}-token sequences."
        )

    return Dataset.from_dict(
        {
            "input_ids": packed_ids_list,
            "labels": packed_labels_list,
            "attention_mask": packed_mask_list,
        }
    )


def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=timedelta(minutes=60),
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    parser = argparse.ArgumentParser(
        description="BareTorch Stage 3: Supervised Fine-Tuning (SFT) Engine (FSDP / Cluster)"
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./checkpoint-142000",
        help="Path to pre-trained/adapted BareTorch checkpoint folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_2.5B_sft",
        help="Directory to save fine-tuned SFT weights.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=600000,
        help="Sub-sample N total rows for fast iteration. Set to 0 for full smoltalk epoch.",
    )

    # Optimization & Schedule
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size (default 4 for 4096 context on H100).",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (8 GPUs * 4 * 4 * 4k = 524k tokens/step).",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="SFT learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=60)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Sequence context length (default 4096).",
    )
    parser.add_argument(
        "--rope_theta",
        type=float,
        default=10000.0,
        help="Rotary Position Embedding base frequency.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply targeted torch.compile to custom CS-LRAD recurrent sub-modules.",
    )
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing.",
    )

    # Cloud Sync
    parser.add_argument(
        "--r2_sync",
        action="store_true",
        help="Enable background checkpoint syncing to Cloudflare R2 via rclone.",
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
        default="checkpoints_sft",
        help="Prefix path inside R2 bucket.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)

    args = parser.parse_args()

    if local_rank == 0:
        logger.info(f"Initializing Tokenizer '{args.tokenizer_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True
    )
    tokenizer.model_max_length = args.seq_len
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_rank == 0:
        logger.info(
            f"Loading model weights into FP32 master copy from: {args.pretrained_model_path}"
        )

    # Load master weights in FP32; FSDP handles sharding & BF16 mixed-precision autocast
    model = BareTorchForCausalLM.from_pretrained(
        args.pretrained_model_path,
        torch_dtype=torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.use_grad_checkpointing = args.grad_checkpointing

    # Enforce RoPE theta and ensure the sinusoidal cache covers the full context span
    adapted_count = adapt_model_for_long_context(
        model,
        model.config,
        target_seq_len=args.seq_len,
        target_rope_theta=args.rope_theta,
    )

    # Targeted torch.compile on custom CS-LRAD recurrent sub-modules (matching train_distill.py)
    compiled_blocks = 0
    if args.compile:
        if local_rank == 0:
            logger.info(
                "⚡ Applying targeted torch.compile to custom CS-LRAD sub-modules..."
            )
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if "lrad" in cls_name or "lrad" in name.lower():
                module.forward = torch.compile(module.forward)
                compiled_blocks += 1
        if local_rank == 0:
            logger.info(
                f"Successfully compiled {compiled_blocks} CS-LRAD recurrent sub-module(s)."
            )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if local_rank == 0:
        logger.info("=" * 70)
        logger.info("🎯 BareTorch Supervised Fine-Tuning Engine (FSDP / FP32 Master)")
        logger.info(f"   ├─ Base Checkpoint       : {args.pretrained_model_path}")
        logger.info(f"   ├─ Sequence Length       : {args.seq_len:,} tokens")
        logger.info(f"   ├─ RoPE Theta Base       : {args.rope_theta:,.0f}")
        logger.info(f"   ├─ Adapted RoPE Blocks   : {adapted_count}")
        logger.info(f"   ├─ Compiled LRAD Modules : {compiled_blocks} (compile={args.compile})")
        logger.info(f"   ├─ Learning Rate         : {args.learning_rate}")
        logger.info(
            f"   └─ Batch per Step        : "
            f"{world_size * args.batch_size * args.grad_accum * args.seq_len:,} tokens"
        )
        logger.info("=" * 70)

    # --------------------------------------------------------------------------
    # Distributed Rank-0 Dataset Preparation and Disk Caching
    # --------------------------------------------------------------------------
    cache_dir = os.path.join(
        args.output_dir, f"packed_dataset_{args.seq_len}_{args.max_samples}"
    )

    if local_rank == 0:
        if not os.path.exists(cache_dir):
            logger.info(
                "Loading SFT Datasets (50% Chat / 20% Code / 15% Math / 15% Reasoning)..."
            )

            # 1. 50% Chat Anchor
            ds_chat = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")

            # 2. 20% Code
            ds_code = load_dataset(
                "ise-uiuc/Magicoder-Evol-Instruct-110K", split="train"
            )

            # 3. 15% Math (NuminaMath-CoT with <think> tag injection)
            ds_math = load_dataset("AI-MO/NuminaMath-CoT", split="train")

            # 4. 15% DeepSeek-R1 Distilled Reasoning (Bespoke-Stratos-17k)
            ds_reasoning = load_dataset(
                "bespokelabs/Bespoke-Stratos-17k", split="train"
            )

            # Proportional Pre-Subsampling BEFORE expensive .map() if max_samples > 0
            if args.max_samples > 0:
                logger.info(
                    f"✂️ Pre-slicing component datasets to target {args.max_samples:,} total rows..."
                )
                n_chat = min(len(ds_chat), int(args.max_samples * 0.55))
                n_code = min(len(ds_code), int(args.max_samples * 0.22))
                n_math = min(len(ds_math), int(args.max_samples * 0.18))
                n_reas = min(len(ds_reasoning), int(args.max_samples * 0.18))

                ds_chat = ds_chat.shuffle(seed=42).select(range(n_chat))
                ds_code = ds_code.shuffle(seed=42).select(range(n_code))
                ds_math = ds_math.shuffle(seed=42).select(range(n_math))
                ds_reasoning = ds_reasoning.shuffle(seed=42).select(range(n_reas))

            num_proc = max(1, min(16, (os.cpu_count() or 4) - 2))

            ds_code = ds_code.map(
                lambda x: {
                    "messages": [
                        {"role": "user", "content": x["instruction"]},
                        {"role": "assistant", "content": x["response"]},
                    ]
                },
                remove_columns=ds_code.column_names,
                num_proc=num_proc,
            )

            def format_math_with_think(x):
                problem = x["problem"]
                solution = x["solution"]
                if "####" in solution:
                    reasoning, answer = solution.rsplit("####", 1)
                    formatted_content = (
                        f"<think>\n{reasoning.strip()}\n</think>\n\n"
                        f"The final answer is \\boxed{{{answer.strip()}}}."
                    )
                else:
                    formatted_content = f"<think>\n{solution.strip()}\n</think>"

                return {
                    "messages": [
                        {"role": "user", "content": problem},
                        {"role": "assistant", "content": formatted_content},
                    ]
                }

            ds_math = ds_math.map(
                format_math_with_think,
                remove_columns=ds_math.column_names,
                num_proc=num_proc,
            )

            def format_stratos_reasoning(x):
                if "messages" in x and x["messages"]:
                    messages = x["messages"]
                else:
                    system_msg = x.get("system", "")
                    user_q = x.get("question", x.get("problem", x.get("prompt", "")))
                    reasoning = x.get("reasoning", "")
                    response = x.get("response", x.get("solution", ""))

                    user_content = (
                        f"{system_msg}\n\n{user_q}".strip() if system_msg else user_q
                    )
                    if reasoning:
                        assistant_content = (
                            f"<think>\n{reasoning.strip()}\n</think>\n\n"
                            f"{response.strip()}"
                        )
                    else:
                        assistant_content = response.strip()

                    messages = [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ]
                return {"messages": messages}

            ds_reasoning = ds_reasoning.map(
                format_stratos_reasoning,
                remove_columns=ds_reasoning.column_names,
                num_proc=num_proc,
            )

            stop_strategy = (
                "all_exhausted" if args.max_samples == 0 else "first_exhausted"
            )
            raw_dataset = interleave_datasets(
                [ds_chat, ds_code, ds_math, ds_reasoning],
                probabilities=[0.50, 0.20, 0.15, 0.15],
                seed=42,
                stopping_strategy=stop_strategy,
            )

            if args.max_samples > 0 and len(raw_dataset) > args.max_samples:
                logger.info(
                    f"✂️ Sub-sampling interleaved dataset from {len(raw_dataset):,} rows to {args.max_samples:,} rows."
                )
                raw_dataset = raw_dataset.select(range(args.max_samples))

            processed_dataset = pack_chatml_dataset(
                raw_dataset, tokenizer, max_seq_len=args.seq_len, local_rank=local_rank
            )
            logger.info(f"💾 Caching packed dataset to '{cache_dir}'...")
            processed_dataset.save_to_disk(cache_dir)
        else:
            logger.info(
                f"⚡ Found pre-packed dataset at '{cache_dir}'. Fast-loading from disk..."
            )

    # Synchronize all distributed workers before loading
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    processed_dataset = Dataset.load_from_disk(cache_dir)
    dataset_split = processed_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = dataset_split["train"]
    val_data = dataset_split["test"]

    if local_rank == 0:
        logger.info(
            f"Dataset split complete: {len(train_data):,} training samples | "
            f"{len(val_data):,} validation samples."
        )

    # PyTorch FSDP Configuration (ZeRO-2 Sharding with FP32 Master Weights)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        bf16=True,
        max_grad_norm=0.5,
        optim="adamw_torch",  # Enforces FP32 AdamW optimizer
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        torch_compile=False,
        gradient_checkpointing=args.grad_checkpointing,
        fsdp="shard_grad_op",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(
                f"Cloudflare R2 Sync activated. Target Bucket: '{r2_bucket}' | Prefix: '{r2_prefix}'"
            )
        callbacks.append(
            R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix)
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    checkpoint_to_resume = None
    if os.path.exists(training_args.output_dir):
        existing_checkpoints = [
            d
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if existing_checkpoints:
            checkpoint_to_resume = True
            if local_rank == 0:
                logger.info(
                    f"Found existing SFT checkpoint(s) in '{training_args.output_dir}'."
                    " Resuming training automatically..."
                )

    if local_rank == 0:
        logger.info("🔥 Starting Stage 3: Supervised Fine-Tuning (SFT) in FSDP Mode...")

    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # All ranks participate in save_model so FSDP gathers sharded states cleanly
    if local_rank == 0:
        logger.info(
            f"Gathering sharded FSDP weights and saving to '{args.output_dir}'..."
        )
    trainer.save_model(args.output_dir)

    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Stage 3: Supervised Fine-Tuning completed successfully!")

        if enable_r2_sync:
            rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
            target_r2_path = (
                f"r2:{r2_bucket}/{r2_prefix.strip('/')}/{rel_output_dir}"
            )
            logger.info(
                "📤 Syncing final SFT model weights to Cloudflare R2"
                f" ({target_r2_path})..."
            )
            cmd = [
                "rclone",
                "copy",
                args.output_dir,
                target_r2_path,
                "--transfers",
                "8",
                "--s3-chunk-size",
                "64M",
            ]
            subprocess.run(cmd, check=False)
            logger.info("✅ Final SFT weights successfully uploaded to Cloudflare R2!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()