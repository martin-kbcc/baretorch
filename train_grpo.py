# /home/martinkb/Desktop/BareTorch_F/train_grpo.py

import argparse
import inspect
import logging
import os
import re
import subprocess
import warnings
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from baretorch import BareTorchConfig, BareTorchForCausalLM

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


# ==============================================================================
#                         Cloudflare R2 Background Sync Callback
# ==============================================================================
class R2CheckpointCallback(TrainerCallback):
    def __init__(
        self,
        bucket_name: str = "baretorch-data",
        remote_name: str = "r2",
        prefix: str = "checkpoints",
    ):
        self.bucket_name = bucket_name
        self.remote_name = remote_name
        self.prefix = prefix.strip("/")

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            checkpoint_dir = f"checkpoint-{state.global_step}"
            local_ckpt_path = os.path.join(args.output_dir, checkpoint_dir)

            if os.path.exists(local_ckpt_path):
                rel_output_dir = os.path.basename(os.path.normpath(args.output_dir))
                target_r2_path = (
                    f"{self.remote_name}:{self.bucket_name}/{self.prefix}/{rel_output_dir}/{checkpoint_dir}"
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


# ==============================================================================
#                               Rule-Based Reward Functions
# ==============================================================================
def format_reward_func(completions, **kwargs):
    """
    Rewards the completion for closing the reasoning block with </think> 
    and providing a boxed final answer \\boxed{...}.
    
    Note: Since <think> is prefixed in the prompt header, completions start 
    directly inside the reasoning trace.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        
        has_think_end = "</think>" in text
        has_boxed = "\\boxed{" in text

        if has_think_end and has_boxed:
            # Check if </think> occurs BEFORE \boxed{
            think_end_idx = text.find("</think>")
            boxed_idx = text.find("\\boxed{")
            if think_end_idx < boxed_idx:
                rewards.append(1.0)
            else:
                rewards.append(0.5)  # Both present, but wrong structural order
        elif has_think_end:
            rewards.append(0.5)
        elif has_boxed:
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


def extract_boxed_answer(text):
    """Extracts answer inside \\boxed{...} tag supporting nested LaTeX braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    num_open_braces = 1
    chars = []
    while i < len(text) and num_open_braces > 0:
        c = text[i]
        if c == '{':
            num_open_braces += 1
        elif c == '}':
            num_open_braces -= 1
        if num_open_braces > 0:
            chars.append(c)
        i += 1
    if num_open_braces == 0:
        return "".join(chars).strip()
    return None


def accuracy_reward_func(completions, answer, **kwargs):
    """Rewards the model (+2.0) if the boxed final answer matches ground truth."""
    rewards = []
    for completion, target in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        pred = extract_boxed_answer(text)
        
        if "####" in target:
            clean_target = target.split("####")[-1].strip()
        else:
            clean_target = target.strip()

        if pred is not None and pred.lower() == clean_target.lower():
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards


# ==============================================================================
#                                  Main Engine
# ==============================================================================
def main():
    if "LOCAL_RANK" in os.environ:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser(
        description="BareTorch Stage 2: Group Relative Policy Optimization (GRPO) Engine"
    )

    # Checkpoint Paths
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="./checkpoints_300m_sft",
        help="Path to SFT-aligned BareTorch checkpoint folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_300m_grpo",
        help="Directory to save RL-trained weights.",
    )

    # Architecture Flags
    parser.add_argument(
        "--tie_word_embeddings",
        action="store_true",
        help="Explicitly tie lm_head.weight to token_embedding.weight.",
    )

    # Tokenizer & Dataset
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="Hugging Face tokenizer identifier.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openai/gsm8k",
        help="Hugging Face dataset identifier for RL training.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        help="Dataset subset/config name.",
    )

    # RL Hyperparameters
    parser.add_argument("--num_generations", type=int, default=4, help="Group size (G) rollouts per prompt.")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="GRPO learning rate.")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty factor relative to reference policy.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU prompt batch size.")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--num_epochs", type=int, default=1)

    # Cloud Storage / Sync
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
        default="checkpoints",
        help="Prefix path inside R2 bucket.",
    )

    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 1. Tokenizer Setup
    if local_rank == 0:
        logger.info(f"Initializing Tokenizer '{args.tokenizer_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Enforce right-padding for CS-LRAD chunk alignment
    tokenizer.padding_side = "right"

    # 2. Policy Model Loading
    if local_rank == 0:
        logger.info(f"Loading SFT Model from: {args.sft_model_path}")

    model = BareTorchForCausalLM.from_pretrained(
        args.sft_model_path,
        torch_dtype=torch.bfloat16,
    )

    if args.tie_word_embeddings:
        model.lm_head.weight = model.model.token_embedding.weight
        model.config.tie_word_embeddings = True
        model.tie_weights()
    elif getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    # 3. Load & Format Dataset
    if local_rank == 0:
        logger.info(f"Loading RL Dataset '{args.dataset_name}'...")

    raw_dataset = load_dataset(args.dataset_name, args.dataset_config, split="train")

    def format_prompt(example):
        system_prompt = (
            "A conversation between User and Assistant. The user asks a question, "
            "and the Assistant solves it step by step. The assistant MUST write its reasoning "
            "process inside <think>...</think> tags and provide the final answer inside \\boxed{}."
        )
        prompt_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        if len(prompt_tokens) > args.max_prompt_length:
            prompt_tokens = prompt_tokens[-args.max_prompt_length:]
            prompt_text = tokenizer.decode(prompt_tokens)

        return {"prompt": prompt_text, "answer": example["answer"]}

    train_dataset = raw_dataset.map(format_prompt, remove_columns=raw_dataset.column_names)

    # 4. GRPO Trainer Configuration
    grpo_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.num_epochs,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "bf16": True,
        "use_vllm": False,
    }

    sig = inspect.signature(GRPOConfig.__init__)
    if "max_prompt_length" in sig.parameters:
        grpo_kwargs["max_prompt_length"] = args.max_prompt_length

    grpo_args = GRPOConfig(**grpo_kwargs)

    enable_r2_sync = args.r2_sync or os.environ.get("R2_SYNC", "0").lower() in ("1", "true", "yes")
    r2_bucket = os.environ.get("R2_BUCKET", args.r2_bucket)
    r2_prefix = os.environ.get("R2_PREFIX", args.r2_prefix)

    callbacks = []
    if enable_r2_sync:
        if local_rank == 0:
            logger.info(f"Cloudflare R2 Sync activated. Target Bucket: '{r2_bucket}'")
        callbacks.append(R2CheckpointCallback(bucket_name=r2_bucket, prefix=r2_prefix))

    # 5. Initialize & Run GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if local_rank == 0:
        logger.info("🧠 Starting Stage 2: Group Relative Policy Optimization (GRPO)...")

    trainer.train()

    if local_rank == 0:
        logger.info(f"Saving final GRPO checkpoint to '{args.output_dir}'...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        logger.info("✅ Stage 2: GRPO RL completed successfully!")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os._exit(0)


if __name__ == "__main__":
    main()