# /home/martinkb/Desktop/BareTorch_F/push_to_hf.py
import os
import json
import shutil
import argparse
from huggingface_hub import HfApi, create_repo

PRIMARY_TASKS = [
    ("hellaswag", "HellaSwag", "acc_norm,none"),
    ("arc_easy", "ARC Easy", "acc_norm,none"),
    ("arc_challenge", "ARC Challenge", "acc_norm,none"),
    ("winogrande", "WinoGrande", "acc,none"),
    ("mmlu", "MMLU (Overall 57-Subject Average)", "acc,none"),
    ("mmlu_stem", "  ├─ MMLU STEM", "acc,none"),
    ("mmlu_humanities", "  ├─ MMLU Humanities", "acc,none"),
    ("mmlu_social_sciences", "  ├─ MMLU Social Sciences", "acc,none"),
    ("mmlu_other", "  └─ MMLU Other", "acc,none"),
]


def format_evaluation_table(eval_json_path: str) -> str:
    """Parses evaluation JSON and returns a formatted Markdown table for core headline benchmarks."""
    if not eval_json_path or not os.path.exists(eval_json_path):
        return "| Benchmark Task | Metric | Score |\n| :--- | :--- | :--- |\n| *Data Not Found* | *N/A* | *N/A* |\n"

    try:
        with open(eval_json_path, "r") as f:
            data = json.load(f)

        table = "| Task / Benchmark | Metric | Score |\n| :--- | :--- | :--- |\n"
        for task_key, display_name, preferred_metric in PRIMARY_TASKS:
            if task_key in data:
                task_metrics = data[task_key]
                score = task_metrics.get(preferred_metric, task_metrics.get("acc,none", "N/A"))
                metric_label = "Acc (Norm)" if "norm" in preferred_metric else "Accuracy"

                if isinstance(score, (float, int)):
                    val_str = f"**{score * 100:.2f}%**"
                else:
                    val_str = str(score)

                table += f"| **{display_name}** | {metric_label} | {val_str} |\n"
        return table
    except Exception as e:
        print(f"⚠️ Error formatting evaluation table from '{eval_json_path}': {e}")
        return "| Benchmark Task | Score |\n| :--- | :--- |\n| *Parsing Error* | *N/A* |\n"


def generate_base_model_card(repo_id: str, eval_json_path: str = None) -> str:
    bt = "```"
    eval_table = format_evaluation_table(eval_json_path)

    return f"""---
language:
- en
license: apache-2.0
tags:
- baretorch
- cs-lrad
- hybrid-attention
- causal-lm
- pure-gemm
- kernel-free
pipeline_tag: text-generation
---

# 🐻🔥 {repo_id.split('/')[-1]}

**BareTorch-500M Base** is a foundational sub-quadratic language model built under a **pure GEMM-compliant, kernel-free paradigm**. The model combines **CS-LRAD** (Chunk-Segmented Low-Rank Associative Delta Engine) recurrent layers with standard Transformer multi-head self-attention in a **3:1 interleaved hybrid topology** ($3\\times\\text{{CS-LRAD}} \\to 1\\times\\text{{Transformer}}$).

By structuring sub-quadratic state updates into block-parallel chunk segments ($C=32$), BareTorch bypasses the compilation and hardware lock-in of custom CUDA or Triton kernels, running with $O(N)$ execution and memory efficiency natively across NVIDIA CUDA, Apple Silicon MLX, WebGPU, and TPUs.

---

## 📐 Model Architecture Specifications

- **Parameters:** ~500M (498.2M active parameters)
- **Hidden Dimension ($d_{{model}}$):** 1,152
- **Total Layers:** 24 (Interleaved 18x CS-LRAD + 6x Transformer)
- **Attention Heads:** 16 ($d_{{head}} = 72$)
- **CS-LRAD Subspace Rank ($r$):** 8
- **Chunk Size ($C$):** 32 tokens
- **Tokenizer:** `HuggingFaceTB/SmolLM2-360M` (Vocab size: 49,152)
- **Context Window:** Up to 32,768 tokens

---

## 🏋️ Pre-Training Runway Specs & Loss Convergence

- **Training Runway:** **100 Billion Tokens** ($190,735$ optimization steps)
- **Hardware Cluster:** $4\\times$ NVIDIA H100 SXM (80GB VRAM)
- **Global Batch Size:** $524,288$ tokens/step ($256$ sequences of length $2048$)
- **Optimizer & LR:** AdamW ($\text{{LR}}_{{peak}} = 6 \\times 10^{{-4}}$, weight decay $0.1$, cosine decay scheduler with $2,000$ warmup steps)

### Pre-Training Evaluation Loss Milestones

| Optimization Step | Tokens Processed | Train Loss | Eval Loss |
| :--- | :--- | :--- | :--- |
| **Step 1,000** | ~0.52 Billion | 5.2876 | -- |
| **Step 10,000** | ~5.24 Billion | 2.7632 | 2.6943 |
| **Step 50,000** | ~26.21 Billion | 2.5275 | 2.4836 |
| **Step 100,000** | ~52.43 Billion | 2.4324 | 2.3965 |
| **Step 150,000** | ~78.64 Billion | 2.3355 | 2.3054 |
| **Step 190,735 (Final)** | **100.0 Billion** | **2.2901** | **2.2690** |

---

## 📊 Zero-Shot Downstream Benchmarks

{eval_table}

---

## ⚡ Long-Context Inference Hardware Scaling (32,768 Context)

BareTorch replaces context-dependent KV-caches with constant-sized $O(1)$ recurrent state updates, eliminating memory bus bottlenecks and out-of-memory crashes on long-context workloads.

### 1. Discrete CUDA GPU (NVIDIA RTX 4090 - 24GB)

| Baseline Pair | Context | Prefill Latency | Local GPU Decode | Peak VRAM | Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen3 0.6B Match** | **32,768** | **314.89 ms** vs 4,343.26 ms | **164.49 tok/s** vs 13.15 tok/s | **1.66 GB** vs 8.58 GB | **12.51x Faster** (-80.6% VRAM) |
| **SmolLM2 1.7B Match** | **32,768** | **1,134.20 ms** vs 4,292.54 ms | **98.92 tok/s** vs 15.79 tok/s | **4.39 GB** vs 15.57 GB | **6.26x Faster** (-71.8% VRAM) |
| **Llama 3.2 1B Match** | **32,768** | **598.36 ms** vs 2,960.03 ms | **169.24 tok/s** vs 24.44 tok/s | **3.02 GB** vs 4.67 GB | **6.92x Faster** (-35.4% VRAM) |
| **Gemma 2 2B Match** | **32,768** | **1,049.72 ms** (Baseline: 💥 OOM) | **101.51 tok/s** (Baseline: 💥 OOM) | **5.96 GB** (Baseline: 💥 OOM) | **Prevents OOM Crashes** |

### 2. Apple Silicon Unified Memory (M1 MacBook Pro 16GB - Native MLX)

| Baseline Pair | Context | Prefill Latency | Local GPU Decode | Peak VRAM | Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SmolLM2 1.7B Match** | **32,768** | **29.56 s** vs 59.03 s | **29.69 tok/s** vs 0.66 tok/s | **3.78 GB** vs 9.97 GB | **44.98x Faster** (-62.1% VRAM) |
| **Llama 3.2 1B Match** | **32,768** | **18.69 s** vs 41.06 s | **28.42 tok/s** vs 3.29 tok/s | **2.81 GB** vs 3.71 GB | **8.64x Faster** (-24.4% VRAM) |
| **Qwen3 0.6B Match** | **32,768** | **7.06 s** (Baseline: 💥 OOM) | **79.55 tok/s** (Baseline: 💥 OOM) | **1.37 GB** (Baseline: 💥 OOM) | **Prevents OOM Crashes** |

---

## 💻 Usage & Code Example

{bt}python
import torch
from transformers import AutoTokenizer
from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
model = BareTorchForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

prompt = "The key innovation of pure GEMM sub-quadratic architectures is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
{bt}

---

## 📜 Citation

{bt}bibtex
@article{{kovacevic2026baretorch,
  title={{BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures}},
  author={{Kovacevic Buvinic, Martin Ignacio}},
  journal={{BareTorch Framework Laboratory Technical Report}},
  year={{2026}}
}}
{bt}
"""


def generate_sft_model_card(repo_id: str, base_repo_id: str, eval_json_path: str = None) -> str:
    bt = "```"
    eval_table = format_evaluation_table(eval_json_path)

    return f"""---
language:
- en
license: apache-2.0
tags:
- baretorch
- cs-lrad
- sft
- chatml
- instruction-tuned
- pure-gemm
base_model: {base_repo_id}
pipeline_tag: text-generation
---

# 💬 {repo_id.split('/')[-1]}

**BareTorch-500M SFT** is the instruction-aligned counterpart of the **BareTorch-500M Base** language model. It was fine-tuned on multi-turn instruction dialogue datasets using ChatML formatting across $2\\times$ NVIDIA RTX 4090 GPUs.

---

## 🏋️ Supervised Fine-Tuning (SFT) Details

- **Base Checkpoint:** [{base_repo_id}](https://huggingface.co/{base_repo_id}) (100B Token Pre-Trained Foundation Model)
- **Dataset:** `HuggingFaceTB/smol-smoltalk` (~485,000 multi-turn instruction samples)
- **Hardware Config:** $2\\times$ NVIDIA GeForce RTX 4090 (24GB)
- **Training Epochs:** 1 Epoch ($2,759$ total steps)
- **Global Batch Size:** 64 sequences ($131,072$ tokens/step, sequence length 2048)
- **Learning Rate:** $1 \\times 10^{{-5}}$ with 100 warmup steps
- **Convergence Loss:** Evaluation loss decreased smoothly from $1.5631$ (step 250) to **$1.4302$** (step 2,759).

---

## 📊 Benchmark Comparison: Base vs. SFT Aligned

Instruction alignment unlocks massive leaps across reasoning, commonsense, and QA benchmarks:

| Benchmark Task | Metric | Base Model (100B) | SFT Aligned Model | Instruction Gain |
| :--- | :---: | :---: | :---: | :---: |
| **HellaSwag** | Acc-Norm | 43.69% | **52.52%** | **+8.83%** |
| **ARC Easy** | Acc-Norm | 53.58% | **59.18%** | **+5.60%** |
| **ARC Challenge** | Acc-Norm | 28.92% | **35.41%** | **+6.49%** |
| **WinoGrande** | Acc | 51.30% | **55.80%** | **+4.50%** |
| **MMLU (Overall)** | Accuracy | 24.70% | **25.57%** | **+0.87%** |

---

## 📊 Detailed SFT Benchmark Breakdown

{eval_table}

---

## ⚡ Long-Context Inference Hardware Scaling (32,768 Context)

BareTorch's kernel-free CS-LRAD hybrid engine delivers massive decoding acceleration and VRAM footprint reduction:

| Platform / Device | Context Window | Decode Velocity | Peak Memory | Speedup vs Standard Transformer |
| :--- | :---: | :---: | :---: | :--- |
| **NVIDIA CUDA (RTX 4090)** | **32,768 Tokens** | **164.49 tok/s** | **1,660 MB** | **12.51x Faster** (vs Qwen3 0.6B) |
| **Apple Silicon (M1 MBP MLX)** | **32,768 Tokens** | **29.69 tok/s** | **3,778 MB** | **44.98x Faster** (vs SmolLM2 1.7B) |

---

## 🤖 Prompt Format (ChatML)

{bt}text
<|im_start|>user
Explain quantum entanglement in two sentences.<|im_end|>
<|im_start|>assistant
Quantum entanglement is a phenomenon where two or more particles become connected such that the state of one instantly influences the state of the other, regardless of distance. Einstein famously referred to this counterintuitive phenomenon as "spooky action at a distance."<|im_end|>
{bt}

---

## 💻 Code Usage Example

{bt}python
import torch
from transformers import AutoTokenizer
from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")
model = BareTorchForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

messages = [
    {{"role": "user", "content": "What are three key benefits of pure-GEMM neural network architectures?"}}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
{bt}

---

## 📜 Citation

{bt}bibtex
@article{{kovacevic2026baretorch,
  title={{BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures}},
  author={{Kovacevic Buvinic, Martin Ignacio}},
  journal={{BareTorch Framework Laboratory Technical Report}},
  year={{2026}}
}}
{bt}
"""


def prepare_and_upload(checkpoint_dir: str, repo_id: str, is_sft: bool, base_repo_id: str = "", eval_json: str = None):
    api = HfApi()

    print(f"\n📦 Preparing repository '{repo_id}' on Hugging Face Hub...")
    create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    readme_path = os.path.join(checkpoint_dir, "README.md")
    if is_sft:
        card_content = generate_sft_model_card(repo_id, base_repo_id, eval_json)
    else:
        card_content = generate_base_model_card(repo_id, eval_json)

    with open(readme_path, "w") as f:
        f.write(card_content)
    print("  ├─ Generated README.md Model Card with Complete Academic & Evaluation Metrics")

    baretorch_source = "/home/martinkb/Desktop/BareTorch_F/baretorch"
    baretorch_target = os.path.join(checkpoint_dir, "baretorch")
    if os.path.exists(baretorch_source) and not os.path.exists(baretorch_target):
        shutil.copytree(baretorch_source, baretorch_target, dirs_exist_ok=True)
        print("  ├─ Bundled custom 'baretorch' model implementation files")

    print(f"🚀 Uploading files to Hugging Face Hub ({repo_id})...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["optimizer.pt", "scheduler.pt", "trainer_state.json", "*.bin.index"]
    )
    print(f"✅ Successfully uploaded '{repo_id}' to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload BareTorch Base & SFT checkpoints to Hugging Face")
    parser.add_argument("--hf_entity", type=str, default="model-rampage", help="Hugging Face organization or username (default: model-rampage).")
    parser.add_argument("--base_eval_json", type=str, default="./base_eval_results_500m.json", help="Path to base model evaluation JSON.")
    parser.add_argument("--sft_eval_json", type=str, default="./sft_eval_results_500m.json", help="Path to SFT model evaluation JSON.")
    args = parser.parse_args()

    base_ckpt = "/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_hybrid_baretorch/checkpoint-190735"
    sft_ckpt = "/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_sft/checkpoint-2759"

    base_repo_id = f"{args.hf_entity}/BareTorch-500M-Base"
    sft_repo_id = f"{args.hf_entity}/BareTorch-500M-SFT"

    if os.path.exists(base_ckpt):
        prepare_and_upload(
            checkpoint_dir=base_ckpt,
            repo_id=base_repo_id,
            is_sft=False,
            eval_json=args.base_eval_json
        )
    else:
        print(f"⚠️ Base checkpoint directory not found at '{base_ckpt}'. Skipping Base upload.")

    if os.path.exists(sft_ckpt):
        prepare_and_upload(
            checkpoint_dir=sft_ckpt,
            repo_id=sft_repo_id,
            is_sft=True,
            base_repo_id=base_repo_id,
            eval_json=args.sft_eval_json
        )
    else:
        print(f"⚠️ SFT checkpoint directory not found at '{sft_ckpt}'. Skipping SFT upload.")


if __name__ == "__main__":
    main()