# /home/martinkb/Desktop/BareTorch_F/push_to_hf.py
import os
import json
import shutil
import argparse
from huggingface_hub import HfApi, create_repo

PRIMARY_TASKS = [
    ("mmlu", "MMLU (Overall 57-Subject Average)", "acc,none"),
    ("arc_challenge", "ARC Challenge", "acc_norm,none"),
    ("arc_easy", "ARC Easy", "acc_norm,none"),
    ("hellaswag", "HellaSwag", "acc_norm,none"),
    ("winogrande", "Winogrande", "acc,none"),
]

def format_evaluation_table(eval_json_path: str) -> str:
    """Parses evaluation JSON and returns a formatted Markdown table for core headline benchmarks."""
    if not eval_json_path or not os.path.exists(eval_json_path):
        return "| Benchmark | Score |\n| :--- | :--- |\n| *Data Not Found* | *N/A* |\n"

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
        return "| Benchmark | Score |\n| :--- | :--- |\n| *Parsing Error* | *N/A* |\n"


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
pipeline_tag: text-generation
---

# 🚀 {repo_id.split('/')[-1]}

**BareTorch-500M Base** is a hybrid language model architecture combining **CS-LRAD** (Chunk-State Low-Rank Associative Delta) recurrent layers and standard Transformer multi-head attention.

## 📐 Architecture Specs
- **Parameters:** ~500M
- **Hidden Dimension ($d_{{model}}$):** 1152
- **Layers:** 24 total (Interleaved 3x CS-LRAD + 1x Transformer)
- **Attention Heads:** 16 ($d_{{head}} = 72$)
- **CS-LRAD Rank:** 8 | **Chunk Size:** 32
- **Vocabulary:** 49,152 (`SmolLM2-360M` base tokenizer)
- **Context Window:** Up to 32,768 tokens

---

## 📊 Core Language Model Benchmarks (Zero-Shot)

{eval_table}

---

## ⚡ Hardware Efficiency & Scaling Benchmarks

BareTorch's CS-LRAD recurrent layers reduce KV-cache growth from $O(N)$ to bounded chunk states, maintaining high decoding speeds and low VRAM footprint even at **32,768 context length**.

### 💚 NVIDIA RTX GPU Benchmark (CUDA / BF16)

| Baseline Model | Context Len | Metric | BareTorch | Baseline Transformer | Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2-1.7B** | **32,768** | **Decode Speed** | **98.9 tok/s** | 15.8 tok/s | **6.26x Faster** |
| | **32,768** | **VRAM** | **4,394 MB** | 15,572 MB | **71.8% Savings** |
| **Qwen3-0.6B** | **32,768** | **Decode Speed** | **164.5 tok/s** | 13.2 tok/s | **12.51x Faster** |
| | **32,768** | **VRAM** | **1,660 MB** | 8,577 MB | **80.6% Savings** |
| **Gemma-2-2B** | **32,768** | **Decode Speed** | **101.5 tok/s** | 💥 OOM | **Saves Execution** |

### 🍎 Apple Silicon Benchmark (M1 MacBook Pro / MLX / FP16)

| Baseline Model | Context Len | Metric | BareTorch | Baseline Transformer | Advantage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SmolLM2-1.7B** | **32,768** | **Decode Speed** | **27.7 tok/s** | 1.6 tok/s | **16.87x Faster** |
| | **32,768** | **VRAM** | **3,778 MB** | 9,973 MB | **62.1% Savings** |
| **Llama-3.2-1B** | **32,768** | **Decode Speed** | **36.9 tok/s** | 6.6 tok/s | **5.57x Faster** |
| **Qwen3-0.6B** | **32,768** | **Execution** | **65.2 tok/s** | 💥 OOM | **Saves Execution** |

---

## 💻 Usage with BareTorch

{bt}python
import torch
from baretorch import BareTorchForCausalLM
from transformers import AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BareTorchForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

prompt = "The future of artificial intelligence lies in"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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
base_model: {base_repo_id}
pipeline_tag: text-generation
---

# 💬 {repo_id.split('/')[-1]}

**BareTorch-500M SFT** is an instruction-fine-tuned alignment checkpoint of the hybrid **CS-LRAD + Transformer** base model, fine-tuned on multi-turn dialogue using ChatML formatting.

---

## 📊 Core Language Model Benchmarks (Zero-Shot)

{eval_table}

---

## ⚡ Speed & Memory Performance Highlights

### 32,768 Long-Context Decoding Efficiency

| Device / Platform | Context Len | BareTorch Decode Speed | VRAM Footprint | Speedup vs Standard Transformer |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA CUDA (BF16)** | **32,768** | **164.5 tok/s** | **1,660 MB** | **12.51x Faster** (vs Qwen3-0.6B) |
| **Apple M1 (MLX/FP16)** | **32,768** | **27.7 tok/s** | **3,778 MB** | **16.87x Faster** (vs SmolLM2-1.7B) |

---

## 🤖 Prompt Format (ChatML)

{bt}text
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
{bt}

---

## 💻 Generation Example

{bt}python
import torch
from baretorch import BareTorchForCausalLM
from transformers import AutoTokenizer

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = BareTorchForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()

messages = [
    {{"role": "user", "content": "Explain gravity in simple terms."}}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
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
    print(f"  ├─ Generated README.md Model Card with Evaluation Metrics")

    baretorch_source = "/home/martinkb/Desktop/BareTorch_F/baretorch"
    baretorch_target = os.path.join(checkpoint_dir, "baretorch")
    if os.path.exists(baretorch_source) and not os.path.exists(baretorch_target):
        shutil.copytree(baretorch_source, baretorch_target, dirs_exist_ok=True)
        print(f"  ├─ Bundled custom 'baretorch' model implementation files")

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
    parser.add_argument("--hf_username", type=str, required=True, help="Your Hugging Face username or organization.")
    parser.add_argument("--base_eval_json", type=str, default="./base_eval_results_500m.json", help="Path to base model evaluation JSON.")
    parser.add_argument("--sft_eval_json", type=str, default="./sft_eval_results_500m.json", help="Path to SFT model evaluation JSON.")
    args = parser.parse_args()

    base_ckpt = "/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_hybrid_baretorch/checkpoint-190735"
    sft_ckpt = "/home/martinkb/Desktop/BareTorch_F/checkpoints_500m_sft/checkpoint-2759"

    base_repo_id = f"{args.hf_username}/BareTorch-500M-Base"
    sft_repo_id = f"{args.hf_username}/BareTorch-500M-SFT"

    if os.path.exists(base_ckpt):
        prepare_and_upload(
            checkpoint_dir=base_ckpt, 
            repo_id=base_repo_id, 
            is_sft=False, 
            eval_json=args.base_eval_json
        )

    if os.path.exists(sft_ckpt):
        prepare_and_upload(
            checkpoint_dir=sft_ckpt, 
            repo_id=sft_repo_id, 
            is_sft=True, 
            base_repo_id=base_repo_id, 
            eval_json=args.sft_eval_json
        )


if __name__ == "__main__":
    main()