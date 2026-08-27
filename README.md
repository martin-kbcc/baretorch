# BareTorch 🐻🔥

> **Kernel-Free, Pure GEMM Sub-Quadratic Sequence Architecture for Universal Hardware**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Rampage-yellow)](https://huggingface.co/model-rampage)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)

**BareTorch** is an open-source development ecosystem and model framework built on a **pure GEMM-compliant, kernel-free paradigm**. 

By structuring sub-quadratic recurrence into block-parallel chunk segments, BareTorch maps sequential historical state updates directly to optimized BLAS/GEMM matrix multiplications natively accelerated by all modern hardware backends. This completely bypasses the compilation and hardware lock-in of custom CUDA or Triton kernels, enabling linear $O(N)$ memory and execution scaling natively across arbitrary accelerators (e.g., NVIDIA CUDA, Apple Silicon MLX, WebGPU, and TPUs).

---

## 🚀 Key Achievements & Highlights

* **100B Token Foundational Scale-Up:** Pre-trained a **~500M parameter hybrid model** on 100 Billion tokens across $4\times$ NVIDIA H100 GPUs ($2.2690$ final evaluation loss).
* **Supervised Instruction Tuning (SFT):** Post-trained on `smol-smoltalk` ($2\times$ RTX 4090 GPUs), unlocking massive reasoning leaps: **52.52% HellaSwag Acc-Norm**, **59.18% ARC-Easy**, and **35.41% ARC-Challenge**.
* **44.98x Faster Apple Silicon MLX Decoding:** At 32,768 context length on an M1 MacBook Pro, BareTorch streams at **29.69 tok/s** while standard MLX chokes at **0.66 tok/s**.
* **12.51x Faster CUDA Decoding:** Reaches **164.49 tok/s** at 32k context on a single NVIDIA RTX 4090 (vs. **13.15 tok/s** baseline).
* **Up to 80.6% VRAM Reduction & Zero OOM Crashes:** Replaces linear KV-cache growth with a constant $O(1)$ state update, running long-context generation seamlessly where baselines crash with Out-Of-Memory errors.

---

## 📊 500M Foundational Model Benchmarks

Our flagship **BareTorch 500M 3:1 Hybrid** ($3 \times \text{CS-LRAD} \to 1 \times \text{Transformer}$) was evaluated across standard downstream reasoning tasks before and after Supervised Fine-Tuning (SFT):

| Benchmark Task | Metric | Base (100B Tokens) | SFT (SmolTalk) | Gain |
| :--- | :---: | :---: | :---: | :---: |
| **HellaSwag** | Acc / Acc-Norm | 34.84% / 43.69% | 41.31% / **52.52%** | **+8.83%** |
| **ARC-Easy** | Acc / Acc-Norm | 56.19% / 53.58% | 66.58% / **59.18%** | **+5.60%** |
| **ARC-Challenge** | Acc / Acc-Norm | 26.11% / 28.92% | 33.70% / **35.41%** | **+6.49%** |
| **WinoGrande** | Acc | 51.30% | **55.80%** | **+4.50%** |
| **MMLU (Overall)** | Acc | 24.70% | **25.57%** | **+0.87%** |

---

## ⚡ Empirical Long-Context Inference Scaling (32k Context)

Inference metrics evaluated at strict parameter parity against open-source baselines across context windows from $512 \to 32,768$ tokens:

### 1. Discrete CUDA GPU (NVIDIA RTX 4090 - 24GB)

| Model Pair | Context ($L$) | Prefill Latency (ms) | Local GPU Decode | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3 0.6B Match** | 32,768 | **314.89 ms** vs 4,343.26 ms | **164.49 tok/s** vs 13.15 tok/s (**12.51x**) | **1.66 GB** vs 8.58 GB (**-80.6%**) |
| **SmolLM2 1.7B Match** | 32,768 | **1,134.20 ms** vs 4,292.54 ms | **98.92 tok/s** vs 15.79 tok/s (**6.26x**) | **4.39 GB** vs 15.57 GB (**-71.8%**) |
| **Llama 3.2 1B Match** | 32,768 | **598.36 ms** vs 2,960.03 ms | **169.24 tok/s** vs 24.44 tok/s (**6.92x**) | **3.02 GB** vs 4.67 GB (**-35.4%**) |
| **Gemma 2 2B Match** | 32,768 | **1,049.72 ms** (Baseline: 💥 OOM) | **101.51 tok/s** (Baseline: 💥 OOM) | **5.96 GB** (Baseline: 💥 OOM) |

### 2. Apple Silicon Unified Memory (M1 MacBook Pro 16GB - Native MLX)

| Model Pair | Context ($L$) | Prefill Latency (ms) | Local GPU Decode | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **SmolLM2 1.7B Match** | 32,768 | **29.56 s** vs 59.03 s | **29.69 tok/s** vs 0.66 tok/s (**44.98x**) | **3.78 GB** vs 9.97 GB (**-62.1%**) |
| **Llama 3.2 1B Match** | 32,768 | **18.69 s** vs 41.06 s | **28.42 tok/s** vs 3.29 tok/s (**8.64x**) | **2.81 GB** vs 3.71 GB (**-24.4%**) |
| **Qwen3 0.6B Match** | 32,768 | **7.06 s** (Baseline: 💥 OOM) | **79.55 tok/s** (Baseline: 💥 OOM) | **1.37 GB** (Baseline: 💥 OOM) |

---

## 🛠️ Quickstart & Usage

### 1. Installation

```bash
git clone [https://github.com/martin-kbcc/baretorch.git](https://github.com/martin-kbcc/baretorch.git)
cd baretorch
pip install -e .
```

### 2. Pre-Training & Supervised Fine-Tuning

Launch distributed training directly using our pre-configured cloud orchestrators:

```bash
# 500M Parameter Pre-Training Runway (4x H100)
bash cloud_runs_0.5B/launch_lrad_hybrid.sh

# 500M Parameter Supervised Fine-Tuning (2x RTX 4090)
bash cloud_runs_0.5B/launch_lrad_hybrid_sft.sh
```

### 3. Hugging Face Integration

BareTorch integrates directly with Hugging Face `transformers` out of the box:

```python
import torch
from transformers import AutoTokenizer
from baretorch.integration.configuration_baretorch import BareTorchConfig
from baretorch.integration.modeling_baretorch import BareTorchForCausalLM

# Load model natively via BareTorch AutoModel classes from model-rampage org
config = BareTorchConfig.from_pretrained("model-rampage/baretorch-500m-sft")
model = BareTorchForCausalLM.from_pretrained("model-rampage/baretorch-500m-sft", torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M")

prompt = "<|user|>\nWhat makes pure-GEMM architectures fast on long context?<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 4. Benchmarking

```bash
# Apple Silicon MLX Multi-Model Suite Evaluation
PYTHONPATH=. python baretorch/benchmarks/apple/benchmark_mlx.py \
  --hf_model_ids HuggingFaceTB/SmolLM2-1.7B meta-llama/Llama-3.2-1B Qwen/Qwen3-0.6B \
  --prompt_lens 512 1024 2048 4096 8192 16384 32768
```

---

## 🧩 Designing Custom Hybrid Architectures

Define sub-quadratic sequence mixing topologies directly from CLI or bash launchers using layer sequences:

```bash
# Launch dynamic DDP pre-training with 3:1 CS-LRAD to Transformer hybrid pattern
torchrun --nproc_per_node=4 train.py \
    --model_type baretorch \
    --layer_sequence "cs_lrad,cs_lrad,cs_lrad,transformer" \
    --num_layers 24 \
    --d_model 1152 \
    --num_heads 16 \
    --chunk_size 32 \
    --rank 8 \
    --compile
```

---

## 📂 Codebase Organization

```text
.
├── baretorch/                      # Core production framework package
│   ├── modeling/                   # Pure-GEMM sequence mixers (cs_lrad.py, cs_ttt.py, transformer.py)
│   ├── integration/                # Hugging Face integration wrappers (configuration_baretorch.py, modeling_baretorch.py)
│   └── benchmarks/
│       └── apple/                  # Native Apple Silicon MLX suite (modeling_mlx.py, benchmark_mlx.py)
├── cloud_runs_0.5B/                # Production launch orchestrators
│   ├── launch_lrad_hybrid.sh       # 500M Foundational Pre-training script (4x H100)
│   └── launch_lrad_hybrid_sft.sh   # 500M SFT Instruction Tuning script (2x RTX 4090)
├── train.py                        # Main DDP pre-training script
└── train_sft.py                    # Main Supervised Fine-Tuning script
```

---

## 📜 Citation

If you use BareTorch in your research or edge deployments, please cite our paper:

```bibtex
@article{kovacevic2026baretorch,
  title={BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures},
  author={Kovacevic Buvinic, Martin Ignacio},
  journal={BareTorch Framework Laboratory Technical Report},
  year={2026}
}
```

---

## ⚖️ License

BareTorch is open-source software licensed under the **Apache License 2.0**.