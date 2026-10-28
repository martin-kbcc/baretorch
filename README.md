# BareTorch 🐻🔥

> **Kernel-Free, Pure GEMM Sub-Quadratic Sequence Architecture for Universal Hardware**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Rampage-yellow)](https://huggingface.co/model-rampage)
[![Live ZeroGPU Playground](https://img.shields.io/badge/%F0%9F%A4%97%20Live_ZeroGPU_Demo-BareTorch_Playground-ff9d00?style=flat&logo=huggingface)](https://huggingface.co/spaces/martin-kb-rampage/BareTorch-Qwen3.5-Playground)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)

**BareTorch** is an open-source development ecosystem and model framework built on a **pure GEMM-compliant, kernel-free paradigm**. 

By structuring sub-quadratic recurrence into block-parallel chunk segments, BareTorch maps sequential historical state updates directly to optimized BLAS/GEMM matrix multiplications natively accelerated by modern hardware backends. This completely bypasses the compilation and hardware lock-in of custom CUDA or Triton kernels, enabling linear memory and execution scaling natively across arbitrary accelerators (e.g., NVIDIA CUDA, Apple Silicon MLX, WebGPU, and TPUs).

---

## 📦 Installation

Install the library in editable mode directly from source:

```bash
git clone [https://github.com/martin-kbcc/baretorch.git](https://github.com/martin-kbcc/baretorch.git)
cd baretorch
pip install -e .
```

---

## ⚡ Performance Benchmarks: BareTorch 2.5B vs. Qwen3.5-2B

All benchmarks were evaluated on a single **NVIDIA GeForce RTX 4090 (24GB VRAM)** across context windows scaling from 512 up to 32,768 tokens, measuring prefill throughput (Time to First Token) and sustained autoregressive generation velocity (64 decode steps averaged over multiple trials).

### A Strict "Apples-to-Apples" Comparison

Cross-architecture benchmarks often distort real-world performance by comparing models with disparate dimensions or mismatched vocabulary granularities. This benchmark enforces a rigorous, controlled evaluation:

1. **Identical Model Scale**: Both models operate at the **~2.5B parameter** tier.
2. **Symmetrical Hybrid Ratio**: Both models utilize an identical **75% sub-quadratic / 25% quadratic attention** layer distribution.
3. **Identical Tokenizer**: Both models evaluate using the exact same tokenizer (`Qwen/Qwen3.5-9B`, ~248k vocabulary). Comparing generation throughput across different tokenizers is inherently flawed because token compression rates differ; with a shared tokenizer, **1 token generated represents the exact same semantic volume of text** in both models.
4. **Apples-to-Apples Memory Optimization**: Both models evaluate with final-logit prefill projection (`num_logits_to_keep=1`), isolating true cache and activation scaling without artificial vocabulary memory penalties.

### The Architectural Paradigm: Pure GEMM vs. Custom Kernels

* **Qwen3.5-2B** relies on bespoke, hand-crafted **C++/CUDA kernels** (Gated DeltaNet / flash-linear-attention) compiled specifically for NVIDIA hardware.
* **BareTorch 2.5B** is **100% kernel-free**. It executes entirely via native PyTorch matrix multiplications optimized with `torch.compile`. Because it does not rely on vendor-locked CUDA or Triton kernels, BareTorch is a **universal framework** designed to run seamlessly across diverse hardware architectures (NVIDIA GPUs, AMD ROCm, Apple Silicon Metal/MLX, TPUs, and edge accelerators).

### Benchmark Results (NVIDIA RTX 4090, BF16)

| Context Length | Model | Compilation / Backend | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak VRAM (GB) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **512** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | **18.22 ms** | **28,108.3** | **66.17** | 4.77 GB |
| 512 | Qwen3.5-2B | Custom CUDA Kernel | 39.93 ms | 12,821.2 | 45.53 | 4.52 GB |
| **1,024** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | **27.95 ms** | **36,631.8** | **65.45** | 4.81 GB |
| 1,024 | Qwen3.5-2B | Custom CUDA Kernel | 38.75 ms | 26,425.4 | 45.29 | 4.56 GB |
| **2,048** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | 50.46 ms | 40,583.0 | **65.41** | 4.88 GB |
| 2,048 | Qwen3.5-2B | Custom CUDA Kernel | **49.93 ms** | **41,020.3** | 45.69 | 4.64 GB |
| **4,096** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | **103.03 ms** | **39,756.7** | **65.39** | 5.03 GB |
| 4,096 | Qwen3.5-2B | Custom CUDA Kernel | 103.54 ms | 39,558.0 | 45.70 | 4.81 GB |
| **8,192** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | **219.34 ms** | **37,348.6** | **65.36** | 5.34 GB |
| 8,192 | Qwen3.5-2B | Custom CUDA Kernel | 231.62 ms | 35,368.6 | 45.57 | 5.14 GB |
| **16,384** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | **461.37 ms** | **35,511.6** | **66.15** | 5.94 GB |
| 16,384 | Qwen3.5-2B | Custom CUDA Kernel | 487.32 ms | 33,620.6 | 46.17 | 5.80 GB |
| **32,768** | **BareTorch 2.5B** | `torch.compile` (Pure GEMM) | **995.40 ms** | **32,919.4** | **63.79** | 7.14 GB |
| 32,768 | Qwen3.5-2B | Custom CUDA Kernel | 1,044.72 ms | 31,365.4 | 45.82 | 7.12 GB |

### Key Observations

* **+44% Faster Autoregressive Generation**: BareTorch delivers consistent **~65.5 tok/s** generation speed versus Qwen's **~45.6 tok/s**. Low-rank associative state factorization ($r=8$) reduces memory-bus traffic per decoding step compared to full-rank linear attention matrices.
* **True $O(1)$ Decode Scaling**: BareTorch's generation throughput remains virtually flat across the entire context range (66.17 tok/s at 512 context vs. 63.79 tok/s at 32k context).
* **Linear Memory Scaling Without Custom Kernels**: At 32k context, BareTorch consumes only **7.14 GB of VRAM** (matching Qwen's 7.12 GB) demonstrating that custom CUDA kernels are not required to achieve linear memory scaling.

---

## 🚀 Quickstart: Inference

BareTorch integrates directly with Hugging Face interfaces. You can load checkpoints using native `baretorch` classes or standard wrappers.

### 1. Single Sequence Generation

```python
import torch
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM

MODEL_ID = "model-rampage/baretorch-2.5b-base"
TOKENIZER_ID = "Qwen/Qwen3.5-9B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = BareTorchForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

# Encode prompt
prompt = "The key architectural advantage of linear recurrence over standard softmax attention is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

# Autoregressive generation
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Batched Generation with Asymmetric Padding

BareTorch supports batched inference with left-padded inputs of variable lengths without attention leakage into padding tokens:

```python
import torch
from transformers import AutoTokenizer
from baretorch import BareTorchForCausalLM

MODEL_ID = "model-rampage/baretorch-2.5b-base"
TOKENIZER_ID = "Qwen/Qwen3.5-9B"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Always enforce left-padding for causal decoder generation
tokenizer.padding_side = "left"

model = BareTorchForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
)
model.eval()

prompts = [
    "Machine learning is",
    "In modern deep learning architectures, hybrid recurrence and attention mechanisms enable models to achieve",
]

# Tokenize batch with dynamic left-padding
batch_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda:0")

with torch.no_grad():
    batch_outputs = model.generate(
        **batch_inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

for i, out_tokens in enumerate(batch_outputs):
    print(f"\n--- Sample {i + 1} ---")
    print(tokenizer.decode(out_tokens, skip_special_tokens=True))
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