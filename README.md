# BareTorch 🐻🔥

> **Enterprise-Grade, Zero-Kernel Sub-Quadratic Sequence Mixing Framework**

BareTorch is a high-performance, cluster-scale language model framework built entirely under a **pure GEMM-compliant, kernel-free paradigm**[cite: 1]. 

By structuring sub-quadratic recurrence into block-parallel chunk segments, BareTorch maps sequential historical state updates directly to optimized BLAS/GEMM routines natively accelerated by all modern hardware backends[cite: 1]. This completely bypasses the compilation and portability lock-in of custom CUDA or Triton kernels, enabling linear $O(N)$ execution scaling natively on arbitrary accelerators (e.g., TPUs, WebGPU, Apple Silicon, or commodity edge devices)[cite: 1].

This repository houses the production core of the BareTorch framework, featuring dynamic hybrid sequencing, gradient checkpointing, and PyTorch compilation integration for scaling up to billions of parameters.

For our historical research diary, baseline comparison files, and proof-of-concept benchmarks, visit the [BareTorch Research Playground](https://github.com/model-rampage/baretorch-experiments).

---

## ⚡ Key Highlights (Symmetric Hybrid Benchmarks)

We benchmarked the BareTorch scalable core on our local workstation (dual RTX 4090s) training over a high-density stream of the **DCLM-100BT** dataset. 

Our **Symmetric Hybrid** configuration—interleaving 3 layers of sub-quadratic **CS-LRAD** with 1 layer of standard **Softmax Attention**—yielded breakthrough modeling density and speed:

* **Massive Perplexity Reduction:** Achieved a validation loss of **`4.586`** (yielding **`98.10` Perplexity**), outperforming the standard Softmax Transformer baseline (`4.783` loss / `119.46` PPL) by over **21 perplexity points** in 10,000 pretraining steps.
* **Retained Hardware Speed:** Maintained **84.3% of the throughput speed** of a highly optimized, hardware-native FlashAttention Transformer (running at **9.50 steps/s** vs. 11.27 steps/s) without utilizing a single line of un-portable low-level kernel code.

---

## 🛠️ Getting Started

### 1. Environment Setup
We recommend setting up a clean virtual environment using Miniconda:

```bash
conda create -n baretorch python=3.11 -y
conda activate baretorch
pip install torch transformers datasets tensorboard
```

### 2. Local Pre-training Test
To verify the framework compiles and trains cleanly on your local workstation, execute our verified hybrid configuration script:

```bash
bash launch_lrad_hybrid_local.sh
```

### 3. Scaling to Multi-GPU Cluster (8x H100)
To launch multi-node or multi-GPU distributed data-parallel pretraining, run our cluster-optimized script:

```bash
bash launch_lrad_hybrid_cluster.sh
```

---

## 🧩 Designing Custom Architectures Natively

BareTorch is designed for research agility. Rather than modifying Python files, you can define entirely custom, deep sub-quadratic configurations directly from your bash launcher scripts.

Simply supply a comma-separated list of layer targets to `--layer_sequence` and scale `--num_layers`. The framework will dynamically assemble and tile the network blocks:

```bash
# Launch an advanced model combining CS-LRAD, CS-TTT, and Attention
torchrun --nproc_per_node=2 train.py \
    --model_type baretorch \
    --layer_sequence "cs_lrad,cs_lrad,cs_ttt,transformer" \
    --num_layers 16 \
    --d_model 768 \
    --compile \
    --grad_checkpointing
```

---

## 📂 Codebase Structure

```text
├── baretorch/           # Production core package containing all modeling files
│   ├── modeling/        # Pure-GEMM mathematical modules (CS-LRAD, CS-TTT)
│   └── configurations/  # Static and dynamic hyperparameter classes
├── train.py             # Highly flexible distributed data-parallel (DDP) pretraining script
├── launch_lrad_local.sh # Local dual-GPU verification script (256 hidden dimensions)
└── launch_lrad_cluster.sh # 500M parameter cluster orchestration script optimized for 8x H100s
```

---

## ⚖️ License

BareTorch is open-source software licensed under the permissive **Apache License 2.0**.

```text
Copyright 2026 Model Rampage

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
```# BareTorch: Hardware-Portable Next-Gen Sequences
