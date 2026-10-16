# BareTorch 🐻🔥

> **Kernel-Free, Pure GEMM Sub-Quadratic Sequence Architecture for Universal Hardware**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Rampage-yellow)](https://huggingface.co/model-rampage)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/martin-kbcc/baretorch-experiments/blob/main/paper.pdf)

**BareTorch** is an open-source development ecosystem and model framework built on a **pure GEMM-compliant, kernel-free paradigm**. 

By structuring sub-quadratic recurrence into block-parallel chunk segments, BareTorch maps sequential historical state updates directly to optimized BLAS/GEMM matrix multiplications natively accelerated by all modern hardware backends. This completely bypasses the compilation and hardware lock-in of custom CUDA or Triton kernels, enabling linear $O(N)$ memory and execution scaling natively across arbitrary accelerators (e.g., NVIDIA CUDA, Apple Silicon MLX, WebGPU, and TPUs).

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