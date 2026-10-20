# /home/martinkb/Desktop/BareTorch_F/setup.py
import os
from setuptools import find_packages, setup

readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="baretorch",
    version="0.2.0",
    author="Martin Ignacio Kovacevic Buvinic",
    author_email="martin@baretorch.ai",
    description="BareTorch: Challenging State-of-the-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martin-kbcc/baretorch",
    packages=find_packages(include=["baretorch", "baretorch.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.14.1",
        "transformers>=5.14.1",
        "accelerate>=1.14.0",
        "datasets>=5.0.0",
        "safetensors>=0.8.0",
        "huggingface-hub>=1.30.0",
        "tokenizers>=0.22.2",
        "sentencepiece>=0.2.2",
        "tiktoken>=0.13.0",
        "numpy>=2.3.5",
        "tqdm>=4.70.0",
        "tensorboard>=2.21.0",
        "lm-eval>=0.4.12",
        "mlx>=0.32.4; sys_platform == 'darwin' and platform_machine == 'arm64'",
    ],
    extras_require={
        "mlx": [
            "mlx>=0.32.4",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)