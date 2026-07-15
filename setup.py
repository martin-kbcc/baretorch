from setuptools import setup, find_packages

setup(
    name="baretorch",
    version="0.1.0",
    author="Martin Ignacio Kovacevic Buvinic",
    author_email="martin@baretorch.ai",  # Placeholder for your startup domain
    description="BareTorch: Challenging State-of-the-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures",
    long_description=open("README.md", "r", encoding="utf-8").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/martinkb/baretorch",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "datasets>=2.15.0",
        "numpy",
        "tensorboard",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)