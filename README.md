# LeanRL

There are several great open source frameworks for training and post training large models for supervised learning and reinforcement learning (RL), but many have become hard to work with in practice. They are often complex, have a steep learning curve, and require you to understand many tightly coupled components before you can safely modify anything. Adapting them to a new research idea or a new setup can be slow, error prone, and frustrating. That complexity also makes reproducibility harder than it should be. When a framework has many moving parts and evolves quickly, experiments can become difficult to track, debug, and reliably reproduce, which is the opposite of what you want when iterating on new ideas.

LeanRL aims to fill this gap. It is a lightweight, modular, production grade framework for post-training/finetuning LLMs, VLMs, and VLAs, without sacrificing much readability, scalability, or flexibility. The guiding principle is a clear separation of concerns. Algorithm code stays algorithmic, and systems code stays systems. There is no free lunch though, and some distributed training and system codes inevitably surface inside algorithm implementations for example. LeanRL keeps these interfaces explicit and clean, so each component’s role remains easy to understand, test, and debug. 

This is the first release, and obviously it comes with its own set of issues. As we add features and capabilities, the goal is to keep LeanRL simple, efficient, and predictable, rather than letting complexity creep in. We are open sourcing it to give the community a framework that is easy to use and reason about, and to build it together with the people who will stress test it, extend it, and make it better.
## Overview

LeanRL is built around three core principles:

- **Efficiency**: Optimized for training models with billions of parameters, using DeepSpeed for distributed training, vLLM for fast inference, and Ray for orchestration at scale.
- **Modularity**: Clear separation between training engines, rollout generation, reward computation, and data handling, which makes it easy to swap components and experiment safely.
- **Research first**: Designed to accelerate research by keeping the codebase simple, hackable, and performant.

## Experiments

To make things easier, we provide scripts and selected results on a few datasets across multiple algorithms. The `experiment` folder contains runnable scripts, and this page summarizes results from some of them.

## Installation

### Prerequisites

- NVIDIA GPU
- NVIDIA driver compatible with CUDA 12.x
- Conda (Miniconda or Anaconda)

Verify CUDA:

```bash
nvidia-smi
```

### Environment Setup

```bash
conda create -n leanrl-env python=3.13.1 -y
conda activate leanrl-env 
```

#### PyTorch (CUDA 12.6)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### Core packages

```bash
pip install pydantic pyyaml huggingface_hub transformers deepspeed datasets vllm mlflow
```

#### CUDA toolkit

```bash
conda install -c nvidia cuda-toolkit=12.2
```

#### FlashAttention

```bash
pip install flash-attn --no-build-isolation --config-settings="--jobs=8" --verbose
```


## Contributing

Contributions are welcome. Please keep changes easy to understand, test, and debug, and follow the existing style.

## FAQ

Check out the [FAQ](FAQ.md) for common questions and answers.

## License

TODO

## Acknowledgments

Some components of this codebase are inspired by practices from open source projects. We try to cite sources wherever we directly reuse exact code. If we missed a citation, please let us know and we will credit the source.