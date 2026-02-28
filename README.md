<h1 align="center">FeynRL</h1>

<p align="center">
  Lightweight, modular, and scalable framework for post-training and fine-tuning large models.
</p>

<p align="center">
  <a href="https://github.com/rasoolfa/FeynRL"><img src="https://img.shields.io/badge/GitHub-FeynRL-181717?style=flat-square&logo=github" alt="GitHub"></a>&nbsp;
  <a href="https://rasoolfa.github.io"><img src="https://img.shields.io/badge/Blog-FeynRL-4285F4?style=flat-square&logo=googlechrome&logoColor=white" alt="Blog"></a>&nbsp;
  <a href="https://github.com/rasoolfa/FeynRL"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License"></a>
</p>

---

**FeynRL** (pronounced “FineRL”) is a lightweight, modular framework for **post-training and fine-tuning** large models (supervised fine-tuning and RL). It is designed for researchers and engineers who need a **robust, scalable distributed training stack** without sacrificing readability, hackability, or reproducibility.

### 🎯 Why We Built This

There are several excellent open-source frameworks for training and post-training large models (supervised fine-tuning and RL), but many are hard to work with in practice. They’ve grown complex, have steep learning curves, and often require understanding tightly coupled components before you can safely change anything. Adapting them to a new research idea or a new setup can be slow, error-prone, and frustrating. That complexity also hurts reproducibility: when a framework has many moving parts and evolves quickly, experiments become harder to track, debug, and reliably reproduce.

FeynRL is built to fill this gap: a **lightweight, modular, scalable** framework for post-training and fine-tuning large models—without sacrificing readability, flexibility, or performance. The core design principle is **separation of concerns**: algorithm code stays algorithmic, and systems code stays systems. In distributed training, some systems details inevitably surface, but FeynRL aims to keep those interfaces as explicit and clean as possible so components remain easy to understand, test, and debug.

This is the first public release, so expect rough edges. As we add features and capabilities, our goal is to keep FeynRL **simple, efficient, and predictable**, rather than letting complexity creep in. We’re open-sourcing it to provide a framework that’s easy to reason about, and to build it alongside the community that will stress-test it, extend it, and improve it.

## 🏗️ Overview

FeynRL is built around three core principles:

- ⚡ **Efficiency**: Designed for training models with billions of parameters, using DeepSpeed for distributed training, vLLM for fast inference, and Ray for orchestration at scale.
- 🧩 **Modularity**: Clear separation between training engines, rollout generation, and data handling, so it’s easy to swap components and experiment safely.
- 🧠 **Algorithm-first**: Designed to accelerate research with a simple, hackable codebase that keeps the focus on the algorithms.

For a more detailed breakdown, see the **[Architecture Overview](docs/ARCHITECTURE.md)**.

### ✅ What’s Included

- 🧪 **Training paradigms**: RL (PPO, SGRPO, CISPO), preference-based learning (DPO), and supervised fine-tuning (SFT)
- 🖥️ **Distributed training**: Multi-GPU and multi-node via DeepSpeed (ZeRO Stage 1/2/3)
- 🎲 **Rollouts / inference**: vLLM-powered rollout engines with tensor parallelism
- 🛰️ **Orchestration**: Ray for scheduling training and rollout workers across nodes
- 🔁 **Training↔rollout scheduling**: Sync and Async modes for flexible execution
- 🔄 **Weight sync**: Fast in-memory transfer with disk fallback
- 🧷 **Parameter-efficient fine-tuning**: LoRA support via PEFT
- 📈 **Experiment tracking**: MLflow and Weights & Biases support
- 🏅 **Evaluation**: Standalone eval pipeline with vLLM engines

FeynRL supports multiple post-training paradigms, including RL methods such as PPO, SGRPO, and CISPO; preference-based learning such as DPO; and supervised fine-tuning (SFT). It runs at scale with multi-GPU and multi-node training via DeepSpeed (ZeRO Stage 1/2/3), and uses vLLM-powered rollout engines for fast inference and evaluation. Ray orchestrates training and rollout workers across nodes, with periodic weight synchronization from training to rollout workers via in-memory transfer when supported (and a disk-based fallback). To improve overall throughput, FeynRL can run in synchronous mode (generate rollouts, then train) or asynchronous mode (overlap generation and training), trading off utilization against how off-policy the collected samples may be. It also supports LoRA fine-tuning, pluggable experiment tracking (MLflow or Weights & Biases), configurable mixed-dataset sampling, and a standalone evaluation pipeline.

## How to Use FeynRL

🛠️ **[Installation & Setup](docs/INSTALL.md)**
Read the guide to configure your environment and dependencies.

🚀 **[Usage & Examples](experiments/README.md)**
Learn how to launch jobs and run experiments.

⚙️ **[Configuration Reference](configs/README.md)**
Full parameter guide for RL, SFT, DPO, and evaluation configs.


## Contributing

Contributions are welcome! Please see our **[Contributing Guidelines](CONTRIBUTING.md)** for details on how to get involved.


## FAQ

Check out the [FAQ](docs/FAQ.md) for common questions and answers.

## Acknowledgments

Some components of this codebase are inspired by practices from open source projects. We try to cite sources wherever we directly reuse exact code. If we missed a citation, please let us know and we will credit the source.