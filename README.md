<h1 align="center">FeynRL</h1>

<p align="center">
  Lightweight, modular, and scalable framework for post-training and fine-tuning large models.
</p>

<p align="center">
  <a href="https://github.com/rasoolfa/FeynRL"><img src="https://img.shields.io/badge/GitHub-FeynRL-181717?style=flat-square&logo=github" alt="GitHub"></a>&nbsp;
  <a href="https://rasoolfa.github.io"><img src="https://img.shields.io/badge/Blog-FeynRL-4285F4?style=flat-square&logo=googlechrome&logoColor=white" alt="Blog"></a>&nbsp;
  <a href="https://discord.gg/HQE9TVXCNS"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License"></a>
</p>

---

<p align="center">
  <em>"What I cannot create, I do not understand."</em> — Richard Feynman
</p>

**FeynRL** (pronounced "FineRL") is a lightweight, modular framework for **post-training and fine-tuning** large models, including supervised fine-tuning (SFT), preference learning (e.g., DPO), and reinforcement learning (e.g., PPO, GRPO). It’s built for researchers and engineers who want a **clean, robust and scalable** distributed training stack without sacrificing readability and hackability.

### 💡 Why use FeynRL?

FeynRL is a good fit if you want to iterate quickly without losing clarity and control.

- 🧩 **Easy to extend**
Most changes stay local, so you can add new ideas without reshaping the stack.
- 🔧 **One workflow across post training**
SFT, DPO, and RL share the same setup, making comparisons simpler.
- 🚀 **Works at any run size**
Use the same pipeline for a single GPU debug run or a multi node job.

### 🎯 Why We Built This

There are several great open source frameworks for training and post training large models. Many are powerful, but as they scale in scope, they can become harder to modify and come with steep learning curves, often requiring you to understand tightly coupled components before you can make changes with confidence. As a result, adapting them to a new research idea or a new setup can be slow, error prone, and frustrating. That complexity can also hurt reproducibility: when a framework has many moving parts and evolves quickly, experiments become harder to track, debug, and reliably reproduce.

FeynRL is designed to reduce that friction: a **lightweight, modular, scalable** framework that keeps algorithm logic clear and keeps systems concerns explicit. The core design principle is **separation of concerns**: algorithm code stays algorithmic, and systems code stays systems, so components remain easier to understand, test, and extend. In distributed training, some systems details inevitably surface, but FeynRL aims to keep those interfaces as explicit and clean as possible so components remain easy to understand, test, and debug.

This is the first public release, so expect rough edges. Our goal is to keep FeynRL **simple, efficient, and predictable**. We're open-sourcing it to provide a framework that's easy to reason about, and to build it alongside the community that will stress-test it, extend it, and improve it.


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
- 🔁 **Training↔rollout scheduling**: Sync and Async modes to further improve throughput
- 🔄 **Weight sync**: Fast in-memory transfer with disk fallback
- 🧷 **Parameter-efficient fine-tuning**: LoRA via PEFT
- 📈 **Experiment tracking**: MLflow and Weights & Biases support
- 🏅 **Evaluation**: Standalone eval pipeline with vLLM engines

For RL, Ray orchestrates the full training loop: it schedules DeepSpeed training workers and vLLM rollout workers across nodes, and coordinates periodic weight synchronization between them via in-memory transfer (with a disk-based fallback). Training and rollout can run synchronously (generate, then train, then sync) or asynchronously (overlap generation with training to reduce GPU idle time, with a configurable staleness bound). SFT and DPO are simpler as they only need a single model and no rollout workers, so they run directly on DeepSpeed without Ray. All paradigms support full fine-tuning and LoRA, and plug into mixed-dataset sampling, experiment tracking, and standalone evaluation without changing the pipeline.

## 📢 News

- **[2026-03-03]** 🎉  We're excited to publicly release FeynRL as a preview! Some features and documentation are still evolving. We welcome feedback, bug reports, and contributions as we continue to build this together.

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
