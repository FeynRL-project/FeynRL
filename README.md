<p align="center">
  <img src="docs/feynrl.png" alt="FeynRL Logo" width="300">
</p>

<p align="center">
  Lightweight, modular, and scalable framework for post-training and fine-tuning large models.
</p>

<p align="center">
  <a href="https://github.com/boson-ai/FeynRL"><img src="https://img.shields.io/badge/GitHub-FeynRL-181717?style=flat-square&logo=github" alt="GitHub"></a>&nbsp;
  <a href="https://rasoolfa.github.io"><img src="https://img.shields.io/badge/Blog-FeynRL-E65100?style=flat-square&logo=googlechrome&logoColor=white" alt="Blog"></a>&nbsp;
  <a href="https://discord.gg/HQE9TVXCNS"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2E7D32?style=flat-square" alt="License"></a>
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

There are several great open source frameworks for training and post training large models. Many are powerful, but as they scale in scope, they can become harder to modify and come with steep learning curves, often requiring you to understand tightly coupled components before you can make changes with confidence. As a result, adapting them to a new research idea or a new setup can be slow, error prone, and frustrating. That complexity can also hurt reliability: when a framework has many moving parts and evolves quickly, failures become harder to diagnose, and subtle bugs can hide in layers of abstraction.

FeynRL is designed to reduce that friction: a **lightweight, modular, scalable** framework that keeps algorithm logic clear and keeps systems concerns explicit. The core design principle is **separation of concerns**: algorithm code stays algorithmic, and systems code stays systems, so components remain easier to understand, test, and extend. In distributed training, some systems details inevitably surface, but FeynRL aims to keep those interfaces as explicit and clean as possible so components remain easy to understand, test, and debug.

This is the first public release, so expect rough edges. Our goal is to keep FeynRL **simple, efficient, and predictable**. We're open-sourcing it to provide a framework that's easy to reason about, and to build it alongside the community that will stress-test it, extend it, and improve it.


## 🏗️ Overview

FeynRL is built around three core principles:

- ⚡ **Efficiency**: Designed for training models with billions of parameters, using DeepSpeed for distributed training, vLLM for fast inference, and Ray for orchestration at scale.
- 🧩 **Modularity**: Clear separation between training engines, rollout generation, and data handling, so it’s easy to swap components and experiment safely.
- 🧠 **Algorithm-first**: Designed to accelerate research with a simple, hackable codebase that keeps the focus on the algorithms.

For a more detailed breakdown, see the **[Architecture Overview](docs/ARCHITECTURE.md)**.

### ✅ What’s Included

- 🧪 **Training paradigms**: RL (PPO, GRPO, CISPO, P3O), preference-based learning (DPO), and supervised fine-tuning (SFT)
- 🖥️ **Distributed training**: Multi-GPU and multi-node via DeepSpeed (ZeRO Stage 1/2/3)
- 🎲 **Rollouts / inference**: vLLM-powered rollout engines with tensor parallelism
- 🛰️ **Orchestration**: Ray for scheduling training and rollout workers across nodes
- 🔀 **Training↔rollout scheduling**: Sync and overlap (async) modes; the overlap engine interleaves chunk-based generation with training, using ESS-driven NCCL weight sync to keep rollout data fresh while significantly reducing GPU idle time
- 🔄 **Weight sync**: NCCL broadcast (fastest), direct in-memory transfer via Ray object store, and disk-based checkpoint reload, with automatic fallback chain (NCCL to direct to disk)
- 🧷 **Parameter-efficient fine-tuning**: LoRA via PEFT
- 📈 **Experiment tracking**: MLflow and Weights & Biases support
- 🏅 **Evaluation**: Standalone eval pipeline with vLLM engines

For RL, Ray orchestrates the full training loop: it schedules DeepSpeed training workers and vLLM rollout workers across nodes, and coordinates weight synchronization between them. In **sync mode**, each epoch generates all rollouts, trains on them, syncs weights, and repeats. In **overlap mode**, generation is dispatched one chunk at a time while training runs concurrently on already-collected data, significantly reducing GPU idle time. An ESS (Effective Sample Size) metric monitors policy divergence and triggers NCCL weight sync at chunk boundaries when the rollout engines are idle, keeping data fresh without stalling the pipeline. Weight sync uses a three-tier fallback chain: NCCL broadcast (fastest, zero-copy), direct transfer via Ray object store, and disk-based checkpoint reload. SFT and DPO are simpler as they only need a single model and no rollout workers, so they run directly on DeepSpeed without Ray. All paradigms support full fine-tuning and LoRA, and plug into mixed-dataset sampling, experiment tracking, and standalone evaluation without changing the pipeline.

## 📢 News

- ![Date](https://img.shields.io/badge/2026--03--03-purple) 🎉  We're excited to publicly release FeynRL as a preview! Some features and documentation are still evolving. We welcome feedback, bug reports, and contributions as we continue to build this together.

## How to Use FeynRL

🛠️ **[Installation & Setup](docs/INSTALL.md)**
Read the guide to configure your environment and dependencies.

🚀 **[Usage & Examples](docs/HOWTO.md)**
Learn how to launch jobs and run experiments.

⚙️ **[Configuration Reference](configs/README.md)**
Full parameter guide for RL, SFT, DPO, and evaluation configs.

## Contributing

Contributions are welcome! Please see our **[Contributing Guidelines](CONTRIBUTING.md)** for details on how to get involved.

## FAQ

Check out the [FAQ](docs/FAQ.md) for common questions and answers.
