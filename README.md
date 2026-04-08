<p align="center">
  <img src="docs/feynrl.png" alt="FeynRL Logo" width="300">
</p>

<p align="center">
  Algorithm-first post-training framework for large models.
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

**FeynRL** (pronounced "FineRL") is an **algorithm-first** framework for **post-training and fine-tuning** large models. It supports supervised fine-tuning (SFT), preference learning (e.g., DPO), and reinforcement learning (e.g., PPO, GRPO, CISPO, P3O), and is built for researchers and engineers who want to understand, modify, and develop new methods without fighting the infrastructure.

The main goal of FeynRL is simple: make new algorithms easy to implement, easy to debug, and still possible to train at scale. The codebase is designed so that **algorithmic logic stays local** and **systems logic stays explicit**, which makes the framework easier to reason about, easier to extend, and more reliable to debug.

### 💡 Why use FeynRL?

FeynRL is a good fit if your goal is not only to run an existing recipe, but to **build and test new post-training methods**.

- **Algorithm-first design** — Most method changes stay local: you can add new objectives, rewards, baselines, or update rules without reshaping the full stack.

- **Clear separation of concerns** — Algorithm code stays algorithmic, and systems code stays systems. That keeps the codebase easier to understand, test, and extend.

- **One framework across post-training** — SFT, DPO, and RL share the same workflow and configuration system, making comparisons easier and reducing duplicated infrastructure.

- **Scales beyond toy settings** — Use the same framework for local single-GPU debugging or large multi-node distributed runs.

FeynRL may not be the best fit if your main priority is the largest built-in feature surface out of the box, or if you mainly want a framework already optimized around a narrow workflow and do not expect to modify it much.

### 🎯 Why We Built This

There are already several strong open-source frameworks for post-training large models. Many are powerful and feature-rich, but they are often optimized around a narrower set of methods or execution patterns, and can become hard to modify once you want to try something new.

FeynRL was built to make a different trade-off. Instead of optimizing first for the largest feature surface, it optimizes first for **clarity, locality of change, and algorithm development**. The codebase is structured so that algorithmic ideas are easy to implement and reason about, while the distributed systems layer remains explicit rather than hidden behind heavy abstractions. In practice, implementing a new algorithm typically means writing a single file with its own loss and update logic, not threading changes through the orchestration, rollout, and data layers.

The framework is designed for scale from the start. It supports large-scale training with DeepSpeed, Ray, and vLLM, including sync and async execution modes, adaptive weight synchronization, and multi-node runs. The goal is to make it possible to do both: **move fast on algorithms and still run realistic experiments at scale**.

This is the first public release, so expect rough edges. We are open-sourcing FeynRL not just as a library, but as a foundation for building new post-training methods with the community.

## ✅ What's Included

For a detailed breakdown of the architecture, see the **[Architecture Overview](docs/ARCHITECTURE.md)**.

- 🧪 **Training paradigms**: RL (PPO, GRPO, CISPO, P3O), preference-based learning (DPO), and supervised fine-tuning (SFT)
- 🖥️ **Distributed training**: Multi-GPU and multi-node via DeepSpeed (ZeRO Stage 1/2/3)
- 🎲 **Rollouts / inference**: vLLM-powered rollout engines with tensor parallelism
- 🛰️ **Orchestration**: Ray for scheduling training and rollout workers across nodes
- 🔀 **Training-rollout scheduling**: Sync and overlap (async) modes. The overlap engine uses a queue-pull architecture: rollout engines self-schedule by pulling prompts from a shared Ray queue while training runs concurrently on already-collected data, with mid-epoch NCCL weight sync triggered by ESS (P3O) or a fixed step interval (PPO/GRPO/CISPO).
- 🔄 **Weight sync**: NCCL broadcast (sync mode supports direct/disk fallbacks; async mode is NCCL-only at runtime, with a built-in NCCL watchdog and fail-fast on communicator destruction).
- 🧷 **Parameter-efficient fine-tuning**: LoRA via PEFT
- 🔢 **Mixed-dataset sampling**: Configurable multi-dataset sampling with ratios within a single training run
- 📈 **Experiment tracking**: MLflow and Weights & Biases support
- 🏅 **Evaluation**: Standalone eval pipeline with vLLM engines

For RL, Ray orchestrates the full training loop: it schedules DeepSpeed training workers and vLLM rollout workers across nodes, and coordinates weight synchronization between them. In **sync mode**, each epoch generates all rollouts, trains on them, syncs weights, and repeats. In **overlap mode**, the driver fills a shared Ray prompt queue at the start of each epoch and rollout engines self-schedule by pulling from it; the driver drains results between training steps so generation and training run concurrently. Mid-epoch NCCL weight sync is triggered by ESS (Effective Sample Size) or a fixed step interval, and pre-launching the next epoch hides the checkpoint-save bubble. Async mode is NCCL-only at runtime; sync mode supports a three-tier NCCL/direct/disk fallback chain. SFT and DPO are simpler because they only require a single model and no rollout workers, so they run directly on DeepSpeed without Ray. All paradigms support full fine-tuning and LoRA, and plug into mixed-dataset sampling, experiment tracking, and standalone evaluation without changing the overall workflow.

## 🗂️ Codebase at a glance

The repository is organized so that algorithmic changes usually stay local:

- `algs/` — Algorithm and optimization logic. Each algorithm (PPO, GRPO, CISPO, P3O, DPO, SFT) has its own module with a README documenting the math and pseudocode.
- `rollouts/` — Rollout generation, vLLM engine wrappers, weight sync, and replay buffer.
- `rewards/` — Pluggable reward functions (GSM8K, math verification, and custom).
- `data_feeds/` — Data loading, sampling, and mixed-dataset support.
- `data_prep/` — Dataset preparation scripts.
- `configs/` — YAML configs for RL, SFT, DPO, and evaluation, with full [parameter reference](configs/README.md).
- `unit_tests/` — Unit and integration tests.

## 📢 News

- ![Date](https://img.shields.io/badge/2026--03--03-purple) We're excited to publicly release FeynRL as a preview! Some features and documentation are still evolving. We welcome feedback, bug reports, and contributions as we continue to build this together.

## 📖 How to Use FeynRL

**[Installation & Setup](docs/INSTALL.md)** — Configure your environment and dependencies.

**[Usage & Examples](docs/HOWTO.md)** — Learn how to launch jobs and run experiments.

**[Configuration Reference](configs/README.md)** — Full parameter guide for RL, SFT, DPO, and evaluation configs.

## 🤝 Contributing

Contributions are welcome! Please see our **[Contributing Guidelines](CONTRIBUTING.md)** for details on how to get involved.

## ❓ FAQ

Check out the [FAQ](docs/FAQ.md) for common questions and answers.
