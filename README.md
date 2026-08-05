<p align="center">
  <img src="docs/feynrl.png" alt="FeynRL Logo" width="200">
</p>

<p align="center">
  Algorithm-first post-training framework for large language models (LLMs) and vision-language models (VLMs).
</p>

<p align="center">
  <a href="https://github.com/FeynRL-project/FeynRL"><img src="https://img.shields.io/badge/GitHub-FeynRL-181717?style=flat-square&logo=github" alt="GitHub"></a>&nbsp;
  <a href="https://feynrl-project.github.io"><img src="https://img.shields.io/badge/Blog-FeynRL-E65100?style=flat-square&logo=googlechrome&logoColor=white" alt="Blog"></a>&nbsp;
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-2E7D32?style=flat-square" alt="License"></a>
</p>

---

<p align="center">
  <em>"What I cannot create, I do not understand."</em> — Richard Feynman
</p>

**FeynRL** (pronounced "FineRL") is an **algorithm-first** framework for **post-training and fine-tuning** large models. It works with both **text-only large language models (LLMs)** and **vision-language models (VLMs)**, and supports supervised fine-tuning (SFT), preference learning (e.g., DPO), and reinforcement learning (e.g., PPO, GRPO, DAPO, CISPO, P3O) across both. It is built for researchers and engineers who want to understand, modify, and develop new methods without fighting the infrastructure.

The main goal of FeynRL is simple: make new algorithms easy to implement, easy to debug, and still possible to train at scale. The codebase is designed so that **algorithmic logic stays local** and **systems logic stays explicit**, which makes the framework easier to reason about, easier to extend, and more reliable to debug.

## 📢 News

- ![Date](https://img.shields.io/badge/2026--07--22-red) FeynRL now supports vision-language models (VLMs) end-to-end for both SFT and RL and a unified config space for both, along with other improvements to stability and performance. We've also added more ready-to-run examples for LLMs and VLMs.
- ![Date](https://img.shields.io/badge/2026--04--27-green) FeynRL is now publicly announced! Since the preview, we've added a new async engine and a collection of tricks and ideas, many not easily found elsewhere, that materially improve training stability and reliability. Thanks to everyone who tried the preview and shared feedback.
- ![Date](https://img.shields.io/badge/2026--03--03-purple) We're excited to publicly release FeynRL as a preview! Some features and documentation are still evolving. We welcome feedback, bug reports, and contributions as we continue to build this together.

## 🎯 Why FeynRL?

Most post-training frameworks optimize for the largest built-in feature surface, which makes them powerful but hard to modify once you want to try something new. FeynRL makes the opposite trade-off: it optimizes for **clarity, locality of change, and algorithm development**. Adding a new method usually means writing a single file with its own loss and update logic, not threading changes through the orchestration, rollout, and data layers.

- **Algorithm-first** — New objectives, rewards, baselines, or update rules stay local. Algorithm code stays algorithmic; systems code stays explicit.
- **One framework across post-training** — SFT, DPO, and RL share the same workflow and config for both LLMs and VLMs, so comparisons are easy and infrastructure isn't duplicated.
- **Scales from the start** — The same code runs single-GPU debugging or large multi-node jobs, backed by DeepSpeed, Ray, and vLLM with sync/async execution and adaptive weight sync.

FeynRL is built for anyone who wants to **deeply understand training of large models, become an expert, and build new methods and ideas**. It's written to be read and learned from, so you can trace exactly how rollouts, losses, and weight syncs fit together, experiment with your own methods, and grow from running recipes to designing them. You can still use it as a ready-made recipe out of the box; the difference is that nothing stays a black box when you're ready to look inside.

This is the first public release, so expect rough edges. We're open-sourcing it as a foundation for post-training research with the community.

## ✅ What's Included

For a detailed breakdown of the architecture, see the **[Architecture Overview](docs/ARCHITECTURE.md)**.

- 🧪 **Training paradigms**: RL (PPO, GRPO, DAPO, CISPO, P3O), preference-based learning (DPO), and supervised fine-tuning (SFT)
- 🧠 **LLMs and VLMs**: train text-only language models or vision-language (image+text) models with the same recipes.
- 🖥️ **Distributed training**: Multi-GPU and multi-node via DeepSpeed (ZeRO Stage 1/2/3)
- 🎲 **Rollouts / inference**: vLLM-powered rollout engines with tensor parallelism
- 🛰️ **Orchestration**: Ray for scheduling training and rollout workers across nodes
- 🔀 **Training-rollout scheduling**: Sync and overlap (async) modes. In overlap mode, rollout generation and training run concurrently on separate GPU pools to reduce idle time, with a configurable staleness budget bounding how off-policy the replay data can drift.
- 🔄 **Weight sync**: direct GPU-memory push with disk fallback in sync mode; NCCL broadcast in async mode, with a built-in NCCL watchdog and fail-fast on communicator destruction.
- 🧷 **Parameter-efficient fine-tuning**: LoRA via PEFT
- 🔢 **Mixed-dataset sampling**: Configurable multi-dataset sampling with ratios within a single training run
- 📈 **Experiment tracking**: MLflow and Weights & Biases support
- 🏅 **Evaluation**: Standalone eval pipeline with vLLM engines

For RL, Ray orchestrates the full training loop: it schedules DeepSpeed training workers and vLLM rollout workers across nodes, and coordinates weight synchronization between them. In **sync mode**, each epoch generates all rollouts, trains on them, syncs weights, and repeats — fully on-policy and easy to reason about. In **overlap mode** (also called async mode), rollout generation and training run concurrently on separate GPU pools so training GPUs don't sit idle waiting for rollouts. Generation is continuous across epoch boundaries and checkpoint saves — the only pauses are brief drains during weight sync, which runs once at the end of every non-final epoch. A configurable staleness budget bounds how off-policy the replay data can drift. Async mode uses NCCL for weight sync; sync mode pushes weights directly through GPU memory and falls back to disk-based reload if that fails. SFT and DPO are simpler because they only require a single model and no rollout workers, so they run directly on DeepSpeed without Ray. All paradigms support full fine-tuning and LoRA, and plug into mixed-dataset sampling, experiment tracking, and standalone evaluation without changing the overall workflow.

## 🗂️ Codebase at a glance

The repository is organized so that algorithmic changes usually stay local:

- `algs/` — Algorithm and optimization logic. Each algorithm (PPO, GRPO, DAPO, CISPO, P3O, DPO, SFT) has its own module with a README documenting the math and pseudocode.
- `rollouts/` — Rollout generation, vLLM engine wrappers, weight sync, and replay buffer.
- `rewards/` — Pluggable reward functions (GSM8K, math verification, and custom).
- `data_feeds/` — Data loading, sampling, and mixed-dataset support.
- `data_prep/` — Dataset preparation scripts.
- `configs/` — YAML configs for RL, SFT, DPO, and evaluation, with full [parameter reference](configs/README.md).
- `unit_tests/` — Unit and integration tests.

## 📖 How to Use FeynRL

**[Installation & Setup](docs/INSTALL.md)** — Configure your environment and dependencies.

**[Quickstart & How-To](docs/HOWTO.md)** — Learn how to launch jobs and run experiments.

**[Experiments](examples/README.md)** — Reference experiment results and the canonical example configs used to reproduce them.

**[Configuration Reference](configs/README.md)** — Full parameter guide for RL, SFT, DPO, and evaluation configs.

**[Troubleshooting](docs/TROUBLESHOOTING.md)** — Diagnose and fix common issues.

## 🤝 Contributing

Contributions are welcome! Please see our **[Contributing Guidelines](CONTRIBUTING.md)** for details on how to get involved.

## ❓ FAQ

Check out the [FAQ](docs/FAQ.md) for common questions and answers.


## 🙏 Acknowledgements

Special thanks to the [Open-Instruct](https://github.com/allenai/open-instruct) and [PipelineRL](https://github.com/ServiceNow/PipelineRL) projects, both of which provided valuable resources for this work and the broader community.

## 📚 Citation

If you use FeynRL in your work, please cite the following:

```bibtex
@misc{FeynRL,
  author       = {Rasool Fakoor and Murdock Aubry and FeynRL Contributors},
  title        = {FeynRL: A Modular and Scalable LLM Post-Training Framework},
  year         = {2026},
  howpublished = {\url{https://github.com/FeynRL-project/FeynRL}},
  note         = {GitHub repository. Corresponding author: Rasool Fakoor}
}

@misc{fakoor2026trustbatchonoffpolicy,
  title         = {Trust the Batch, On- or Off-Policy: Adaptive Policy Optimization for RL Post-Training},
  author        = {Rasool Fakoor and Murdock Aubry and Nicholas Stranges and Alexander J. Smola},
  year          = {2026},
  eprint        = {2605.12380},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.12380}
}
```
