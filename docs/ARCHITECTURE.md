# FeynRL Architecture Overview

FeynRL is designed with a **separation of concerns** between algorithmic logic and system-level orchestration. This modularity allows researchers and engineers to focus on developing new methods while leveraging a scalable, high-performance training stack.

## System Components

### 🛰️ Orchestration (Ray)
Ray serves as the central orchestrator, managing the lifecycle of distributed workers across a cluster. It schedules:
- **Training Workers**: Handle distributed training using DeepSpeed.
- **Rollout Workers**: Generate trajectories using vLLM rollout engines.

### 🖥️ Training Engine (DeepSpeed)
The training engine utilizes **DeepSpeed** for distributed training, supporting:
- **ZeRO Stage 1/2/3**: Efficient parameter, gradient, and optimizer state partitioning.
- **CPU Offloading**: Optional offloading of optimizer states and parameters to CPU memory to handle larger models.
- **LoRA Support**: Parameter-efficient fine-tuning via PEFT integration.

### 🎲 Rollout Engine (vLLM)
Trajectory generation is powered by **vLLM**, which provides:
- **High Throughput Generation**: Optimized kernels and PagedAttention for fast inference.
- **Tensor Parallelism**: Capability to shard large models across multiple GPUs for rollout.
- **Dynamic Loading**: Support for directly updating policy weights during training.

### 🔄 Weight Synchronization
FeynRL supports two methods for syncing weights from the training engine to the rollout workers:
1. **Direct Sync**: Weights are gathered from training engines and pushed to rollout workers via Ray's shared-memory object store, avoiding disk I/O entirely.
2. **Disk Sync**: Weights are saved as a checkpoint to disk, and rollout workers reload from the saved path. This also serves as an automatic fallback if direct sync fails.

### 🔁 Training↔Rollout Scheduling
FeynRL supports two execution modes that control how rollout generation and training are scheduled:

1. **Synchronous**: Each epoch generates rollouts, trains on them, syncs weights, and repeats. Simple, fully "on-policy", and easy to debug. Training and rollout workers are scheduled sequentially on separate GPUs.
2. **Overlap (async prefetch)**: When enabled, the next epoch's rollouts are scheduled *during* the current training step, running concurrently on separate GPUs. A configurable maximum lag controls how many policy versions the rollout engine can fall behind before a weight sync is forced. This improves GPU utilization by reducing idle time, at the cost of slightly staler rollout data.

## 🧩 Modularity & Extensibility

- **Algorithm Agnostic**: The system is designed to support various algorithms by providing a common interface for data handling and model updates.
- **Pluggable Rewards**: Custom reward functions can be easily integrated in the configuration.
- **Flexible Data Processing**: The data pipeline supports mixed-dataset sampling with configurable ratios, allowing for complex training recipes.

## 📂 Repository Structure

```text
FeynRL/
├── algs/               # Implementation of various algorithms such as PPO, GRPO, CISPO, DPO, SFT
├── configs/            # YAML configuration files and Pydantic schema validation
├── data_feeds/         # Data loading, mixed-dataset sampling, and dataset construction
├── data_prep/          # Scripts for processing raw datasets
├── docs/               # Documentation files (Installation, FAQ, Architecture, Troubleshooting)
├── experiments/        # Experiment configurations and documentation
├── misc/               # Utility modules (logging, trackers, helpers)
├── rewards/            # Reward functions for RL training
├── rollouts/           # vLLM-powered rollout engine and weight synchronization
├── main_rl.py          # Entry point for Reinforcement Learning training
├── main_sl.py          # Entry point for Supervised Fine-Tuning (SFT)
├── main_cl.py          # Entry point for Preference Learning (e.g., DPO)
├── main_eval.py        # Entry point for standalone model evaluation
├── requirements.txt    # Project dependencies
├── CONTRIBUTING.md     # Contribution guidelines
├── LICENSE             # Project license
├── .gitignore          # Git ignore rules
└── README.md           # Main project landing page
```