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
FeynRL supports three methods for syncing weights from the training engine to the rollout workers:
1. **NCCL Broadcast** (fastest): Training rank 0 broadcasts weights directly to all rollout engine TP workers over a dedicated NCCL process group. Zero-copy, no serialization overhead. The custom NCCL process group is namespaced via PrefixStore to avoid conflicting with DeepSpeed's default group.
2. **Direct Sync**: Weights are gathered from training engines via DeepSpeed's ZeRO gather and pushed to rollout workers through Ray's shared-memory object store. No disk I/O, but involves serialization.
3. **Disk Sync**: Weights are saved as a checkpoint to disk, and rollout workers reload from the saved path. Slowest, but the most robust.

In **sync mode** (`run_rl_sync.py`), `weight_sync_method` must be `"direct"` or `"disk"` — the config validator at [`configs/load.py:684-686`](../configs/load.py#L684-L686) rejects `"nccl"` in sync mode because the non-async vLLM engine has no NCCL weight-sync path. If `"direct"` fails mid-run, FeynRL falls back to a `"disk"` save + rollout reload so the rollout engines always receive updated weights. In **overlap mode** (`run_rl_async.py`), the config validator forces `weight_sync_method: nccl` and there is **no runtime fallback chain** — async mode pairs NCCL with a watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_TIMEOUT=1800`, set by [`misc/nccl_env.py`](../misc/nccl_env.py)) that aborts wedged collectives so the job fails fast with a clear error rather than silently retrying on a destroyed communicator. (`direct` is still used by `load_checkpoint_for_resume` exactly once, before the NCCL group has been initialized.)

### 🔁 Training↔Rollout Scheduling
FeynRL supports two execution modes, dispatched from `main_rl.py` based on `config.overlap.enabled`:

1. **Synchronous** (`run_rl_sync.py`): Each epoch generates all rollouts (blocking), trains on them, syncs weights, and repeats. Fully on-policy, simple to debug, and the right choice when data freshness matters more than throughput. Sync mode supports the `direct` → `disk` fallback chain for weight sync (the validator forbids `nccl` in sync mode; see [`configs/load.py:684-686`](../configs/load.py#L684-L686)).

2. **Overlap / async** (`run_rl_async.py`): Generation and training run concurrently on separate GPU pools, so neither waits for the other to finish. Rollout engines stream samples into a bounded replay buffer while training consumes from it; weights are synced once per epoch via NCCL. This significantly reduces GPU idle time when generation is the bottleneck. Off-policy drift is bounded by `overlap.max_lag` (max policy-version distance between the buffer's oldest and newest samples). Async mode requires `weight_sync_method: nccl` and pairs it with an NCCL watchdog so wedged collectives fail fast instead of hanging. Our async engine design is inspired by PipelineRL [[1]](#references), but FeynRL diverges in two significant ways: it keeps a bounded replay buffer with version-based eviction whereas PipelineRL is purely streaming with no replay; and it syncs once per epoch whereas PipelineRL applies weight updates mid-sequence. PipelineRL stays as on-policy as possible; FeynRL deliberately works with older, off-policy data because controlled off-policy reuse can be a valuable training signal in itself.

   **When to use each mode:**
   - Use **overlap (async) mode** when generation is the bottleneck (large models, long sequences) and you want to fill training GPU idle time with useful work.
   - Use **sync mode** when you need strict on-policy data, are debugging, when training and generation take roughly the same time, or when you want the `direct` → `disk` weight-sync fallback chain (sync-only).

## 🧩 Modularity & Extensibility

- **Algorithm Agnostic**: The system is designed to support various algorithms by providing a common interface for data handling and model updates.
- **Pluggable Rewards**: Custom reward functions can be easily integrated in the configuration.
- **Flexible Data Processing**: The data pipeline supports mixed-dataset sampling with configurable ratios, allowing for complex training recipes.

## 📂 Repository Structure

```text
FeynRL/
├── algs/               # Implementation of various algorithms such as PPO, GRPO, CISPO, P3O, DPO, SFT
├── configs/            # YAML configuration files and Pydantic schema validation
├── core/               # RL training/rollout engine primitives (NCCL gather/broadcast, dataloaders)
├── data_feeds/         # Data loading, mixed-dataset sampling, and dataset construction
├── data_prep/          # Scripts for processing raw datasets
├── docs/               # Documentation files (Installation, FAQ, Architecture, Troubleshooting)
├── examples/           # End-to-end recipes for different tasks with full results
├── misc/               # Utility modules (logging, trackers, NCCL helpers)
├── rewards/            # Reward functions for RL training
├── rollouts/           # vLLM-powered rollout engine, replay buffer, and weight synchronization
├── unit_tests/         # Unit and integration tests
├── main_rl.py          # Entry point for RL (dispatches to run_rl_sync or run_rl_async)
├── main_sl.py          # Entry point for Supervised Fine-Tuning (SFT)
├── main_cl.py          # Entry point for Preference Learning (e.g., DPO)
├── main_eval.py        # Entry point for standalone model evaluation
├── run_rl_sync.py      # Sync training↔rollout loop
├── run_rl_async.py     # Overlap (async) training↔rollout loop
├── requirements.txt    # Project dependencies
├── pyproject.toml      # Project metadata and build config
├── CONTRIBUTING.md     # Contribution guidelines
├── LICENSE             # Project license
├── .gitignore          # Git ignore rules
└── README.md           # Main project landing page
```

## References

[1] A. Piché, E. Kamalloo, R. Pardinas, X. Chen, and D. Bahdanau. *PipelineRL: Faster On-policy Reinforcement Learning for Long Sequence Generation.* arXiv:2509.19128, 2025. [https://arxiv.org/abs/2509.19128](https://arxiv.org/abs/2509.19128). Code: [https://github.com/ServiceNow/PipelineRL](https://github.com/ServiceNow/PipelineRL).