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

In **sync mode** (`run_rl_sync.py`), all three methods are available and FeynRL automatically falls back from NCCL to direct to disk if a sync fails. In **overlap mode** (`run_rl_async.py`), the config validator forces `weight_sync_method: nccl` and there is **no runtime fallback chain** — async mode pairs NCCL with a watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`) that aborts wedged collectives so the job fails fast with a clear error rather than silently retrying on a destroyed communicator. (`direct` is still used by `load_checkpoint_for_resume` exactly once, before the NCCL group has been initialized.)

### 🔁 Training↔Rollout Scheduling
FeynRL supports two execution modes, dispatched from `main_rl.py` based on `config.overlap.enabled`:

1. **Synchronous** (`run_rl_sync.py`): Each epoch generates all rollouts (blocking), trains on them, syncs weights, and repeats. Fully on-policy, simple to debug, and the right choice when data freshness matters more than throughput. Sync mode supports the full NCCL→direct→disk fallback chain for weight sync.

2. **Overlap / async** (`run_rl_async.py`): Generation and training run concurrently on separate GPU pools within the same epoch. This mode significantly reduces GPU idle time. It works as follows:

   - **Queue-pull architecture**: At the start of each epoch the driver fills a shared Ray `prompt_queue` with all sharded prompt batches plus one POISON_PILL sentinel per rollout engine. Each rollout engine runs a long-lived `run_pull_loop` that pulls shards from the queue and pushes results to a bounded `results_queue`. There is no central dispatcher as engines self-schedule, so a fast engine naturally processes more shards than a slow one. The driver drains `results_queue` between training steps and rebuilds DeepSpeed shards from the replay buffer as new samples arrive.
   - **Pipelined generation**: Inside `run_pull_loop`, one shard is always in flight on vLLM via `submit_generation`/`complete_generation`, so new prompts join the running continuous batch before the previous shard's tail finishes. This eliminates the shard-boundary throughput gap.
   - **Mid-epoch weight sync**: Triggered either by ESS (Effective Sample Size, for P3O) when it drops below `ess_sync_threshold`, or by a fixed step interval (`fixed_sync_interval`, required for PPO/GRPO/CISPO since they don't expose ESS). When triggered, `perform_inline_sync` drains the prompt queue, pushes poison pills, waits for in-flight `generate()` calls to finish, runs the three-phase NCCL sync (gather → broadcast → finalize, with sender- and receiver-side NaN/Inf checks and partial-load detection), then requeues leftover shards and relaunches the pull loops with the new policy version.
   - **Staleness control**: A configurable `max_lag` bounds how many policy versions the rollout data can lag behind. End-of-epoch sync fires when `lag ≥ max_lag`; otherwise it is skipped. The replay buffer evicts samples older than `policy_version - max_lag` at the end of each epoch, retaining recent samples across epochs and eliminating the cold-start bubble.
   - **Pre-launched next epoch**: After end-of-epoch sync but before `save_checkpoint`, the driver pre-launches the next epoch's queues and pull loops so rollout engines generate the next epoch's data while the driver writes the checkpoint to disk, hiding the 5-60s save bubble.
   - **NCCL-only at runtime**: Async mode forces `weight_sync_method: nccl` via the config validator. There is no runtime fallback to direct/disk. Instead, FeynRL pairs NCCL with the watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_TIMEOUT=1800`) and pattern-matches fatal NCCL errors (communicator destruction, Ray timeout, actor death) to fail fast rather than retry on a broken communicator.
   - **Hang resistance**: Two-stage health check (`ping` runs in a separate Ray concurrency group to bypass the mailbox FIFO; `ping_mailbox` runs in the default group to detect wedged actors); explicit POISON_PILL sentinels at the queue tail eliminate polling timeouts; bounded `results_queue` provides natural back-pressure; every distributed wait is bounded by `init_timeout`, `train_step_timeout`, `sync_timeout`, or `rollout_timeout`.

   **When to use each mode:**
   - Use **overlap (async) mode** when generation is the bottleneck (large models, long sequences) and you want to fill training GPU idle time with useful work.
   - Use **sync mode** when you need strict on-policy data, are debugging, when training and generation take roughly the same time, or when you want the NCCL→direct→disk fallback chain.

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