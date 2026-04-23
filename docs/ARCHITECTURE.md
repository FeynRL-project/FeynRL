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

2. **Overlap / async** (`run_rl_async.py`): Generation and training run concurrently on separate GPU pools. This mode significantly reduces GPU idle time. The key mechanics:

   - **Persistent producer / pull-loop architecture**: Pull loops, the bounded `prompt_queue`, the bounded `results_queue`, and a daemon `ShardProducer` thread are all built **once** in `main()` and persist for the entire run. The producer thread continuously tops up `prompt_queue` from an `InfiniteShardIterator` (which reshuffles the dataloader on each pass); rollout engines run a long-lived `run_pull_loop` that blocks on `prompt_queue.get()` and exits only on a `POISON_PILL` sentinel. The driver drains `results_queue` into the replay buffer at the start of each round and rebuilds DeepSpeed shards before training. Generation never pauses for epoch boundaries or for `save_checkpoint`.
   - **Pipelined generation**: Inside `run_pull_loop`, one shard is always in flight on vLLM via `submit_generation`/`complete_generation`, so new prompts join the running continuous batch before the previous shard's tail finishes. This eliminates the shard-boundary throughput gap.
   - **End-of-epoch weight sync**: Sync runs once at the end of every non-final epoch (no mid-epoch sync triggers). When it runs, `perform_inline_sync` stops the producer, drains the prompt queue and pushes poison pills, then **fires the ZeRO-3 gather on all training engines** (`start_nccl_gather`) immediately — the gather is a training-side-only collective that doesn't involve rollout engines, so it runs concurrently with the pull-loop drain that follows. The driver waits for in-flight `complete_generation` calls to finish (continuously draining `results_queue` so engines blocked on `put` can exit) while the gather runs in the background on training GPUs. Once drain completes, the driver waits for the gather result, runs the broadcast and finalize phases (`broadcast_and_finalize_nccl`, with partial-load detection), relaunches the pull loops with the new policy version, and restarts the producer. Leftover prompts are dropped — the infinite iterator replays equivalent work on restart. Overlapping the gather with the drain reduces the sync bubble by the gather latency (~5–20 s for large models).
   - **Staleness control via `overlap.max_lag`**: After each round's drain, items older than `current_policy_version − max_lag` are evicted from the replay buffer (version-based, in addition to FIFO size eviction). The replay buffer is a hard-capped `deque` whose size is derived from `rollout_samples_per_epoch`, `n_samples`, and `max_lag`. The *effective* lag observed at training time can be smaller than the configured `max_lag` when rollout is faster than training, because FIFO trims oldest items before age-based eviction does — the `rollout/speed_ratio` metric (logged at the end of each round) reports actual items-per-cycle ÷ theoretical items-per-round; a ratio above 1.0 indicates the rollout-faster regime. Recent samples persist across epochs, eliminating the cold-start bubble.
   - **Carryover-aware round drain**: Items that arrived in `results_queue` during the previous epoch's sync are counted toward the next round's drain target, so the round drain skips blocking on rollout if pull-loops already produced enough during sync. This preserves the async wall-clock benefit when training is slower than rollout.
   - **Foreground checkpoint save**: Because pull loops, queues, and the producer are all persistent, `save_checkpoint` runs in the foreground while generation continues uninterrupted in the background — no pre-launch hand-off is needed.
   - **NCCL-only at runtime**: Async mode forces `weight_sync_method: nccl` via the config validator. There is no runtime fallback to direct/disk. Instead, FeynRL pairs NCCL with the watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_TIMEOUT=1800`) and pattern-matches fatal NCCL errors (communicator destruction, Ray timeout, actor death) to fail fast rather than retry on a broken communicator.
   - **Hang resistance**: Two-stage health check — the Ray actor is declared with three concurrency groups (`health`, `pull`, `mailbox`), one slot each ([`rollouts/vllm_engine_async.py:19`](../rollouts/vllm_engine_async.py#L19)). `ping` runs in `health` to bypass the default mailbox FIFO and catch dead processes. `ping_mailbox` runs in the dedicated `mailbox` group — it does not queue behind `run_pull_loop`; instead it checks the staleness of `_last_pull_progress` (updated each iteration by `run_pull_loop`) and raises if the stamp hasn't advanced within `wedge_threshold_s` (default 300 s), catching wedged actors whose pull loop is stuck in `complete_generation` or blocked on a saturated `results_queue`. Beyond the health checks: fast-fail `ray.wait(pull_refs, timeout=0)` at every training-loop iteration surfaces a crashed pull loop before the next weight sync; producer thread crashes are captured in `self.error` and re-raised on the main thread via `producer.check_error()`; bounded `results_queue` and `prompt_queue` provide natural back-pressure; every distributed wait is bounded by `init_timeout`, `train_step_timeout`, `sync_timeout`, or `rollout_timeout`.
   - **Pipeline diagnostics**: Per-round console logs report `Round start: results_queue_qsize=…`, `Drain: …shards`, `Buffer at train: items_added_this_round=…, lag_max=…, lag_mean=…, unique_versions=…`, `evict_stale(min_v=…): -K items`, `Sync drain: +K items (N shards) into buffer`, and `Pipeline regime: items_per_cycle=…, rollout_speed_ratio=…`. The corresponding tracker metrics are logged once per round under the `rollout/` namespace: `items_per_cycle`, `speed_ratio`, `buffer_lag_max`, `buffer_lag_mean`, `buffer_unique_versions`, plus the standard rollout summary stats.

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