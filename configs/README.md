# Configuration Reference

All experiments are configured via YAML files validated by Pydantic schemas in `load.py`. Template configs are provided for each experiment type.

## Experiment Types
1. **Reinforcement Learning (RL)**: `rl_args.yaml` — supports GRPO, CISPO, PPO, P3O
2. **Supervised Learning (SL)**: `sl_args.yaml` — Supervised Fine-Tuning (SFT)
3. **Contrastive Learning (CL)**: `cl_args.yaml` — Direct Preference Optimization (DPO)
4. **Evaluation**: `eval_args.yaml` — inference and scoring

## Command-Line Arguments

All `main_*.py` entry points accept the following arguments:

| Argument | Description | Default |
|:---|:---|:---|
| `--config_file` | Path to the YAML config file | `"./config/myexp_rl.yaml"`|
| `--experiment_id` | Unique experiment identifier | `"run_1"` |
| `--log_level` | Logging level | `"INFO"` |
| `--resume_from` | Path to a checkpoint directory to resume training (not available in `main_eval.py`) | `None` |

**Resuming from a checkpoint:** The checkpoint directory must contain a `CHECKPOINT_COMPLETE` marker. The run configuration such as number of GPUs, DeepSpeed settings, etc., must exactly match the original run.

---

## `run` — Experiment Settings

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `seed` | Random seed to help with reproducibility | Integer ≥ 0 | `42`, `1337` |
| `project_name` | Project name in experiment tracker | String | `"FeynRL"`, `"my-project"` |
| `logger_type` | Tracker type | `"mlflow"` \| `"wandb"` | `"mlflow"` |
| `tracking_uri` | URI for the tracking server | String (URL) - only for mlflow | `"http://mlflow:8080/"` |
| `method` | Set automatically by entry point | `"rl"` \| `"sl"` \| `"cl"` \| `"eval"` | `"rl"` |
| `checkpoint_dir` | Directory for saving checkpoints | Path string | `"./ckps"`, `"/data/ckps"` |
| `checkpoint_save_interval` | Save checkpoint every N epochs; 0 = end only (default: `1`) | Integer ≥ 0 | `1`, `5`, `0` |

#### NCCL / Multi-Node

These settings apply to all experiment types (SL, CL, RL). Leave as `null` for single-node or when NCCL auto-detection works.

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `nccl_socket_ifname` | Network interface for inter-node traffic. Run `ip addr` to find it. | String \| `null` | `"eth0"`, `"bond0"` |
| `nccl_ib_hca` | InfiniBand HCA device(s) to use or exclude. Run `ibstat` to find it. | String \| `null` | `"mlx5_0"`, `"^mlx5_2,mlx5_3"` |

If you encounter NCCL InfiniBand connection errors (`ibv_modify_qp failed with 110 Connection timed out`), see the [Troubleshooting Guide — NCCL InfiniBand connection timeout](../docs/TROUBLESHOOTING.md#nccl-infiniband-connection-timeout) for diagnosis and fix steps.

#### RL-Specific Run Settings

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `training_gpus` | GPUs for DeepSpeed training engines | Integer ≥ 1 | `1`, `3`, `8` |
| `rollout_gpus` | GPUs for vLLM rollout engines (also used in eval) | Integer ≥ 1 | `2`, `4`, `7` |
| `ray_address` | Ray cluster address |`"auto"` if multi-node, `null` if single-node | `"auto"`  |
| `ray_master_port` | Port for torch distributed rendezvous | Integer \| `null` | `29500` |
| `weight_sync_method` | Weight sync method. `"nccl"` is **both required and exclusive** to overlap mode: `overlap.enabled=True` requires `"nccl"`, and `"nccl"` requires `overlap.enabled=True`. Sync mode must use `"direct"` or `"disk"`. | `"direct"` \| `"disk"` \| `"nccl"` | `"direct"` |
| `nccl_sync_port` | Port for NCCL weight sync rendezvous (default: `ray_master_port + 100`) | Integer \| `null` | `29600` |
| `nccl_sync_backend` | Backend for weight sync broadcast. `"nccl"` uses GPU-to-GPU transfer via vLLM's PyNcclCommunicator (fast, recommended for full-model sync). `"gloo"` uses CPU-based transfer via torch.distributed (robust fallback). | `"nccl"` \| `"gloo"` | `"nccl"` |

#### RL Timeouts (seconds)

| Parameter | Description | Type | Examples |
|:---|:---|:---|:---|
| `init_timeout` | Training engine initialization | Seconds (Integer ≥ 0) | `3600`  |
| `rollout_timeout` | Rollout generation per batch | Seconds (Integer ≥ 0) | `7200` |
| `train_step_timeout` | Single training step | Seconds (Integer ≥ 0) | `1200` |
| `save_timeout` | Checkpoint save |  Seconds (Integer ≥ 0) | `1800` |
| `sync_timeout` | Weight sync operations | Seconds (Integer ≥ 0) | `1800` |

---

## `overlap` — Overlap Engine (RL only)

Controls concurrent rollout generation and training. When `enabled: true`, rollout and training run on separate GPU pools to reduce idle time, and `weight_sync_method` must be `"nccl"` (validator-enforced). A configurable staleness budget (`max_lag`) bounds how off-policy the replay data can drift. See the [Architecture Overview](../docs/ARCHITECTURE.md#-trainingrollout-scheduling) for the full mechanics.

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `enabled` | Enable the async overlap engine. When `True`, `run.weight_sync_method` must be `"nccl"`. | Boolean | `False`, `True` |
| `max_lag` | Tolerated staleness, in policy versions. Bounds (1) the replay-buffer capacity (sized as `~max_lag` rounds of rollout output) and (2) the age-based eviction threshold applied after each round (items older than `current_version − max_lag` are dropped). The *effective* in-buffer lag can be smaller than `max_lag` when rollout is faster than training (FIFO eviction trims older items first); the `rollout/speed_ratio` metric reports the actual ratio of items produced per cycle vs. one theoretical round. | Integer ≥ 1 | `1`, `2`, `3` |
| `behave_imp_weight_cap` | Cap on the behavioral importance weight in the decoupled-PPO loss path used by GRPO / PPO / CISPO. Clamps the IS correction to a finite range to prevent runaway weights on stale data. **No-op for P3O** (which uses an ESS-driven trust-region KL instead). The validator requires `> 1.0` for non-P3O algorithms. | Float > 1.0 \| `null` | `null`, `5.0` |

---

## `train` — Training Configuration

### Optimizer

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `optimizer_name` | Base optimizer | `"adamw"` \| `"adam"` | `"adamw"` |
| `lr` | Learning rate | Float > 0 | `1e-5`, `5e-7` |
| `adam_epsilon` | Adam epsilon | Float | `1e-8` |
| `betas` | Adam beta parameters | List of 2 floats | `[0.9, 0.95]` |
| `weight_decay` | Weight decay | Float ≥ 0 | `0.01` |
| `warmup_steps_ratio` | Fraction of total steps for warmup | 0<= Float <=1 | `0.1` |
| `clip_grad_norm` | Max gradient norm | Float > 0 | `1.0` |
| `lr_scheduler` | LR scheduler type | `"WarmupCosineLR"` | `"WarmupCosineLR"` |

### Training Loop

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `alg_name` | Algorithm name | RL: `"grpo"` \| `"cispo"` \| `"ppo"` \| `"p3o"`, SL: `"sft"`, CL: `"dpo"` | `"grpo"` |
| `total_number_of_epochs` | Total training epochs | Integer ≥ 1 | `30`, `100` |
| `train_steps_per_epoch` | RL: optimizer steps per epoch | Integer ≥ 1 | `5`, `10` |
| `micro_batches_per_epoch` | SL/CL: micro-batch iterations per epoch | Integer ≥ 1 | `1000` |
| `train_batch_size_per_gpu` | Micro-batch size per GPU | Integer ≥ 1 | `2`, `4` |
| `gradient_accumulation_steps` | Gradient accumulation steps | Integer ≥ 1 | `1`, `4` |
| `val_batch_size_per_gpu` | Validation batch size per GPU | Integer ≥ 1 | `16` |
| `dynamic_ratio_every_step` | Recalculate dataset mix ratios every step | Boolean | `False` |
| `normalize_loss` | `True`: global per-token mean. `False`: global sum. See [SFT README — Loss Normalization](../algs/SFT/README.md#loss-normalization). | Boolean | `True` |

### RL Policy Arguments

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `kl_coeff` | KL divergence penalty weight | Float ≥ 0 | `0.0`, `0.001` |
| `clip_low` / `clip_high` | Policy ratio clipping bounds | Float ≥ 0 | `0.2` |
| `entropy_coeff` | Entropy bonus coefficient | Float ≥ 0 | `0.0`, `0.01` |
| `update_after_full_replay` | Update only after full replay buffer pass | Boolean | `True` |

### PPO-Specific

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `tau` | GAE lambda | Float \| `null` | `0.95` |
| `gamma` | Discount factor | Float \| `null` | `0.99` |
| `value_lr` | Value model LR; defaults to policy `lr` if `null` | Float \| `null` | `1e-5` |
| `value_weight_decay` | Value model weight decay; defaults to policy `weight_decay` if `null` | Float \| `null` | `0.01` |
| `value_clip_grad_norm` | Value model gradient clipping; defaults to policy `clip_grad_norm` if `null` | Float \| `null` | `1.0` |

### CL/DPO-Specific

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `cl_beta` | Beta for DPO objective | Float > 0 | `0.1` |

---

## `model` — Model Configuration

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `name` | Base model (HF ID or local path) | String | `"google/gemma-3-1b-it"` |
| `dtype` | Model precision | `"bfloat16"` \| `"bf16"` \| `"float16"` \| `"fp16"` \| `"float32"` \| `"fp32"` | `"bfloat16"` |
| `ref_model` | Reference model for KL/DPO | String \| `null` | `"google/gemma-3-1b-it"` |
| `value_model` | Value model path (PPO only) | String \| `null` | `"google/gemma-3-1b-it"` |
| `ref_model_offload_to_cpu` | Offload ref model to CPU | Boolean (default: `false`) | `true` |
| `trust_remote_code` | Allow HF remote code | Boolean | `false` |
| `model_class` | Model class identifier | String \| `null` | `"llm"` |
| `attn_implementation` | Attention backend | `"flash_attention_2"` \| `"eager"` \| `null` | `"flash_attention_2"` |
| `gradient_checkpointing` | Enable gradient checkpointing | Boolean \| `null` | `true` |

---

## `data` — Data Configuration

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `train_files_path` | Training Parquet file paths | List of strings | `["./data1.parquet", "./d2.parquet"]` |
| `val_files_path` | Validation Parquet file paths | List of strings | `["./d3.parquet"]` |
| `test_files_path` | Test Parquet file path (eval only) | String \| `null` | `"./test.parquet"` |
| `train_ratios` | Per-dataset sampling ratios | Dict (basename → float) | `{"data1": 8.0, "d2": 0.2}` |
| `num_workers` | DataLoader worker count | Integer ≥ 0 | `4` |
| `max_seq_len` | Max total sequence length (prompt + response) | Integer > 0 | `512`, `2048` |
| `prompt_key` | Prompt column name in Parquet | String | `"prompt"` |
| `answer_key` | Answer column name (SFT target) | String | `"answer"` |
| `solution_key` | Ground truth column for RL reward calculation | String \| `null` | `"solution"` |

---

## `rollout` — Rollout Generation (RL and Eval)

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `n_samples` | Completions per prompt | Integer ≥ 1 | `1`, `5`, `16` |
| `max_tokens` | Max generation tokens (must be < `max_seq_len`) | Integer ≥ 1 | `512`, `1024` |
| `rollout_samples_per_epoch` | Prompts to process per epoch (RL only) | Integer ≥ 1 | `100`, `500` |
| `rollout_batch_size_per_gpu` | Batch size for prompt dataloader | Integer ≥ 1 | `2`, `4` |
| `temperature` | Sampling temperature | Float ≥ 0 | `1.0`, `0.7` |
| `top_p` | Nucleus sampling threshold | Float 0–1 | `1.0`, `0.95` |
| `top_k` | Top-k sampling (`-1` = disabled) | Integer | `-1`, `50` |
| `tensor_parallel_size` | vLLM tensor parallelism (GPUs per engine) | Integer ≥ 1 | `1`, `2` |
| `gpu_memory_utilization` | vLLM GPU memory fraction | Float 0–1 | `0.5`, `0.9` |
| `force_strict_on_policy` | Enforce strict on-policy rollouts | Boolean | `true` |
| `ignore_eos` | Continue generation past EOS | Boolean | `false` |
| `stop` | Stop string for generation | String | `""` |
| `stop_token_ids` | Token IDs that trigger stop | List of integers | `[]` |
| `prompt_logprobs` | Return prompt token logprobs (memory intensive) | Boolean | `false` |
| `batch_invariant` | Force batch-invariant kernels (See [vLLM Reproducibility Doc](https://docs.vllm.ai/en/stable/examples/offline_inference/reproducibility/)) | Boolean | `false` |
| `max_model_len` | Override maximum context length for vLLM. Useful for models with complex RoPE scaling (e.g. YaRN) where vLLM fails to infer it. Otherwise, leave `null`. | Integer \| `null` | `8192`, `null` |

---

## `reward` — Reward Configuration (RL and Eval)

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `reward_func` | Reward function name in `rewards/` module | String | `"gsm8k_reward_func"` |
| `broadcast` | Broadcast scalar reward across response tokens | Boolean | `false` |

---

## `peft` — Parameter-Efficient Fine-Tuning

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `use_peft` | Enable LoRA | Boolean | `true`, `false` |
| `peft_type` | PEFT method | String | `"lora"` |
| `task_type` | Task type | String | `"CAUSAL_LM"` |
| `lora_rank` | LoRA rank | Integer ≥ 1 (when enabled) | `8`, `16`, `64` |
| `lora_alpha` | LoRA alpha scaling | Integer ≥ 1 (when enabled) | `16`, `32` |
| `lora_dropout` | LoRA dropout rate | Float 0–1 | `0.0`, `0.05` |
| `lora_target_modules` | Target modules; | List of strings \|  `null` = all linear layers | `["q_proj", "v_proj"]` |

---

## `deepspeed` — DeepSpeed Configuration

DeepSpeed settings are defined under the `deepspeed` key. Some parameters such as `train_batch_size`, `train_micro_batch_size_per_gpu`, `gradient_accumulation_steps`, `gradient_clipping`, `optimizer`, `scheduler`, `fp16`/`bf16` are **automatically synced** from the `train` and `model` sections, hence do not set them manually in the DeepSpeed block.

A separate `deepspeed_ref` section will be configured automatically for inference-only DeepSpeed for the reference model (RL/CL).
A separate `deepspeed_value` section will be configured automatically for DeepSpeed for the value model (PPO).

### ZeRO Optimization (`zero_optimization`)

| Parameter | Description | Type / Constraint | Default |
|:---|:---|:---|:---|
| `stage` | ZeRO stage | `0` (disabled) \| `1` (optimizer state) \| `2` (optimizer + gradients) \| `3` (optimizer + gradients + parameters) | `3` |
| `stage3_param_persistence_threshold` | Threshold for keeping small parameters on GPU to avoid transfer thrashing. `"auto"` sets it to `hidden_size * 10`. | Float \| `"auto"` | `"auto"` |
| `stage3_prefetch_bucket_size` | How much data is pre-fetched from CPU to GPU. `"auto"` sizes based on model `hidden_size`, avoiding excessive VRAM usage. | Float \| `"auto"` | `"auto"` |
| `stage3_max_live_parameters` | Max parameters live on GPU simultaneously during forward/backward. `"auto"` picks a conservative value (the hardcoded default of 1e9 is often too large). | Float \| `"auto"` | `"auto"` |
| `stage3_max_reuse_distance` | When to release gathered parameters. `"auto"` releases sooner for memory savings. | Float \| `"auto"` | `"auto"` |
| `sub_group_size` | Sub-group size for gradient partitioning. `"auto"` lets DeepSpeed optimize. | Float \| `"auto"` | `"auto"` |
| `zero_hpz_partition_size` | **ZeRO++ (multi-node only).** Hierarchical partitioning: keeps a full parameter copy within each node so all-gather stays intra-node (fast NVLink) instead of crossing the network. Set to the number of GPUs per node. No effect on single-node setups. | Int | Not set |
| `reduce_bucket_size` | Chunk size for gradient reduce between GPUs. `"auto"` sizes based on `hidden_size` (~`hidden_size²`), avoiding excessive peak memory that can cause allocator cache flushes. | Float \| `"auto"` | `"auto"` |
| `allgather_bucket_size` | Chunk size for allgather between GPUs. Same sizing rationale as `reduce_bucket_size`. | Float \| `"auto"` | `"auto"` |
| `contiguous_gradients` | Copy gradients to a contiguous buffer to reduce fragmentation | Boolean | `true` |
| `overlap_comm` | Overlap gradient reduction with backward pass computation | Boolean | `true` |
| `reduce_scatter` | Use reduce-scatter to average and scatter gradients to the responsible GPU | Boolean | `true` |
| `stage3_gather_16bit_weights_on_model_save` | Gather fragmented ZeRO-3 weights into a single FP16 file when saving | Boolean | `true` |

#### CPU Offloading

| Parameter | Description | Type / Constraint | Default |
|:---|:---|:---|:---|
| `offload_optimizer.device` | Offload optimizer state to CPU. Use `"none"` if GPU VRAM suffices. | `"cpu"` \| `"none"` | `"none"` |
| `offload_optimizer.pin_memory` | Pin CPU memory for faster transfers | Boolean | `true` |
| `offload_param.device` | Offload parameters to CPU. High performance penalty; use `"none"` if possible. | `"cpu"` \| `"none"` | `"none"` |
| `offload_param.pin_memory` | Pin CPU memory for faster transfers | Boolean | `true` |

### Top-Level DeepSpeed Settings

| Parameter | Description | Type / Constraint | Default |
|:---|:---|:---|:---|
| `prescale_gradients` | Pre-scale loss before backward. Set to `false` to scale after backward instead (matches open-instruct and OpenRLHF). | Boolean | `false` |
| `data_types.grad_accum_dtype` | Accumulate gradients in this dtype for numerical stability under bf16/fp16 training. | `"fp32"` \| `"fp16"` \| `"bf16"` | `"fp32"` |


> **Note on `"auto"` values:** Most ZeRO-3 tuning knobs now default to `"auto"`, which lets DeepSpeed size buffers based on the model's `hidden_size`. This avoids hardcoded values (e.g., 500MB bucket sizes) that can cause excessive peak memory, allocator cache flushes, and corrupt bf16 persistent parameters during the optimizer step.

---