# Configuration Reference

All experiments are configured via YAML files validated by Pydantic schemas in `load.py`. Template configs are provided for each experiment type.

## Experiment Types
1. **Reinforcement Learning (RL)**: `rl_args.yaml` — supports SGRPO, CISPO, PPO
2. **Supervised Learning (SL)**: `sl_args.yaml` — Supervised Fine-Tuning (SFT)
3. **Contrastive Learning (CL)**: `cl_args.yaml` — Direct Preference Optimization (DPO)
4. **Evaluation**: `eval_args.yaml` — inference and scoring

---

## `run` — Experiment Settings

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `experiment_id` | Unique identifier for the run | String | `"rl_exp_01"`, `"sft_v2"` |
| `seed` | Random seed to help with reproducibility | Integer ≥ 0 | `42`, `1337` |
| `project_name` | Project name in experiment tracker | String | `"FeynRL"`, `"my-project"` |
| `logger_type` | Tracker type | `"mlflow"` \| `"wandb"` | `"mlflow"` |
| `tracking_uri` | URI for the tracking server | String (URL) - only for mlflow | `"http://mlflow:8080/"` |
| `method` | Set automatically by entry point | `"rl"` \| `"sl"` \| `"cl"` \| `"eval"` | `"rl"` |
| `checkpoint_dir` | Directory for saving checkpoints | Path string | `"./ckps"`, `"/data/ckps"` |

#### RL-Specific Run Settings

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `training_gpus` | GPUs for DeepSpeed training engines | Integer ≥ 1 | `1`, `3`, `8` |
| `rollout_gpus` | GPUs for vLLM rollout engines (also used in eval) | Integer ≥ 1 | `2`, `4`, `7` |
| `ray_address` | Ray cluster address |`"auto"` if multi-node, `null` if single-node | `"auto"`  |
| `ray_master_port` | Port for torch distributed rendezvous | Integer \| `null` | `29500` |
| `overlap_enabled` | Enable async overlap of rollout and training |  `True`, `False` | `True` |
| `overlap_max_lag` | Max policy versions rollout can lag behind | Integer ≥ 1 | `1`, `2`, `3` |
| `weight_sync_method` | Weight sync method (default: `"direct"`) | `"direct"` \| `"disk"` | `"direct"` |
| `checkpoint_save_interval` | Save checkpoint every N epochs; 0 = end only (default: `1`) | Integer ≥ 0 | `1`, `5`, `0` |

#### NCCL / Multi-Node

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `nccl_socket_ifname` | Network interface for inter-node traffic | String \| `null` | `"eth0"`, `"bond0"` |
| `nccl_ib_hca` | InfiniBand HCA device | String \| `null` | `"mlx5_0"` |

#### Timeouts (seconds)

| Parameter | Description | Type | Examples |
|:---|:---|:---|:---|
| `init_timeout` | Training engine initialization | Seconds (Integer ≥ 0) | `3600`  |
| `rollout_timeout` | Rollout generation per batch | Seconds (Integer ≥ 0) | `7200` |
| `train_step_timeout` | Single training step | Seconds (Integer ≥ 0) | `1200` |
| `save_timeout` | Checkpoint save |  Seconds (Integer ≥ 0) | `1800` |
| `sync_timeout` | Weight sync operations | Seconds (Integer ≥ 0) | `1800` |

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
| `alg_name` | Algorithm name | RL: `"sgrpo"` \| `"cispo"` \| `"ppo"`, SL: `"sft"`, CL: `"dpo"` | `"sgrpo"` |
| `total_number_of_epochs` | Total training epochs | Integer ≥ 1 | `30`, `100` |
| `train_steps_per_epoch` | RL: optimizer steps per epoch | Integer ≥ 1 | `5`, `10` |
| `micro_batches_per_epoch` | SL/CL: micro-batch iterations per epoch | Integer ≥ 1 | `1000` |
| `train_batch_size_per_gpu` | Micro-batch size per GPU | Integer ≥ 1 | `2`, `4` |
| `gradient_accumulation_steps` | Gradient accumulation steps | Integer ≥ 1 | `1`, `4` |
| `val_batch_size_per_gpu` | Validation batch size per GPU | Integer ≥ 1 | `16` |
| `dynamic_ratio_every_step` | Recalculate dataset mix ratios every step | Boolean | `False` |
| `normalize_loss` | Normalize loss by token count | Boolean | `True` |

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
| `value_lr` | Value model LR | Float \| `null` | `1e-5` |
| `value_weight_decay` | Value model weight decay | Float \| `null` | `0.01` |
| `value_clip_grad_norm` | Value model gradient clipping | Float \| `null` | `1.0` |

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

---

## `reward` — Reward Configuration (RL and Eval)

| Parameter | Description | Type / Constraint | Examples |
|:---|:---|:---|:---|
| `reward_func` | Reward function name in `rewards/` module | String | `"gsm8k_reward_func"` |
| `broadcast` | Broadcast scalar reward across response tokens | Boolean | `false` |
| `eps_reward_norm` | Epsilon for reward normalization | Float > 0 | `1e-8` |

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

---