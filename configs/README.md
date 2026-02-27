# Configuration Files Guide

This directory contains the configuration files for training and evaluating models with FeynRL. All experiments are configured via YAML files that are parsed and validated using Pydantic schemas defined in `configs/load.py`.

## Core Principle
**No default values are set within the workspace.** Every parameter required for an experiment must be explicitly defined in the configuration file. The configuration files provided here serve as complete templates for each experiment type.

## Experiment Types
1. **Reinforcement Learning (RL)**: Configured via `rl_args.yaml`. Supports algorithms like GRPO, PPO, and CISPO.
2. **Supervised Learning (SL)**: Configured via `sl_args.yaml`. Primarily used for Supervised Fine-Tuning (SFT).
3. **Continual Learning (CL)**: Configured via `cl_args.yaml`. Primarily used for Direct Preference Optimization (DPO).
4. **Evaluation**: Configured via `eval_args.yaml`. Used to run inference and score model outputs.

---

## Configuration Parameter Tables

### 1. General Configuration
These parameters are consistent across all experiment types and are used for experiment tracking, model initialization, and resource management.

| Parameter                | Section | Description                                          | Type / Constraint                                   | Examples                       |
| :----------------------- | :------ | :--------------------------------------------------- | :-------------------------------------------------- | :----------------------------- |
| `experiment_id`          | `run`   | Unique string identifier for the experiment run      | String                                              | `"rl_exp_01"`, `"sft_v2"`      |
| `seed`                   | `run`   | Random seed                                          | Integer ≥ 0                                         | `42`, `1337`                   |
| `project_name`           | `run`   | Name of the project in the experiment tracker        | String                                              | `"FeynRL"`, `"Research-Alpha"` |
| `logger_type`            | `run`   | Type of experiment tracker to use                    | Allowed values: `"mlflow"`, `"wandb"`               | `"mlflow"`                     |
| `tracking_uri`           | `run`   | URI/URL for the tracking server                      | String (valid URL)                                  | `"http://mlflow-server:8080/"` |
| `checkpoint_dir`         | `run`   | Local directory for saving model checkpoints         | Absolute path                                       | `"/path/to/ckps"`              |
| `optimizer_name`         | `train` | Base optimizer architecture                          | Allowed values: `"adamw"`, `"adam"`                 | `"adamw"`                      |
| `lr`                     | `train` | Learning rate for the main model                     | Float > 0                                           | `1e-5`, `5e-7`                 |
| `lr_scheduler`           | `train` | Learning rate scheduler type                         | Allowed values: `"WarmupCosineLR"`                  | `"WarmupCosineLR"`             |
| `total_number_of_epochs` | `train` | Total training epochs                                | Integer ≥ 1                                         | `30`                           |
| `name`                   | `model` | Base model identifier (HuggingFace ID or local path) | String                                              | `"google/gemma-3-1b-it"`       |
| `dtype`                  | `model` | Model precision datatype                             | Allowed values: `"bfloat16"`, `"float16"`, `"fp32"` | `"bfloat16"`                   |
| `trust_remote_code`      | `model` | Allow remote code execution from HuggingFace         | Boolean                                             | `true`, `false`                |
| `use_peft`               | `peft`  | Enable Parameter-Efficient Fine-Tuning (LoRA)        | Boolean                                             | `true`, `false`                |
| `train_files_path`       | `data`  | List of paths to training Parquet files              | List of strings                                     | `["/data/train_1.parquet"]`    |
| `max_seq_len`            | `data`  | Maximum tokens for input sequences                   | Integer > 0                                         | `1024`, `2048`, `4096`         |

### 2. RL-Specific Parameters
| Parameter                | Section   | Description                                      | Type / Constraint                             | Examples                 |
| :----------------------- | :-------- | :----------------------------------------------- | :-------------------------------------------- | :----------------------- |
| `training_gpus`          | `run`     | Number of GPUs for model optimization            | Integer ≥ 1                                   | `1`, `2`, `3`            |
| `rollout_gpus`           | `run`     | Number of GPUs for sampling/rollout generation   | Integer ≥ 1                                   | `1`, `2`, `3`, `4`,'7'   |
| `weight_sync_method`     | `run`     | Method to sync weights between train and rollout | Allowed values: `"direct"`, `"disk"`          | `"direct"`               |
| `alg_name`               | `train`   | Reinforcement learning algorithm                 | Allowed values: `"sgrpo"`, `"cispo"`, `"ppo"` | `"ppo"`                  |
| `train_steps_per_epoch`  | `train`   | Optimizer steps to perform per epoch             | Integer ≥ 1                                   | `5`                      |
| `kl_coeff`               | `train`   | Weight of KL divergence penalty                  | Float ≥ 0                                     | `0.001`                  |
| `clip_low` / `clip_high` | `train`   | PPO/GRPO policy clipping bounds                  | Float ≥ 0                                     | `0.2`                    |
| `reward_func`            | `reward`  | Reward function name in `rewards/` module        | String                                        | `"gsm8k_reward_func"`    |
| `n_samples`              | `rollout` | Number of completions per prompt                 | Integer ≥ 1                                   | `4`, `8`, `16`           |
| `max_tokens`             | `rollout` | Max completion tokens per rollout                | Integer ≥ 1                                   | `512`, `1024`            |
| `tensor_parallel_size`   | `rollout` | Model partitioning for vLLM rollout workers      | Integer ≥ 1 and ≤ `rollout_gpus`              | `2`                      |
| `ref_model`              | `model`   | Path to frozen reference model                   | String (HF ID or local path)                  | `"google/gemma-3-1b-it"` |
| `value_model`            | `model`   | Path to value model (only for PPO)               | String (HF ID or local path)                  | `"google/gemma-3-1b-it"` |

### 3. SL & CL Specific Parameters
| Parameter                 | Section | Description                                  | Type / Constraint            | Examples                 |
| :------------------------ | :------ | :------------------------------------------- | :--------------------------- | :----------------------- |
| `micro_batches_per_epoch` | `train` | Total micro-batch iterations per epoch       | Integer ≥ 1                  | `1000`                   |
| `cl_beta`                 | `train` | Beta parameter for DPO objective             | Float ≥ 0                    | `0.1`                    |
| `ref_model`               | `model` | Reference model for DPO (Continual Learning) | String (HF ID or local path) | `"google/gemma-3-1b-it"` |

---

## Dynamic Placeholders
The configuration files use the following paths which **must be replaced** with your specific workspace paths before running:

- `tracking_uri`: Your MLflow or WandB tracking server address.
- `checkpoint_dir`: Absolute path to where you want to store model weights.
- `train_files_path` / `val_files_path`: Absolute paths to your data in `.parquet` format.
- `name` / `ref_model` / `value_model`: Paths to base models.

## Training Hardware Efficiency (ZeRO)
All training configs default to **Deepspeed ZeRO Stage 3** with optional CPU offloading.
- Set `offload_optimizer.device` to `"cpu"` if you experience OOM on GPU.
- Adjust `train_batch_size_per_gpu` and `gradient_accumulation_steps` to fit your hardware.