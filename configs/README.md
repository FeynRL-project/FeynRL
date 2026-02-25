# Configuration Files Guide

This directory contains the configuration files for training and evaluating models with LeanRL. All experiments are configured via YAML files that are parsed and validated using Pydantic schemas defined in `configs/load.py`.

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

| Parameter | Section | Description | Allowable Inputs / Examples |
| :--- | :--- | :--- | :--- |
| `experiment_id` | `run` | Unique string identifier for the experiment run. | `"rl_exp_01"`, `"sft_v2"` |
| `seed` | `run` | Random seed for reproducibility. | `42`, `1337` |
| `project_name` | `run` | Name of the project in the experiment tracker. | `"LeanRL"`, `"Research-Alpha"` |
| `logger_type` | `run` | Type of experiment tracker to use. | `"mlflow"`, `"wandb"` |
| `tracking_uri` | `run` | URI/URL for the tracking server. | `"http://mlflow-server:5000/"` |
| `checkpoint_dir` | `run` | Local directory for saving model checkpoints. | Absolute path (e.g., `/path/to/ckps`) |
| `optimizer_name`| `train` | Base optimizer architecture. | `"adamw"`, `"adam"` |
| `lr` | `train` | Learning rate for the main model. | Float (e.g., `1e-5`, `5e-7`) |
| `lr_scheduler` | `train` | Learning rate scheduler type. | `"WarmupCosineLR"` |
| `total_number_of_epochs` | `train` | Total training epochs (defined by iteration count). | Integer (e.g., `30`) |
| `name` | `model` | Base model identifier (HuggingFace ID or local path). | `"google/gemma-3-1b-it"` |
| `dtype` | `model` | Model precision datatype. | `"bfloat16"`, `"float16"`, `"fp32"` |
| `trust_remote_code`| `model` | Whether to allow remote code execution from HF. | `true`, `false` |
| `use_peft` | `peft` | Enable Parameter-Efficient Fine-Tuning (LoRA). | `true`, `false` |
| `train_files_path`| `data` | List of paths to training Parquet files. | `["/data/train_1.parquet"]` |
| `max_seq_len` | `data` | Maximum tokens for input sequences. | `1024`, `2048`, `4096` |

### 2. RL-Specific Parameters
These parameters control the reinforcement learning loop, rollout generation, and multi-resource allocation.

| Parameter | Section | Description | Allowable Inputs / Examples |
| :--- | :--- | :--- | :--- |
| `training_gpus` | `run` | Number of GPUs for model optimization. | `1`, `2`, `4`, `8` |
| `rollout_gpus` | `run` | Number of GPUs for sampling/rollout generation. | `1`, `2`, `4`, `8` |
| `weight_sync_method`| `run` | Method to sync weights between train and rollout. | `"direct"` (GPU), `"disk"` |
| `alg_name` | `train` | Reinforcement learning algorithm. | `"sgrpo"`, `"cispo"`, `"ppo"` |
| `train_steps_per_epoch`| `train` | Optimizer steps to perform in each epoch. | Integer (e.g., `5`) |
| `kl_coeff` | `train` | Weight of the KL divergence penalty. | Float (e.g., `0.001`) |
| `clip_low`/`clip_high`| `train` | PPO/GRPO policy clipping bounds. | Float (e.g., `0.2`) |
| `reward_func` | `reward` | Name of the reward function in the `rewards/` module. | `"gsm8k_reward_func"` |
| `n_samples` | `rollout`| Number of completions per prompt. | `4`, `8`, `16` |
| `max_tokens` | `rollout`| Max completion tokens allowed per rollout. | `512`, `1024` |
| `tensor_parallel_size`| `rollout`| Model partitioning for vLLM rollout workers. | Integer (must be $\le$ `rollout_gpus`) |
| `ref_model` | `model` | Path to the frozen reference model. | String (HF ID or local path) |
| `value_model` | `model` | Path to the value model (required only for PPO). | String (HF ID or local path) |

### 3. SL & CL Specific Parameters
These parameters are specific to Supervised Learning (SFT) and preference-based optimization (DPO).

| Parameter | Section | Description | Allowable Inputs / Examples |
| :--- | :--- | :--- | :--- |
| `micro_batches_per_epoch` | `train` | Total micro-batch iterations per epoch. | Integer (e.g., `1000`) |
| `cl_beta` | `train` | Beta parameter for DPO objective. | Float (e.g., `0.1`) |
| `ref_model` | `model` | Reference model for DPO (Continual Learning). | String (HF ID or local path) |

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
