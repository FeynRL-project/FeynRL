# LLM Experiments

Text-only language model experiments using FeynRL's SFT and RL (GRPO) pipelines on mathematical reasoning datasets.

> **Note:** Commands below pass `--experiment_id EXPNAME` — replace `EXPNAME` with your own experiment name/ID. It's used to name the output directory for logs, checkpoints, and metrics.

## Directory Layout

```text
llm/
├── eval/
│   ├── eval_shared_base.yaml           # Base config for the Shared Evaluation Protocol
│   └── run_shared_eval.sh              # Runs the Shared Evaluation Protocol across all 10 benchmarks
├── sft/
│   └── gsm8k/
│       └── gemma-2-2b-it/              # SFT on GSM8K with Gemma-2-2B-it
├── rl/
│   └── gsm8k/
│       ├── qwen2.5-1.5b-instruct/      # GRPO on GSM8K with Qwen2.5-1.5B-Instruct
│       └── qwen3-4b-thinking-2507/     # GRPO on DeepScaler with Qwen3-4B-Thinking-2507
└── README.md
```

---

## Shared Evaluation Protocol

Downstream evaluation reports pass@1 and pass@16 across 10 mathematical reasoning benchmarks using `n=16` samples per prompt and temperature `1.0`. Used by the RL experiments below.

| Benchmark     | Dataset Card                                                                           | Benchmark     | Dataset Card                                                                           |
| ------------- | -------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------- |
| GSM8K         | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                           | AIME 2024     | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)     |
| AIME 2025     | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)             | AIME 2026     | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026)             |
| AMC           | [rawsh/2024_AMC12](https://huggingface.co/datasets/rawsh/2024_AMC12)                   | AMO           | [meituan-longcat/AMO-Bench](https://huggingface.co/datasets/meituan-longcat/AMO-Bench) |
| Brumo         | [MathArena/brumo_2025](https://huggingface.co/datasets/MathArena/brumo_2025)           | HMMT February | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025)     |
| HMMT November | [MathArena/hmmt_nov_2025](https://huggingface.co/datasets/MathArena/hmmt_nov_2025)     | Olympiad      | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench)           |

### Data Preparation

```bash
python data_prep/shared_eval_benchmarks.py --local_dir ./data
```

Downloads each of the 10 benchmarks above directly from its HuggingFace dataset card and packs it into a separate `./data/{benchmark}_test.parquet` file. Pass `--variant wsp` to prepend a shared system prompt instead of the default no-system-prompt (`ns`) variant, or `--benchmarks gsm8k aime_2024 ...` to pack a subset instead of all 10.

### Running the Protocol

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/llm/eval/run_shared_eval.sh --model <path_or_hf_id> --experiment_id EXPNAME
```

Runs `main_eval.py` once per benchmark in sequence against [`eval/eval_shared_base.yaml`](eval/eval_shared_base.yaml) — a single base config (n_samples=16, temperature=1.0, matching the protocol above) shared across all 10 runs, with only the model, test file, and checkpoint dir swapped per benchmark. Each benchmark's `rollout_stats.json` (containing pass@1..pass@16) lands under `./ckps/eval/EXPNAME/{benchmark}/`. Requires `./data/{benchmark}_test.parquet` for each benchmark (see Data Preparation above).

A slurm equivalent is available at [`scripts/slurm/launch_shared_eval.sh`](../../scripts/slurm/launch_shared_eval.sh): `sbatch scripts/slurm/launch_shared_eval.sh --model <path_or_hf_id> --experiment_id EXPNAME`.

---

## Data Preparation

### GSM8K

```bash
python data_prep/gsm8k.py --local_dir ./data --system_prompt ""
```

Downloads [GSM8K](https://huggingface.co/datasets/openai/gsm8k) and writes `gsm8k_processed_{run_id}_ns_{train,val,test}.parquet` under `./data/`. Used by both the SFT and RL GSM8K experiments below — point each config's `data.train_files_path` / `data.val_files_path` / `data.test_files_path` at the matching file. Also rename the `data.train_ratios` key in the training config to match the new train file's basename exactly (e.g. `gsm8k_processed_{run_id}_ns_train`), or startup fails with `Dataset/ratio key mismatch`.

### DeepScaleR

```bash
python data_prep/deepscaler.py --local_dir ./data --system_prompt ""
```

Downloads [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) (single `train` split, no built-in val/test) and writes `deepscaler_preview_processed_{run_id}_ns_{train,val,test}.parquet` under `./data/`, splitting 80/10/10 via `--val_ratio`/`--test_ratio`. Used by the Qwen3-4B-Thinking-2507 RL experiment below — same `data.train_files_path` / `data.val_files_path` / `data.train_ratios` conventions as GSM8K above.

---

## Gemma-2-2B-it — GSM8K (SFT)

| Item              | Value                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------- |
| Model              | `google/gemma-2-2b-it`                                                                  |
| Training dataset   | [GSM8K](https://huggingface.co/datasets/openai/gsm8k)                                  |
| Algorithm          | SFT (supervised fine-tuning)                                                             |
| DeepSpeed          | ZeRO stage 3, bf16                                                                       |
| Training config    | [`sft/gsm8k/gemma-2-2b-it/train.yaml`](sft/gsm8k/gemma-2-2b-it/train.yaml)             |
| Evaluation config  | [`sft/gsm8k/gemma-2-2b-it/eval.yaml`](sft/gsm8k/gemma-2-2b-it/eval.yaml)               |

Data: prepared via [`data_prep/gsm8k.py`](#gsm8k) above.

### Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main_sl.py --config examples/llm/sft/gsm8k/gemma-2-2b-it/train.yaml --experiment_id EXPNAME
```

![FeynRL loss curve](sft/gsm8k/gemma-2-2b-it/feynrl_loss_curve.png)

### Evaluation Results

Evaluated on the GSM8K test set with `n_samples=8`, temperature `1.0`.

| Model  | GSM8K pass@1 |
| ------ | -----------: |
| Base   |       21.81% |
| FeynRL |   **32.59%** |

SFT improves pass@1 by **+10.78 pp** over the base model.

### Key Training Settings

| Parameter             | Value                       |
| ---------------------- | --------------------------- |
| Model                  | google/gemma-2-2b-it        |
| Dataset                | GSM8K                       |
| Learning rate          | 1e-5                        |
| LR scheduler           | WarmupCosineLR (10% warmup) |
| Train batch per GPU    | 1                           |
| Gradient accumulation  | 16                          |
| Micro batches / epoch  | 416                         |
| Max sequence length    | 4096                        |
| DeepSpeed              | ZeRO stage 3, bf16          |
| LoRA                   | disabled (full fine-tune)   |
| Total epochs           | 2                           |

### Reproducing Evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_eval.py --config examples/llm/sft/gsm8k/gemma-2-2b-it/eval.yaml --experiment_id EXPNAME
```

Replace `model.name` with your checkpoint path and `data.test_files_path` with your target benchmark parquet.

---

## RL Shared Setup

The following applies to the RL (GRPO) experiments below.

- **Algorithm:** GRPO
- **DeepSpeed:** ZeRO stage 2/3, bf16
- **Hardware:** 8×H100 GPUs with CUDA v12.4
- **Training:** `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/llm/rl/<dataset>/<model>/train_sync.yaml --experiment_id EXPNAME`
- **Evaluation:** see [Shared Evaluation Protocol](#shared-evaluation-protocol) above — all RL experiments below are evaluated through it rather than a per-experiment eval config.

---

## Qwen2.5-1.5B-Instruct — GSM8K

| Item                  | Value                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| Model                 | `Qwen/Qwen2.5-1.5B-Instruct`                                                                          |
| Training dataset      | [GSM8K](https://huggingface.co/datasets/openai/gsm8k)                                                 |
| GPU split             | 6 training GPUs / 2 rollout GPUs                                                                       |
| Sync training config  | [`rl/gsm8k/qwen2.5-1.5b-instruct/train_sync.yaml`](rl/gsm8k/qwen2.5-1.5b-instruct/train_sync.yaml)   |
| Async training config | [`rl/gsm8k/qwen2.5-1.5b-instruct/train_async.yaml`](rl/gsm8k/qwen2.5-1.5b-instruct/train_async.yaml) |

Data: prepared via [`data_prep/gsm8k.py`](#gsm8k) above (uses `data.train_files_path` / `data.val_files_path` only).

### Training

```bash
# Synchronous (no overlap)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/llm/rl/gsm8k/qwen2.5-1.5b-instruct/train_sync.yaml --experiment_id EXPNAME

# Asynchronous (with overlap)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/llm/rl/gsm8k/qwen2.5-1.5b-instruct/train_async.yaml --experiment_id EXPNAME
```

The reward curves below overlay the sync and async runs over the first hour of wall-clock training time.

![FeynRL reward curve](rl/gsm8k/qwen2.5-1.5b-instruct/feynrl_reward_curve.png)

At 1 hour, the sync run reaches **0.894** reward and the async run reaches **0.858**.

### Evaluation Results

| Model  | Avg pass@1 | Avg pass@16 |
| ------ | ---------: | ----------: |
| Base   |      12.0% |       26.4% |
| FeynRL |      12.2% |       27.0% |

### Key Training Settings

| Parameter              | Value                                                        |
| ---------------------- | ------------------------------------------------------------ |
| Model                  | Qwen/Qwen2.5-1.5B-Instruct                                   |
| Dataset                | GSM8K                                                        |
| GPU split              | 6 training / 2 rollout                                       |
| Weight sync            | direct                                                       |
| Overlap                | disabled in `train_sync.yaml`, enabled in `train_async.yaml` |
| Learning rate          | 1e-5                                                         |
| LR scheduler           | WarmupCosineLR (10% warmup)                                  |
| KL coefficient         | 0.0                                                          |
| Clip (low / high)      | 0.4 / 0.4                                                    |
| Train batch per GPU    | 8                                                            |
| Gradient accumulation  | 1                                                            |
| Rollout samples/prompt | 4                                                            |
| Rollout samples/epoch  | 512                                                          |
| Rollout max tokens     | 1024                                                         |
| Context length         | 1024                                                         |
| Total epochs           | 100                                                          |

---

## Qwen3-4B-Thinking-2507 — DeepScaler

| Item                    | Value                                                                                                      |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| Model                   | `Qwen/Qwen3-4B-Thinking-2507`                                                                              |
| Training dataset        | [DeepScaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)                      |
| GPU split               | 4 training GPUs / 4 rollout GPUs                                                                           |
| Sync training config    | [`rl/gsm8k/qwen3-4b-thinking-2507/train_sync.yaml`](rl/gsm8k/qwen3-4b-thinking-2507/train_sync.yaml)     |
| Async training config   | [`rl/gsm8k/qwen3-4b-thinking-2507/train_async.yaml`](rl/gsm8k/qwen3-4b-thinking-2507/train_async.yaml)   |

Data: prepared via [`data_prep/deepscaler.py`](#deepscaler) above.

### Training

```bash
# Synchronous (no overlap)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/llm/rl/gsm8k/qwen3-4b-thinking-2507/train_sync.yaml --experiment_id EXPNAME

# Asynchronous (with overlap)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/llm/rl/gsm8k/qwen3-4b-thinking-2507/train_async.yaml --experiment_id EXPNAME
```

The reward curves below overlay the sync and async runs over the first 8 hours of wall-clock training time.

![FeynRL Qwen3 reward curve](rl/gsm8k/qwen3-4b-thinking-2507/feynrl_reward_curve_qwen3.png)

At 8 hours, the sync run is at **0.526** reward and the async run is at **0.584**.

### Evaluation Results

The trained checkpoint (`iter000075`) was evaluated using the shared protocol (with-system-prompt variant):

| Model  | Avg pass@1 | Avg pass@16 |
| ------ | ---------: | ----------: |
| Base   |      12.2% |       19.7% |
| FeynRL |      27.0% |       40.2% |

FeynRL improves average pass@1 by **+12.9 pp** and pass@16 by **+17.1 pp** over the base model.

### Key Training Settings

| Parameter              | Value                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------- |
| Model                  | Qwen/Qwen3-4B-Thinking-2507                                                           |
| Dataset                | DeepScaler                                                                            |
| GPU split              | 4 training / 4 rollout                                                                |
| Weight sync            | direct (sync) / NCCL (async)                                                          |
| Overlap                | disabled in `train_sync.yaml`, enabled in `train_async.yaml`                          |
| Learning rate          | 1e-5                                                                                  |
| LR scheduler           | WarmupCosineLR (10% warmup)                                                           |
| KL coefficient         | 0.0                                                                                   |
| Clip (low / high)      | 0.4 / 0.4                                                                             |
| Train batch per GPU    | 4                                                                                     |
| Gradient accumulation  | 2                                                                                     |
| Rollout samples/prompt | 4                                                                                     |
| Rollout samples/epoch  | 256                                                                                   |
| Rollout max tokens     | 2048                                                                                  |
| Context length         | 4096                                                                                  |
| Total epochs           | 500                                                                                   |
