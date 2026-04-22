# FeynRL — Results

This document summarizes the main training and evaluation results for FeynRL on mathematical reasoning tasks, together with the canonical configs used to reproduce them.

## Shared Setup

The experiments below share the same high-level training and evaluation setup:

- **Algorithm:** GRPO
- **DeepSpeed:** ZeRO stage 3, bf16
- **Tracking:** MLflow
- **Training reproduction:** `python main_rl.py --config examples/<experiment>/<train_config>.yaml`
- **Evaluation reproduction:** `python main_eval.py --config examples/<experiment>/eval.yaml`

GPU allocation differs by experiment and is listed in the corresponding section below.

### Shared Evaluation Protocol

Unless noted otherwise, downstream evaluation:

- covers 10 mathematical reasoning benchmarks
- reports pass@1 and pass@16
- uses `n=16` samples per prompt
- uses temperature `1.0`
- averages use each framework's available benchmarks
- gains vs. `base` are computed on the benchmarks shared with `base`

| Benchmark     | Dataset Card                                                                           | Benchmark     | Dataset Card                                                                           |
| ------------- | -------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------- |
| GSM8K         | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)                           | AIME 2024     | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)     |
| AIME 2025     | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)             | AIME 2026     | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026)             |
| AMC           | [AI-MO/aimo-validation-amc](https://huggingface.co/datasets/AI-MO/aimo-validation-amc) | AMO           | [meituan-longcat/AMO-Bench](https://huggingface.co/datasets/meituan-longcat/AMO-Bench) |
| Brumo         | [MathArena/brumo_2025](https://huggingface.co/datasets/MathArena/brumo_2025)           | HMMT February | [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025)     |
| HMMT November | [MathArena/hmmt_nov_2025](https://huggingface.co/datasets/MathArena/hmmt_nov_2025)     | Olympiad      | [Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench)           |

For reproduction, use the single canonical `examples/<experiment>/eval.yaml` for each experiment:

- substitute your own checkpoint, output, tracking, and cache paths as needed
- set the `{benchmark}` placeholder in fields like `run.experiment_id`, `run.checkpoint_dir`, and `data.test_files_path`
- point `data.test_files_path` at the specific benchmark dataset you want to evaluate
- if your setup differs, adjust any benchmark-specific prompt formatting or preprocessing to match your model

---

## Qwen2.5-1.5B-Instruct

| Item                  | Value                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| Model                 | `Qwen/Qwen2.5-1.5B-Instruct`                                                                         |
| Training dataset      | [GSM8K](https://huggingface.co/datasets/openai/gsm8k)                                                |
| GPU split             | 6 training GPUs / 2 rollout GPUs                                                                     |
| Sync training config  | [`examples/qwen2.5-1.5b-instruct/train_sync.yaml`](examples/qwen2.5-1.5b-instruct/train_sync.yaml)   |
| Async training config | [`examples/qwen2.5-1.5b-instruct/train_async.yaml`](examples/qwen2.5-1.5b-instruct/train_async.yaml) |
| Evaluation config     | [`examples/qwen2.5-1.5b-instruct/eval.yaml`](examples/qwen2.5-1.5b-instruct/eval.yaml)               |

### Training

The reward curve below overlays the dedicated sync and async FeynRL runs, using `rollout/avg_reward` over the first hour of wall-clock training time.

![FeynRL reward curve](examples/feynrl_reward_curve.png)

At 1 hour, the sync run reaches **0.894** reward and the async run reaches **0.858**.

To prepare the GSM8K parquet files for this experiment with the repo's `data_prep` scripts:

```bash
python data_prep/gsm8k.py --local_dir ./data --run_id 123245 --system_prompt ""
```

This produces `./data/gsm8k_processed_123245_ns_train.parquet` and `./data/gsm8k_processed_123245_ns_val.parquet`, so update `data.train_files_path` and `data.val_files_path` in the training config if you use these generated files directly.

To reproduce the dedicated sync and async runs:

```bash
python main_rl.py --config examples/qwen2.5-1.5b-instruct/train_sync.yaml
python main_rl.py --config examples/qwen2.5-1.5b-instruct/train_async.yaml
```

To regenerate the figure:

```bash
python examples/plot_reward_curve.py
```

### Downstream Evaluation

The trained checkpoint was evaluated using the shared protocol above.

Average results across the reported benchmarks:

| Model  | Avg pass@1 | Avg pass@16 |
| ------ | ---------: | ----------: |
| Base   |      12.0% |       26.4% |
| FeynRL |      12.2% |       27.0% |

Relative to the base model, this corresponds to a modest improvement of **+0.2 pp** at pass@1 and **+0.6 pp** at pass@16.

### Reproducing Evaluation

Use the canonical config [`examples/qwen2.5-1.5b-instruct/eval.yaml`](examples/qwen2.5-1.5b-instruct/eval.yaml). Update `model.name`, `run.checkpoint_dir`, and any tracking/output settings you need, then substitute `{benchmark}` in the templated fields and point `data.test_files_path` at the benchmark parquet you want to evaluate.

```bash
python main_eval.py --config examples/qwen2.5-1.5b-instruct/eval.yaml
```

### Key Training Settings

| Parameter              | Value                                                        |
| ---------------------- | ------------------------------------------------------------ |
| Model                  | Qwen/Qwen2.5-1.5B-Instruct                                   |
| Dataset                | [GSM8K](https://huggingface.co/datasets/openai/gsm8k)        |
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

## Qwen3-4B-Thinking-2507

| Item                    | Value                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| Model                   | `Qwen/Qwen3-4B-Thinking-2507`                                                                          |
| Training dataset        | [DeepScaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)                  |
| GPU split               | 4 training GPUs / 4 rollout GPUs                                                                       |
| Primary training config | [`examples/qwen3-4b-thinking-2507/train.yaml`](examples/qwen3-4b-thinking-2507/train.yaml)             |
| Sync training config    | [`examples/qwen3-4b-thinking-2507/train_sync.yaml`](examples/qwen3-4b-thinking-2507/train_sync.yaml)   |
| Async training config   | [`examples/qwen3-4b-thinking-2507/train_async.yaml`](examples/qwen3-4b-thinking-2507/train_async.yaml) |
| Evaluation config       | [`examples/qwen3-4b-thinking-2507/eval.yaml`](examples/qwen3-4b-thinking-2507/eval.yaml)               |

### Training

The reward curve below overlays the dedicated sync and async FeynRL runs, using `rollout/avg_reward` over the first 8 hours of wall-clock training time.

![FeynRL Qwen3 reward curve](examples/feynrl_reward_curve_qwen3.png)

At 8 hours, the sync run is at **0.526** reward and the async run is at **0.584**.

To reproduce the primary training run:

```bash
python main_rl.py --config examples/qwen3-4b-thinking-2507/train.yaml
```

To reproduce the dedicated sync and async comparison runs:

```bash
python main_rl.py --config examples/qwen3-4b-thinking-2507/train_sync.yaml
python main_rl.py --config examples/qwen3-4b-thinking-2507/train_async.yaml
```

To regenerate the figure:

```bash
python examples/plot_reward_curve_qwen3.py
```

### Downstream Evaluation

The trained checkpoint (`iter000075`) was evaluated using the shared protocol above. The reported benchmark data use the with-system-prompt (`wsp`) variant to match the model's instruction format.

Average results across the reported benchmarks:

| Model  | Avg pass@1 | Avg pass@16 |
| ------ | ---------: | ----------: |
| Base   |      12.2% |       19.7% |
| FeynRL |      27.0% |       40.2% |

On the benchmarks shared with `base`, FeynRL improves average pass@1 by **+12.9 pp** and pass@16 by **+17.1 pp** over the base model.

### Reproducing Evaluation

Use the canonical config [`examples/qwen3-4b-thinking-2507/eval.yaml`](examples/qwen3-4b-thinking-2507/eval.yaml). Update `model.name`, `run.checkpoint_dir`, and any tracking/output settings you need, then substitute `{benchmark}` in the templated fields and point `data.test_files_path` at the benchmark parquet you want to evaluate. Keep the benchmark data and prompt formatting aligned with the `wsp` evaluation setup used for the reported numbers.

```bash
python main_eval.py --config examples/qwen3-4b-thinking-2507/eval.yaml
```

### Primary Training Settings

These settings correspond to [`examples/qwen3-4b-thinking-2507/train.yaml`](examples/qwen3-4b-thinking-2507/train.yaml).

| Parameter              | Value                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------- |
| Model                  | Qwen/Qwen3-4B-Thinking-2507                                                           |
| Dataset                | [DeepScaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) |
| GPU split              | 4 training / 4 rollout                                                                |
| Weight sync            | NCCL                                                                                  |
| Overlap                | enabled                                                                               |
| Learning rate          | 1e-6                                                                                  |
| LR scheduler           | WarmupCosineLR (10% warmup)                                                           |
| KL coefficient         | 0.001                                                                                 |
| Clip (low / high)      | 0.2 / 0.2                                                                             |
| Training batch per GPU | 8                                                                                     |
| Gradient accumulation  | 4                                                                                     |
| Rollout samples/prompt | 8                                                                                     |
| Rollout samples/epoch  | 256                                                                                   |
| Rollout max tokens     | 2048                                                                                  |
| Context length         | 4069                                                                                  |
| Total epochs           | 100                                                                                   |
| Steps per epoch        | 4                                                                                     |

### Dedicated Sync/Async Comparison Settings

These settings correspond to [`examples/qwen3-4b-thinking-2507/train_sync.yaml`](examples/qwen3-4b-thinking-2507/train_sync.yaml) and [`examples/qwen3-4b-thinking-2507/train_async.yaml`](examples/qwen3-4b-thinking-2507/train_async.yaml).

| Parameter              | Value                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Model                  | Qwen/Qwen3-4B-Thinking-2507                                                                                              |
| Dataset variant        | [DeepScaler](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) with `wsp` train/val parquet files |
| GPU split              | 4 training / 4 rollout                                                                                                   |
| Weight sync            | `direct` in `train_sync.yaml`, `nccl` in `train_async.yaml`                                                              |
| Overlap                | disabled in `train_sync.yaml`, enabled in `train_async.yaml`                                                             |
| Learning rate          | 1e-5                                                                                                                     |
| LR scheduler           | WarmupCosineLR (10% warmup)                                                                                              |
| KL coefficient         | 0.0                                                                                                                      |
| Clip (low / high)      | 0.4 / 0.4                                                                                                                |
| Train batch per GPU    | 4                                                                                                                        |
| Gradient accumulation  | 2                                                                                                                        |
| Rollout samples/prompt | 4                                                                                                                        |
| Rollout samples/epoch  | 256                                                                                                                      |
| Rollout max tokens     | 2048                                                                                                                     |
| Context length         | 4096                                                                                                                     |
| Total epochs           | 500                                                                                                                      |
