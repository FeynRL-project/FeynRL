# Examples

A curated set of FeynRL experiments organized by model type, algorithm, and dataset. Each experiment includes training configs, evaluation configs, and reproduction instructions.

> **Note:** Commands below pass `--experiment_id EXPNAME` — replace `EXPNAME` with your own experiment name/ID. It's used to name the output directory for logs, checkpoints, and metrics.

## Directory Structure

```text
examples/
├── llm/                                              # Text-only language models
│   ├── sft/
│   │   └── gsm8k/
│   │       └── gemma-2-2b-it/                       # SFT on GSM8K
│   └── rl/
│       └── gsm8k/
│           ├── qwen2.5-1.5b-instruct/               # GRPO on GSM8K
│           └── qwen3-4b-thinking-2507/              # GRPO on DeepScaler
├── vlm/                                             # Visual language models (image + text)
│   ├── sft/
│   │   ├── mm_math/
│   │   │   └── qwen2.5-vl-3b-instruct/             # SFT on MM-Math
│   │   └── snake/
│   │       ├── snake.py                             # Snake environment and BFS oracle
│   │       └── smolvlm2-500m-video-instruct/        # Snake game SFT
│   └── rl/
│       └── mm_math/
│           └── qwen2.5-vl-3b-instruct/             # GRPO on MM-Math
└── README.md
```

---

## LLM — Mathematical Reasoning

Text-only models fine-tuned on math reasoning datasets with SFT and GRPO. See [`llm/README.md`](llm/README.md) for full details and reproduction instructions.

### SFT

| Model         | Dataset | GSM8K pass@1 (base → FeynRL) |
| ------------- | ------- | ------------------------------: |
| Gemma-2-2B-it | GSM8K   |            21.81% → **32.59%** |

**Quick start:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main_sl.py --config examples/llm/sft/gsm8k/gemma-2-2b-it/train.yaml --experiment_id EXPNAME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_eval.py --config examples/llm/sft/gsm8k/gemma-2-2b-it/eval.yaml --experiment_id EXPNAME
```

### RL (GRPO)

| Model                   | Dataset    | Avg pass@1 (base → FeynRL) | Avg pass@16 (base → FeynRL) |
| ----------------------- | ---------- | -------------------------: | --------------------------: |
| Qwen2.5-1.5B-Instruct   | GSM8K      |          12.0% → **12.2%** |           26.4% → **27.0%** |
| Qwen3-4B-Thinking-2507  | DeepScaler |          12.2% → **27.0%** |           19.7% → **40.2%** |

**Quick start:**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/llm/rl/gsm8k/qwen2.5-1.5b-instruct/train_sync.yaml --experiment_id EXPNAME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/llm/eval/run_shared_eval.sh --model ./ckps/qwen2.5-1.5b-instruct/checkpoint --experiment_id EXPNAME
```

---

## VLM — Multimodal Tasks

Vision-language models fine-tuned on math reasoning and game control. See [`vlm/README.md`](vlm/README.md) for full details and reproduction instructions.

### MM-Math (Qwen2.5-VL-3B-Instruct)

| Method | MM-Math | Geometry3K | MathVista |
| ------ | ------: | ---------: | --------: |
| Base   |   23.0% |      28.5% |     52.0% |
| SFT    |   18.0% |      27.5% |     60.6% |
| GRPO   | **34.0%** | **34.1%** | **62.0%** |

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_rl.py --config examples/vlm/rl/mm_math/qwen2.5-vl-3b-instruct/train.yaml --experiment_id EXPNAME
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_eval.py --config examples/vlm/rl/mm_math/qwen2.5-vl-3b-instruct/eval.yaml --experiment_id EXPNAME
```

### Snake Game (SmolVLM2-500M-Video-Instruct)

A 500 M VLM learns to play Snake from pixel observations via supervised imitation of a BFS oracle.

| Method | Avg Score | Avg Steps |
| ------ | --------: | --------: |
| SFT    |      31.8 |     296.6 |

```bash
python examples/vlm/sft/snake/prepare_data.py \
    --output_dir ./data/vla-games/snake_sft
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main_sl.py --config examples/vlm/sft/snake/smolvlm2-500m-video-instruct/train.yaml --experiment_id EXPNAME
```

See the [experiment README](vlm/sft/snake/README.md) for the training progression, rollout instructions, and the interactive UI.

---

## Reproducing Any Experiment

1. **Prepare data** — run the dataset-specific prep script or download a pre-built parquet
2. **Train** — `python main_rl.py --config <train_config>` (RL) or `python main_sl.py --config <train_config>` (SFT)
3. **Evaluate** — `python main_eval.py --config <eval_config>` (math benchmarks) or the rollout script (game tasks)

Update `model.name`, `run.checkpoint_dir`, and `data.*_files_path` in the configs to point at your local paths before running.
