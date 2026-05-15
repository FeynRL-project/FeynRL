# Examples

Canonical training and evaluation configs, data, and plots for the experiments reported on [feynrl.github.io](https://feynrl.github.io).

## Structure

```
examples/
├── generate_plots.py          # unified plotting script (see Plotting below)
├── plot.sh                    # sample usage
├── math/                      # mathematical reasoning experiments
│   ├── README.md
│   ├── data/                  # MLflow-exported CSVs (reward vs wall-clock time)
│   ├── plots/                 # generated PNG reward curves
│   ├── qwen2.5-1.5b-instruct/ # training + eval configs
│   └── qwen3-4b-thinking-2507/
├── healthbench/               # HealthBench rubric-graded medical-question experiments
│   ├── README.md
│   └── qwen2.5-1.5b-instruct/
└── rar_science/               # RAR-b Science rubric-graded experiments
    ├── README.md
    └── qwen2.5-1.5b-instruct/
```

## Experiments

### [Math](math/README.md)

GRPO training on mathematical reasoning benchmarks (GSM8K, DeepScaler). Evaluates reward vs wall-clock time for sync and async FeynRL modes.

| Experiment | Model | Dataset |
| --- | --- | --- |
| [Qwen2.5-1.5B-Instruct](math/README.md#qwen25-15b-instruct) | Qwen/Qwen2.5-1.5B-Instruct | GSM8K |
| [Qwen3-4B-Thinking-2507](math/README.md#qwen3-4b-thinking-2507) | Qwen/Qwen3-4B-Thinking-2507 | DeepScaler |

### [HealthBench](healthbench/README.md)

GRPO and P3O training on [HealthBench](https://openai.com/index/healthbench/), a rubric-graded medical-question benchmark. Reward is provided by an LLM judge.

| Experiment | Model | Algorithm |
| --- | --- | --- |
| [Qwen2.5-1.5B-Instruct](healthbench/README.md) | Qwen/Qwen2.5-1.5B-Instruct | GRPO, P3O |

### [RAR-b Science](rar_science/README.md)

GRPO and P3O training on RAR-b Science, a rubric-graded science-question benchmark. Reward is provided by an LLM judge.

| Experiment | Model | Algorithm |
| --- | --- | --- |
| [Qwen2.5-1.5B-Instruct](rar_science/README.md) | Qwen/Qwen2.5-1.5B-Instruct | GRPO, P3O |

## Plotting

`generate_plots.py` generates reward curves for all registered experiments.

```bash
# All experiments
python generate_plots.py

# Specific experiment(s)
python generate_plots.py --experiment math_qwen3_4b
python generate_plots.py --experiment math_qwen3_4b --experiment math_qwen2_5_1b

# Override smoothing and time window
python generate_plots.py --smooth 10 --max-hours 4

# Write PNGs to a custom directory
python generate_plots.py --output-dir /tmp/plots

# List available experiment keys
python generate_plots.py --list
```

To add a new experiment: drop its CSV into the appropriate `data/` subfolder, add an entry to `EXPERIMENTS` in `generate_plots.py`, then run the script.

## Reproducing Training

All experiments share the same entry point:

```bash
python main_rl.py --config examples/<experiment>/<model>/train_<variant>.yaml
```

See each experiment's README for model-specific settings, data preparation steps, and evaluation instructions.
