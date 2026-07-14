# Snake Game

![Snake training progression](snake_progression.gif)

_Left to right: base model (score 0.3) -> epoch 1 (score 4.4) -> epoch 50 (score 31.8). The final game reaches score 40 in 393 steps._

Teaching a vision-language model to play classic Snake from pixel observations via supervised imitation of a BFS oracle.

---

## Files

```text
snake/
├── snake_progression.gif                 # GitHub-renderable training progression
├── snake.py                              # Environment, renderer, and BFS oracle
├── prepare_data.py                       # Generate the oracle-imitation dataset
└── smolvlm2-500m-video-instruct/         # SFT experiment with SmolVLM2-500M-Video-Instruct
    ├── train.yaml                        # Training config
    ├── rollout.py                        # Evaluate a checkpoint by playing games
    ├── ui.py                             # Interactive browser UI
    └── results/                          # Pre-generated rollout videos
```

---

## The Environment (`snake.py`)

The game runs as a simple Python class with no gym dependency. `snake.py` contains the environment, the pixel renderer, and the BFS oracle.

**Layout:** 10x10 interior playfield surrounded by a 1-cell wall border, rendered at 28 px/cell for **336x336 px PNG** frames.

**Actions:** `UP`, `DOWN`, `LEFT`, `RIGHT`, and `NONE` to continue the current direction. The model predicts exactly one token per step.

**Oracle:** BFS finds the shortest path from the snake's head to food. When no safe food path exists, it falls back to a flood-fill survival heuristic that maximizes reachable empty cells.

---

## Data Preparation (`prepare_data.py`)

The dataset is generated entirely from oracle rollouts.

```bash
python examples/vlm/sft/snake/prepare_data.py \
    --output_dir         ./data/snake_sft \
    --num_train_episodes 4000 \
    --num_val_episodes   400  \
    --max_steps          500  \
    --seed               42
```

For each episode the oracle plays from start to death or `--max_steps`. Every frame-action pair becomes one parquet row:

| Column | Type | Description |
| ------ | ---- | ----------- |
| `prompt` | `list[dict]` | Single-turn chat message with the task instruction |
| `answer` | `str` | Oracle action: `UP`, `DOWN`, `LEFT`, `RIGHT`, or `NONE` |
| `image_bytes` | `bytes` | PNG-encoded 336x336 frame before the action |
| `episode_id` | `str` | Unique episode identifier, e.g. `ep_000042` |
| `frame_index` | `int` | Step number within the episode |
| `score` | `int` | Food eaten so far at this step |
| `snake_length` | `int` | Current snake length |
| `game_state` | `str` | JSON snapshot of body positions and food location |

Prompt used for every frame:

> *"You are playing the classic Snake game. Move the green snake to eat the red food and grow. Avoid hitting the gray walls or your own body or the game ends. Reply with exactly one word: UP, DOWN, LEFT, RIGHT, or NONE (to continue in your current direction)."*

Output layout:

```text
./data/snake_sft/
  train.parquet
  val.parquet
  dataset_info.json
```

At 4,000 training episodes the dataset is about 879k rows. The training config processes one tenth of this per epoch (`micro_batches_per_epoch: 5496`), which makes early learning visible after the first epoch.

---

## SmolVLM2-500M-Video-Instruct SFT

A 500M VLM learns to play Snake by predicting the next action from a single rendered frame.

### Training Progression

| Checkpoint | Avg Score | Avg Steps |
| ---------- | --------: | --------: |
| Base model (no training) | 0.3 | 6.0 |
| Epoch 1 | 4.4 | 46.6 |
| Epoch 50 (final) | **31.8** | **296.6** |

### Quickstart

Generate data:

```bash
python examples/vlm/sft/snake/prepare_data.py \
    --output_dir ./data/snake_sft \
    --num_train_episodes 4000 \
    --num_val_episodes 400
```

Train:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 main_sl.py --config examples/vlm/sft/snake/smolvlm2-500m-video-instruct/train.yaml
```

Evaluate:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/rollout.py \
    --checkpoint_dir ./ckps/snake-sft/smolvlm2-500m/iter000050 \
    --output_dir     ./rollouts/snake \
    --num_games      5 \
    --max_steps      500 \
    --fps            8
```

Interactive UI:

```bash
python examples/vlm/sft/snake/smolvlm2-500m-video-instruct/ui.py \
    --checkpoint_dir ./ckps/snake-sft/smolvlm2-500m/iter000050
```

Then open `http://localhost:5002`. Arrow keys or WASD steer, `A` toggles oracle auto-play, and `R` resets.

### Evaluation Results

Final checkpoint (epoch 50):

| Game | Score | Steps | Snake Length |
| ---- | ----: | ----: | -----------: |
| 0    |    38 |   395 |           41 |
| 1    |    30 |   232 |           33 |
| 2    |    18 |   169 |           21 |
| 3    |    40 |   393 |           43 |
| 4    |    33 |   294 |           36 |
| **Avg** | **31.8** | **296.6** | -- |

Epoch 1:

| Game | Score | Steps | Snake Length |
| ---- | ----: | ----: | -----------: |
| 0    |     1 |     6 |            4 |
| 1    |     9 |   103 |           12 |
| 2    |     2 |    19 |            5 |
| 3    |     7 |    66 |           10 |
| 4    |     3 |    39 |            6 |
| **Avg** | **4.4** | **46.6** | -- |

Base model:

| Game | Score | Steps | Snake Length |
| ---- | ----: | ----: | -----------: |
| 0    |     1 |     6 |            4 |
| 1    |     0 |     6 |            3 |
| 2    |     0 |     6 |            3 |
| **Avg** | **0.3** | **6.0** | -- |

### Key Training Settings

| Parameter | Value |
| --------- | ----- |
| Model | HuggingFaceTB/SmolVLM2-500M-Video-Instruct |
| Epochs | 50 (each epoch = 1/10 of data) |
| Effective batch size | 32 (2 per GPU x 8 GPUs x 2 grad accum) |
| Learning rate | 1e-5 with WarmupCosineLR (5% warmup) |
| Max sequence length | 2048 |
| LoRA rank | 16 (`alpha=32`, `dropout=0.05`, `q/k/v/o_proj`) |
| DeepSpeed | ZeRO stage 2, bf16 |
| Full dataset size | ~879,301 steps across 4,000 episodes |
| Hardware | 8xA100 40 GB GPUs |
