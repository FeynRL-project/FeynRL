# models/

This package handles two orthogonal concerns for every model family:

1. **Loading** — instantiate a model + tokenizer (+ optional processor) from a checkpoint.
2. **Forward adapters** — translate a modality-specific batch into a canonical
   `(logits, target_ids, loss_mask)` triple that algorithms consume.

These two concerns are deliberately split so that the training engines (SFT, DPO, GRPO, …) never
call any model directly and remain completely modality-agnostic. The only coupling between a model
family and an algorithm is the `model_class` string set in your config.

---

## Directory structure

```
models/
├── __init__.py              # public API: load()
├── registry.py              # @register / get_loader() — loader registry
│
├── transformers/            # one module per model family (loading only)
│   ├── __init__.py          # side-effect imports to trigger @register decorators
│   ├── hf_common.py         # generic causal LM loader + llm / qwen2_5 / gemma3 registrations
│   ├── hf_vlm.py            # qwen2_5_vl registration
│   └── hf_audio.py          # qwen2_audio registration
│
└── adapters/                # one module per model family (forward pass only)
    ├── __init__.py          # get_adapter() + __all__
    ├── base.py              # ForwardOutput dataclass + ModelAdapter Protocol
    ├── text_causal_lm.py    # standard text LLM adapter (default)
    ├── qwen2_5_vl.py        # vision-language adapter (Qwen2.5-VL)
    └── qwen2_audio.py       # audio adapter (Qwen2-Audio)
```

---

## Registered model families

`model_class` is the single string that selects the loader, adapter, and data feed for a run.

| `model_class`  | Loader (`transformers/`)            | Adapter (`adapters/`)       | Returns processor? | Modality    |
|----------------|-------------------------------------|-----------------------------|--------------------|-------------|
| `llm`          | `AutoModelForCausalLM`              | `TextCausalLMAdapter`       | No                 | Text        |
| `qwen2_5`      | `AutoModelForCausalLM` (Qwen2.5)    | `TextCausalLMAdapter`       | No                 | Text        |
| `gemma3`       | `AutoModelForCausalLM` (Gemma 3)    | `TextCausalLMAdapter`       | No                 | Text        |
| `qwen2_5_vl`   | `Qwen2_5_VLForConditionalGeneration`| `Qwen2_5VLAdapter`          | Yes                | Vision+Text |
| `qwen2_audio`  | `Qwen2AudioForConditionalGeneration`| `Qwen2AudioAdapter`         | Yes                | Audio+Text  |

---

## Component responsibilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  models/transformers/   LOADING ONLY                                        │
│                                                                             │
│  • Load weights from HF checkpoint (dtype, attn_impl, trust_remote_code)   │
│  • Return (model, tokenizer, processor)                                     │
│  • processor is None for text-only families                                 │
│  • Never called after init — weights are wrapped in DeepSpeed immediately   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  models/adapters/       FORWARD PASS ONLY                                   │
│                                                                             │
│  • forward(engine, batch) → ForwardOutput(logits, target_ids, loss_mask)   │
│  • Extracts modality-specific tensors from batch["multi_modal_inputs"]      │
│  • Passes them as **kwargs to the HF model (pixel_values, input_features …) │
│  • to_device(batch, device) — moves tensors; handles nested dicts/lists     │
│  • build_multi_modal_inputs(processor, mm_items) — converts raw Python      │
│    objects (PIL images, audio waveforms) into batched tensors               │
│  • Stateless — no mutable instance state                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  data_feeds/factory.py  FEED SELECTION                                      │
│                                                                             │
│  • make_sft_feed(model_class, ...)       → dataset class + kwargs           │
│  • make_preference_feed(model_class, …)  → dataset class + kwargs + collate │
│  • make_rollout_feed(model_class, …)     → dataset class + kwargs           │
│  • Branches on model_class; returns text feed by default                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multimodal data: two representations

There are two distinct forms of multimodal data in the pipeline. They serve different purposes and
live at different stages.

| Name                  | Type                   | Where it lives                        | Contents                                      |
|-----------------------|------------------------|---------------------------------------|-----------------------------------------------|
| `multi_modal_data`    | Raw Python objects     | vLLM prompts, `ReplayBuffer` items    | `{"image": PIL.Image}` or `{"audio": (waveform, sr)}` |
| `multi_modal_inputs`  | Batched tensors        | Training micro-batches                | `{"vision": {"pixel_values": …, "image_grid_thw": …}}` or `{"audio": {"input_features": …}}` |

`multi_modal_data` is what the rollout engine and data feeds produce. It is never passed to the
model. `multi_modal_inputs` is what the adapter's `forward()` consumes. The conversion happens in
`ReplayBuffer.collate_fn()` (RL) or inside the SFT/preference dataset (SFT, DPO), via
`adapter.build_multi_modal_inputs(processor, mm_items)`.

---

## Full data pipeline by training mode

### SFT (`main_sl.py`)

```
config.model.model_class
        │
        ├──▶ models.load()          loads (model, tokenizer, processor)
        │
        ├──▶ make_sft_feed()        selects dataset class by model_class
        │         │
        │    text: PairedFeed            per-sample: {input_ids, attn_mask, loss_mask}
        │    VLM:  ImagePairedFeed       per-sample: {input_ids, attn_mask, loss_mask,
        │    audio: AudioPairedFeed                   multi_modal_inputs}
        │         │
        │    DataLoader (default collate_fn)
        │         │
        │         ▼  batch: {input_ids [B,T], attn_mask [B,T], loss_mask [B,T-1],
        │                    multi_modal_inputs (if multimodal)}
        │
        ├──▶ get_adapter(model_class)   selects adapter instance
        │
        └──▶ SFT(model_engine, optimizer, model_adapter=adapter)
                  │
                  ▼ SFT.forward(batch)
                        adapter.forward(engine, batch) → ForwardOutput
```

**Entry points adjusted for multimodal (`main_sl.py`):**
- `model_class = config.model.model_class` (line ~216)
- `model_adapter = get_adapter(model_class)` (line ~217)
- `make_sft_feed(model_class, params, processor)` (line ~121) — selects `ImagePairedFeed` or `AudioPairedFeed`
- `SFT(..., model_adapter=model_adapter)` (line ~293) — adapter passed to algorithm

---

### DPO / Contrastive Learning (`main_cl.py`)

```
config.model.model_class
        │
        ├──▶ models.load()          loads (model, tokenizer, processor)
        │
        ├──▶ make_preference_feed() selects dataset class + collate_fn by model_class
        │         │
        │    text: PreferenceFeed           per-sample: {input_ids [2,T], attn_mask [2,T],
        │    VLM:  ImagePreferenceFeed                   loss_mask [2,T-1]}
        │         │                          +VLM:       {multi_modal_inputs}
        │         │
        │    collate_fn
        │    text: default (stack → [B,2,T])
        │    VLM:  _preference_vision_collate
        │              stacks tensors + interleaves vision to align with DPO's
        │              [B,2,T] → [2B,T] flattening (repeat_interleave(2))
        │         │
        │         ▼  batch: {input_ids [B,2,T], attn_mask [B,2,T], loss_mask [B,2,T-1],
        │                    multi_modal_inputs (if VLM)}
        │
        ├──▶ get_adapter(model_class)
        │
        └──▶ DPO(model_engine, ref_engine, ..., model_adapter=adapter)
                  │
                  ▼ DPO.step(batch)
                        if model_adapter:  adapter.forward(engine, batch) → ForwardOutput
                        else:              direct HF call (text-only legacy path)
```

**Entry points adjusted for multimodal (`main_cl.py`):**
- `model_class = config.model.model_class` (line ~223)
- `model_adapter = get_adapter(model_class)` (line ~305)
- `make_preference_feed(model_class, params, processor)` (line ~127) — selects `ImagePreferenceFeed` + custom collator
- `DPO(..., model_adapter=model_adapter)` (line ~312)

**Note:** `AudioPreferenceFeed` is not yet implemented; `qwen2_audio` DPO falls back to text preference feed.

---

### RL training (`run_rl_sync.py` / `run_rl_async.py`)

The RL pipeline has two distinct phases: **rollout** (generation) and **training** (gradient steps).
Multimodal data crosses the boundary between them via the `ReplayBuffer`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ROLLOUT PHASE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

config.model.model_class
        │
        ├──▶ make_rollout_feed()    selects prompt dataset by model_class
        │         │
        │    text:  PromptsFeed          per-sample: {"prompt": str}
        │    VLM:   ImagePromptsFeed     per-sample: {"prompt": str,
        │    audio: AudioPromptsFeed                  "multi_modal_data": {"image": PIL}}
        │                                             "multi_modal_data": {"audio": (waveform, sr)}}
        │         │
        │         ▼  prompts list fed to vLLM
        │
        ├──▶ VLLMRolloutEngine / VLLMRolloutEngineAsync
        │         │
        │    vLLM generates responses; multi_modal_data passed natively to vLLM
        │    for vision/audio encoding during generation
        │         │
        │    per-sample output: {input_ids [T], pred_rewards, pred_masks,
        │                        pred_old_logprobs, ...,
        │                        multi_modal_data}   ← raw objects passed through
        │         │
        │         ▼
        ├──▶ ReplayBuffer.add_batch_seqs()
        │         stores items as CPU tensors; multi_modal_data kept as raw Python objects
        │         (PIL images, audio tuples) — NOT converted to tensors here
        │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  TRAINING PHASE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

        ├──▶ ReplayBuffer.collate_fn()  (called by DataLoader per micro-batch)
        │         │
        │    pads input_ids, attn_masks, etc. to batch max length
        │         │
        │    if any(mm_items are not None):
        │         adapter.build_multi_modal_inputs(processor, mm_items)
        │             converts raw objects → batched tensors
        │             e.g. PIL images → {pixel_values [B*P,D], image_grid_thw [B*I,3]}
        │         → out["multi_modal_inputs"] added to batch dict
        │         │
        │         ▼  micro_batch: {input_ids [B,T], attn_mask [B,T], mask [B,T],
        │                          old_logprobs [B,T], rewards [B,T], ...
        │                          multi_modal_inputs (if multimodal)}
        │
        └──▶ GRPO / PPO / P3O / P4O / CISPO
                  │
                  │  algorithm unpacks: input_ids, att_mask, pos_ids from micro_batch
                  │  (algorithm code is modality-agnostic — no mm logic here)
                  │
                  ▼ COMMON.policy_forward(input_ids, att_mask, pos_ids,
                                          micro_batch=micro_batch)
                         reads micro_batch.get("multi_modal_inputs") → batch dict
                         adapter.to_device(batch, device)
                         adapter.forward(policy_engine, batch) → ForwardOutput

                    COMMON.ref_forward(input_ids, att_mask, pos_ids,
                                       micro_batch=micro_batch)
                         same path via ref_model_engine

                    COMMON.snapshot_prox_logprobs(micro_batches, ...)
                         also passes micro_batch=micro_batch per iteration
                         (P3O, P4O, CISPO prox snapshots are multimodal-aware)
```

**Entry points adjusted for multimodal (RL runners):**
- `model_class = config.model.model_class` (run_rl_sync ~178, run_rl_async ~1105)
- `ReplayBuffer(..., model_class=model_class, processor=processor)` — wires adapter + processor for collation
- `make_rollout_feed(model_class, params)` — selects `ImagePromptsFeed` / `AudioPromptsFeed`
- `GRPO/PPO/...(model_class=model_class)` via `core/rl_engines.py` — wires adapter inside `COMMON._get_cached_adapter()`

---

## Batch contract at each stage

### SFT / DPO batch (into algorithm)

| Key                  | Shape         | Notes                                      |
|----------------------|---------------|--------------------------------------------|
| `input_ids`          | `[B, T]`      |                                            |
| `attn_mask`          | `[B, T]`      |                                            |
| `loss_mask`          | `[B, T-1]`    | 1 = train on token, 0 = ignore             |
| `position_ids`       | `[B, T]`      | optional                                   |
| `multi_modal_inputs` | nested dict   | absent for text; `{"vision": {…}}` for VLM |

### RL micro-batch (out of `ReplayBuffer.collate_fn`)

| Key                  | Shape         | Notes                                      |
|----------------------|---------------|--------------------------------------------|
| `input_ids`          | `[B, T]`      | prompt + response tokens                   |
| `attn_mask`          | `[B, T]`      |                                            |
| `mask`               | `[B, T]`      | 1 = response token (train), 0 = prompt/pad |
| `old_logprobs`       | `[B, T]`      | from rollout engine                        |
| `rewards`            | `[B, T]`      |                                            |
| `done`               | `[B, T]`      |                                            |
| `zscore`             | `[B, T]`      |                                            |
| `multi_modal_inputs` | nested dict   | absent for text; built by adapter during collation |

### Adapter batch (into `adapter.forward`)

| Key                  | Shape         | Notes                                      |
|----------------------|---------------|--------------------------------------------|
| `input_ids`          | `[B, T]`      |                                            |
| `attn_mask`          | `[B, T]`      |                                            |
| `loss_mask`          | `[B, T-1]`    | ones for RL (no token masking at this layer)|
| `position_ids`       | `[B, T]`      | optional                                   |
| `multi_modal_inputs` | nested dict   | `None` is safe — text adapter ignores it   |

---

## Adapter details

### `TextCausalLMAdapter`
- Used by: `llm`, `qwen2_5`, `gemma3`
- `forward`: calls `model_engine(input_ids, attention_mask, position_ids, use_cache=False)`
- `multi_modal_inputs`: never read; ignored whether absent or `None`

### `Qwen2_5VLAdapter`
- Used by: `qwen2_5_vl`
- `forward`: extracts `pixel_values` and `image_grid_thw` from `batch["multi_modal_inputs"]["vision"]`;
  flattens patch dim if needed `[B, P, D] → [B*P, D]`; passes as `**kwargs` to model
- `build_multi_modal_inputs`: calls `processor(text, images)` per sample; cats patch tensors across batch
- `to_device`: recursive move via `move_to_device` (handles nested dicts)

### `Qwen2AudioAdapter`
- Used by: `qwen2_audio`
- `forward`: extracts `input_features` and `feature_attention_mask` from `batch["multi_modal_inputs"]["audio"]`
- `build_multi_modal_inputs`: calls processor on `(waveform, sr)` per sample; stacks features

---

## Loaders (`models/transformers/`)

A loader has the signature:

```python
def load_my_model(cfg, rank: int = 0) -> tuple[Model, Tokenizer, Processor | None]:
    ...
```

`cfg` fields used: `cfg.name`, `cfg.dtype`, `cfg.trust_remote_code`, `cfg.attn_implementation`,
`cfg.processor_name_or_path` (optional; multimodal only).

The public entry point:

```python
import models
bundle = models.load(model_cfg, rank=rank)
model, tokenizer, processor = bundle.model, bundle.tokenizer, bundle.processor
```

---

## Adding a new family (checklist)

1. **Loader** — `models/transformers/hf_*.py` + `@register("your_key")` + add import to `transformers/__init__.py`
2. **Adapter** — `models/adapters/your_key.py` + add branch in `adapters/__init__.py:get_adapter()`
3. **Data feeds** — add branches in `data_feeds/factory.py` for `make_sft_feed`, `make_preference_feed`, `make_rollout_feed`
4. **Config** — add `model_class: your_key` to your YAML

If the family is text-only, step 2 can be skipped (`TextCausalLMAdapter` is the default).

---

## Invariants

- **Algorithms are modality-agnostic.** No algorithm file (`algs/*/`) imports from `data_feeds/` or
  references `pixel_values`, `input_features`, or any modality-specific key. The adapter absorbs all
  modality logic.
- **Adapters are stateless.** An adapter instance holds no mutable state; `forward` and `to_device`
  are pure functions of their arguments.
- **`multi_modal_inputs=None` is always safe.** Text adapters ignore the key entirely; multimodal
  adapters guard with `isinstance(mm, dict)`. Algorithms always pass `micro_batch=micro_batch` to
  `policy_forward` / `ref_forward` and the value is `None` for text runs.
- **`loss_mask` is never altered by the adapter.** The data feed or RL layer computes it; the
  adapter returns it unchanged inside `ForwardOutput`.
- **`model_class` is the sole coupling point.** Loader selection, adapter selection, and data feed
  selection all key off the same string. Adding a family means touching the four locations above and
  nothing else.
