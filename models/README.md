# models/

This package handles two orthogonal concerns for every model family:

1. **Loading** — instantiate a model + tokenizer (+ optional processor) from a checkpoint.
2. **Forward adapters** — translate a modality-specific batch into a canonical
   `(logits, target_ids, loss_mask)` triple that algorithms consume.

These two concerns are deliberately split so that the training engines (SFT, DPO, GRPO, …) never
call any model directly and remain completely data-agnostic. The only coupling between a model
family and an algorithm is the `model_class` string set in your config.

---

## Directory structure

```
models/
├── __init__.py              # public API: load_model_and_tokenizer()
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

## Loaders (`models/transformers/`)

A loader is a plain function with the signature:

```python
def load_my_model(cfg, rank: int = 0) -> tuple[Model, Tokenizer, Processor | None]:
    ...
```

`cfg` is your validated config object (see `configs/load.py`) and includes at minimum:
`cfg.name`, `cfg.dtype`, `cfg.trust_remote_code`, `cfg.attn_implementation`, and optionally
`cfg.processor_name_or_path`.

Loaders are registered with `@register("model_class_key")` from `models/registry.py`.
`models/__init__.py` imports `models.transformers` on startup (for side-effects), which runs all
`@register` decorators so every family is available before any call to `get_loader()`.

The public entry point is:

```python
from models import load_model_and_tokenizer
model, tokenizer, processor = load_model_and_tokenizer(model_cfg, rank=rank)
```

**Registered keys and what they load**

| `model_class`  | Model class loaded                           | Returns processor? |
|----------------|----------------------------------------------|--------------------|
| `llm`          | `AutoModelForCausalLM`                       | No                 |
| `qwen2_5`      | `AutoModelForCausalLM` (Qwen2.5 text)        | No                 |
| `gemma3`       | `AutoModelForCausalLM` (Gemma 3 text)        | No                 |
| `qwen2_5_vl`   | `Qwen2_5_VLForConditionalGeneration`         | Yes                |
| `qwen2_audio`  | `Qwen2AudioForConditionalGeneration`         | Yes                |

---

## Adapters (`models/adapters/`)

An adapter satisfies the `ModelAdapter` Protocol defined in `adapters/base.py`:

```python
class ModelAdapter(Protocol):
    def forward(self, model_engine: Any, batch: Dict[str, Any]) -> ForwardOutput: ...
    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]: ...
```

`ForwardOutput` is a frozen dataclass:

```python
@dataclass(frozen=True)
class ForwardOutput:
    logits:     torch.Tensor   # [B, T-1, V]
    target_ids: torch.Tensor   # [B, T-1]
    loss_mask:  torch.Tensor   # [B, T-1]
```

SFT and DPO call `adapter.forward(engine, batch)` and work exclusively with `ForwardOutput`.
No algorithm code inspects `pixel_values`, `input_features`, or any other modality-specific tensor.

The adapter is selected at run time via `get_adapter(model_class)` from `adapters/__init__.py`.

**Batch contract**

All adapters receive a batch dict with at minimum:

| Key                  | Shape        | Notes                                    |
|----------------------|--------------|------------------------------------------|
| `input_ids`          | `[B, T]`     |                                          |
| `attn_mask`          | `[B, T]`     |                                          |
| `loss_mask`          | `[B, T-1]`   | 1 = compute loss, 0 = ignore             |
| `position_ids`       | `[B, T]`     | optional; HF computes it if absent       |
| `multi_modal_inputs` | nested dict  | optional; modality-specific tensors only |

Adapters for non-text models read `batch["multi_modal_inputs"]` for extra tensors (e.g.
`pixel_values`, `image_grid_thw`, `input_features`) and pass them as `**kwargs` to the model.
The text adapter ignores `multi_modal_inputs` entirely.

**`to_device`** moves the whole batch to a target device. For most adapters this is a recursive
tensor move via `misc.batch_utils.move_to_device`. Call it before `forward` whenever the batch
arrives from a DataLoader (CPU tensors).

---

## Data flow: from config to algorithm

```
config.model.model_class = "qwen2_5_vl"
          │
          ▼
models/transformers/hf_vlm.py
  @register("qwen2_5_vl")
  load_qwen2_5_vl(cfg) → (model, tokenizer, processor)
          │
          ├── model  →  wrapped in DeepSpeed engine
          ├── tokenizer  →  DataFeed, rollout tokenisation
          └── processor  →  ImagePairedFeed, ImagePreferenceFeed, rollout pre-processing
          │
          ▼
models/adapters/qwen2_5_vl.py
  get_adapter("qwen2_5_vl") → Qwen2_5VLAdapter()
          │
          ▼
SFT / DPO / GRPO / ...
  adapter.forward(engine, batch) → ForwardOutput(logits, target_ids, loss_mask)
  # algorithm sees only ForwardOutput — no pixel_values, no modality logic
```

---

## What the algorithms see (and don't see)

The RL training engines (`algs/GRPO`, `algs/CISPO`, `algs/PPO`, etc.) call
`COMMON.policy_forward` and `COMMON.ref_forward` directly rather than going through the adapter.
This is intentional: RL training operates on token sequences from the replay buffer, which contain
no image or audio tensors. Multimodal content is consumed only at rollout time by the vLLM
engines, which handle it natively via `multi_modal_data`. If you need the RL *training* step to
see multimodal inputs (e.g. for a vision reward signal baked into the policy gradient), you would
need to extend `policy_forward` in `common.py` to call the adapter instead of the model directly.

In this repo's entrypoints, SFT and DPO go through the adapter for all model classes.

---

## Adding a new family (quick checklist)

- Add a loader: `models/transformers/hf_*.py` + `@register("your_key")`.
- Add an adapter (only if needed): `models/adapters/*.py` + `get_adapter()` branch.
- Add data feeds/collators (only if needed): `data_feeds/factory.py` branches.

---

## Invariants to preserve (short)

- **Algorithms are data-agnostic.** No algorithm file may import from `data_feeds/` or reference
  `pixel_values`, `input_features`, or any other modality-specific key.
- **Adapters are stateless.** An adapter instance holds no mutable state; `forward` and
  `to_device` are pure functions of their arguments.
- **`loss_mask` is never altered by the adapter.** The data feed computes it; the adapter passes
  it through unchanged inside `ForwardOutput`.
- **`model_class` is the sole coupling point.** Loader selection, adapter selection, and data
  feed selection all key off the same string. Adding a new family means touching three files
  (`transformers/`, `adapters/__init__.py`, `data_feeds/factory.py`) plus writing the new
  loader and adapter modules.
