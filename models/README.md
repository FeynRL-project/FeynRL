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

## The two sub-systems

### 1. Loaders (`models/transformers/`)

A loader is a plain function with the signature:

```python
def load_my_model(cfg, rank: int = 0) -> tuple[Model, Tokenizer, Processor | None]:
    ...
```

`cfg` is a namespace with at minimum: `cfg.name`, `cfg.dtype`, `cfg.trust_remote_code`,
`cfg.attn_implementation`, and optionally `cfg.processor_name_or_path`.

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

### 2. Adapters (`models/adapters/`)

An adapter satisfies the `ModelAdapter` Protocol defined in `base.py`:

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

Every algorithm calls `adapter.forward(engine, batch)` and works exclusively with
`ForwardOutput`. No algorithm ever inspects `pixel_values`, `input_features`, or any other
modality-specific tensor.

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

## Adding a new model or modality

There are two independent questions to answer:

**Q1: Does loading the model require anything beyond `AutoModelForCausalLM.from_pretrained`?**
If yes, write a new loader. If no, just add a `@register` alias in `hf_common.py`.

**Q2: Does the forward pass require anything beyond `(input_ids, attn_mask, position_ids)`?**
If yes, write a new adapter. If no, `TextCausalLMAdapter` already covers it — just add the
`model_class` key to the `("llm", "qwen2_5", "gemma3", "")` branch in `get_adapter`.

The two decisions are independent. A new text model with a custom architecture only needs a new
loader. A standard-architecture model that happens to ingest images needs a new adapter and a new
loader. A generic causal LM with no special tokens needs neither.

---

### Step-by-step: adding a new multimodal model

This walkthrough adds a hypothetical `llava` model class that takes image pixel values.

#### Step 1 — Write the loader (`models/transformers/hf_llava.py`)

```python
from __future__ import annotations
from typing import Any, Tuple
from misc.utils import safe_string_to_torch_dtype
from models.registry import register
from models.transformers.hf_common import normalize_pad_token

@register("llava")
def load_llava(cfg: Any, rank: int = 0) -> Tuple[Any, Any, Any]:
    dtype = safe_string_to_torch_dtype(cfg.dtype)
    processor_name = getattr(cfg, "processor_name_or_path", None) or cfg.name

    from transformers import LlavaForConditionalGeneration, AutoProcessor

    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.name,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=cfg.trust_remote_code)
    tokenizer = processor.tokenizer
    normalize_pad_token(model, tokenizer, rank=rank)
    return model, tokenizer, processor
```

Key rules:
- Always use `torch_dtype=dtype` (not `dtype=`).
- Always return a 3-tuple `(model, tokenizer, processor)`. Return `None` for processor if not
  applicable (matches the `hf_common.py` loaders).
- Call `normalize_pad_token` so pad token is set consistently before DeepSpeed wrapping.

#### Step 2 — Register the loader (`models/transformers/__init__.py`)

Add the import so the `@register` decorator fires at startup:

```python
from models.transformers import hf_llava as _hf_llava  # noqa: F401
```

#### Step 3 — Write the adapter (`models/adapters/llava.py`)

```python
from __future__ import annotations
from typing import Any, Dict
import torch
from misc.batch_utils import move_to_device
from models.adapters.base import ForwardOutput, ModelAdapter

class LlavaAdapter(ModelAdapter):

    def forward(self, model_engine: Any, batch: Dict[str, Any]) -> ForwardOutput:
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        loss_mask = batch["loss_mask"]

        pos_ids = batch.get("position_ids", None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(attn_mask.device)

        mm_kwargs: Dict[str, Any] = {}
        mm = batch.get("multi_modal_inputs", None)
        if isinstance(mm, dict):
            vision = mm.get("vision", None)
            if isinstance(vision, dict):
                pv = vision.get("pixel_values", None)
                if pv is not None and torch.is_tensor(pv):
                    mm_kwargs["pixel_values"] = pv

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=False,
            **mm_kwargs,
        )

        logits = outputs.logits[:, :-1, :].contiguous()   # [B, T-1, V]
        target_ids = input_ids[:, 1:].contiguous()         # [B, T-1]
        return ForwardOutput(logits=logits, target_ids=target_ids, loss_mask=loss_mask)

    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return move_to_device(batch, device)
```

Key rules:
- Always slice `logits[:, :-1, :]` and `input_ids[:, 1:]` — the standard causal shift. Do not
  alter `loss_mask`; it arrives pre-computed from the data feed.
- Extract extra tensors only from `batch["multi_modal_inputs"]`. Never read modality-specific keys
  at the top level of the batch.
- Always pass `use_cache=False`.
- Do not modify the `loss_mask` — that is the data feed's responsibility.

#### Step 4 — Register the adapter (`models/adapters/__init__.py`)

Import and add to `get_adapter`:

```python
from models.adapters.llava import LlavaAdapter   # add this import

def get_adapter(model_class: str | None) -> ModelAdapter:
    model_class = model_class or "llm"
    if model_class in ("llm", "qwen2_5", "gemma3", ""):
        return TextCausalLMAdapter()
    if model_class == "qwen2_5_vl":
        return Qwen2_5VLAdapter()
    if model_class == "qwen2_audio":
        return Qwen2AudioAdapter()
    if model_class == "llava":                    # add this branch
        return LlavaAdapter()
    raise ValueError(...)
```

Also add `"LlavaAdapter"` to `__all__`.

#### Step 5 — Write the data feeds

The data feeds live in `data_feeds/`, not `models/`. See `data_feeds/factory.py` for the
`make_sft_feed`, `make_preference_feed`, and `make_rollout_feed` factories and add a branch for
`"llava"` in each one that is relevant to your use case.

The feeds must populate `batch["multi_modal_inputs"]["vision"]["pixel_values"]` (or whatever
key your adapter reads) so the adapter can find it.

For DPO preference data, also update `_preference_vision_collate` in `factory.py` if the vision
tensors need special batching logic — or add a new collate function for your modality.

#### Step 6 — Config

Set `model.model_class: llava` in your YAML config. That single string routes both the loader
and the adapter.

---

## What the algorithms see (and don't see)

The RL training engines (`algs/GRPO`, `algs/CISPO`, `algs/PPO`, etc.) call
`COMMON.policy_forward` and `COMMON.ref_forward` directly rather than going through the adapter.
This is intentional: RL training operates on token sequences from the replay buffer, which contain
no image or audio tensors. Multimodal content is consumed only at rollout time by the vLLM
engines, which handle it natively via `multi_modal_data`. If you need the RL *training* step to
see multimodal inputs (e.g. for a vision reward signal baked into the policy gradient), you would
need to extend `policy_forward` in `common.py` to call the adapter instead of the model directly.

SFT and DPO go through the adapter unconditionally for all model classes.

---

## Invariants to preserve

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
