"""
Model loading + forward adapters.

This package separates:
- Algorithms (losses) from
- Model-family loading and forward plumbing (adapters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import models.transformers  # noqa: F401 — trigger @register decorators in all family modules
from models.registry import get_loader, list_loaders


@dataclass(frozen=True)
class ModelBundle:
    model: Any | None
    tokenizer: Any | None
    processor: Any | None


Components = Iterable[Literal["model", "tokenizer", "processor"]]


def load(model_cfg: Any, *, rank: int = 0, components: Components = ("model", "tokenizer", "processor")) -> ModelBundle:
    """
    Unified loader for (model, tokenizer, processor).

    - If "model" is requested, dispatches through the registry keyed by `model_cfg.model_class`.
      This typically loads model weights.
    - If only tokenizer/processor are requested, loads them without instantiating the HF model.
      Intended for RL drivers and dataset plumbing.
    """
    want = set(components)
    if not want.issubset({"model", "tokenizer", "processor"}):
        raise ValueError(f"Invalid components={sorted(want)}")

    model_class = getattr(model_cfg, "model_class", None)
    if not model_class:
        raise ValueError("model.model_class must be set in your config.")
    name = getattr(model_cfg, "name", None)
    if not name:
        raise ValueError("model.name must be set in your config.")

    trust_remote_code = bool(getattr(model_cfg, "trust_remote_code", False))
    processor_name = getattr(model_cfg, "processor_name_or_path", None) or name

    model = tokenizer = processor = None

    if "model" in want:
        loader = get_loader(model_class)
        result = loader(model_cfg, rank=rank)
        # Registry loaders conventionally return (model, tokenizer, processor_or_none)
        if isinstance(result, tuple):
            if len(result) == 3:
                model, tokenizer, processor = result
            elif len(result) == 2:
                model, tokenizer = result
                processor = None
            else:
                raise RuntimeError(f"Unexpected loader return tuple length: {len(result)}")
        else:
            raise RuntimeError(f"Unexpected loader return type: {type(result)}")
        if "tokenizer" not in want:
            tokenizer = None
        if "processor" not in want:
            processor = None
        return ModelBundle(model=model, tokenizer=tokenizer, processor=processor)

    # Tokenizer/processor-only path (no model weights).
    if model_class in ("qwen2_5_vl", "qwen2_audio"):
        from transformers import AutoProcessor  # type: ignore

        processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=trust_remote_code)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer  # type: ignore

            tokenizer = AutoTokenizer.from_pretrained(processor_name, trust_remote_code=trust_remote_code)
    else:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        processor = None

    if "tokenizer" not in want:
        tokenizer = None
    if "processor" not in want:
        processor = None
    return ModelBundle(model=None, tokenizer=tokenizer, processor=processor)


__all__ = ["ModelBundle", "load", "get_loader", "list_loaders"]