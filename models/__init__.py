"""
Model loading + forward adapters.

This package intentionally separates:
- *Algorithms* (e.g. SFT/PPO/GRPO losses) from
- *Model-family plumbing* (how to call forward, what extra kwargs exist, how to
  map outputs to loss inputs).
"""

import models.transformers  # noqa: F401 — trigger @register decorators in all family modules
from models.registry import get_loader, list_loaders

__all__ = ["load_model_and_tokenizer", "load_sft_model_and_tokenizer"]


def load_model_and_tokenizer(model_cfg, rank: int = 0):
    """
    Dispatch model + tokenizer loading via model_cfg.model_class.

    Returns whatever the registered loader returns (model, tokenizer).
    """
    model_class = getattr(model_cfg, "model_class", None)
    if not model_class:
        raise ValueError("model.model_class must be set in your config.")
    loader = get_loader(model_class)
    return loader(model_cfg, rank=rank)


# Back-compat alias for older SFT-specific naming.
def load_sft_model_and_tokenizer(model_cfg, rank: int = 0):
    return load_model_and_tokenizer(model_cfg, rank=rank)
