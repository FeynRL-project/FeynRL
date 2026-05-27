from __future__ import annotations
from typing import Any, Dict, Iterable
import torch


def _is_batchfeature(x: Any) -> bool:
    # Avoid importing transformers at import time.
    return hasattr(x, "to") and hasattr(x, "keys") and hasattr(x, "__getitem__")


def move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Recursively move tensors (and common tensor containers) to `device`.

    Supports:
      - torch.Tensor
      - dict / list / tuple
      - HF BatchFeature / BatchEncoding-like objects (anything with `.to()` + mapping interface)
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    if _is_batchfeature(obj):
        try:
            return obj.to(device)
        except Exception:
            # Fall back to mapping-based move.
            return {k: move_to_device(obj[k], device) for k in obj.keys()}

    return obj


def stack_text_tensors(samples: Iterable[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    samples = list(samples)
    return {
        "input_ids": torch.stack([s["input_ids"] for s in samples], dim=0),
        "attn_mask": torch.stack([s["attn_mask"] for s in samples], dim=0),
        "loss_mask": torch.stack([s["loss_mask"] for s in samples], dim=0),
    }
