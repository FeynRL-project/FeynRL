from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Tuple

import torch


@dataclass(frozen=True)
class ForwardOutput:
    """
    Canonical output for token-level supervised objectives.

    - `logits`: [B, T-1, V]
    - `target_ids`: [B, T-1]
    - `loss_mask`: [B, T-1]
    """

    logits: torch.Tensor
    target_ids: torch.Tensor
    loss_mask: torch.Tensor


class ModelAdapter(Protocol):
    """
    Adapter that turns a (possibly multimodal) batch into an algorithm-ready
    `(logits, targets, mask)` triple.
    """

    def forward(self, model_engine: Any, batch: Dict[str, Any]) -> ForwardOutput:
        ...

    def get_mm_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract modality-specific tensors from `batch` into a flat kwargs dict
        ready to be splatted into a model forward call.

        Text-only adapters return ``{}``.  Multimodal adapters return e.g.
        ``{"pixel_values": ..., "image_grid_thw": ...}``.

        Algorithms call this once at the top of ``forward()`` and pass the
        result as ``**mm_kwargs`` alongside ``input_ids``, ``attention_mask``,
        etc.  For DPO, the batch-level MM tensors must already be in the
        flattened ``[2B, ...]`` form produced by the preference collator before
        ``get_mm_kwargs`` is called.
        """
        ...

    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Move the parts of `batch` that should live on `device` to `device`.
        Text-only adapters can simply move tensors. Multimodal adapters can
        handle nested dicts/lists and leave non-tensor payloads untouched.
        """
        ...

