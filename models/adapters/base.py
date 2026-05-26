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

    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Move the parts of `batch` that should live on `device` to `device`.
        Text-only adapters can simply move tensors. Multimodal adapters can
        handle nested dicts/lists and leave non-tensor payloads untouched.
        """
        ...
