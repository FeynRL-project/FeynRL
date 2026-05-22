from __future__ import annotations

from models.adapters.base import ForwardOutput, ModelAdapter
from models.adapters.qwen_2 import Qwen2Adapter

__all__ = [
    "ForwardOutput",
    "ModelAdapter",
    "Qwen2Adapter",
    "get_sft_adapter",
]


def get_sft_adapter(model_class: str):
    return Qwen2Adapter()
