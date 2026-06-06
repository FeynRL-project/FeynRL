from __future__ import annotations
from models.adapters.base import ForwardOutput, ModelAdapter
from models.adapters.text_causal_lm import TextCausalLMAdapter
from models.adapters.qwen2_5_vl import Qwen2_5VLAdapter

__all__ = [
    "ForwardOutput",
    "ModelAdapter",
    "TextCausalLMAdapter",
    "Qwen2_5VLAdapter",
    "get_adapter",
]


def get_adapter(model_class: str | None) -> ModelAdapter:
    model_class = model_class or "llm"
    if model_class in ("llm", "qwen2_5", "gemma3", ""):
        return TextCausalLMAdapter()
    if model_class in ("qwen2_vl", "qwen2_5_vl"):
        return Qwen2_5VLAdapter()
    raise ValueError(
        f"Unknown model_class '{model_class}'. "
        f"Supported: 'llm', 'qwen2_5', 'gemma3', 'qwen2_vl', 'qwen2_5_vl'."
    )
