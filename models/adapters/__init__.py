from __future__ import annotations

from models.adapters.base import ForwardOutput, ModelAdapter
from models.adapters.text_causal_lm import TextCausalLMAdapter
from models.adapters.qwen2_5_vl import Qwen2_5VLAdapter
from models.adapters.qwen2_audio import Qwen2AudioAdapter

__all__ = [
    "ForwardOutput",
    "ModelAdapter",
    "TextCausalLMAdapter",
    "Qwen2_5VLAdapter",
    "Qwen2AudioAdapter",
    "get_sft_adapter",
    "get_adapter",
]


def get_sft_adapter(model_class: str):
    if model_class in ("llm", "", None):
        return TextCausalLMAdapter()
    if model_class == "qwen2_5_vl":
        return Qwen2_5VLAdapter()
    if model_class == "qwen2_audio":
        return Qwen2AudioAdapter()
    raise ValueError(
        f"Unsupported model_class '{model_class}' for SFT. "
        f"Supported: 'llm', 'qwen2_5_vl', 'qwen2_audio'."
    )


def get_adapter(model_class: str | None):
    model_class = model_class or "llm"
    if model_class in ("llm", ""):
        return TextCausalLMAdapter()
    if model_class == "qwen2_5_vl":
        return Qwen2_5VLAdapter()
    if model_class == "qwen2_audio":
        return Qwen2AudioAdapter()
    raise ValueError(f"Unknown model_class '{model_class}' for adapter dispatch")
