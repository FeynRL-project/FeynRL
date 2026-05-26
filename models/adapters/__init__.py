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
    """
    Keep SFT adapters simple for now: we support a single text LLM adapter.

    Convention:
      - set `model.model_class: llm` in configs for text-only HF causal LMs
      - future model-specific / multimodal adapters can be added later
    """
    if model_class not in ("llm", "", None):
        raise ValueError(
            f"Unsupported model_class '{model_class}' for SFT adapter in this PR. "
            "Use model.model_class: llm for text-only runs."
        )
    return TextCausalLMAdapter()


def get_adapter(model_class: str | None):
    model_class = model_class or "llm"
    if model_class in ("llm", ""):
        return TextCausalLMAdapter()
    if model_class == "qwen2_5_vl":
        return Qwen2_5VLAdapter()
    if model_class == "qwen2_audio":
        return Qwen2AudioAdapter()
    raise ValueError(f"Unknown model_class '{model_class}' for adapter dispatch")
