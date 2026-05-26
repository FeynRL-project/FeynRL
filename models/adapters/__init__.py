from __future__ import annotations

from models.adapters.base import ForwardOutput, ModelAdapter
from models.adapters.text_causal_lm import TextCausalLMAdapter

__all__ = [
    "ForwardOutput",
    "ModelAdapter",
    "TextCausalLMAdapter",
    "get_sft_adapter",
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
