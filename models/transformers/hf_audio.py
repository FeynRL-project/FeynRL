from __future__ import annotations
from typing import Any, Tuple
from misc.utils import safe_string_to_torch_dtype
from models.registry import register
from models.transformers.hf_common import normalize_pad_token


@register("qwen2_audio")
def load_qwen2_audio(cfg: Any, rank: int = 0) -> Tuple[Any, Any, Any]:
    """Load Qwen2-Audio model + tokenizer + processor."""
    assert cfg.dtype != "auto", "dtype must not be auto to avoid precision issues"
    attn_impl = getattr(cfg, "attn_implementation", None)
    dtype = safe_string_to_torch_dtype(cfg.dtype)
    name = cfg.name
    processor_name = getattr(cfg, "processor_name_or_path", None) or name

    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration  # type: ignore

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        name,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=None if attn_impl == "" else attn_impl,
    )
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=cfg.trust_remote_code)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(processor_name, trust_remote_code=cfg.trust_remote_code)

    normalize_pad_token(model, tokenizer, rank=rank)
    return model, tokenizer, processor
