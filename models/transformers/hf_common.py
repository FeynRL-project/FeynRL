from __future__ import annotations

from typing import Any, Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from misc.utils import safe_string_to_torch_dtype
from models.registry import register


def normalize_pad_token(model: Any, tokenizer: Any, rank: int = 0) -> None:
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            if rank == 0:
                print("Warning: pad token not set; using eos_token as pad token")
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        else:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_hf_causal_lm(cfg: Any, rank: int = 0) -> Tuple[Any, Any, None]:
    assert cfg.dtype != "auto", "dtype must not be auto to avoid precision issues"
    assert getattr(cfg, "attn_implementation", None) in (None, "", "eager", "flash_attention_2"), (
        "attn_implementation must be one of None, '', 'eager', 'flash_attention_2'"
    )
    attn_impl = getattr(cfg, "attn_implementation", None)
    dtype = safe_string_to_torch_dtype(cfg.dtype)
    config = AutoConfig.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
        config=config,
        attn_implementation=None if attn_impl == "" else attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    normalize_pad_token(model, tokenizer, rank=rank)
    return model, tokenizer, None


# Back-compat alias.
load_hf_text_model = load_hf_causal_lm


@register("llm")
def _load_llm(cfg: Any, rank: int = 0) -> Tuple[Any, Any, None]:
    return load_hf_causal_lm(cfg, rank=rank)


@register("qwen2_5")
def _load_qwen2_5(cfg: Any, rank: int = 0) -> Tuple[Any, Any, None]:
    return load_hf_causal_lm(cfg, rank=rank)


@register("gemma3")
def _load_gemma3(cfg: Any, rank: int = 0) -> Tuple[Any, Any, None]:
    return load_hf_causal_lm(cfg, rank=rank)
