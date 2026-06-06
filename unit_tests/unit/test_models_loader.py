import os
import tempfile
from unittest.mock import MagicMock
import pytest
import torch


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_register_and_get():
    from models.registry import get_loader, register

    @register("_unit_test_dummy")
    def _dummy(cfg, rank):
        return "model", "tok"

    assert get_loader("_unit_test_dummy") is _dummy


def test_registry_unknown_class_raises():
    from models.registry import get_loader

    with pytest.raises(ValueError, match="Unknown model_class"):
        get_loader("__does_not_exist__")


def test_registry_list_loaders_contains_expected_keys():
    from models.registry import list_loaders

    keys = set(list_loaders())
    assert "llm" in keys
    assert "qwen2_5" in keys
    assert "gemma3" in keys
    assert "qwen2_5_vl" in keys


# ---------------------------------------------------------------------------
# models.load dispatch
# ---------------------------------------------------------------------------


def test_load_dispatches_to_registered_loader():
    import models
    from models.registry import register

    sentinel = (MagicMock(), MagicMock())

    @register("_unit_test_dispatch")
    def _loader(cfg, rank):
        return sentinel

    model_cfg = MagicMock()
    model_cfg.model_class = "_unit_test_dispatch"
    model_cfg.name = "dummy"
    model_cfg.trust_remote_code = False
    out = models.load(model_cfg, rank=0, components=("model", "tokenizer"))
    assert (out.model, out.tokenizer) == sentinel


@pytest.mark.parametrize("model_class", [None, ""])
def test_load_falsy_model_class_defaults_to_llm(model_class):
    import models
    from models.registry import register, get_loader

    sentinel = (MagicMock(), MagicMock())
    original_llm = get_loader("llm")

    @register("llm")
    def _mock_llm(cfg, rank):
        return sentinel

    try:
        model_cfg = MagicMock()
        model_cfg.model_class = model_class
        model_cfg.name = "dummy"
        model_cfg.trust_remote_code = False
        out = models.load(model_cfg, rank=0, components=("model", "tokenizer"))
        assert (out.model, out.tokenizer) == sentinel
    finally:
        register("llm")(original_llm)


# ---------------------------------------------------------------------------
# hf_common validation (transformers is mocked by conftest.py)
# ---------------------------------------------------------------------------


def test_hf_common_dtype_auto_raises():
    from models.transformers.hf_common import load_hf_causal_lm

    cfg = MagicMock()
    cfg.dtype = "auto"
    with pytest.raises(AssertionError):
        load_hf_causal_lm(cfg)


def test_hf_common_invalid_attn_impl_raises():
    from models.transformers.hf_common import load_hf_causal_lm

    cfg = MagicMock()
    cfg.dtype = "bfloat16"
    cfg.attn_implementation = "unsupported_backend"
    with pytest.raises(AssertionError):
        load_hf_causal_lm(cfg)


def test_hf_common_valid_attn_impl_none():
    from models.transformers.hf_common import load_hf_causal_lm

    cfg = MagicMock()
    cfg.dtype = "bfloat16"
    cfg.attn_implementation = None
    load_hf_causal_lm(cfg)  # should not raise


def test_hf_common_valid_attn_impl_empty_string():
    from models.transformers.hf_common import load_hf_causal_lm

    cfg = MagicMock()
    cfg.dtype = "bfloat16"
    cfg.attn_implementation = ""
    load_hf_causal_lm(cfg)  # should not raise


# ---------------------------------------------------------------------------
# normalize_pad_token
# ---------------------------------------------------------------------------


def test_normalize_pad_token_sets_from_eos_token():
    from models.transformers.hf_common import normalize_pad_token

    model = MagicMock()
    model.config.pad_token_id = None

    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 2

    normalize_pad_token(model, tokenizer, rank=0)
    tokenizer.add_special_tokens.assert_called_once_with({"pad_token": "<eos>"})


def test_normalize_pad_token_falls_back_to_eos_token_id():
    from models.transformers.hf_common import normalize_pad_token

    model = MagicMock()
    model.config.pad_token_id = None

    tokenizer = MagicMock()
    tokenizer.pad_token_id = None
    tokenizer.eos_token = None
    tokenizer.eos_token_id = 3

    normalize_pad_token(model, tokenizer, rank=0)
    assert tokenizer.pad_token_id == 3


def test_normalize_pad_token_syncs_to_model_config():
    from models.transformers.hf_common import normalize_pad_token

    model = MagicMock()
    model.config.pad_token_id = None

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 5

    normalize_pad_token(model, tokenizer, rank=0)
    assert model.config.pad_token_id == 5


def test_normalize_pad_token_skips_when_already_set():
    from models.transformers.hf_common import normalize_pad_token

    model = MagicMock()
    model.config.pad_token_id = 1

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 1

    normalize_pad_token(model, tokenizer, rank=0)
    tokenizer.add_special_tokens.assert_not_called()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_get_adapter_dispatches_by_model_class():
    from models.adapters import TextCausalLMAdapter, get_adapter

    assert isinstance(get_adapter("llm"), TextCausalLMAdapter)


def test_get_adapter_unknown_model_class_raises():
    from models.adapters import get_adapter

    with pytest.raises(ValueError, match="Unknown model_class"):
        get_adapter("__does_not_exist__")


def test_llm_loader_is_registered_and_callable():
    from models.registry import get_loader

    loader = get_loader("llm")
    assert callable(loader)


def test_qwen2_5_loader_is_registered_and_callable():
    from models.registry import get_loader

    loader = get_loader("qwen2_5")
    assert callable(loader)


def test_gemma3_loader_is_registered_and_callable():
    from models.registry import get_loader

    loader = get_loader("gemma3")
    assert callable(loader)


# ---------------------------------------------------------------------------
# Synthetic data script
# ---------------------------------------------------------------------------
