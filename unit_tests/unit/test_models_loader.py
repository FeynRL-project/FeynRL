import os
import tempfile
from unittest.mock import MagicMock

import pandas as pd
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
    # Provided by models/transformers/hf_common.py for convenience
    assert "transformers_qwen2_5_sft_text" in keys
    assert "transformers_gemma3_sft_text" in keys


# ---------------------------------------------------------------------------
# load_model_and_tokenizer dispatch
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
    assert models.load_model_and_tokenizer(model_cfg, rank=0) is sentinel
    # Back-compat alias should behave identically.
    assert models.load_sft_model_and_tokenizer(model_cfg, rank=0) is sentinel


def test_load_none_model_class_raises():
    import models

    model_cfg = MagicMock()
    model_cfg.model_class = None
    with pytest.raises(ValueError, match="model.model_class must be set"):
        models.load_model_and_tokenizer(model_cfg)


def test_load_empty_model_class_raises():
    import models

    model_cfg = MagicMock()
    model_cfg.model_class = ""
    with pytest.raises(ValueError, match="model.model_class must be set"):
        models.load_model_and_tokenizer(model_cfg)


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

    loader = get_loader("transformers_qwen2_5_sft_text")
    assert callable(loader)


def test_gemma3_loader_is_registered_and_callable():
    from models.registry import get_loader

    loader = get_loader("transformers_gemma3_sft_text")
    assert callable(loader)


# ---------------------------------------------------------------------------
# Synthetic data script
# ---------------------------------------------------------------------------


def test_synthetic_dataframe_schema():
    from data_prep.synthetic import build_synthetic_dataframe

    df = build_synthetic_dataframe(n=4)
    assert list(df.columns) == ["prompt", "answer"]
    assert len(df) == 4
    for prompt in df["prompt"]:
        assert isinstance(prompt, list) and len(prompt) >= 1
        assert prompt[-1]["role"] == "user"
    for answer in df["answer"]:
        assert isinstance(answer, str) and len(answer) > 0


def test_synthetic_dataframe_with_system_prompt():
    from data_prep.synthetic import build_synthetic_dataframe

    df = build_synthetic_dataframe(n=4, system_prompt="Be concise.")
    for prompt in df["prompt"]:
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"


def test_synthetic_dataframe_repeats_to_fill_n():
    from data_prep.synthetic import build_synthetic_dataframe

    df = build_synthetic_dataframe(n=32)
    assert len(df) == 32


def test_synthetic_script_writes_valid_parquet(tmp_path):
    from data_prep.synthetic import build_synthetic_dataframe

    path = str(tmp_path / "smoke.parquet")
    df = build_synthetic_dataframe(n=8)
    df.to_parquet(path, index=False)

    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == ["prompt", "answer"]
    assert len(loaded) == 8

