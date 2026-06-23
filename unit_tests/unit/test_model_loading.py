import pytest
from unittest.mock import patch
import misc.model_loading as ml


def test_build_hf_model_selects_vlm_class():
    '''model_class="vlm" must load via AutoModelForImageTextToText, not the causal LM.
    attn_impl="eager" is forwarded as-is.'''
    with patch.object(ml, "AutoConfig"), \
         patch.object(ml, "AutoModelForImageTextToText") as vlm_cls, \
         patch.object(ml, "AutoModelForCausalLM") as llm_cls:
        ml.build_hf_model("some/path", "bfloat16", "vlm", False, "eager")
        vlm_cls.from_pretrained.assert_called_once()
        llm_cls.from_pretrained.assert_not_called()
        assert vlm_cls.from_pretrained.call_args.kwargs["attn_implementation"] == "eager"


def test_build_hf_model_selects_llm_class():
    '''model_class="llm" must load via AutoModelForCausalLM; attn_impl "" maps to None.'''
    with patch.object(ml, "AutoConfig"), \
         patch.object(ml, "AutoModelForImageTextToText") as vlm_cls, \
         patch.object(ml, "AutoModelForCausalLM") as llm_cls:
        ml.build_hf_model("some/path", "bfloat16", "llm", False, "")
        llm_cls.from_pretrained.assert_called_once()
        vlm_cls.from_pretrained.assert_not_called()
        assert llm_cls.from_pretrained.call_args.kwargs["attn_implementation"] is None


def test_build_hf_model_rejects_unknown_class():
    with patch.object(ml, "AutoConfig"):
        with pytest.raises(ValueError, match="Unsupported model_class"):
            ml.build_hf_model("some/path", "bfloat16", "audio", False, None)


def test_common_load_single_model_delegates_to_build_hf_model():
    '''RL's COMMON.load_single_model must route through the shared build_hf_model with
    the alg's model_class/attn_impl, instead of selecting the class itself.'''
    from types import SimpleNamespace
    from algs.RL.common import COMMON

    c = COMMON()
    c.model_class = "vlm"
    c.trust_remote_code = False
    c.attn_impl = "eager"
    c.alg_name = "TEST"
    c.peft_config = SimpleNamespace(use_peft=False)
    c.gradient_checkpointing = False

    import algs.RL.common as common_mod
    import torch
    with patch.object(common_mod, "build_hf_model") as bhm:
        out = c.load_single_model("some/path", torch.float32, "ref")
        bhm.assert_called_once()
        kw = bhm.call_args.kwargs
        assert kw["model_class"] == "vlm"
        assert kw["attn_impl"] == "eager"
        assert kw["model_path"] == "some/path"
        assert out is bhm.return_value


def test_common_load_single_model_accepts_dict_attn_impl():
    '''VLM per-subconfig dict attn_impl (validated in load.py) must pass the assertion
    and be forwarded unchanged to build_hf_model.'''
    from types import SimpleNamespace
    from algs.RL.common import COMMON

    c = COMMON()
    c.model_class = "vlm"
    c.trust_remote_code = False
    c.attn_impl = {"text_config": "flash_attention_2", "vision_config": "eager"}
    c.alg_name = "TEST"
    c.peft_config = SimpleNamespace(use_peft=False)
    c.gradient_checkpointing = False

    import algs.RL.common as common_mod
    import torch
    with patch.object(common_mod, "build_hf_model") as bhm:
        c.load_single_model("some/path", torch.float32, "ref")
        assert bhm.call_args.kwargs["attn_impl"] == {
            "text_config": "flash_attention_2", "vision_config": "eager"}
