import torch
from types import SimpleNamespace
from algs.RL.common import COMMON


def _common(mapping):
    # COMMON has no __init__; remap reads _checkpoint_conversion_mapping off the
    # DeepSpeed-wrapped HF model at self.policy_engine.module.
    c = COMMON()
    c.policy_engine = SimpleNamespace(module=SimpleNamespace(_checkpoint_conversion_mapping=mapping))
    return c


# Real transformers 4.57 _checkpoint_conversion_mapping values.
QWEN_MAPPING = {
    "^visual": "model.visual",
    "^model(?!\\.(language_model|visual))": "model.language_model",
}
GEMMA_MAPPING = {
    "^language_model.model": "model.language_model",
    "^vision_tower": "model.vision_tower",
    "^multi_modal_projector": "model.multi_modal_projector",
    "^language_model.lm_head": "lm_head",
}


def test_remap_qwen_layout():
    '''
        Qwen2/2.5-VL: model.visual.* -> visual.*, model.language_model.* -> model.*,
        lm_head untouched. Matches what save_pretrained writes (and vLLM loads).
    '''
    c = _common(QWEN_MAPPING)
    sd = {
        "model.visual.patch_embed.proj.weight": torch.zeros(1),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.language_model.norm.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }
    out = c.remap_vlm_keys_for_vllm(sd)
    assert set(out.keys()) == {
        "visual.patch_embed.proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    assert out["lm_head.weight"] is sd["lm_head.weight"]


def test_remap_gemma_layout():
    '''
        Gemma-3 has a DIFFERENT mapping than Qwen — the hardcoded Qwen rules would be
        wrong here, which is why the remap is driven by the model's own mapping.
    '''
    c = _common(GEMMA_MAPPING)
    sd = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight": torch.zeros(1),
        "model.multi_modal_projector.mm_input_projection_weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }
    out = c.remap_vlm_keys_for_vllm(sd)
    assert set(out.keys()) == {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
        "multi_modal_projector.mm_input_projection_weight",
        "language_model.lm_head.weight",
    }


def test_remap_noop_for_llm():
    '''
        LLMs have an empty _checkpoint_conversion_mapping -> dict returned unchanged.
    '''
    c = _common({})
    sd = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.norm.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
    }
    out = c.remap_vlm_keys_for_vllm(sd)
    assert out is sd  # short-circuit: same object


def test_get_mapping_unwraps_peft():
    '''
        PEFT path: HF model (with the mapping) sits under .base_model.model.
    '''
    hf = SimpleNamespace(_checkpoint_conversion_mapping=QWEN_MAPPING)
    peft_wrapper = SimpleNamespace(base_model=SimpleNamespace(model=hf))
    c = COMMON()
    c.policy_engine = SimpleNamespace(module=peft_wrapper)
    assert c.get_checkpoint_conversion_mapping() == QWEN_MAPPING


def _transformers_save_reverse(mapping, keys):
    '''
        Verbatim copy of transformers PreTrainedModel.save_pretrained's reverse remap,
        used as the reference oracle our implementation must match exactly.
    '''
    import re
    reverse = {v: k for k, v in mapping.items()}
    out = []
    for key in keys:
        for pattern, replacement in reverse.items():
            replacement = replacement.lstrip("^")
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n = re.subn(pattern, replacement, key)
            if n > 0:
                break
        out.append(key)
    return set(out)


def test_remap_matches_transformers_for_installed_vlms():
    '''
        The remap must produce byte-for-byte the same names transformers' own
        save_pretrained writes to disk (which vLLM is known to load). Pull the REAL
        _checkpoint_conversion_mapping from whatever transformers is installed and compare
        against an independent copy of transformers' reverse logic for each VLM family.
    '''
    import pytest
    transformers = pytest.importorskip("transformers")

    samples = {
        "Qwen2_5_VLForConditionalGeneration": [
            "model.visual.blocks.0.attn.qkv.weight",
            "model.language_model.layers.3.mlp.down_proj.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
        ],
        "Qwen2VLForConditionalGeneration": [
            "model.visual.patch_embed.proj.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        ],
        "Gemma3ForConditionalGeneration": [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.vision_tower.vision_model.encoder.layers.2.mlp.fc1.weight",
            "model.multi_modal_projector.mm_input_projection_weight",
            "lm_head.weight",
        ],
    }

    checked = 0
    for cls_name, keys in samples.items():
        cls = getattr(transformers, cls_name, None)
        mapping = getattr(cls, "_checkpoint_conversion_mapping", None) if cls else None
        if not mapping:
            continue  # this VLM family / mapping not present in the installed version
        c = _common(mapping)
        ours = set(c.remap_vlm_keys_for_vllm({k: torch.zeros(1) for k in keys}).keys())
        assert ours == _transformers_save_reverse(mapping, keys), f"mismatch for {cls_name}"
        checked += 1

    assert checked > 0, "no VLM _checkpoint_conversion_mapping found to verify against"