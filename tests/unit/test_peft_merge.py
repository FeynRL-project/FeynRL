import torch
import pytest
from types import SimpleNamespace
from algs.RL.common import COMMON

def test_merge_peft_state_dict_logic():
    dummy_self = SimpleNamespace()
    dummy_self.peft_config = SimpleNamespace(
        lora_alpha=32,
        lora_rank=8
    )
    
    # Scaling = 32 / 8 = 4.0
    
    # Create a dummy state dict
    # We want to merge lora into a linear layer
    # module_path: "model.layers.0.self_attn.q_proj"
    # base_weight_key: "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    # lora_A_key: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    # lora_B_key: "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    
    H, D, R = 16, 16, 8
    base_weight = torch.ones(H, D)
    lora_A = torch.ones(R, D)
    lora_B = torch.ones(H, R)
    
    # delta = (B @ A) * 4.0 = (ones(H,R) @ ones(R,D)) * 4.0 = ones(H,D) * R * 4.0 = 8 * 4 = 32.0
    # expected_merged = 1.0 + 32.0 = 33.0
    
    prefix = "base_model.model."
    module_path = "model.layers.0.self_attn.q_proj"
    
    raw_sd = {
        prefix + module_path + ".base_layer.weight": base_weight,
        prefix + module_path + ".lora_A.default.weight": lora_A,
        prefix + module_path + ".lora_B.default.weight": lora_B,
        prefix + "model.embed_tokens.weight": torch.ones(10, 10) # Non-lora param
    }
    
    merged = COMMON.merge_peft_state_dict(dummy_self, raw_sd)
    
    # HF names should be stripped of prefix and .base_layer.
    assert "model.layers.0.self_attn.q_proj.weight" in merged
    assert "model.embed_tokens.weight" in merged
    
    assert torch.allclose(merged["model.layers.0.self_attn.q_proj.weight"], torch.tensor(33.0))
    assert torch.allclose(merged["model.embed_tokens.weight"], torch.tensor(1.0))

def test_policy_forward_logic():
    dummy_self = SimpleNamespace()
    dummy_self.ent_coeff = 0.1
    # Mocking self.cross_entropy which is used in policy_forward
    # Actually PPO might define cross_entropy. COMMON uses self.cross_entropy?
    # No, ppo.py defines cross_entropy = nn.CrossEntropyLoss(reduction='none') in __init__
    # Wait, algs/RL/common.py:172 uses self.cross_entropy
    dummy_self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
    dummy_self.alg_name = "test"
    
    B, T, V = 2, 4, 10
    input_ids = torch.randint(0, V, (B, T))
    att_mask = torch.ones(B, T)
    
    # Mock policy_engine
    mock_logits = torch.randn(B, T, V)
    dummy_self.policy_engine = MagicMock(return_value=SimpleNamespace(logits=mock_logits))
    
    logprobs, entropies, target_ids = COMMON.policy_forward(dummy_self, input_ids, att_mask, None)
    
    assert logprobs.shape == (B, T-1)
    assert entropies.shape == (B, T-1)
    assert target_ids.shape == (B, T-1)
    assert torch.equal(target_ids, input_ids[:, 1:])

from unittest.mock import MagicMock
