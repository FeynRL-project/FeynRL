import torch
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from algs.PPO.ppo import PPO

# Use the original class
ppo_logic = PPO

def test_value_forward_logic():
    # Mocking self
    dummy_self = SimpleNamespace()
    
    # Mock value_engine
    B, T = 2, 5
    mock_logits = torch.arange(B * T, dtype=torch.float32).view(B, T, 1)
    # Row 0: [0, 1, 2, 3, 4]
    # Row 1: [5, 6, 7, 8, 9]
    
    dummy_self.value_engine = MagicMock(return_value=SimpleNamespace(logits=mock_logits))
    
    input_ids = torch.zeros(B, T)
    # att_mask: first row has 3 tokens, second has 5
    att_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1]
    ])
    
    values, last_value = ppo_logic.value_forward(dummy_self, input_ids, att_mask, None)
    
    # values should be logits[:, :-1] squeezed: [B, T-1]
    assert values.shape == (2, 4)
    assert torch.allclose(values[0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    
    # last_value should be the value at last_real_idx
    # Row 0: att_mask sum=3, last_idx=2, value=2.0
    # Row 1: att_mask sum=5, last_idx=4, value=9.0
    assert torch.allclose(last_value, torch.tensor([2.0, 9.0]))

def test_value_forward_empty_mask_error():
    dummy_self = SimpleNamespace()
    dummy_self.value_engine = MagicMock(return_value=SimpleNamespace(logits=torch.zeros(1, 5, 1)))
    att_mask = torch.zeros(1, 5) # All padding
    
    with pytest.raises(ValueError, match="att_mask has rows with zero valid tokens"):
        ppo_logic.value_forward(dummy_self, torch.zeros(1, 5), att_mask, None)

def test_compute_value_loss_correctness():
    dummy_self = SimpleNamespace()
    
    B, T_minus_1 = 2, 3
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    returns = torch.tensor([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    
    loss, v_denom, metrics = ppo_logic.compute_value_loss(dummy_self, values, returns, mask)

    # (values - returns)^2 = [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]
    # mask * errors^2 = [[0.25, 0.25, 0.0], [0.25, 0.25, 0.25]]
    # Sum = 0.25 * 5 = 1.25
    # Denom = mask.sum() = 5
    # v_loss_sum (raw sum) = 0.5 * 1.25 = 0.625
    # metrics['loss_v'] = v_loss_sum / v_denom = 0.625 / 5 = 0.125

    assert np.isclose(loss.item(), 0.625)
    assert np.isclose(v_denom.item(), 5.0)
    assert np.isclose(metrics['loss_v'], 0.125)

def test_calculate_gae():
    dummy_self = SimpleNamespace()
    dummy_self.value_engine = MagicMock()
    dummy_self.value_engine.device = torch.device('cpu')
    
    # Mock methods being called
    dummy_self.value_forward = MagicMock(return_value=(torch.zeros(1, 3), torch.zeros(1)))
    dummy_self.compute_advantages = MagicMock(return_value=(torch.ones(1, 3), torch.ones(1, 3)))
    # Bind get_global_stats so calculate_gae can call self.get_global_stats
    import types
    dummy_self.get_global_stats = types.MethodType(ppo_logic.get_global_stats, dummy_self)
    
    micro_batches = [
        {
            'input_ids': torch.zeros(1, 4),
            'attn_mask': torch.ones(1, 4),
            'rewards': torch.zeros(1, 4),
            'done': torch.zeros(1, 4),
            'mask': torch.ones(1, 4),
        }
    ]
    
    result = ppo_logic.calculate_gae(dummy_self, micro_batches)
    
    assert len(result) == 1
    rets, advs = result[0]
    assert torch.all(rets == 1.0)
    # calculate_gae normalizes advantages across the full batch.
    # All-ones advantages have zero variance, so they normalize to zero.
    assert torch.allclose(advs, torch.zeros(1, 3))
    
    dummy_self.value_forward.assert_called_once()
    dummy_self.compute_advantages.assert_called_once()
