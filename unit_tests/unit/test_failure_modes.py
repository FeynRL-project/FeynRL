import torch
import pytest
from algs.PPO.ppo import PPO
from rollouts.replay_buffer import ReplayBuffer
from types import SimpleNamespace

def test_shape_mismatch_error():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 2, 4
    # Mismatched rewards shape
    rewards = torch.randn(B, T + 1)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.ones(B, T)
    
    # In ppo.py, it checks if len(all_len) != 1.
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_nan_reward_error():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 2, 4
    rewards = torch.tensor([[1.0, 2.0, float('nan'), 4.0], [1.0, 2.0, 3.0, 4.0]])
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.ones(B, T)
    
    with pytest.raises(ValueError, match="rewards or values contain NaN"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_empty_batch_error():
    # ReplayBuffer is a normal class
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    with pytest.raises(ValueError, match="collate_fn received an empty batch"):
        rb.collate_fn([])

def test_invalid_mask_holes():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 1, 5
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    
    # ppo.py checks for holes (rises & (drops.cumsum(dim=1) > 0)).any()
    with pytest.raises(ValueError, match="mask has non-contiguous valid regions"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)
