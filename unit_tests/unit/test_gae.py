import torch
import pytest
import numpy as np
from types import SimpleNamespace
from algs.PPO.ppo import PPO

def test_gae_correctness():
    # Use PPO directly since conftest.py mocks @ray.remote to be a no-op
    ppo_logic = PPO 
    dummy_self = SimpleNamespace(gamma=0.9, tau=0.5)
    
    # Simple deterministic test case
    rewards = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    values = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    done = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    
    # Utilizing codebase logic directly
    rets, advs = ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)
    
    # Hand calculation:
    # T=1: mask=1, done=1, next_val=0.0
    #      delta = 2.0 + 0 - 0.5 = 1.5
    #      last_adv = 1 * (1.5 + 0) = 1.5
    #      advs[1] = 1.5, next_val = values[1] = 0.5
    # T=0: mask=1, done=0, next_val=0.5
    #      delta = 1.0 + 0.9*0.5*1.0 - 0.5 = 0.95
    #      last_adv = 1 * (0.95 + 0.9*0.5*1.5) = 1.625
    #      advs[0] = 1.625
    
    assert torch.allclose(advs[0, 0], torch.tensor(1.625))
    assert torch.allclose(advs[0, 1], torch.tensor(1.5))
    assert torch.allclose(rets, advs + values)

def test_gae_mask_behavior():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 1, 4
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    
    rets, advs = ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)
    
    # Masked tokens should have zero advantage
    assert (advs[0, 2:] == 0.0).all()

def test_gae_nan_failure():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.9, tau=0.5)
    rewards = torch.tensor([[1.0, float('nan')]])
    values = torch.tensor([[0.5, 0.5]])
    done = torch.tensor([[0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]])
    
    with pytest.raises(ValueError, match="rewards or values contain NaN"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_gae_mask_holes():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.9, tau=0.5)
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    done = torch.zeros_like(rewards)
    mask = torch.tensor([[1.0, 0.0, 1.0]]) 
    
    with pytest.raises(ValueError, match="mask has non-contiguous valid regions"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)
