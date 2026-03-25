import torch
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
from algs.PPO.ppo import PPO

def test_ppo_loss_clipped_objective():
    # Use PPO directly since conftest.py mocks @ray.remote to be a no-op
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[-0.1, -0.2]])
    old_logprobs = torch.tensor([[-0.1, -0.2]])
    # Ratio = exp(0) = 1.0
    # Advantages must be pre-normalized.
    # Mean-zero advantages with ratio=1 give pi_loss=0.
    advantages = torch.tensor([[-1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    # codebase logic
    loss, denom, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)

    assert np.isclose(metrics['pi_loss'], 0.0, atol=1e-6)
    assert np.isclose(metrics['clipfrac'], 0.0)
    assert np.isclose(metrics['approx_kl'], 0.0)

def test_ppo_loss_clipping():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[10.0]]) # Ratio will be exp(10)
    old_logprobs = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])
    
    loss, denom, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    
    # Single-token advantages normalize to zero variance case; loss should stay finite.
    assert np.isfinite(metrics['pi_loss'])
    assert np.isclose(metrics['clipfrac'], 1.0)

def test_ppo_loss_entropy():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.1, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[-0.5]])
    old_logprobs = torch.tensor([[-0.5]])
    advantages = torch.tensor([[0.0]])
    mask = torch.tensor([[1.0]])
    entropies = torch.tensor([[0.7]])
    
    loss, denom, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, entropies, None)
    
    # Loss_pi = 0 (adv = 0)
    # Loss_ent = 0.7
    # Loss_total = 0 - 0.1 * 0.7 = -0.07
    assert np.isclose(metrics['loss_total'], -0.07)

def test_ppo_loss_gradient_flow():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.0,
        alg_name="Mock"
    )
    
    logprobs = torch.tensor([[-0.5]], requires_grad=True)
    old_logprobs = torch.tensor([[-0.5]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])
    
    loss, denom, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    loss.backward()
    
    assert logprobs.grad is not None
    assert not torch.isnan(logprobs.grad).any()

def test_ppo_loss_kl_ref():
    ppo_logic = PPO
    # Mock self with kl_coeff > 0
    dummy_self = SimpleNamespace(
        clip_low=0.2, 
        clip_high=0.2, 
        ent_coeff=0.0, 
        kl_coeff=0.5,
        alg_name="Mock"
    )
    # mock compute_kl_distance on dummy_self
    dummy_self.compute_kl_distance = MagicMock(return_value=torch.tensor([[0.2, 0.4]]))
    
    logprobs = torch.tensor([[-0.1, -0.2]])
    old_logprobs = torch.tensor([[-0.1, -0.2]])
    advantages = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0]])
    ref_logprobs = torch.tensor([[-0.1, -0.2]]) # Values don't matter as kl_dist is mocked
    
    loss, denom, metrics = ppo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, ref_logprobs)

    # kl_ref = (0.2 + 0.4) / 2 = 0.3
    # loss_total_sum (raw sum) = 0.5 * (0.2 + 0.4) = 0.3
    assert np.isclose(metrics['kl_ref'], 0.3)
    assert np.isclose(loss.item(), 0.3)
    dummy_self.compute_kl_distance.assert_called_once()
