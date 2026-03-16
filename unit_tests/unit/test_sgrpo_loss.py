import torch
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
from algs.SGRPO.sgrpo import SGRPO

def test_sgrpo_loss_clipped_objective():
    sgrpo_logic = SGRPO
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
    advantages = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0]])
    
    loss, metrics = sgrpo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    
    # Loss = - (1.0 * 1.0 + 1.0 * 2.0) / 2 = -1.5
    assert np.isclose(metrics['pi_loss'], -1.5)
    assert np.isclose(metrics['clipfrac'], 0.0)
    assert np.isclose(metrics['approx_kl'], 0.0)

def test_sgrpo_loss_clipping():
    sgrpo_logic = SGRPO
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
    
    loss, metrics = sgrpo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    
    # Ratio = exp(10) >> 1.2
    # Clipped = 1.2 * 1.0 = 1.2
    # Unclipped = exp(10) * 1.0 = VERY LARGE
    # Loss = -min(VERY LARGE, 1.2) = -1.2
    assert np.isclose(metrics['pi_loss'], -1.2)
    assert np.isclose(metrics['clipfrac'], 1.0)

def test_sgrpo_loss_entropy():
    sgrpo_logic = SGRPO
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
    
    loss, metrics = sgrpo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, entropies, None)
    
    # Loss_pi = 0 (adv = 0)
    # Loss_ent = 0.7
    # Loss_total = 0 - 0.1 * 0.7 = -0.07
    assert np.isclose(metrics['loss_total'], -0.07)

def test_sgrpo_loss_gradient_flow():
    sgrpo_logic = SGRPO
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
    
    loss, metrics = sgrpo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)
    loss.backward()
    
    assert logprobs.grad is not None
    assert not torch.isnan(logprobs.grad).any()
    assert logprobs.grad.item() != 0.0

def test_sgrpo_loss_kl_ref():
    sgrpo_logic = SGRPO
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
    
    loss, metrics = sgrpo_logic.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, ref_logprobs)
    
    # kl_ref = (0.2 + 0.4) / 2 = 0.3
    # Loss = 0 (pi) + 0.5 * 0.3 (kl) = 0.15
    assert np.isclose(metrics['kl_ref'], 0.3)
    assert np.isclose(loss.item(), 0.15)
    dummy_self.compute_kl_distance.assert_called_once()