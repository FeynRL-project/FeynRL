import torch
import pytest
from algs.RL.common import COMMON
from types import SimpleNamespace

# COMMON is a normal class
common_logic = COMMON

def test_kl_distance_correctness():
    dummy_self = SimpleNamespace()
    logprobs = torch.tensor([[-0.1, -0.2]])
    ref_logprobs = torch.tensor([[-0.15, -0.25]])
    
    kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs)
    
    # expected kl (per token) = log(pi/pi_ref) + pi_ref/pi - 1
    #    = (log_pi - log_ref) + exp(log_ref - log_pi) - 1
    
    # t=0: (-0.1 - (-0.15)) + exp(-0.15 - (-0.1)) - 1
    #      = 0.05 + exp(-0.05) - 1 = 0.05 + 0.951229 - 1 = 0.001229
    # t=1: (-0.2 - (-0.25)) + exp(-0.25 - (-0.2)) - 1
    #      = 0.05 + exp(-0.05) - 1 = 0.001229
    
    expected = (0.05 + torch.exp(torch.tensor(-0.05)) - 1.0)
    assert torch.allclose(kl, expected, atol=1e-6)

def test_kl_non_negative():
    dummy_self = SimpleNamespace()
    # KL should always be >= 0 for this variant (log(x) + 1/x - 1 >= 0 for x > 0)
    # where x = pi/pi_ref
    for _ in range(5):
        logprobs = torch.randn(2, 4)
        ref_logprobs = torch.randn(2, 4)
        kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs)
        assert (kl >= -1e-7).all()

def test_kl_zero_when_equal():
    dummy_self = SimpleNamespace()
    logprobs = torch.randn(2, 4)
    kl = common_logic.compute_kl_distance(dummy_self, logprobs, logprobs)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7)

def test_kl_shapes():
    dummy_self = SimpleNamespace()
    B, T = 8, 16
    logprobs = torch.randn(B, T)
    ref_logprobs = torch.randn(B, T)
    kl = common_logic.compute_kl_distance(dummy_self, logprobs, ref_logprobs)
    assert kl.shape == (B, T)
