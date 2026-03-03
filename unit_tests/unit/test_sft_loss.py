import torch
import pytest
import numpy as np
from unittest.mock import MagicMock
from algs.SFT.sft import SFT

def test_sft_compute_loss():
    model_engine = MagicMock()
    optimizer = MagicMock()
    
    sft = SFT(model_engine, optimizer, normalize_loss=False)
    
    # B=1, T-1=2, vocab_size=3
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    target_ids = torch.tensor([[0, 1]])
    loss_mask = torch.tensor([[1.0, 1.0]])
    
    loss, loss_sum, num_tokens = sft.compute_loss(logits, target_ids, loss_mask)
    
    # Each token logit is ~10, so logprob is ~0, loss is ~0
    assert np.isclose(loss.item(), 0.0, atol=1e-3)
    assert np.isclose(loss_sum, 0.0, atol=1e-3)
    assert num_tokens == 2.0

def test_sft_compute_loss_with_mask():
    sft = SFT(MagicMock(), MagicMock(), normalize_loss=False)
    
    # One correct, one incorrect but masked
    logits = torch.tensor([[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    target_ids = torch.tensor([[0, 1]]) # second token is wrong
    loss_mask = torch.tensor([[1.0, 0.0]]) # second token masked
    
    loss, loss_sum, num_tokens = sft.compute_loss(logits, target_ids, loss_mask)
    
    # loss should be only from the first token (~0)
    assert np.isclose(loss.item(), 0.0, atol=1e-3)
    assert num_tokens == 1.0

def test_sft_normalize_loss():
    sft = SFT(MagicMock(), MagicMock(), normalize_loss=True)
    
    # loss_sum = S, total_possible_tokens = B * (T-1)
    logits = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) # 2 tokens
    target_ids = torch.tensor([[0, 0]])
    loss_mask = torch.tensor([[1.0, 1.0]])
    
    loss, loss_sum, num_tokens = sft.compute_loss(logits, target_ids, loss_mask)
    
    # tokens = 2. Loss = loss_sum / 2
    assert np.isclose(loss.item(), loss_sum / 2.0)

def test_sft_gradient_flow():
    sft = SFT(MagicMock(), MagicMock())
    
    logits = torch.randn(1, 2, 3, requires_grad=True)
    target_ids = torch.zeros(1, 2, dtype=torch.long)
    loss_mask = torch.ones(1, 2)
    
    loss, loss_sum, num_tokens = sft.compute_loss(logits, target_ids, loss_mask)
    loss.backward()
    
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
