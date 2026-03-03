import torch
import pytest
import numpy as np
from unittest.mock import MagicMock
from algs.DPO.dpo import DPO

def test_dpo_compute_loss():
    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()
    
    beta = 0.1
    dpo = DPO(model_engine, ref_model_engine, optimizer, beta)
    
    # B = 1, T-1 = 2, vocab_size = 3
    # logits shape: [2B, T-1, vocab_size] = [2, 2, 3]
    logits = torch.tensor([
        [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], # chosen: high logit for index 0
        [[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]], # rejected: high logit for index 2
    ])
    
    # target_ids: index 0 for all
    target_ids = torch.tensor([[0, 0], [0, 0]])
    
    # ref_logprobs: all zeros
    ref_logprobs = torch.zeros(2, 2)
    
    loss_mask = torch.ones(2, 2)
    
    loss, metrics = dpo.compute_loss(logits, ref_logprobs, target_ids, loss_mask)
    
    # chosen_logprobs will be ~0 (since exp(10) is much larger than others)
    # rejected_logprobs will be ~ -10 (logit for index 0 is 0, logsumexp is ~10)
    # chosen_rewards = (0 - 0) / 1 = 0
    # rejected_rewards = (-10 - 0) / 1 = -10
    # loss = -logsigmoid(0.1 * (0 - (-10))) = -logsigmoid(1.0)
    
    expected_loss = -torch.nn.functional.logsigmoid(torch.tensor(1.0)).item()
    assert np.isclose(loss.item(), expected_loss, atol=1e-3)
    assert metrics['reward_accuracies'] == 1.0

def test_dpo_gradient_flow():
    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()
    
    dpo = DPO(model_engine, ref_model_engine, optimizer, beta=0.1)
    
    logits = torch.randn(2, 2, 3, requires_grad=True)
    target_ids = torch.zeros(2, 2, dtype=torch.long)
    ref_logprobs = torch.zeros(2, 2)
    loss_mask = torch.ones(2, 2)
    
    loss, metrics = dpo.compute_loss(logits, ref_logprobs, target_ids, loss_mask)
    loss.backward()
    
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
