import torch
import pytest
import numpy as np
from unittest.mock import MagicMock
from algs.DPO.dpo import DPO

def _logits_to_logprobs(logits, target_ids):
    '''
        Convert logits [2B, T-1, V] and target_ids [2B, T-1] to per-token
        log-probs [2B, T-1], matching what DPO.forward() produces.
    '''
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    two_B, T_minus_1, V = logits.shape
    neg_lp = ce(logits.float().view(-1, V), target_ids.view(-1))
    return -neg_lp.view(two_B, T_minus_1)

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

    # compute_loss expects logprobs [2B, T-1], not raw logits
    logprobs = _logits_to_logprobs(logits, target_ids)

    # ref_logprobs: all zeros
    ref_logprobs = torch.zeros(2, 2)

    loss_mask = torch.ones(2, 2)

    loss, metrics = dpo.compute_loss(logprobs, ref_logprobs, loss_mask)

    # chosen_logprobs ≈ [~0, ~0] (exp(10) dominates softmax for index 0)
    # rejected_logprobs ≈ [-10, -10] (logit for index 0 is 0, logsumexp ≈ 10)
    # chosen_rewards  = sum([~0, ~0]) / 2 = ~0
    # rejected_rewards = sum([-10, -10]) / 2 = -10
    # loss = -logsigmoid(0.1 * (0 - (-10))) = -logsigmoid(1.0)

    expected_loss = -torch.nn.functional.logsigmoid(torch.tensor(1.0)).item()
    assert np.isclose(loss.item(), expected_loss, atol=1e-3)
    assert metrics['reward_accuracies'] == 1.0

def test_dpo_gradient_flow():
    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()

    dpo = DPO(model_engine, ref_model_engine, optimizer, beta=0.1)

    # Build logprobs that retain gradient through the cross-entropy computation
    logits = torch.randn(2, 2, 3, requires_grad=True)
    target_ids = torch.zeros(2, 2, dtype=torch.long)
    logprobs = _logits_to_logprobs(logits, target_ids)

    ref_logprobs = torch.zeros(2, 2)
    loss_mask = torch.ones(2, 2)

    loss, metrics = dpo.compute_loss(logprobs, ref_logprobs, loss_mask)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
