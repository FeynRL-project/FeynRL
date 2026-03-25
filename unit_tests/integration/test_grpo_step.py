import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from unit_tests.models import TinyModel

def test_grpo_integration_step():
    '''
        Minimal integration test for a single GRPO update step.
        GRPO uses z-score advantages (no value network) with a clipped policy gradient.
        Verifies that gradients are computed, parameters are updated, and loss is finite.
    '''
    torch.manual_seed(42)

    # 1. Model and optimizer
    vocab_size = 50
    hidden_dim = 16
    policy_net = TinyModel(vocab_size=vocab_size, hidden_dim=hidden_dim)

    orig_policy_param = policy_net.lm_head.weight.detach().clone()
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    # 2. Mock batch data [B, T]
    B, T = 2, 4
    input_ids = torch.randint(0, vocab_size, (B, T))
    old_logprobs = torch.randn(B, T-1)
    mask = torch.ones(B, T-1)
    # GRPO uses z-score normalized rewards as advantages directly
    zscores = torch.randn(B, T-1)

    # 3. Policy update (GRPO style: clipped ratio * advantages, no value net)
    optimizer.zero_grad()
    p_output = policy_net(input_ids)
    logits = p_output.logits[:, :-1, :].contiguous()
    target_ids = input_ids[:, 1:].contiguous()

    logprobs = -nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1),
        reduction='none'
    ).reshape(B, T-1)

    # Clipped loss (same formula as GRPO.compute_policy_loss)
    ratio = torch.exp(logprobs - old_logprobs)
    clip_eps = 0.2
    unclipped = ratio * zscores
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * zscores
    pi_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / mask.sum().clamp(min=1.0)

    assert torch.isfinite(pi_loss), "Policy loss is not finite"
    pi_loss.backward()
    optimizer.step()

    # 4. Assertions
    new_policy_param = policy_net.lm_head.weight.detach().clone()

    # Weights should change
    assert not torch.allclose(orig_policy_param, new_policy_param), "Policy weights did not change"

    # No NaNs in grads
    for p in policy_net.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN found in policy gradients"