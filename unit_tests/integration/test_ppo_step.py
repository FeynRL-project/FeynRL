import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from unit_tests.models import TinyModel, TinyValueModel

def test_ppo_integration_step():
    """
    Minimal integration test for a single PPO update step. 
    Verifies that gradients are computed, parameters are updated, and loss is finite.
    """
    torch.manual_seed(42)
    
    # 1. Models and optimizer
    vocab_size = 50
    hidden_dim = 16
    policy_net = TinyModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    value_net = TinyValueModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
    # Target parameter for update check
    orig_policy_param = policy_net.lm_head.weight.detach().clone()
    orig_value_param = value_net.value_head.weight.detach().clone()
    
    optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)
    
    # 2. Mock batch data [B, T]
    B, T = 2, 4
    input_ids = torch.randint(0, vocab_size, (B, T))
    # attn_mask is used for value_forward to find last real token
    attn_mask = torch.ones(B, T)
    old_logprobs = torch.randn(B, T-1)
    mask = torch.ones(B, T-1)
    rewards = torch.randn(B, T-1)
    done = torch.zeros(B, T-1)
    
    # 3. Simulate training flow
    
    # Forward pass through value net
    v_output = value_net(input_ids)
    values_full = v_output.logits.squeeze(-1) # [B, T]
    values = values_full[:, :-1]
    last_v = values_full[:, -1]
    
    # In integration test, we use simplified version of GAE
    # but the requirement says "Assert: Parameters change, Loss is finite, No NaNs"
    
    # Simplified returns and advantages
    advs = torch.randn(B, T-1)
    returns = advs + values.detach()
    
    # Policy update
    optimizer_policy.zero_grad()
    p_output = policy_net(input_ids)
    logits_full = p_output.logits # [B, T, V]
    logits = logits_full[:, :-1, :].contiguous() # [B, T-1, V]
    target_ids = input_ids[:, 1:].contiguous() # [B, T-1]
    
    # Careful with view/reshape
    logprobs = -nn.functional.cross_entropy(
        logits.view(-1, vocab_size), 
        target_ids.view(-1), 
        reduction='none'
    ).reshape(B, T-1)
    
    # Clipped loss
    ratio = torch.exp(logprobs - old_logprobs)
    clip_eps = 0.2
    unclipped = ratio * advs
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advs
    pi_loss = -(torch.minimum(unclipped, clipped) * mask).mean()
    
    assert torch.isfinite(pi_loss), "Policy loss is not finite"
    pi_loss.backward()
    optimizer_policy.step()
    
    # Value update
    optimizer_value.zero_grad()
    v_output_new = value_net(input_ids)
    current_values = v_output_new.logits.squeeze(-1)[:, :-1]
    v_loss = 0.5 * ((current_values - returns)**2 * mask).mean()
    
    assert torch.isfinite(v_loss), "Value loss is not finite"
    v_loss.backward()
    optimizer_value.step()
    
    # 4. Assertions
    new_policy_param = policy_net.lm_head.weight.detach().clone()
    new_value_param = value_net.value_head.weight.detach().clone()
    
    # Weights should change
    assert not torch.allclose(orig_policy_param, new_policy_param), "Policy weights did not change"
    assert not torch.allclose(orig_value_param, new_value_param), "Value weights did not change"
    
    # No NaNs in grads
    for p in policy_net.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN found in policy gradients"
            
    for p in value_net.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN found in value gradients"
