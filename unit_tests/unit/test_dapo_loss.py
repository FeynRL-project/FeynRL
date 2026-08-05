import torch
import numpy as np
from types import SimpleNamespace
from algs.DAPO.dapo import DAPO

def make_self(clip_low=0.2, clip_high=0.28, use_decoupled_loss=False, cap=None):
    '''DAPO's paper setting: asymmetric clip (clip-higher), no entropy, no KL.'''
    return SimpleNamespace(
        clip_low=clip_low,
        clip_high=clip_high,
        ent_coeff=0.0,
        kl_coeff=0.0,
        use_decoupled_loss=use_decoupled_loss,
        behave_imp_weight_cap=cap,
        alg_name="Mock",
    )

def test_dapo_loss_unclipped_objective():
    dummy_self = make_self()

    logprobs = torch.tensor([[-0.1, -0.2]])
    old_logprobs = torch.tensor([[-0.1, -0.2]])
    # Ratio = exp(0) = 1.0
    advantages = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0]])

    loss, denom, metrics = DAPO.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)

    # Loss = - (1.0 * 1.0 + 1.0 * 2.0) / 2 = -1.5
    assert np.isclose(metrics['pi_loss'], -1.5)
    assert np.isclose(metrics['clipfrac'], 0.0)
    assert np.isclose(metrics['approx_kl'], 0.0)

def test_dapo_clip_higher_gives_exploration_headroom():
    '''ratio = 1.25 sits ABOVE GRPO's symmetric 1.2 ceiling but BELOW DAPO's
    1.28: with clip-higher the unclipped term survives and nothing is flagged
    clipped. This is the defining behavioral difference vs. clip_high=0.2.'''
    dummy_self = make_self(clip_low=0.2, clip_high=0.28)

    ratio = 1.25
    logprobs = torch.tensor([[float(np.log(ratio))]])
    old_logprobs = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])

    loss, denom, metrics = DAPO.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)

    assert np.isclose(metrics['pi_loss'], -1.25, atol=1e-6)
    assert np.isclose(metrics['clipfrac'], 0.0)

    # sanity: the same ratio under a symmetric 0.2 clip IS clipped to 1.2
    sym_self = make_self(clip_low=0.2, clip_high=0.2)
    _, _, sym_metrics = DAPO.compute_policy_loss(sym_self, logprobs, old_logprobs, advantages, mask, None, None)
    assert np.isclose(sym_metrics['pi_loss'], -1.2, atol=1e-6)
    assert np.isclose(sym_metrics['clipfrac'], 1.0)

def test_dapo_loss_upper_ceiling():
    dummy_self = make_self()

    logprobs = torch.tensor([[10.0]])  # Ratio = exp(10) >> 1.28
    old_logprobs = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])

    loss, denom, metrics = DAPO.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)

    # Clipped at 1 + clip_high = 1.28
    assert np.isclose(metrics['pi_loss'], -1.28)
    assert np.isclose(metrics['clipfrac'], 1.0)

def test_dapo_loss_downside_clip_stays_tight():
    '''clip-higher only raises the ceiling; the downside clip stays at
    1 - clip_low = 0.8.'''
    dummy_self = make_self()

    logprobs = torch.tensor([[-10.0]])  # Ratio = exp(-10) ~ 0
    old_logprobs = torch.tensor([[0.0]])
    advantages = torch.tensor([[-1.0]])
    mask = torch.tensor([[1.0]])

    loss, denom, metrics = DAPO.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)

    # unclipped = ratio * (-1) ~ 0; clipped = 0.8 * (-1) = -0.8
    # min(~0, -0.8) = -0.8 -> loss = +0.8
    assert np.isclose(metrics['pi_loss'], 0.8, atol=1e-4)
    assert np.isclose(metrics['clipfrac'], 1.0)

def test_dapo_loss_mask_zeroes_tokens():
    dummy_self = make_self()

    logprobs = torch.tensor([[0.0, 10.0]])
    old_logprobs = torch.tensor([[0.0, 0.0]])
    advantages = torch.tensor([[1.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0]])  # second token masked out

    loss, denom, metrics = DAPO.compute_policy_loss(dummy_self, logprobs, old_logprobs, advantages, mask, None, None)

    # only the first token contributes: -1.0 / denom(=1) = -1.0
    assert np.isclose(metrics['pi_loss'], -1.0)
    assert denom.item() == 1.0

def test_dapo_decoupled_loss_matches_standard_when_prox_equals_behav():
    '''Async (overlap) mode trains DAPO with the decoupled loss. When
    pi_prox == pi_behav the behavioral weight is 1 and the result must equal
    the standard clipped objective.'''
    std = make_self()
    dec = make_self(use_decoupled_loss=True, cap=None)

    logprobs = torch.tensor([[0.1, -0.3]])
    old_logprobs = torch.tensor([[0.0, 0.0]])
    advantages = torch.tensor([[1.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0]])

    _, _, m_std = DAPO.compute_policy_loss(std, logprobs, old_logprobs, advantages, mask, None, None)
    _, _, m_dec = DAPO.compute_policy_loss(dec, logprobs, old_logprobs, advantages, mask, None, None,
                                           prox_logprobs=old_logprobs)

    assert np.isclose(m_std['pi_loss'], m_dec['pi_loss'], atol=1e-6)
    assert np.isclose(m_dec['behave_w_mean'], 1.0, atol=1e-6)

def test_dapo_decoupled_loss_behavioral_cap():
    '''behave_imp_weight_cap must clamp pi_prox/pi_behav (stale async data).'''
    dec = make_self(use_decoupled_loss=True, cap=2.0)

    # pi == pi_prox (r_prox = 1), pi_prox/pi_behav = exp(3) >> cap
    prox = torch.tensor([[0.0]])
    logprobs = torch.tensor([[0.0]])
    old_logprobs = torch.tensor([[-3.0]])
    advantages = torch.tensor([[1.0]])
    mask = torch.tensor([[1.0]])

    loss, denom, metrics = DAPO.compute_policy_loss(dec, logprobs, old_logprobs, advantages, mask, None, None,
                                                    prox_logprobs=prox)

    # r_prox = 1 -> min(1*A, clip(1)*A) = 1; weighted by capped w = 2.0
    assert np.isclose(metrics['pi_loss'], -2.0, atol=1e-6)
    assert np.isclose(metrics['behave_w_max'], 2.0, atol=1e-6)
    assert np.isclose(metrics['behave_w_capfrac'], 1.0)