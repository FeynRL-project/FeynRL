import torch
import pytest
import numpy as np
from types import SimpleNamespace

from algs.GRPO.grpo import GRPO
from algs.CISPO.cispo import CISPO
from algs.DAPO.dapo import DAPO
from algs.P3O.p3o import P3O
from algs.P4O.p4o import P4O
from algs.PPO.ppo import PPO

ALGS = [GRPO, CISPO, DAPO, P3O, P4O, PPO]
# P3O has no proximal snapshot; P4O always takes one
PROX_ALGS = [GRPO, CISPO, DAPO, P4O, PPO]

B, T      = 2, 8
NUM_MICRO = 5
GA_STEPS  = 2

# Live action positions per row (columns 1..n of mask[:, :-1]) of each micro-batch, keyed by its tag. The counts
# differ so that a denominator built from the wrong micro-batches (e.g. the pre-shuffle order) gives a wrong value.
LIVE_PER_ROW = {1: 3, 2: 5, 3: 2, 4: 6, 5: 4}

# [B, T-1] positions of the injected NaNs per source, keyed by micro-batch tag
POLICY_NANS = {1: [(r, c) for r in range(B) for c in range(1, LIVE_PER_ROW[1] + 1)],  # every live token: fully sanitized
               2: [(0, 2)],                                                           # one live token
               4: [(1, 0)]}                                                           # outside the mask: never counted
REF_NANS    = {3: [(1, 1)]}
PROX_NANS   = {5: [(0, 3)]}

# per-micro-batch token counts of an emulated second rank (no NaN there), in its own processing order
OTHER_RANK_COUNTS = [7.0, 3.0, 11.0, 5.0, 9.0]

class RecordingEngine:
    '''
        Stand-in for a DeepSpeed engine that records every loss handed to backward,
        so the scale applied to each micro-batch can be checked against the denominator.
    '''
    device = torch.device("cpu")
    gradient_accumulation_steps = GA_STEPS

    def __init__(self):
        self.losses = []
        self.module = SimpleNamespace(named_parameters=lambda: [])

    def train(self): pass
    def eval(self): pass
    def zero_grad(self): pass
    def set_gradient_accumulation_boundary(self, flag): pass
    def step(self): pass

    def backward(self, loss):
        self.losses.append(float(loss.detach()))

def valid(tag):
    return B * LIVE_PER_ROW[tag]

def dropped(tag, ref_nan=False, prox_nan=False):
    '''Live tokens of this micro-batch that sanitization removes from the loss mask.'''
    nans = set(POLICY_NANS.get(tag, []))
    if ref_nan:
        nans |= set(REF_NANS.get(tag, []))

    if prox_nan:
        nans |= set(PROX_NANS.get(tag, []))

    return sum(1 for _, col in nans if 1 <= col <= LIVE_PER_ROW[tag])

def make_micro_batch(tag):
    input_ids = torch.full((B, T), 7, dtype=torch.long)
    # lets the stubbed forwards recognize a micro-batch after train_step shuffles them
    input_ids[0, 0] = tag
    mask = torch.zeros(B, T)
    mask[:, 1:LIVE_PER_ROW[tag] + 1] = 1.0
    return {"input_ids": input_ids,
            "attn_mask": torch.ones(B, T, dtype=torch.long),
            "mask": mask,
            "old_logprobs": torch.full((B, T), -1.0),
            "zscore": torch.ones(B, T),
            "rewards": torch.zeros(B, T),
            "done": torch.zeros(B, T)}

def logprobs_with_nans(positions):
    lp = torch.full((B, T - 1), -0.9)
    for row, col in positions:
        lp[row, col] = float("nan")

    return lp

def make_alg(alg_cls, normalize_loss=True, full_replay=True, ref_nan=False, prox_nan=False, ent_coeff=0.0):
    '''
        Bare algorithm instance (no DeepSpeed init) running the real train_step, loss, sanitizer and
        denominator helpers. Only the forwards and the engines are stubbed, and each micro-batch's
        loss inputs and outputs are recorded in processing order.
    '''
    alg = alg_cls.__new__(alg_cls)
    alg.alg_name = alg_cls.__name__
    alg.seed = 0
    alg.normalize_loss = normalize_loss
    alg.update_only_after_full_replay = full_replay
    alg.clip_low = 0.2
    alg.clip_high = 0.2
    alg.ent_coeff = ent_coeff
    alg.kl_coeff = 0.1 if ref_nan else 0.0
    alg.use_decoupled_loss = prox_nan
    alg.behave_imp_weight_cap = None
    alg.train_steps_per_epoch = 1
    alg.cached_prox_logprobs = None
    alg.cached_prox_nan_masks = None
    alg.policy_engine = RecordingEngine()
    alg.ref_model_engine = object() if ref_nan else None

    alg.order = []
    def policy_forward(input_ids, att_mask, pos_ids, mm_kwargs=None, need_entropy=True):
        tag  = int(input_ids[0, 0])
        nans = list(POLICY_NANS.get(tag, []))
        if torch.is_grad_enabled():
            # the training forward, once per micro-batch in processing order
            alg.order.append(tag)

        elif prox_nan:
            # the prox snapshot runs under no_grad, so a NaN injected only here is a proximal NaN
            nans += PROX_NANS.get(tag, [])

        entropies = None
        if need_entropy and ent_coeff > 0.0:
            # as if a non-finite logits row reached Categorical: NaN entropy where its logprob is NaN
            entropies = torch.full((B, T - 1), 0.7)
            for row, col in POLICY_NANS.get(tag, []):
                entropies[row, col] = float("nan")

        return logprobs_with_nans(nans), entropies, input_ids[:, 1:]

    alg.policy_forward = policy_forward
    alg.ref_forward = lambda input_ids, att_mask, pos_ids, mm_kwargs=None: \
        logprobs_with_nans(REF_NANS.get(int(input_ids[0, 0]), []))

    # (raw loss sum, local token count, metrics dict) per micro-batch; train_step adds nan_tokens to the dict later
    alg.records = []
    def compute_policy_loss(*args, **kwargs):
        loss_total_sum, denom, metrics = alg_cls.compute_policy_loss(alg, *args, **kwargs)
        alg.records.append((float(loss_total_sum.detach()), float(denom), metrics))
        return loss_total_sum, denom, metrics

    alg.compute_policy_loss = compute_policy_loss

    # what each micro-batch hands to the all-masked watchdog; it must be the raw (unclamped) token count
    alg.masked_counts = []
    def check_all_masked(**kwargs):
        alg.masked_counts.append(float(kwargs['masked_token_count']))
        return alg_cls.check_all_masked(alg, **kwargs)

    alg.check_all_masked = check_all_masked

    if alg_cls is PPO:
        alg.value_engine = RecordingEngine()
        alg.calculate_gae = lambda micro_batches: [(torch.zeros(B, T - 1), mb['zscore'][:, :-1].clone())
                                                   for mb in micro_batches]
        alg.value_forward = lambda input_ids, att_mask, pos_ids, mm_kwargs=None: (torch.ones(B, T - 1), torch.zeros(B))
        alg.value_records = []
        def compute_value_loss(*args, **kwargs):
            v_loss_sum, v_denom, metrics = PPO.compute_value_loss(alg, *args, **kwargs)
            alg.value_records.append((float(v_loss_sum.detach()), float(v_denom)))
            return v_loss_sum, v_denom, metrics

        alg.compute_value_loss = compute_value_loss

    return alg

def run_train_step(alg_cls, **kwargs):
    alg = make_alg(alg_cls, **kwargs)
    out = alg_cls.train_step(alg, engine_id=1, micro_batches=[make_micro_batch(tag) for tag in LIVE_PER_ROW])
    return alg, out

def step_denominators(order, full_replay, other_rank_counts=None):
    '''
        The shared denominator Z of each micro-batch's optimizer step, in processing order: valid tokens
        BEFORE sanitization, summed over the step's micro-batches and over the emulated second rank, whose
        counts the all_reduce adds position by position.
    '''
    counts = [float(valid(tag)) for tag in order]
    if other_rank_counts is not None:
        counts = [c + o for c, o in zip(counts, other_rank_counts)]

    if full_replay:
        return [sum(counts)] * len(counts)

    return [sum(counts[(step // GA_STEPS) * GA_STEPS:(step // GA_STEPS + 1) * GA_STEPS]) for step in range(len(counts))]

def local_mean_factor(step, full_replay):
    '''GA rescaling applied on top of the per-micro-batch mean when normalize_loss=False.'''
    if full_replay:
        return GA_STEPS / NUM_MICRO

    remainder = NUM_MICRO % GA_STEPS
    if remainder != 0 and step >= NUM_MICRO - remainder:
        return GA_STEPS / remainder

    return 1.0

def check_train_step(alg, out, normalize_loss, full_replay, ref_nan=False, prox_nan=False,
                     world_size=1, other_rank_counts=None):
    '''
        Checks every micro-batch of one train_step: the loss saw the sanitized mask, nan_tokens counts exactly
        the live tokens it lost, and the loss handed to backward is scaled by dp_scale / Z with Z shared by the
        whole optimizer step (normalize_loss=True), or is the local mean times the GA factor (normalize_loss=False).
    '''
    assert len(alg.order) == NUM_MICRO and len(alg.records) == NUM_MICRO
    assert len(alg.policy_engine.losses) == NUM_MICRO
    denoms   = step_denominators(alg.order, full_replay, other_rank_counts)
    dp_scale = GA_STEPS * world_size

    def scaled(step, loss_sum, local_denom):
        if normalize_loss:
            return loss_sum * dp_scale / denoms[step]

        return loss_sum / local_denom * local_mean_factor(step, full_replay)

    for step, tag in enumerate(alg.order):
        loss_sum, local_denom, metrics = alg.records[step]
        survivors = valid(tag) - dropped(tag, ref_nan, prox_nan)
        assert metrics['nan_tokens'] == dropped(tag, ref_nan, prox_nan)
        assert local_denom == max(survivors, 1)
        assert alg.masked_counts[step] == survivors
        assert (loss_sum == 0.0) == (survivors == 0)
        assert alg.policy_engine.losses[step] == pytest.approx(scaled(step, loss_sum, local_denom), rel=1e-6)

    if isinstance(alg, PPO):
        # the value loss uses the same sanitized mask and the same denominator as the policy loss
        assert len(alg.value_engine.losses) == NUM_MICRO and len(alg.value_records) == NUM_MICRO
        for step, tag in enumerate(alg.order):
            v_loss_sum, v_denom = alg.value_records[step]
            assert v_denom == max(valid(tag) - dropped(tag, ref_nan, prox_nan), 1)
            assert alg.value_engine.losses[step] == pytest.approx(scaled(step, v_loss_sum, v_denom), rel=1e-6)

    assert out['nan_tokens'] == pytest.approx(np.mean([dropped(tag, ref_nan, prox_nan) for tag in alg.order]))

modes    = pytest.mark.parametrize("full_replay", [True, False], ids=["full_replay", "per_group"])
all_algs = pytest.mark.parametrize("alg_cls", ALGS, ids=lambda c: c.__name__)

@modes
@all_algs
def test_every_micro_batch_shares_the_step_denominator(alg_cls, full_replay):
    '''
        With normalize_loss=True every micro-batch divides by the same all-reduced token count of its
        optimizer step, including the one that lost a live token and the one that lost all of them.
        Subtracting a micro-batch's own NaN count (private to this rank and micro-batch) would make its
        surviving tokens weigh more than every other token of the step.
    '''
    alg, out = run_train_step(alg_cls, normalize_loss=True, full_replay=full_replay)
    check_train_step(alg, out, normalize_loss=True, full_replay=full_replay)

@modes
@all_algs
def test_local_mean_path_is_unchanged(alg_cls, full_replay):
    '''
        normalize_loss=False keeps the per-micro-batch mean with GA rescaling and is not touched by the
        shared denominator; nan_tokens is reported on this path as well.
    '''
    alg, out = run_train_step(alg_cls, normalize_loss=False, full_replay=full_replay)
    check_train_step(alg, out, normalize_loss=False, full_replay=full_replay)

@modes
@all_algs
def test_reference_nan_counts_without_moving_the_denominator(alg_cls, full_replay):
    '''A live token lost to a NaN reference logprob counts in nan_tokens and leaves Z alone.'''
    alg, out = run_train_step(alg_cls, full_replay=full_replay, ref_nan=True)
    check_train_step(alg, out, normalize_loss=True, full_replay=full_replay, ref_nan=True)

@modes
@pytest.mark.parametrize("alg_cls", PROX_ALGS, ids=lambda c: c.__name__)
def test_proximal_nan_counts_without_moving_the_denominator(alg_cls, full_replay):
    '''
        A live token lost to a NaN in the proximal snapshot counts in nan_tokens and leaves Z alone.
        The policy NaNs also appear in the snapshot but are counted once.
    '''
    alg, out = run_train_step(alg_cls, full_replay=full_replay, prox_nan=True)
    check_train_step(alg, out, normalize_loss=True, full_replay=full_replay, prox_nan=True)

@modes
@all_algs
def test_nan_entropy_at_sanitized_positions_does_not_poison_the_loss(alg_cls, full_replay):
    '''
        An entropy that is NaN at a masked position (a non-finite logits row that reached Categorical
        without being neutralized) must not poison the loss, since NaN * 0 is NaN. With ent_coeff > 0 the
        entropy term zeroes those positions through the mask (like the KL term), both at a sanitized live
        position and at one that was never in the mask.
    '''
    alg, out = run_train_step(alg_cls, full_replay=full_replay, ent_coeff=0.1)
    assert np.isfinite(alg.policy_engine.losses).all()
    check_train_step(alg, out, normalize_loss=True, full_replay=full_replay)

@pytest.fixture
def second_rank(monkeypatch):
    '''
        Emulates a second data-parallel rank inside one process. The token-count all_reduces add that
        rank's counts (it has no NaN), and any other collective (P3O/P4O's ESS sums) sees a rank with
        identical statistics.
    '''
    def all_reduce(tensor, op=None, group=None, async_op=False):
        if tensor.dim() == 0:
            # compute_global_token_denom
            tensor += sum(OTHER_RANK_COUNTS)

        elif tensor.dtype == torch.float64 and tuple(tensor.shape) == (NUM_MICRO,):
            # compute_per_group_token_denoms: per-micro-batch counts, position by position
            tensor += torch.tensor(OTHER_RANK_COUNTS, dtype=tensor.dtype)

        else:
            tensor *= 2

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

@modes
@all_algs
def test_nan_on_one_rank_leaves_the_global_denominator_alone(alg_cls, full_replay, second_rank):
    '''
        The case the shared denominator exists for: this rank sanitizes tokens while the other rank does
        not. Every micro-batch still divides by the count summed over both ranks, and dp_scale includes
        the world size.
    '''
    alg, out = run_train_step(alg_cls, normalize_loss=True, full_replay=full_replay)
    check_train_step(alg, out, normalize_loss=True, full_replay=full_replay,
                     world_size=2, other_rank_counts=OTHER_RANK_COUNTS)
