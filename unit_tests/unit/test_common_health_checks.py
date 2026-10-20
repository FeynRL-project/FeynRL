import torch
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from algs.RL.common import COMMON


def make_dummy_common(alg_name="MockAlg"):
    '''Create a minimal COMMON-like object for testing.'''
    obj = SimpleNamespace(alg_name=alg_name)
    # real method, needed by every weight scan; a plain param has numel > 0
    # so the ZeRO-3 shard fallback stays inert in these tests
    obj.local_param_shard = COMMON.local_param_shard.__get__(obj)
    return obj


# ─── compute_per_group_token_denoms ───

def test_per_group_denoms_basic():
    '''Basic case: 12 micro-batches, ga_steps=3 → 4 groups.'''
    obj = make_dummy_common()
    # Each micro-batch has 10 valid tokens
    micro_batches = [{'mask': torch.ones(1, 11)} for _ in range(12)]
    ga_denoms, dp_scale = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=3, device='cpu')

    assert len(ga_denoms) == 4
    assert all(d == 30.0 for d in ga_denoms)  # 3 MBs × 10 tokens each
    assert dp_scale == 3  # ga_steps * world_size(1)


def test_per_group_denoms_remainder():
    '''Non-divisible: 7 micro-batches, ga_steps=3 → 3 groups (last has 1 MB).'''
    obj = make_dummy_common()
    micro_batches = [{'mask': torch.ones(1, 11)} for _ in range(7)]
    ga_denoms, dp_scale = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=3, device='cpu')

    assert len(ga_denoms) == 3
    assert ga_denoms[0] == 30.0  # MBs 0,1,2
    assert ga_denoms[1] == 30.0  # MBs 3,4,5
    assert ga_denoms[2] == 10.0  # MB 6 only


def test_per_group_denoms_ga_equals_1():
    '''ga_steps=1 → each micro-batch is its own group.'''
    obj = make_dummy_common()
    micro_batches = [{'mask': torch.ones(1, 11)} for _ in range(5)]
    ga_denoms, dp_scale = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=1, device='cpu')

    assert len(ga_denoms) == 5
    assert all(d == 10.0 for d in ga_denoms)
    assert dp_scale == 1


def test_per_group_denoms_ga_exceeds_num_micro():
    '''ga_steps > num_micro → single group containing all MBs.'''
    obj = make_dummy_common()
    micro_batches = [{'mask': torch.ones(1, 11)} for _ in range(3)]
    ga_denoms, dp_scale = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=100, device='cpu')

    assert len(ga_denoms) == 1
    assert ga_denoms[0] == 30.0


def test_per_group_denoms_variable_token_counts():
    '''Micro-batches with different numbers of valid tokens.'''
    obj = make_dummy_common()
    # mask[:, :-1] is used, so mask shape [B, T] → valid tokens = (mask[:, :-1] > 0.5).sum()
    micro_batches = [
        {'mask': torch.ones(1, 101)},   # 100 valid tokens
        {'mask': torch.ones(1, 51)},    # 50 valid tokens
        {'mask': torch.ones(1, 201)},   # 200 valid tokens
        {'mask': torch.ones(1, 11)},    # 10 valid tokens
    ]
    ga_denoms, dp_scale = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=2, device='cpu')

    assert len(ga_denoms) == 2
    assert ga_denoms[0] == 150.0  # 100 + 50
    assert ga_denoms[1] == 210.0  # 200 + 10


def test_per_group_denoms_empty_masks():
    '''Micro-batches with all-zero masks.'''
    obj = make_dummy_common()
    micro_batches = [
        {'mask': torch.zeros(1, 11)},
        {'mask': torch.zeros(1, 11)},
    ]
    ga_denoms, dp_scale = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=2, device='cpu')

    assert len(ga_denoms) == 1
    assert ga_denoms[0] == 1.0  # max(0, 1.0) prevents division by zero


def test_per_group_index_within_bounds():
    '''Verify group_idx = step // ga_steps always indexes within ga_denoms.'''
    obj = make_dummy_common()
    num_micro = 7
    ga_steps = 3
    micro_batches = [{'mask': torch.ones(1, 11)} for _ in range(num_micro)]
    ga_denoms, _ = COMMON.compute_per_group_token_denoms(
        obj, micro_batches=micro_batches, ga_steps=ga_steps, device='cpu')

    for step in range(num_micro):
        group_idx = step // ga_steps
        assert group_idx < len(ga_denoms), \
            f"step={step}, group_idx={group_idx} out of bounds for {len(ga_denoms)} groups"


# ─── compute_global_token_denom ───

def test_global_denom_basic():
    '''Basic global token count.'''
    obj = make_dummy_common()
    micro_batches = [{'mask': torch.ones(1, 11)} for _ in range(4)]
    ga_denom, dp_scale = COMMON.compute_global_token_denom(
        obj, micro_batches=micro_batches, ga_steps=1, device='cpu')

    assert ga_denom == 40.0  # 4 MBs × 10 tokens
    assert dp_scale == 1


def test_global_denom_raises_on_zero():
    '''Should raise if all masks are zero.'''
    obj = make_dummy_common()
    micro_batches = [{'mask': torch.zeros(1, 11)} for _ in range(2)]
    with pytest.raises(RuntimeError, match="Invalid global token denominator"):
        COMMON.compute_global_token_denom(
            obj, micro_batches=micro_batches, ga_steps=1, device='cpu')


# ─── check_weights_health ───

def test_check_weights_health_clean():
    '''No NaN in weights → no exception.'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([3.0, 4.0], requires_grad=True)
    module.named_parameters.return_value = [('w1', p1), ('w2', p2)]
    obj.policy_engine = SimpleNamespace(module=module)

    # Should not raise
    COMMON.check_weights_health(obj, engine_id=0, location="TEST")


def test_check_weights_health_nan():
    '''NaN in weights → RuntimeError.'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([1.0, float('nan')], requires_grad=True)
    module.named_parameters.return_value = [('w1', p1)]
    obj.policy_engine = SimpleNamespace(module=module)

    with pytest.raises(RuntimeError, match="contain NaN TEST"):
        COMMON.check_weights_health(obj, engine_id=0, location="TEST")


def test_check_weights_health_skips_frozen():
    '''Frozen params (requires_grad=False) should be skipped.'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([float('nan')], requires_grad=False)  # frozen, has NaN
    p2 = torch.tensor([1.0], requires_grad=True)             # trainable, clean
    module.named_parameters.return_value = [('frozen', p1), ('trainable', p2)]
    obj.policy_engine = SimpleNamespace(module=module)

    # Should not raise — NaN is in frozen param
    COMMON.check_weights_health(obj, engine_id=0, location="TEST")


def test_check_weights_health_skips_empty():
    '''Empty params (numel=0) should be skipped (ZeRO-3 shards).'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([], requires_grad=True)  # empty shard
    module.named_parameters.return_value = [('empty_shard', p1)]
    obj.policy_engine = SimpleNamespace(module=module)

    # Should not raise
    COMMON.check_weights_health(obj, engine_id=0, location="TEST")


def test_weight_scan_checks_zero3_shard():
    '''A ZeRO-3-style param (numel 0 outside gather, NaN in its own slice) must
       be caught; before the shard lookup these scans were vacuous.'''
    obj = make_dummy_common()
    p = torch.nn.Parameter(torch.empty(0))
    p.ds_tensor = torch.tensor([float("nan"), 1.0])
    p.ds_numel = 2                       # both slots are real data
    module = SimpleNamespace(named_parameters=lambda: [("w", p)])
    obj.policy_engine = SimpleNamespace(module=module)
    with pytest.raises(RuntimeError):
        COMMON.check_weights_health(obj, engine_id=0, location="test")


def test_weight_scan_ignores_zero3_alignment_padding():
    '''DeepSpeed allocates each slice with torch.empty and pads the parameter to
       a multiple of the partition count, so the tail of the last rank's slice is
       never written. Scanning it would abort a healthy run.'''
    obj = make_dummy_common()
    p = torch.nn.Parameter(torch.empty(0))
    # slice of 4, but only 3 elements are real; the 4th is uninitialised padding
    p.ds_tensor = torch.tensor([1.0, 2.0, 3.0, float("nan")])
    p.ds_numel = 3
    module = SimpleNamespace(named_parameters=lambda: [("w", p)])
    obj.policy_engine = SimpleNamespace(module=module)
    COMMON.check_weights_health(obj, engine_id=0, location="test")   # must NOT raise

    assert COMMON.local_param_shard(obj, p).numel() == 3


# ─── check_all_masked ───

def test_check_all_masked_valid_tokens():
    '''Non-zero denom → counter resets to 0.'''
    obj = make_dummy_common()
    denom = torch.tensor(100.0)
    result = COMMON.check_all_masked(
        obj, engine_id=0, step=0, num_micro=10, masked_token_count=denom,
        consecutive_nan_steps=5)
    assert result == 0


def test_check_all_masked_increments():
    '''Zero denom → counter increments.'''
    obj = make_dummy_common()
    denom = torch.tensor(0.0)
    result = COMMON.check_all_masked(
        obj, engine_id=1, step=0, num_micro=10, masked_token_count=denom,
        consecutive_nan_steps=3)
    assert result == 4


def test_check_all_masked_aborts_at_threshold():
    '''Reaching threshold → RuntimeError.'''
    obj = make_dummy_common()
    denom = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="model has collapsed"):
        COMMON.check_all_masked(
            obj, engine_id=0, step=0, num_micro=10, masked_token_count=denom,
            consecutive_nan_steps=19, nan_abort_threshold=20)


def test_check_all_masked_below_threshold():
    '''Just below threshold → no exception.'''
    obj = make_dummy_common()
    denom = torch.tensor(0.0)
    result = COMMON.check_all_masked(
        obj, engine_id=1, step=0, num_micro=10, masked_token_count=denom,
        consecutive_nan_steps=18, nan_abort_threshold=20)
    assert result == 19


def test_check_all_masked_takes_raw_count_not_clamped_denom():
    '''The denom compute_policy_loss returns is clamped to min 1.0; passing it
       here would make the guard dead code. The guard fires on the raw count.'''
    obj = make_dummy_common()
    clamped = torch.tensor(0.0).clamp(min=1.0)  # what the old code passed
    raw = torch.tensor(0.0)                     # what the callers pass now
    assert COMMON.check_all_masked(obj, engine_id=0, step=0, num_micro=1,
                                   masked_token_count=clamped,
                                   consecutive_nan_steps=0) == 0
    assert COMMON.check_all_masked(obj, engine_id=0, step=0, num_micro=1,
                                   masked_token_count=raw,
                                   consecutive_nan_steps=0) == 1


# ─── sanitize_logprobs ───

def test_sanitize_logprobs_clean():
    '''No NaN/Inf → logprobs unchanged, mask all False.'''
    obj = make_dummy_common()
    logprobs = torch.tensor([[-1.0, -2.0, -0.5]])
    result, nan_mask = COMMON.sanitize_logprobs(
        obj, logprobs=logprobs, engine_id=0, step=0, num_micro=10)

    assert torch.equal(result, logprobs)
    assert nan_mask.sum().item() == 0


def test_sanitize_logprobs_nan():
    '''NaN → replaced with 1.0, mask marks positions.'''
    obj = make_dummy_common()
    logprobs = torch.tensor([[-1.0, float('nan'), -0.5]])
    result, nan_mask = COMMON.sanitize_logprobs(
        obj, logprobs=logprobs, engine_id=0, step=0, num_micro=10)

    assert result[0, 1].item() == 1.0  # sentinel
    assert nan_mask[0, 1].item() == True
    assert nan_mask.sum().item() == 1


def test_sanitize_logprobs_inf():
    '''Inf → replaced with 1.0, mask marks positions.'''
    obj = make_dummy_common()
    logprobs = torch.tensor([[float('inf'), -2.0, float('-inf')]])
    result, nan_mask = COMMON.sanitize_logprobs(
        obj, logprobs=logprobs, engine_id=0, step=0, num_micro=10)

    assert result[0, 0].item() == 1.0
    assert result[0, 2].item() == 1.0
    assert nan_mask.sum().item() == 2


# ─── check_logit_health ───

def test_check_logit_health_clean():
    '''Clean logits → no exception.'''
    obj = make_dummy_common()
    logits = torch.randn(2, 10, 100)  # [B, T-1, V]
    # Should not raise
    COMMON.check_logit_health(obj, logits)


def test_check_logit_health_nan_with_corrupted_weights():
    '''NaN in logits + NaN in weights → RuntimeError (fatal).'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([float('nan')], requires_grad=True)
    module.named_parameters.return_value = [('bad_param', p1)]
    obj.policy_engine = SimpleNamespace(module=module)

    logits = torch.randn(2, 10, 100)
    logits[0, 5, 50] = float('nan')

    with pytest.raises(RuntimeError, match="NaN in logits AND weights"):
        COMMON.check_logit_health(obj, logits)


def test_check_logit_health_inf_with_clean_weights():
    '''Inf in logits + clean weights → not fatal (bf16 overflow), no exception.'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    module.named_parameters.return_value = [('clean_param', p1)]
    obj.policy_engine = SimpleNamespace(module=module)

    logits = torch.randn(2, 10, 100)
    logits[1, 3, 0] = float('inf')

    # Should NOT raise — weights are clean, so this is a plain bf16 overflow (reported downstream)
    COMMON.check_logit_health(obj, logits)


def test_check_logit_health_nan_with_clean_weights():
    '''NaN in logits + clean weights → not fatal.'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([1.0], requires_grad=True)
    module.named_parameters.return_value = [('clean_param', p1)]
    obj.policy_engine = SimpleNamespace(module=module)

    logits = torch.randn(2, 10, 100)
    logits[0, 5, 50] = float('nan')

    # Should NOT raise — weights are clean, so it's bf16 overflow
    COMMON.check_logit_health(obj, logits)


# ─── DeepSpeed config defaults ───

def test_deepspeed_config_defaults():
    '''Verify new DeepSpeed config fields have correct defaults.'''
    from configs.load import DeepSpeed as DSConfig

    ds = DSConfig(zero_optimization={"stage": 3})
    dumped = ds.model_dump()

    assert dumped['data_types'] == {"grad_accum_dtype": "fp32"}
    assert dumped['prescale_gradients'] == False


def test_deepspeed_config_override():
    '''Verify DeepSpeed config fields can be overridden.'''
    from configs.load import DeepSpeed as DSConfig

    ds = DSConfig(
        zero_optimization={"stage": 2},
        data_types={"grad_accum_dtype": "bf16"},
        prescale_gradients=True,
    )
    dumped = ds.model_dump()

    assert dumped['data_types'] == {"grad_accum_dtype": "bf16"}
    assert dumped['prescale_gradients'] == True


# ─── policy_forward need_entropy ───

def test_policy_forward_need_entropy_false_skips_entropy():
    '''need_entropy=False returns no entropies even when ent_coeff > 0.'''
    obj = make_dummy_common()
    obj.ent_coeff = 0.1
    logits = torch.randn(1, 4, 5)
    obj.policy_engine = lambda input_ids, attention_mask, position_ids=None, use_cache=False, **kw: SimpleNamespace(logits=logits)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    att_mask  = torch.ones(1, 4, dtype=torch.long)

    assert COMMON.policy_forward(obj, input_ids, att_mask, None)[1] is not None
    assert COMMON.policy_forward(obj, input_ids, att_mask, None, need_entropy=False)[1] is None


def test_prox_snapshot_does_not_request_entropy():
    '''The no-grad proximal snapshot never uses entropies, so it asks policy_forward to skip them.'''
    obj = make_dummy_common()
    requested = []
    def policy_forward(input_ids, att_mask, pos_ids, mm_kwargs=None, need_entropy=True):
        requested.append(need_entropy)
        return torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1), None, input_ids[:, 1:]

    obj.policy_forward = policy_forward
    obj.extract_mm_kwargs = lambda micro_batch, device: {}
    obj.sanitize_logprobs = COMMON.sanitize_logprobs.__get__(obj)
    mb = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attn_mask": torch.ones(2, 4, dtype=torch.long)}
    COMMON.snapshot_prox_logprobs(obj, micro_batches=[mb, mb], engine_id=0, device=torch.device("cpu"))

    assert requested == [False, False]


# ─── policy_forward neutralizes non-finite logit rows ───

class _LogitsEngine:
    '''Stand-in policy engine: returns fixed logits, exposes clean params for check_logit_health.'''
    def __init__(self, logits):
        self._logits = logits
        self.module = SimpleNamespace(named_parameters=lambda: [("w", torch.nn.Parameter(torch.ones(2)))])

    def __call__(self, input_ids, attention_mask, position_ids=None, use_cache=False, **kwargs):
        return SimpleNamespace(logits=self._logits)


def _policy_forward_on(logits, ent_coeff=0.0):
    obj = make_dummy_common()
    obj.ent_coeff = ent_coeff
    obj.policy_engine = _LogitsEngine(logits)
    obj.check_logit_health = COMMON.check_logit_health.__get__(obj)
    B, T, _ = logits.shape
    input_ids = torch.zeros(B, T, dtype=torch.long)          # targets are all token 0
    att_mask  = torch.ones(B, T, dtype=torch.long)
    return COMMON.policy_forward(obj, input_ids, att_mask, None)


def test_policy_forward_clean_logits_unchanged():
    '''No bad row: logprobs are the plain log_softmax + gather, nothing else happens.'''
    logits = torch.randn(2, 5, 7)
    logprobs, entropies, target_ids = _policy_forward_on(logits)
    ref = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)[..., 0]     # target token 0
    assert torch.allclose(logprobs, ref, atol=1e-6)
    assert entropies is None and target_ids.shape == (2, 4)


def test_policy_forward_neutralizes_non_finite_logit_rows():
    '''A +inf (bf16 overflow) or NaN logit gives a finite masked loss but a NaN gradient
       through the log_softmax backward, and a NaN Categorical entropy (torch.compile turns
       distribution validation off). policy_forward must zero those rows first and hand them
       to sanitize_logprobs as NaN logprobs.'''
    base   = torch.randn(1, 4, 6, requires_grad=True)
    inject = torch.zeros(1, 4, 6)
    inject[0, 1, 2] = float("inf")            # row 1 (after the [:, :-1] slice): overflow
    inject[0, 2, 0] = float("nan")            # row 2: NaN
    logprobs, entropies, _ = _policy_forward_on(base + inject, ent_coeff=0.1)

    # bad rows come back as NaN (the channel sanitize_logprobs reads); the clean row is finite
    assert torch.isfinite(logprobs[0, 0])
    assert torch.isnan(logprobs[0, 1]) and torch.isnan(logprobs[0, 2])
    # entropy was computed (ent_coeff > 0) from the neutralized rows, and is finite
    assert entropies is not None and torch.isfinite(entropies).all()

    # the masked loss, built the way the algorithms build it, has a FINITE gradient
    nan_mask = ~torch.isfinite(logprobs)
    mask     = (~nan_mask).float()
    loss = -(logprobs.masked_fill(nan_mask, 1.0) * mask).sum() - 0.1 * (entropies * mask).sum()
    loss.backward()
    assert torch.isfinite(base.grad).all()
    assert torch.count_nonzero(base.grad[0, 1]) == 0 and torch.count_nonzero(base.grad[0, 2]) == 0


def test_policy_forward_neutralizes_all_neg_inf_row_only():
    '''A row that is entirely -inf has a -inf max and NaN log_softmax, so it is neutralized like
       the others. A row with some -inf entries keeps a finite max and is left alone.'''
    base   = torch.randn(1, 4, 6, requires_grad=True)
    inject = torch.zeros(1, 4, 6)
    inject[0, 1, :] = float("-inf")           # row 1: entirely -inf
    inject[0, 2, 3] = float("-inf")           # row 2: one -inf entry, target token 0 is finite
    logprobs, entropies, _ = _policy_forward_on(base + inject, ent_coeff=0.1)

    assert torch.isnan(logprobs[0, 1])
    assert torch.isfinite(logprobs[0, [0, 2]]).all()
    assert torch.isfinite(entropies).all()

    nan_mask = ~torch.isfinite(logprobs)
    mask     = (~nan_mask).float()
    loss = -(logprobs.masked_fill(nan_mask, 1.0) * mask).sum() - 0.1 * (entropies * mask).sum()
    loss.backward()
    assert torch.isfinite(base.grad).all()
    assert torch.count_nonzero(base.grad[0, 1]) == 0


def test_policy_forward_bad_row_with_nan_weights_is_fatal():
    '''check_logit_health is wired in: a bad row while the weights are NaN must abort.'''
    logits = torch.randn(1, 3, 5)
    logits[0, 0, 1] = float("inf")
    obj = make_dummy_common()
    obj.ent_coeff = 0.0
    obj.policy_engine = _LogitsEngine(logits)
    obj.policy_engine.module = SimpleNamespace(
        named_parameters=lambda: [("w", torch.nn.Parameter(torch.tensor([float("nan"), 1.0])))])
    obj.check_logit_health = COMMON.check_logit_health.__get__(obj)
    with pytest.raises(RuntimeError):
        COMMON.policy_forward(obj, torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long), None)


def test_policy_forward_neutralizes_rows_in_a_batch_copy():
    '''B > 1 makes the [:, :-1] slice non-contiguous, so logits is a fresh copy and the
       in-place fill lands on it; the clean sample in the same batch must be untouched.'''
    base   = torch.randn(2, 4, 6, requires_grad=True)
    inject = torch.zeros(2, 4, 6)
    inject[0, 1, 2] = float("inf")            # sample 0, row 1: overflow
    logprobs, _, _ = _policy_forward_on(base + inject)

    assert torch.isnan(logprobs[0, 1])
    assert torch.isfinite(logprobs[0, [0, 2]]).all() and torch.isfinite(logprobs[1]).all()
    # sample 1 is exactly what a plain forward on its own logits gives
    ref = torch.log_softmax(base[1:, :-1, :].detach().float(), dim=-1)[..., 0]
    assert torch.allclose(logprobs[1], ref, atol=1e-6)

    nan_mask = ~torch.isfinite(logprobs)
    loss = -(logprobs.masked_fill(nan_mask, 1.0) * (~nan_mask).float()).sum()
    loss.backward()
    assert torch.isfinite(base.grad).all()
    assert torch.count_nonzero(base.grad[0, 1]) == 0


def test_check_logit_health_returns_bad_row_count():
    '''The guard is silent on clean weights and hands the count back to the caller.'''
    obj = make_dummy_common()
    obj.policy_engine = SimpleNamespace(module=SimpleNamespace(
        named_parameters=lambda: [("w", torch.nn.Parameter(torch.ones(2)))]))
    logits = torch.randn(2, 5, 4)
    assert COMMON.check_logit_health(obj, logits) == 0
    logits[0, 1, 0] = float("inf")
    logits[1, 3, 2] = float("nan")
    assert COMMON.check_logit_health(obj, logits) == 2
