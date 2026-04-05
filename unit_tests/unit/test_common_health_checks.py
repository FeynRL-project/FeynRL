import torch
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from algs.RL.common import COMMON


def make_dummy_common(alg_name="MockAlg"):
    '''Create a minimal COMMON-like object for testing.'''
    obj = SimpleNamespace(alg_name=alg_name)
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


# ─── check_all_masked ───

def test_check_all_masked_valid_tokens():
    '''Non-zero denom → counter resets to 0.'''
    obj = make_dummy_common()
    denom = torch.tensor(100.0)
    result = COMMON.check_all_masked(
        obj, engine_id=0, step=0, num_micro=10, local_denom=denom,
        consecutive_nan_steps=5)
    assert result == 0


def test_check_all_masked_increments():
    '''Zero denom → counter increments.'''
    obj = make_dummy_common()
    denom = torch.tensor(0.0)
    result = COMMON.check_all_masked(
        obj, engine_id=1, step=0, num_micro=10, local_denom=denom,
        consecutive_nan_steps=3)
    assert result == 4


def test_check_all_masked_aborts_at_threshold():
    '''Reaching threshold → RuntimeError.'''
    obj = make_dummy_common()
    denom = torch.tensor(0.0)
    with pytest.raises(RuntimeError, match="model has collapsed"):
        COMMON.check_all_masked(
            obj, engine_id=0, step=0, num_micro=10, local_denom=denom,
            consecutive_nan_steps=19, nan_abort_threshold=20)


def test_check_all_masked_below_threshold():
    '''Just below threshold → no exception.'''
    obj = make_dummy_common()
    denom = torch.tensor(0.0)
    result = COMMON.check_all_masked(
        obj, engine_id=1, step=0, num_micro=10, local_denom=denom,
        consecutive_nan_steps=18, nan_abort_threshold=20)
    assert result == 19


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
    '''Inf in logits + clean weights → warning only (bf16 overflow), no exception.'''
    obj = make_dummy_common()
    module = MagicMock()
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    module.named_parameters.return_value = [('clean_param', p1)]
    obj.policy_engine = SimpleNamespace(module=module)

    logits = torch.randn(2, 10, 100)
    logits[1, 3, 0] = float('inf')

    # Should NOT raise — just prints a warning
    COMMON.check_logit_health(obj, logits)


def test_check_logit_health_nan_with_clean_weights():
    '''NaN in logits + clean weights → warning only (not fatal).'''
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
