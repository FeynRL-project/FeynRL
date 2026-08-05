'''
    DAPO feature tests: soft overlong punishment and degenerate-group flagging
    (rollouts/base.py), dynamic-sampling drop/restore in the replay buffer,
    and the config validation matrix for both engines (sync + async/overlap).
'''
import os
import sys
import copy
import yaml
import torch
import pytest
from unittest.mock import MagicMock

# rollouts.base imports vllm for SamplingParams (unused in the methods under
# test); stub it when vllm isn't installed. setdefault keeps the real module
# when it is.
sys.modules.setdefault("vllm", MagicMock())

from rollouts.base import Base
from rollouts.replay_buffer import ReplayBuffer
from configs.load import load_and_verify


#################
# Base: compute_overlong_penalty + normalize_rewards
#################

class FakeEngine(Base):
    '''Minimal attribute surface both rollout engines expose to
    Base.normalize_rewards / compute_overlong_penalty.'''
    def __init__(self, overlong_buffer_tokens=0, overlong_penalty_factor=1.0,
                 reward_broadcast=True, max_tokens=512):
        self.max_tokens = max_tokens
        self.overlong_buffer_tokens = int(overlong_buffer_tokens)
        self.overlong_penalty_factor = float(overlong_penalty_factor)
        self.reward_broadcast = bool(reward_broadcast)


def make_sample(prompt_len, response_len, reward):
    T = prompt_len + response_len
    token_rewards = torch.zeros(T)
    token_rewards[-1] = reward
    pred_rewards = torch.zeros(T)
    pred_rewards[-2] = reward
    return {"token_rewards": token_rewards,
            "pred_rewards": pred_rewards,
            "token_zscores": token_rewards.clone(),
            "pred_zscores": pred_rewards.clone(),
            "response_len": response_len}


def run_normalize(engine, rewards_and_lens, prompt_len=4):
    samples = [make_sample(prompt_len, L, r) for r, L in rewards_and_lens]
    stats = {"rewards": [r for r, _ in rewards_and_lens],
             "lengths": [L for _, L in rewards_and_lens],
             "correct_threshold": [0.0] * len(rewards_and_lens)}
    engine.normalize_rewards(samples=samples, stats=stats,
                             prompt_len=prompt_len, is_per_token=False)
    return samples


def test_overlong_penalty_piecewise_linear():
    eng = FakeEngine(overlong_buffer_tokens=100, overlong_penalty_factor=1.0, max_tokens=512)
    assert eng.compute_overlong_penalty(300) == 0.0                 # below buffer zone
    assert eng.compute_overlong_penalty(412) == 0.0                 # exactly at threshold
    assert abs(eng.compute_overlong_penalty(462) - (-0.5)) < 1e-9   # midpoint
    assert eng.compute_overlong_penalty(512) == -1.0                # full -c at max_tokens
    assert eng.compute_overlong_penalty(600) == -1.0                # capped beyond


def test_overlong_penalty_scales_with_factor():
    eng = FakeEngine(overlong_buffer_tokens=100, overlong_penalty_factor=0.5, max_tokens=512)
    assert abs(eng.compute_overlong_penalty(462) - (-0.25)) < 1e-9
    assert eng.compute_overlong_penalty(512) == -0.5


def test_normalize_rewards_folds_penalty_and_stamps_sample():
    eng = FakeEngine(overlong_buffer_tokens=100)
    s = run_normalize(eng, [(1.0, 462), (1.0, 10)])
    # 1.0 + (-0.5) folded into both alignments; raw penalty stamped
    assert abs(s[0]["token_rewards"][-1].item() - 0.5) < 1e-6
    assert abs(s[0]["pred_rewards"][-2].item() - 0.5) < 1e-6
    assert s[0]["overlong_penalty"] == -0.5
    # unpenalized sample untouched and unstamped
    assert s[1]["token_rewards"][-1].item() == 1.0
    assert "overlong_penalty" not in s[1]


def test_normalize_rewards_zscores_follow_shaped_rewards():
    eng = FakeEngine(overlong_buffer_tokens=100)
    s = run_normalize(eng, [(1.0, 462), (1.0, 10)])
    # penalty differentiates an all-equal group: z-scores are +-, not zero
    assert s[0]["token_zscores"][-1].item() < 0 < s[1]["token_zscores"][-1].item()


def test_degenerate_group_flagging():
    eng = FakeEngine(overlong_buffer_tokens=0)
    # all-equal rewards -> every sample flagged degenerate
    s = run_normalize(eng, [(1.0, 10), (1.0, 20), (1.0, 30)])
    assert all(x["degenerate_prompt_groups"] for x in s)
    # mixed rewards -> no sample flagged
    s = run_normalize(eng, [(1.0, 10), (0.0, 20), (1.0, 30)])
    assert not any(x["degenerate_prompt_groups"] for x in s)


def test_degenerate_flag_uses_shaped_rewards():
    '''An all-equal group whose overlong penalties differ is NOT degenerate:
    the length punishment still carries gradient.'''
    eng = FakeEngine(overlong_buffer_tokens=100)
    s = run_normalize(eng, [(1.0, 462), (1.0, 10)])
    assert not any(x["degenerate_prompt_groups"] for x in s)


def test_disabled_penalty_leaves_rewards_untouched():
    '''overlong_buffer_tokens=0 (the async engine default when not configured)
    must be a strict no-op.'''
    eng = FakeEngine(overlong_buffer_tokens=0)
    s = run_normalize(eng, [(1.0, 462), (1.0, 10)])
    assert s[0]["token_rewards"][-1].item() == 1.0
    assert "overlong_penalty" not in s[0]
    assert all(x["degenerate_prompt_groups"] for x in s)


#################
# ReplayBuffer: dynamic-sampling drop / keep / restore
#################

def buffer_samples(n, degenerate, seq_len=8):
    out = []
    for _ in range(n):
        t = torch.zeros(seq_len)
        out.append({"input_ids": torch.ones(seq_len, dtype=torch.int64),
                    "pred_rewards": t.clone(), "pred_zscores": t.clone(),
                    "pred_masks": torch.ones(seq_len, dtype=torch.int32),
                    "pred_dones": torch.zeros(seq_len, dtype=torch.int32),
                    "pred_old_logprobs": t.clone(),
                    "policy_version": 0, "response_len": 4,
                    "degenerate_prompt_groups": degenerate})
    return out


def test_buffer_flag_off_keeps_degenerate_samples():
    '''run_rl_async constructs the buffer WITHOUT drop_zero_advantage_groups:
    degenerate-flagged samples must be stored, not dropped.'''
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=16, max_size=100)
    rb.add_batch_seqs(buffer_samples(4, degenerate=True))
    assert len(rb) == 4
    assert rb.zero_adv_dropped == 0


def test_buffer_flag_on_drops_degenerate_keeps_informative():
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=16, drop_zero_advantage_groups=True)
    rb.add_batch_seqs(buffer_samples(4, degenerate=True))
    assert len(rb) == 0
    assert rb.zero_adv_dropped == 4
    rb.add_batch_seqs(buffer_samples(2, degenerate=False))
    assert len(rb) == 2
    # spill is discarded once real samples land
    assert rb.zero_adv_spill == []


def test_buffer_restore_zero_advantage_spill():
    '''All-degenerate epoch: restore puts the dropped samples back so the
    training step does not crash on an empty buffer.'''
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=16, drop_zero_advantage_groups=True)
    rb.add_batch_seqs(buffer_samples(3, degenerate=True))
    assert len(rb) == 0
    restored = rb.restore_zero_advantage_spill()
    assert restored == 3
    assert len(rb) == 3
    # the flag must be back on after the restore round-trip
    assert rb.drop_zero_advantage_groups is True


def test_buffer_reset_clears_dynamic_sampling_state():
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=16, drop_zero_advantage_groups=True)
    rb.add_batch_seqs(buffer_samples(2, degenerate=True))
    rb.reset()
    assert rb.zero_adv_dropped == 0
    assert rb.zero_adv_restored == 0
    assert rb.zero_adv_spill == []


#################
# Config validation matrix (sync + async)
#################

BASE_YAML = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "rl_args.yaml")


def deep_set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def load_case(tmp_path, overrides):
    with open(BASE_YAML) as f:
        case = yaml.safe_load(f)
    deep_set(case, "run.checkpoint_dir", str(tmp_path / "ckps"))
    for k, v in overrides.items():
        deep_set(case, k, v)
    path = tmp_path / "case.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(case, f)
    # rank=1 skips the config dump / checkpoint_dir writes
    return load_and_verify(method="rl", input_yaml=str(path), experiment_id="t", rank=1)


ASYNC = {"overlap.enabled": True, "run.weight_sync_method": "nccl"}
DAPO = {"train.alg_name": "dapo", "reward.broadcast": True, "train.normalize_loss": True}


def test_config_dapo_sync_passes(tmp_path):
    cfg = load_case(tmp_path, {**DAPO})
    assert cfg.train.alg_name == "dapo"


def test_config_dapo_async_passes(tmp_path):
    '''DAPO on the async (overlap) engine is supported.'''
    cfg = load_case(tmp_path, {**ASYNC, **DAPO})
    assert cfg.overlap.enabled is True


def test_config_dapo_async_with_overlong_passes(tmp_path):
    cfg = load_case(tmp_path, {**ASYNC, **DAPO, "rollout.overlong_buffer_tokens": 102})
    assert cfg.rollout.overlong_buffer_tokens == 102


def test_config_grpo_async_with_overlong_passes(tmp_path):
    load_case(tmp_path, {**ASYNC, "rollout.overlong_buffer_tokens": 102})


def test_config_dapo_async_explicit_dynamic_sampling_rejected(tmp_path):
    with pytest.raises(ValueError, match="only supported by the sync engine"):
        load_case(tmp_path, {**ASYNC, **DAPO, "rollout.dynamic_sampling": True})


def test_config_dapo_async_auto_dynamic_sampling_stays_off(tmp_path):
    '''dynamic_sampling: null must NOT auto-enable (and then be rejected)
    under overlap; the run must load.'''
    cfg = load_case(tmp_path, {**ASYNC, **DAPO, "rollout.dynamic_sampling": None})
    assert cfg.rollout.dynamic_sampling is None


def test_config_dapo_requires_normalize_loss_both_engines(tmp_path):
    with pytest.raises(ValueError, match="normalize_loss must be True"):
        load_case(tmp_path, {**DAPO, "train.normalize_loss": False})
    with pytest.raises(ValueError, match="normalize_loss must be True"):
        load_case(tmp_path, {**ASYNC, **DAPO, "train.normalize_loss": False})


def test_config_dapo_requires_broadcast_both_engines(tmp_path):
    with pytest.raises(ValueError, match="reward.broadcast=True"):
        load_case(tmp_path, {**DAPO, "reward.broadcast": False})
    with pytest.raises(ValueError, match="reward.broadcast=True"):
        load_case(tmp_path, {**ASYNC, **DAPO, "reward.broadcast": False})


def test_config_dapo_sync_n_samples_1_rejected_async_allowed(tmp_path):
    # sync: auto dynamic sampling needs n_samples > 1
    with pytest.raises(ValueError, match="n_samples > 1"):
        load_case(tmp_path, {**DAPO, "rollout.n_samples": 1})
    # async: dynamic sampling is off, so n_samples=1 loads (same as grpo)
    load_case(tmp_path, {**ASYNC, **DAPO, "rollout.n_samples": 1})


def test_config_dynamic_sampling_rejected_for_ppo(tmp_path):
    with pytest.raises(ValueError, match="disable dynamic_sampling for ppo"):
        load_case(tmp_path, {"train.alg_name": "ppo", "train.tau": 0.95,
                             "train.gamma": 0.99, "model.value_model": "m",
                             "rollout.dynamic_sampling": True})


def test_config_overlong_bounds_apply_in_both_engines(tmp_path):
    for mode in ({}, ASYNC):
        with pytest.raises(ValueError, match="must be < rollout.max_tokens"):
            load_case(tmp_path, {**mode, "rollout.overlong_buffer_tokens": 512})
        with pytest.raises(ValueError, match="overlong_penalty_factor must be > 0"):
            load_case(tmp_path, {**mode, "rollout.overlong_buffer_tokens": 102,
                                 "rollout.overlong_penalty_factor": 0.0})


def test_config_dapo_async_quantization_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="sync rollout engine"):
        load_case(tmp_path, {**ASYNC, **DAPO, "rollout.quantization": "fp8"})