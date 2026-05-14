import sys
import torch
import pytest
from unittest.mock import MagicMock, patch
from algs.PPO.ppo import PPO
from rollouts.replay_buffer import ReplayBuffer
from types import SimpleNamespace

def test_shape_mismatch_error():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 2, 4
    # Mismatched rewards shape
    rewards = torch.randn(B, T + 1)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.ones(B, T)
    
    # In ppo.py, it checks if len(all_len) != 1.
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_nan_reward_error():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 2, 4
    rewards = torch.tensor([[1.0, 2.0, float('nan'), 4.0], [1.0, 2.0, 3.0, 4.0]])
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.ones(B, T)
    
    with pytest.raises(ValueError, match="rewards or values contain NaN"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_empty_batch_error():
    # ReplayBuffer is a normal class
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    with pytest.raises(ValueError, match="collate_fn received an empty batch"):
        rb.collate_fn([])

def test_invalid_mask_holes():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 1, 5
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    
    # ppo.py checks for holes (rises & (drops.cumsum(dim=1) > 0)).any()
    with pytest.raises(ValueError, match="mask has non-contiguous valid regions"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)


def _import_shard_and_put():
    """Import shard_and_put while stubbing out the entire heavy import chain."""
    heavy = [
        "datasets", "ray",
        "data_feeds", "data_feeds.prompts", "data_feeds.mixed_sampler",
        "rollouts", "rollouts.vllm_engine", "rollouts.vllm_engine_async",
        "rollouts.base", "rollouts.replay_buffer",
        "misc", "misc.utils", "misc.nccl_env", "misc.rollout_stats",
        "vllm", "transformers",
    ]
    stubs = {k: MagicMock() for k in heavy}
    # Remove any previously cached core.rl_engines so reload picks up stubs
    sys.modules.pop("core.rl_engines", None)
    with patch.dict(sys.modules, stubs):
        import core.rl_engines as mod
        return mod.shard_and_put, mod


def test_shard_and_put_raises_on_size_mismatch():
    """shard_and_put must raise RuntimeError (not print) when shard sizes differ.
    Unequal shards guarantee a ZeRO-3 collective deadlock; hard failure is the only
    safe response."""
    shard_and_put, mod = _import_shard_and_put()
    batches = [object() for _ in range(5)]  # 5 batches, 2 engines → shards of 3 and 2
    fake_ray = MagicMock()
    fake_ray.put.side_effect = lambda x: x

    with patch.object(mod, "ray", fake_ray):
        with pytest.raises(RuntimeError, match="SHARD SIZE MISMATCH"):
            shard_and_put(batches, num_engines=2)


def test_shard_and_put_even_split_does_not_raise():
    """shard_and_put must not raise when every engine gets the same number of batches."""
    shard_and_put, mod = _import_shard_and_put()
    batches = [object() for _ in range(6)]  # 6 batches, 2 engines → 3 each
    fake_ray = MagicMock()
    fake_ray.put.side_effect = lambda x: x

    with patch.object(mod, "ray", fake_ray):
        refs = shard_and_put(batches, num_engines=2)
    assert len(refs) == 2
