import sys
import time
import torch
import pytest
from unittest.mock import MagicMock, patch

# Additional mocks needed for importing main_rl 
# conftest.py already mocks ray/transformers/deepspeed/peft/safetensors.
# main_rl transitively imports vllm, mlflow, wandb, datasets.
_extra_mocks = [
    "vllm", "vllm.sampling_params", "vllm.lora", "vllm.lora.request",
    "vllm.config", "vllm.engine", "vllm.engine.arg_utils",
    "mlflow", "mlflow.tracking",
    "wandb",
    "datasets",
]
for _mod in _extra_mocks:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from main_rl import (
    ChunkFuture,
    chunk_is_ready,
    dispatch_one_chunk,
    finalize_chunk,
    aggregate_chunk_stats,
    prepare_training_batches,
    shard_and_put,
    shard_batch_for_engines,
)
from rollouts.replay_buffer import ReplayBuffer
import misc.rollout_stats as rollout_stats

def _make_replay_buffer(n_samples=20, seq_len=8, max_seq_len=16,
                        pad_token_id=0, policy_version=0):
    """Create a replay buffer pre-populated with n_samples dummy entries."""
    torch.manual_seed(0)
    rb = ReplayBuffer(pad_token_id=pad_token_id, max_seq_len=max_seq_len)
    for _ in range(n_samples):
        rb.add(
            input_ids=torch.randint(1, 100, (seq_len,)),
            rewards=torch.randn(seq_len),
            zscores=torch.randn(seq_len),
            masks=torch.ones(seq_len),
            dones=torch.zeros(seq_len),
            old_logprobs=torch.randn(seq_len),
            policy_version=policy_version,
        )
    return rb

def _make_stats(n=1, reward=1.0, response_len=4, tokens=8):
    """Create a rollout stats dict compatible with rollout_stats.accumulate."""
    return {
        'total_samples_generated': n,
        'all_rewards': [reward] * n,
        'all_zscores': [0.0] * n,
        'all_response_lens': [response_len] * n,
        'min_response_len': response_len,
        'max_response_len': response_len,
        'total_tokens': tokens * n,
        'total_truncated': 0,
        'total_eos': n,
        'total_finish_stop': n,
        'total_prompt_len': 4 * n,
        'prompt_response_groups': {},
        'total_logprob_sum': -1.0 * n,
        'total_logprob_tokens': response_len * n,
    }

def _make_sample(seq_len=8, policy_version=0):
    """Create a single rollout sample compatible with ReplayBuffer.add_batch_seqs."""
    return {
        "response_len": seq_len // 2,
        "input_ids": torch.randint(1, 100, (seq_len,)),
        "pred_rewards": torch.randn(seq_len),
        "pred_zscores": torch.randn(seq_len),
        "pred_masks": torch.cat([torch.zeros(seq_len // 2),
                                  torch.ones(seq_len // 2)]),
        "pred_dones": torch.zeros(seq_len),
        "pred_old_logprobs": torch.randn(seq_len),
        "policy_version": policy_version,
    }

class TestChunkIsReady:
    def test_all_futures_ready(self):
        refs = [MagicMock(name=f"ref_{i}") for i in range(6)]
        chunk = ChunkFuture(
            futures=[[refs[0], refs[1], refs[2]], [refs[3], refs[4], refs[5]]],
            dispatch_time=0.0,
            chunk_idx=0,
        )
        with patch("main_rl.ray") as mock_ray:
            mock_ray.wait.return_value = (refs, [])
            assert chunk_is_ready(chunk) is True

        mock_ray.wait.assert_called_once()
        _, kwargs = mock_ray.wait.call_args
        assert kwargs["timeout"] == 0
        assert kwargs["num_returns"] == 6

    def test_some_futures_pending(self):
        refs = [MagicMock(name=f"ref_{i}") for i in range(4)]
        chunk = ChunkFuture(
            futures=[[refs[0], refs[1]], [refs[2], refs[3]]],
            dispatch_time=0.0,
            chunk_idx=1,
        )
        with patch("main_rl.ray") as mock_ray:
            mock_ray.wait.return_value = (refs[:3], refs[3:])
            assert chunk_is_ready(chunk) is False

    def test_single_batch_single_engine(self):
        ref = MagicMock()
        chunk = ChunkFuture(futures=[[ref]], dispatch_time=0.0, chunk_idx=0)
        with patch("main_rl.ray") as mock_ray:
            mock_ray.wait.return_value = ([ref], [])
            assert chunk_is_ready(chunk) is True

class TestDispatchOneChunk:
    def test_dispatches_chunk_size_batches(self):
        logger = MagicMock()
        engines = [MagicMock(), MagicMock()]
        for e in engines:
            e.generate.remote = MagicMock(return_value=MagicMock())

        # Each item in the dataloader is a list of prompt dicts (a batch)
        data = [[{"prompt": f"q{i}"}] for i in range(10)]
        it = iter(data)

        chunk = dispatch_one_chunk(it, engines, epoch=0, policy_version=1,
                                   chunk_size=3, chunk_idx=0, logger=logger)

        assert chunk is not None
        assert chunk.chunk_idx == 0
        assert len(chunk.futures) == 3
        assert chunk.dispatch_time > 0
        # Each batch dispatches to 2 engines (1 prompt sharded → 1 non-empty shard)
        for batch_futures in chunk.futures:
            assert len(batch_futures) >= 1

    def test_exhausted_before_chunk_size(self):
        logger = MagicMock()
        engines = [MagicMock()]
        engines[0].generate.remote = MagicMock(return_value=MagicMock())

        it = iter([[{"prompt": "only_one"}]])
        chunk = dispatch_one_chunk(it, engines, epoch=0, policy_version=0,
                                   chunk_size=5, chunk_idx=0, logger=logger)

        assert chunk is not None
        assert len(chunk.futures) == 1

    def test_empty_dataloader_returns_none(self):
        logger = MagicMock()
        engines = [MagicMock()]

        chunk = dispatch_one_chunk(iter([]), engines, epoch=0, policy_version=0,
                                   chunk_size=3, chunk_idx=0, logger=logger)
        assert chunk is None

    def test_chunk_metadata_correct(self):
        logger = MagicMock()
        engines = [MagicMock()]
        engines[0].generate.remote = MagicMock(return_value=MagicMock())

        before = time.time()
        chunk = dispatch_one_chunk(iter([[{"p": "x"}]]), engines, epoch=5,
                                   policy_version=3, chunk_size=1,
                                   chunk_idx=7, logger=logger)
        after = time.time()

        assert chunk.chunk_idx == 7
        assert before <= chunk.dispatch_time <= after

class TestShardBatchForEngines:
    def test_even_split(self):
        batch = [{"p": i} for i in range(6)]
        shards = shard_batch_for_engines(batch, num_rollout_engines=3)
        assert len(shards) == 3
        assert all(len(s) == 2 for s in shards)

    def test_uneven_split_no_empty_shards(self):
        batch = [{"p": i} for i in range(5)]
        shards = shard_batch_for_engines(batch, num_rollout_engines=3)
        assert all(len(s) > 0 for s in shards)
        assert sum(len(s) for s in shards) == 5

    def test_empty_batch(self):
        assert shard_batch_for_engines([], num_rollout_engines=2) == []

    def test_single_engine_gets_all(self):
        batch = [{"p": i} for i in range(4)]
        shards = shard_batch_for_engines(batch, num_rollout_engines=1)
        assert len(shards) == 1
        assert len(shards[0]) == 4

    def test_more_engines_than_items(self):
        batch = [{"p": 0}]
        shards = shard_batch_for_engines(batch, num_rollout_engines=4)
        # Empty shards are filtered out
        assert len(shards) == 1
        assert len(shards[0]) == 1

class TestPrepareTrainingBatches:
    def test_returns_padded_for_num_engines(self):
        rb = _make_replay_buffer(n_samples=7, seq_len=4)
        # 7 samples / bs=3 = 3 batches → pad to 4 for 2 engines
        batches = prepare_training_batches(rb, batch_size=3, num_engines=2,
                                           seed=0, epoch=0)
        assert len(batches) % 2 == 0

    def test_no_padding_needed(self):
        rb = _make_replay_buffer(n_samples=8, seq_len=4)
        # 8 samples / bs=4 = 2 batches, already divisible by 2 engines
        batches = prepare_training_batches(rb, batch_size=4, num_engines=2,
                                           seed=0, epoch=0)
        assert len(batches) == 2

    def test_deterministic_same_seed(self):
        rb = _make_replay_buffer(n_samples=20, seq_len=8)
        b1 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=42, epoch=0)
        b2 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=42, epoch=0)
        for a, b in zip(b1, b2):
            assert torch.equal(a["input_ids"], b["input_ids"])

    def test_different_epoch_different_shuffle(self):
        rb = _make_replay_buffer(n_samples=20, seq_len=8)
        b1 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=42, epoch=0)
        b2 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=42, epoch=1)
        differs = any(
            not torch.equal(a["input_ids"], b["input_ids"])
            for a, b in zip(b1, b2)
        )
        assert differs, "Different epoch must produce different shuffle"

    def test_shard_rebuild_seeding_uniqueness(self):
        """The seeding fix: epoch * total_chunks + shard_rebuild_count
        must produce different shuffles for different rebuild counts."""
        rb = _make_replay_buffer(n_samples=40, seq_len=8)
        total_chunks = 5
        epoch = 0
        b0 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=42,
                                      epoch=epoch * total_chunks + 0)
        b1 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=42,
                                      epoch=epoch * total_chunks + 1)
        differs = any(
            not torch.equal(a["input_ids"], b["input_ids"])
            for a, b in zip(b0, b1)
        )
        assert differs, "Different rebuild count must produce different shuffle"

    def test_cross_epoch_no_seed_collision(self):
        """Epoch 0 rebuild 3 vs epoch 1 rebuild 0 must not collide."""
        rb = _make_replay_buffer(n_samples=40, seq_len=8)
        total_chunks = 5
        # epoch=0, rebuild=3 → effective_epoch = 0*5 + 3 = 3
        # epoch=1, rebuild=0 → effective_epoch = 1*5 + 0 = 5
        b_a = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                       seed=42, epoch=3)
        b_b = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                       seed=42, epoch=5)
        differs = any(
            not torch.equal(a["input_ids"], b["input_ids"])
            for a, b in zip(b_a, b_b)
        )
        assert differs

class TestShardAndPut:
    def test_interleaved_distribution(self):
        batches = ["b0", "b1", "b2", "b3", "b4", "b5"]
        with patch("main_rl.ray") as mock_ray:
            mock_ray.put.side_effect = lambda x: f"ref_{x[0]}"
            refs = shard_and_put(batches, num_engines=2)

        assert len(refs) == 2
        calls = mock_ray.put.call_args_list
        # Engine 0 gets [b0, b2, b4], Engine 1 gets [b1, b3, b5]
        assert calls[0][0][0] == ["b0", "b2", "b4"]
        assert calls[1][0][0] == ["b1", "b3", "b5"]

    def test_single_engine_gets_all(self):
        batches = ["b0", "b1", "b2"]
        with patch("main_rl.ray") as mock_ray:
            mock_ray.put.return_value = "ref"
            refs = shard_and_put(batches, num_engines=1)

        assert len(refs) == 1
        mock_ray.put.assert_called_once_with(["b0", "b1", "b2"])

    def test_empty_shard_raises(self):
        with patch("main_rl.ray"):
            with pytest.raises(AssertionError, match="empty shard"):
                shard_and_put(["b0"], num_engines=2)

    def test_three_engines_six_batches(self):
        batches = list(range(6))
        with patch("main_rl.ray") as mock_ray:
            mock_ray.put.side_effect = lambda x: f"ref_{x[0]}"
            refs = shard_and_put(batches, num_engines=3)

        assert len(refs) == 3
        calls = mock_ray.put.call_args_list
        assert calls[0][0][0] == [0, 3]  # engine 0
        assert calls[1][0][0] == [1, 4]  # engine 1
        assert calls[2][0][0] == [2, 5]  # engine 2

class TestFinalizeChunk:
    def test_single_batch_adds_to_buffer(self):
        rb = ReplayBuffer(pad_token_id=0, max_seq_len=16)
        logger = MagicMock()
        sample = _make_sample(seq_len=8, policy_version=0)
        stats = _make_stats(n=1)

        chunk = ChunkFuture(futures=[["fake_ref"]], dispatch_time=0.0, chunk_idx=0)

        with patch("main_rl.ray_get_with_timeout", return_value=[[sample]]), \
             patch("main_rl.merge_rollout_with_stats", return_value=([sample], stats)):
            acc = finalize_chunk(chunk, rb, logger, rollout_timeout=30)

        assert len(rb) == 1
        assert acc['total_samples_generated'] == 1

    def test_multi_batch_accumulates(self):
        rb = ReplayBuffer(pad_token_id=0, max_seq_len=16)
        logger = MagicMock()

        samples_a = [_make_sample(seq_len=8) for _ in range(3)]
        samples_b = [_make_sample(seq_len=8) for _ in range(2)]
        stats_a = _make_stats(n=3)
        stats_b = _make_stats(n=2)

        chunk = ChunkFuture(
            futures=[["ref_a1", "ref_a2"], ["ref_b1", "ref_b2"]],
            dispatch_time=0.0,
            chunk_idx=0,
        )

        call_count = [0]
        def mock_merge(results):
            if call_count[0] == 0:
                call_count[0] += 1
                return samples_a, stats_a
            return samples_b, stats_b

        with patch("main_rl.ray_get_with_timeout", return_value=[[]]), \
             patch("main_rl.merge_rollout_with_stats", side_effect=mock_merge):
            acc = finalize_chunk(chunk, rb, logger, rollout_timeout=30)

        assert len(rb) == 5
        assert acc['total_samples_generated'] == 5

class TestAggregateChunkStats:
    def test_combines_two_chunks(self):
        s1 = _make_stats(n=10, reward=1.0, response_len=5, tokens=10)
        s2 = _make_stats(n=5, reward=2.0, response_len=8, tokens=12)

        result = aggregate_chunk_stats([s1, s2], generation_time=10.0,
                                       wall_time=15.0)

        assert result["total_samples_generated"] == 15
        assert result["rollout_time_with_overlap"] == 15.0
        assert result["rollout_time"] == 10.0

    def test_empty_list(self):
        result = aggregate_chunk_stats([], generation_time=0.0, wall_time=0.0)
        assert result["total_samples_generated"] == 0
        assert result["rollout_time_with_overlap"] == 0.0

class TestShardRefreshOnBufferGrowth:
    def test_condition_triggers_on_growth(self):
        """The refresh condition: len(rb) >= bs AND len(rb) != shard_buffer_size."""
        rb = _make_replay_buffer(n_samples=10, seq_len=4)
        batch_size = 4
        shard_buffer_size = 0

        # First build triggers
        assert len(rb) >= batch_size and len(rb) != shard_buffer_size
        shard_buffer_size = len(rb)

        # Same size → no trigger
        assert not (len(rb) >= batch_size and len(rb) != shard_buffer_size)

        # Growth → triggers again
        rb.add(input_ids=torch.randint(1, 100, (4,)),
               rewards=torch.randn(4), zscores=torch.randn(4),
               masks=torch.ones(4), dones=torch.zeros(4),
               old_logprobs=torch.randn(4), policy_version=1)
        assert len(rb) >= batch_size and len(rb) != shard_buffer_size

    def test_rebuilt_shards_reflect_full_buffer(self):
        """After adding more data, prepare_training_batches produces batches
        covering all samples, not just the initial subset."""
        rb = _make_replay_buffer(n_samples=8, seq_len=4)
        b1 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=0, epoch=0)
        total_b1 = sum(b["input_ids"].shape[0] for b in b1)
        assert total_b1 == 8

        # Double the buffer
        for _ in range(8):
            rb.add(input_ids=torch.randint(1, 100, (4,)),
                   rewards=torch.randn(4), zscores=torch.randn(4),
                   masks=torch.ones(4), dones=torch.zeros(4),
                   old_logprobs=torch.randn(4), policy_version=1)

        b2 = prepare_training_batches(rb, batch_size=4, num_engines=1,
                                      seed=0, epoch=1)
        total_b2 = sum(b["input_ids"].shape[0] for b in b2)
        assert total_b2 == 16
        assert total_b2 > total_b1

    def test_evict_stale_shrinks_buffer(self):
        """After eviction, shard_buffer_size won't match → triggers rebuild."""
        rb = _make_replay_buffer(n_samples=10, seq_len=4, policy_version=0)
        shard_buffer_size = len(rb)  # 10

        # Add fresh samples
        for _ in range(5):
            rb.add(input_ids=torch.randint(1, 100, (4,)),
                   rewards=torch.randn(4), zscores=torch.randn(4),
                   masks=torch.ones(4), dones=torch.zeros(4),
                   old_logprobs=torch.randn(4), policy_version=1)
        assert len(rb) == 15

        # Evict old samples (version < 1)
        evicted = rb.evict_stale(min_version=1)
        assert evicted == 10
        assert len(rb) == 5
        # Buffer shrank → mismatch with shard_buffer_size
        assert len(rb) != shard_buffer_size

class TestOverlapMetrics:
    def _compute_ratio(self, train_sec, wait_sec):
        total = train_sec + wait_sec
        return train_sec / total if total > 0 else 0.0

    def test_perfect_overlap(self):
        '''Gen always finishes before training → gen_wait=0 → ratio=1.0.'''
        assert self._compute_ratio(10.0, 0.0) == 1.0

    def test_no_overlap(self):
        '''No training during generation → ratio=0.0.'''
        assert self._compute_ratio(0.0, 10.0) == 0.0

    def test_balanced(self):
        assert self._compute_ratio(5.0, 5.0) == pytest.approx(0.5)

    def test_zero_total_safe(self):
        """Empty dataloader, no interleaving → both zero → ratio=0.0."""
        assert self._compute_ratio(0.0, 0.0) == 0.0

    def test_typical_overlap(self):
        """Training fills 80% of time, 20% waiting → ratio=0.8."""
        assert self._compute_ratio(8.0, 2.0) == pytest.approx(0.8)
