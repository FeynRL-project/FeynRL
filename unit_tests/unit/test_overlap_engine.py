'''
    Unit tests for the queue-pull async overlap engine helpers.

    The current architecture uses a persistent ShardProducer daemon thread
    feeding a bounded prompt_queue, run_pull_loop on rollout actors, and
    drain_results / poison-pill coordination on the driver. The previous
    chunk-based dispatch model and the pre-launch / teardown_prelaunched
    helpers no longer exist; their tests have been removed.

    Heavy dependencies (ray, deepspeed, transformers) are mocked in
    unit_tests/conftest.py. Functions exercised here are pure-Python
    (no Ray actors, no NCCL, no GPUs).
'''
import sys
import types
from unittest.mock import MagicMock
import pytest

# Stub heavy transitive imports BEFORE importing run_rl_async. The conftest
# mocks the obvious ones (transformers, deepspeed, peft, etc) but the
# core.rl_engines / data_feeds chain pulls in datasets + huggingface_hub
# which break under MagicMock. We replace the leaf modules with empty
# stubs that expose the names run_rl_async actually imports.
for mod in ("mlflow", "wandb", "tensorboardX", "datasets"):
    sys.modules.setdefault(mod, MagicMock())

# data_feeds.prompts → exposes PromptsFeed
_df_prompts = types.ModuleType("data_feeds.prompts")
_df_prompts.PromptsFeed = MagicMock()
sys.modules.setdefault("data_feeds", types.ModuleType("data_feeds"))
sys.modules.setdefault("data_feeds.prompts", _df_prompts)

# data_feeds.mixed_sampler → exposes create_prompt_dataset_and_sampler
_df_mixed = types.ModuleType("data_feeds.mixed_sampler")
_df_mixed.create_prompt_dataset_and_sampler = MagicMock()
sys.modules.setdefault("data_feeds.mixed_sampler", _df_mixed)

# rollouts.vllm_engine and rollouts.vllm_engine_async — stub the actor
# classes since they'd otherwise pull in vLLM which isn't installed in
# the test env. The async one is heavy.
_rl_vllm = types.ModuleType("rollouts.vllm_engine")
_rl_vllm.VLLMRolloutEngine = MagicMock()
sys.modules.setdefault("rollouts.vllm_engine", _rl_vllm)
_rl_vllm_async = types.ModuleType("rollouts.vllm_engine_async")
_rl_vllm_async.VLLMRolloutEngineAsync = MagicMock()
sys.modules.setdefault("rollouts.vllm_engine_async", _rl_vllm_async)

# vllm itself — for the rare paths that import directly.
sys.modules.setdefault("vllm", MagicMock())

# ray.util.queue is not in the conftest mocks; install a minimal stub
# before importing run_rl_async so the `from ray.util.queue import Queue
# as RayQueue, Empty as RayQueueEmpty` line resolves.
class _FakeRayQueueEmpty(Exception):
    pass
_ray_util = MagicMock()
_ray_util.queue.Queue = MagicMock()
_ray_util.queue.Empty = _FakeRayQueueEmpty
sys.modules.setdefault("ray.util", _ray_util)
sys.modules.setdefault("ray.util.queue", _ray_util.queue)

import run_rl_async
from run_rl_async import (
    POISON_PILL,
    drain_prompt_queue,
    stop_engines_and_drain,
    compute_pipeline_capacities,
    InfiniteShardIterator,
)
from misc.nccl_utils import is_nccl_fatal_error
from misc.nccl_env import nccl_watchdog_env_vars

# Make the module's RayQueueEmpty match our fake so the helpers see it
run_rl_async.RayQueueEmpty = _FakeRayQueueEmpty


class FakePromptQueue:
    '''
        Minimal in-memory queue with the same get/put surface as ray.util.queue.Queue,
        sufficient for the helpers under test. get(block=False) raises the
        same Empty type the module imported.
    '''
    def __init__(self, items=None):
        self._items = list(items or [])

    def put(self, item, block=True, timeout=None):
        self._items.append(item)

    def get(self, block=False, timeout=None):
        if not self._items:
            raise _FakeRayQueueEmpty()
        return self._items.pop(0)

    def __len__(self):
        return len(self._items)

    def snapshot(self):
        return list(self._items)


# ---------------------------------------------------------------------------
# POISON_PILL constant
# ---------------------------------------------------------------------------

class TestPoisonPillConstant:
    def test_poison_pill_value(self):
        '''Pull loop on the engine side hard-codes "__STOP__" — must match.'''
        assert POISON_PILL == "__STOP__"

    def test_poison_pill_is_string(self):
        '''isinstance(item, str) check in run_pull_loop assumes string type.'''
        assert isinstance(POISON_PILL, str)


# ---------------------------------------------------------------------------
# drain_prompt_queue
# ---------------------------------------------------------------------------

class TestDrainPromptQueue:
    def test_empty_queue_returns_zero(self):
        q = FakePromptQueue([])
        assert drain_prompt_queue(q) == 0

    def test_drains_real_shards(self):
        shards = [{"id": 0}, {"id": 1}, {"id": 2}]
        q = FakePromptQueue(shards)
        n = drain_prompt_queue(q)
        assert n == 3
        assert len(q) == 0

    def test_drains_pills_too(self):
        '''Pills are dropped along with everything else.'''
        q = FakePromptQueue([POISON_PILL, POISON_PILL, POISON_PILL])
        n = drain_prompt_queue(q)
        assert n == 3
        assert len(q) == 0

    def test_mixed_pills_and_shards(self):
        items = [{"a": 0}, POISON_PILL, {"a": 1}, POISON_PILL, {"a": 2}]
        q = FakePromptQueue(items)
        n = drain_prompt_queue(q)
        assert n == 5
        assert len(q) == 0


# ---------------------------------------------------------------------------
# stop_engines_and_drain
# ---------------------------------------------------------------------------

class TestStopEnginesAndDrain:
    def test_drains_then_pushes_one_pill_per_engine(self):
        q = FakePromptQueue([{"a": 0}, {"a": 1}])
        stop_engines_and_drain(q, num_rollout_engines=3, logger=MagicMock())
        # After: queue contains exactly 3 pills, no real shards
        snap = q.snapshot()
        assert len(snap) == 3
        assert all(x == POISON_PILL for x in snap)

    def test_discards_stale_pills_before_pushing_new(self):
        '''If a previous round left pills, they should be discarded too.'''
        q = FakePromptQueue([POISON_PILL, {"a": 0}, POISON_PILL])
        stop_engines_and_drain(q, num_rollout_engines=2, logger=MagicMock())
        snap = q.snapshot()
        assert snap == [POISON_PILL, POISON_PILL]

    def test_empty_queue_pushes_pills(self):
        q = FakePromptQueue([])
        stop_engines_and_drain(q, num_rollout_engines=4, logger=MagicMock())
        assert q.snapshot() == [POISON_PILL] * 4


# ---------------------------------------------------------------------------
# compute_pipeline_capacities
# ---------------------------------------------------------------------------

class TestComputePipelineCapacities:
    def _call(self, **kwargs):
        '''Helper with sensible defaults. Signature is
        (rollout_samples_per_epoch, rollout_batch_size_per_gpu,
         num_rollout_engines, n_samples, max_lag).
        '''
        defaults = dict(
            rollout_samples_per_epoch=1024,
            rollout_batch_size_per_gpu=8,
            num_rollout_engines=4,
            n_samples=16,
            max_lag=4,
        )
        defaults.update(kwargs)
        return compute_pipeline_capacities(**defaults)

    def test_returns_four_values(self):
        buf, results_q, prompt_q, items_per_round = self._call()
        assert isinstance(buf, int)
        assert isinstance(results_q, int)
        assert isinstance(prompt_q, int)
        assert isinstance(items_per_round, int)

    def test_buffer_uses_rounded_prompts(self):
        '''prompt_per_pass rounds up to bsz_rollout boundary: items_per_round × max_lag.'''
        # rollout_samples_per_epoch=20, bsz=4*8=32 → ceil(20/32)*32 = 32
        buf, _, _, _ = self._call(rollout_samples_per_epoch=20,
                                  rollout_batch_size_per_gpu=8,
                                  num_rollout_engines=4, n_samples=8, max_lag=2)
        # items_per_round = 32 * 8 = 256; buffer = 256 * 2 = 512
        assert buf == 512

    def test_buffer_exact_multiple(self):
        '''No rounding when rollout_samples_per_epoch is exact multiple of bsz.'''
        buf, _, _, _ = self._call(rollout_samples_per_epoch=128,
                                  rollout_batch_size_per_gpu=16,
                                  num_rollout_engines=4, n_samples=16, max_lag=4)
        # bsz=64, ceil(128/64)*64=128, items=128*16=2048, buffer=2048*4=8192
        assert buf == 8192

    def test_scales_linearly_with_max_lag(self):
        '''Buffer is exactly items_per_round × max_lag.'''
        buf2 = self._call(max_lag=2)[0]
        buf8 = self._call(max_lag=8)[0]
        assert buf8 == 4 * buf2

    def test_scales_with_rollout_samples(self):
        small = self._call(rollout_samples_per_epoch=256)[0]
        big   = self._call(rollout_samples_per_epoch=2048)[0]
        assert big > small

    def test_max_lag_one_gives_one_round(self):
        '''max_lag=1 means exactly one round in the buffer (no floor now).'''
        # rollout_samples=64, bsz=32, n_samples=4: items_per_round = 64*4 = 256
        buf, _, _, _ = self._call(rollout_samples_per_epoch=64,
                                  rollout_batch_size_per_gpu=8,
                                  num_rollout_engines=4, n_samples=4, max_lag=1)
        assert buf == 256

    def test_prompt_queue_has_floor_of_two_bursts(self):
        '''prompt_queue = num_engines × max(2, max_lag) — floor ensures 2 bursts lookahead.'''
        _, _, pq1, _ = self._call(num_rollout_engines=4, max_lag=1)
        _, _, pq2, _ = self._call(num_rollout_engines=4, max_lag=2)
        assert pq1 == pq2 == 4 * 2

    def test_prompt_queue_scales_with_engines(self):
        _, _, few, _  = self._call(num_rollout_engines=2)
        _, _, many, _ = self._call(num_rollout_engines=16)
        assert many > few

    def test_results_queue_at_least_prompt_queue(self):
        for max_lag in [1, 2, 4, 8]:
            _, results_q, prompt_q, _ = self._call(max_lag=max_lag)
            assert results_q >= prompt_q

    def test_results_queue_scales_with_buffer(self):
        '''Bigger buffer → bigger results_queue.'''
        _, small_rq, _, _ = self._call(rollout_samples_per_epoch=128)
        _, big_rq, _, _   = self._call(rollout_samples_per_epoch=4096)
        assert big_rq > small_rq

    def test_items_per_round_matches_buffer_over_max_lag(self):
        '''items_per_round × max_lag == buffer size (the knob's stated semantic).'''
        for ml in [1, 2, 5, 10]:
            buf, _, _, items_per_round = self._call(max_lag=ml)
            assert items_per_round * ml == buf

    def test_items_per_round_independent_of_max_lag(self):
        '''items_per_round is one dataloader pass; max_lag doesn't change it.'''
        _, _, _, ipr1 = self._call(max_lag=1)
        _, _, _, ipr8 = self._call(max_lag=8)
        assert ipr1 == ipr8


# ---------------------------------------------------------------------------
# InfiniteShardIterator
# ---------------------------------------------------------------------------

class TestInfiniteShardIterator:
    def test_rejects_empty_dataloader(self):
        '''Empty dataloader at construction time → fail fast.'''
        empty = MagicMock()
        empty.__len__ = lambda self: 0
        with pytest.raises(ValueError, match="zero batches"):
            InfiniteShardIterator(dataloader=empty, num_rollout_engines=4)

    def test_yields_shards_then_advances_epoch(self):
        '''On dataloader exhaustion, increment epoch and reshuffle.'''
        batch_sampler = MagicMock()
        batch_sampler.set_epoch = MagicMock()
        # 2-batch dataloader, each batch is a list of 2 prompts
        loader = MagicMock()
        loader.__len__ = lambda self: 2
        loader.batch_sampler = batch_sampler
        # iter() returns a fresh iterator each call (mimics DataLoader)
        loader.__iter__ = lambda self: iter([[{"p": 0}, {"p": 1}],
                                             [{"p": 2}, {"p": 3}]])
        it = InfiniteShardIterator(dataloader=loader, num_rollout_engines=2, start_epoch=0)
        assert it.epoch == 0

        # Each next_shards() returns one batch worth of shards. With
        # num_rollout_engines=2 and a 2-prompt batch, each call returns
        # 2 shards of 1 prompt each.
        shards_a = it.next_shards()
        shards_b = it.next_shards()
        assert len(shards_a) == 2
        assert len(shards_b) == 2
        assert it.epoch == 0  # still on epoch 0 — both batches consumed but no StopIteration yet

        # The third call exhausts the iterator, triggering reset_for_new_epoch.
        shards_c = it.next_shards()
        assert len(shards_c) == 2
        assert it.epoch == 1
        # set_epoch was called: once at construction (epoch=0), once on reset (epoch=1)
        assert batch_sampler.set_epoch.call_count == 2


# ---------------------------------------------------------------------------
# Removed-symbol coverage
# ---------------------------------------------------------------------------
# The following helpers used to be tested here but have been deleted from
# run_rl_async.py as part of the persistent-queue refactor:
#   - compute_results_queue_maxsize  (replaced by compute_pipeline_queue_sizes)
#   - teardown_prelaunched           (no more pre-launch path)
#   - fill_prompt_queue              (replaced by ShardProducer thread)
# Their tests were removed. The new helpers above cover the equivalent
# functionality.


# ---------------------------------------------------------------------------
# is_nccl_fatal_error
# ---------------------------------------------------------------------------

class TestIsNcclFatalError:
    @pytest.mark.parametrize("msg", [
        "NCCL communicator was aborted",
        "ProcessGroupNCCL: communicator is aborted",
        "ncclCommAbort returned",
        "communicator is destroyed",
        "NCCL error: internal failure",
        "internal error in NCCL group",
        "NCCL unhandled cuda error: out of memory",
        "Watchdog caught collective operation timeout",
    ])
    def test_classic_fatal_patterns(self, msg):
        assert is_nccl_fatal_error(RuntimeError(msg)) is True

    @pytest.mark.parametrize("msg", [
        "NCCL broadcast v5 timed out after 900s. Check actor logs.",
        "finalize weight sync v3 timed out after 60s",
    ])
    def test_ray_timeout_wrapper_is_fatal(self, msg):
        '''
            ray_get_with_timeout wraps GetTimeoutError as
            "<description> timed out after Xs. ..." which means a NCCL
            collective is wedged on a worker's CUDA stream. Treat as fatal.
        '''
        assert is_nccl_fatal_error(RuntimeError(msg)) is True

    @pytest.mark.parametrize("msg", [
        "NCCL gather v2 failed because a Ray actor died: ActorDiedError",
        "actor died unexpectedly",
    ])
    def test_actor_death_is_fatal(self, msg):
        '''A dead actor mid-sync means a missing rank — unrecoverable.'''
        assert is_nccl_fatal_error(RuntimeError(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Partial weight load on engines [3]: finalize_weight_nccl returned False",
        "ValueError: shape mismatch in tensor",
        "Some random unrelated error",
        "",
    ])
    def test_recoverable_or_unrelated_errors_not_fatal(self, msg):
        '''
            Partial-load errors and generic exceptions must NOT be classified
            as fatal — the driver retries them at end-of-epoch.
        '''
        assert is_nccl_fatal_error(RuntimeError(msg)) is False

    def test_case_insensitive(self):
        '''Patterns are matched case-insensitively against str(exc).lower().'''
        assert is_nccl_fatal_error(RuntimeError("COMMUNICATOR WAS ABORTED")) is True


# ---------------------------------------------------------------------------
# nccl_watchdog_env_vars
# ---------------------------------------------------------------------------

class TestNcclWatchdogEnvVars:
    def test_returns_dict_with_expected_keys(self):
        env = nccl_watchdog_env_vars()
        assert "TORCH_NCCL_ASYNC_ERROR_HANDLING" in env
        assert "TORCH_NCCL_BLOCKING_WAIT" in env
        assert "NCCL_TIMEOUT" in env

    def test_async_error_handling_enabled(self):
        env = nccl_watchdog_env_vars()
        assert env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "1"

    def test_blocking_wait_disabled(self):
        '''Watchdog handles errors asynchronously rather than blocking.'''
        env = nccl_watchdog_env_vars()
        assert env["TORCH_NCCL_BLOCKING_WAIT"] == "0"

    def test_timeout_default_seconds(self):
        env = nccl_watchdog_env_vars()
        # Default is 1800 seconds (30 minutes)
        assert env["NCCL_TIMEOUT"] == "1800"

    def test_timeout_overridable(self):
        env = nccl_watchdog_env_vars(timeout_seconds=600)
        assert env["NCCL_TIMEOUT"] == "600"

    def test_all_values_are_strings(self):
        '''Ray runtime_env requires str values.'''
        env = nccl_watchdog_env_vars()
        for k, v in env.items():
            assert isinstance(v, str), f"{k} is not a string"
