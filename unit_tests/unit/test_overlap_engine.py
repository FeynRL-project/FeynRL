'''
    Unit tests for the queue-pull async overlap engine.

    Replaces the previous tests for the chunk-based dispatch model
    (ChunkFuture / dispatch_one_chunk / finalize_chunk / cycle_chunk)
    which no longer exist. The current architecture uses
    fill_prompt_queue / run_pull_loop / drain_results / poison-pill
    coordination.

    Heavy dependencies (ray, deepspeed, transformers) are mocked in
    unit_tests/conftest.py. Functions exercised here are pure-Python
    (no Ray actors, no NCCL, no GPUs).
'''
import sys
import types
from unittest.mock import MagicMock, patch
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
    compute_results_queue_maxsize,
    check_ess_sync,
    teardown_prelaunched,
    fill_prompt_queue,
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

    def put(self, item):
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
    def test_empty_queue_returns_empty_list(self):
        q = FakePromptQueue([])
        assert drain_prompt_queue(q) == []

    def test_only_real_shards(self):
        shards = [{"id": 0}, {"id": 1}, {"id": 2}]
        q = FakePromptQueue(shards)
        result = drain_prompt_queue(q)
        assert result == shards
        assert len(q) == 0

    def test_only_pills_returns_empty_list_and_drains(self):
        '''Pills must be discarded, not returned.'''
        q = FakePromptQueue([POISON_PILL, POISON_PILL, POISON_PILL])
        result = drain_prompt_queue(q)
        assert result == []
        assert len(q) == 0

    def test_mixed_pills_and_shards_preserves_shard_order(self):
        '''Real shards must come back in FIFO order; pills discarded.'''
        items = [{"a": 0}, POISON_PILL, {"a": 1}, POISON_PILL, {"a": 2}]
        q = FakePromptQueue(items)
        result = drain_prompt_queue(q)
        assert result == [{"a": 0}, {"a": 1}, {"a": 2}]
        assert len(q) == 0

    def test_drain_does_not_remove_pills_after_get_raises(self):
        '''If get raises Empty, drain returns whatever it has so far.'''
        q = FakePromptQueue([{"a": 0}])
        result = drain_prompt_queue(q)
        assert result == [{"a": 0}]


# ---------------------------------------------------------------------------
# stop_engines_and_drain
# ---------------------------------------------------------------------------

class TestStopEnginesAndDrain:
    def test_drains_then_pushes_one_pill_per_engine(self):
        shards = [{"a": 0}, {"a": 1}]
        q = FakePromptQueue(shards)
        leftover = stop_engines_and_drain(q, num_rollout_engines=3)
        assert leftover == shards
        # After: queue contains exactly 3 pills, no real shards
        snap = q.snapshot()
        assert len(snap) == 3
        assert all(x == POISON_PILL for x in snap)

    def test_discards_stale_pills_before_pushing_new(self):
        '''If a previous round left pills, they should be discarded.'''
        q = FakePromptQueue([POISON_PILL, {"a": 0}, POISON_PILL])
        leftover = stop_engines_and_drain(q, num_rollout_engines=2)
        assert leftover == [{"a": 0}]
        snap = q.snapshot()
        assert snap == [POISON_PILL, POISON_PILL]

    def test_empty_queue_yields_empty_leftover_and_pushes_pills(self):
        q = FakePromptQueue([])
        leftover = stop_engines_and_drain(q, num_rollout_engines=4)
        assert leftover == []
        assert q.snapshot() == [POISON_PILL] * 4


# ---------------------------------------------------------------------------
# compute_results_queue_maxsize
# ---------------------------------------------------------------------------

class TestComputeResultsQueueMaxsize:
    def test_minimum_floor(self):
        '''Even with max_lag=1, each engine should get at least the floor.'''
        size = compute_results_queue_maxsize(num_rollout_engines=2, max_lag=1)
        assert size >= 2 * 8  # at least num_engines * 8

    def test_scales_with_max_lag(self):
        small = compute_results_queue_maxsize(num_rollout_engines=4, max_lag=2)
        big   = compute_results_queue_maxsize(num_rollout_engines=4, max_lag=8)
        assert big >= small

    def test_scales_with_num_engines(self):
        few  = compute_results_queue_maxsize(num_rollout_engines=2, max_lag=4)
        many = compute_results_queue_maxsize(num_rollout_engines=8, max_lag=4)
        assert many > few

    def test_returns_positive(self):
        for n in [1, 2, 8, 32]:
            for lag in [1, 2, 4]:
                assert compute_results_queue_maxsize(n, lag) > 0


# ---------------------------------------------------------------------------
# check_ess_sync
# ---------------------------------------------------------------------------

class TestCheckEssSync:
    def test_p3o_ess_below_threshold_triggers(self):
        '''P3O path: ESS < threshold → True. One-shot per epoch.'''
        should_sync, ess = check_ess_sync(
            train_metrics={"ess_factor": 0.3},
            train_step_count=10,
            ess_sync_threshold=0.5,
            fixed_sync_interval=None,
            sync_triggered_this_epoch=False,
        )
        assert should_sync is True
        assert ess == 0.3

    def test_p3o_ess_above_threshold_does_not_trigger(self):
        should_sync, ess = check_ess_sync(
            train_metrics={"ess_factor": 0.8},
            train_step_count=10,
            ess_sync_threshold=0.5,
            fixed_sync_interval=None,
            sync_triggered_this_epoch=False,
        )
        assert should_sync is False
        assert ess == 0.8

    def test_p3o_already_synced_this_epoch_does_not_retrigger(self):
        '''ESS-driven sync is gated to one-shot per epoch.'''
        should_sync, _ = check_ess_sync(
            train_metrics={"ess_factor": 0.1},
            train_step_count=10,
            ess_sync_threshold=0.5,
            fixed_sync_interval=None,
            sync_triggered_this_epoch=True,
        )
        assert should_sync is False

    def test_no_ess_metric_no_p3o_trigger(self):
        '''Non-P3O algorithms don't expose ess_factor.'''
        should_sync, ess = check_ess_sync(
            train_metrics={"loss": 0.5},
            train_step_count=10,
            ess_sync_threshold=0.5,
            fixed_sync_interval=None,
            sync_triggered_this_epoch=False,
        )
        assert should_sync is False
        assert ess is None

    def test_fixed_interval_triggers_at_boundary(self):
        '''Fixed interval: deterministic, NOT gated by sync_triggered_this_epoch.'''
        should_sync, _ = check_ess_sync(
            train_metrics={},
            train_step_count=10,
            ess_sync_threshold=0.5,
            fixed_sync_interval=10,
            sync_triggered_this_epoch=False,
        )
        assert should_sync is True

    def test_fixed_interval_not_at_boundary(self):
        should_sync, _ = check_ess_sync(
            train_metrics={},
            train_step_count=11,
            ess_sync_threshold=0.5,
            fixed_sync_interval=10,
            sync_triggered_this_epoch=False,
        )
        assert should_sync is False

    def test_fixed_interval_step_zero_does_not_trigger(self):
        '''step % interval == 0 at step=0 must NOT fire (would sync at start).'''
        should_sync, _ = check_ess_sync(
            train_metrics={},
            train_step_count=0,
            ess_sync_threshold=0.5,
            fixed_sync_interval=10,
            sync_triggered_this_epoch=False,
        )
        assert should_sync is False

    def test_fixed_interval_fires_even_after_already_synced(self):
        '''Fixed interval is NOT one-shot — can fire multiple times per epoch.'''
        should_sync, _ = check_ess_sync(
            train_metrics={},
            train_step_count=20,
            ess_sync_threshold=0.5,
            fixed_sync_interval=10,
            sync_triggered_this_epoch=True,
        )
        assert should_sync is True


# ---------------------------------------------------------------------------
# teardown_prelaunched
# ---------------------------------------------------------------------------

class TestTeardownPrelaunched:
    def test_none_state_is_noop(self):
        '''Common case: epoch not pre-launched.'''
        logger = MagicMock()
        teardown_prelaunched(None, num_rollout_engines=4, logger=logger)
        logger.warning.assert_not_called()

    def test_drains_real_shards_then_pushes_pills(self):
        '''
            Recent fix: teardown must drain unread shards FIRST so the pills
            land at the head of the queue. Otherwise pull loops would process
            every leftover shard before reaching their pill.
        '''
        q = FakePromptQueue([{"a": 0}, {"a": 1}, {"a": 2}])
        with patch.object(run_rl_async, "ray") as mock_ray:
            mock_ray.wait.return_value = ([], [])
            teardown_prelaunched(
                prelaunched_state={
                    'prefilled_prompt_queue': q,
                    'prefilled_pull_refs': [MagicMock(), MagicMock()],
                },
                num_rollout_engines=2,
                logger=MagicMock(),
            )
        # After teardown, queue should contain only pills (real shards drained)
        snap = q.snapshot()
        assert all(x == POISON_PILL for x in snap)
        assert len(snap) == 2

    def test_handles_missing_keys_gracefully(self):
        '''A best-effort cleanup must not raise on a partial state dict.'''
        logger = MagicMock()
        teardown_prelaunched(
            prelaunched_state={'prefilled_pull_refs': []},
            num_rollout_engines=2,
            logger=logger,
        )
        # Should not raise; warning may or may not fire

    def test_swallows_exceptions(self):
        '''Cleanup is best-effort and must never propagate.'''
        bad_q = MagicMock()
        bad_q.get.side_effect = RuntimeError("network down")
        bad_q.put.side_effect = RuntimeError("network down")
        logger = MagicMock()
        with patch.object(run_rl_async, "ray"):
            teardown_prelaunched(
                prelaunched_state={
                    'prefilled_prompt_queue': bad_q,
                    'prefilled_pull_refs': [],
                },
                num_rollout_engines=1,
                logger=logger,
            )
        # No exception escaped


# ---------------------------------------------------------------------------
# fill_prompt_queue
# ---------------------------------------------------------------------------

class TestFillPromptQueue:
    def test_all_shards_enqueued_plus_end_of_epoch_pills(self):
        '''
            Sharded across engines, every non-empty shard goes to the queue.
            After all shards, fill_prompt_queue pushes exactly num_engines
            POISON_PILL sentinels (the explicit "epoch done" signal that
            replaces the previous 5-second polling timeout in run_pull_loop).
        '''
        q = FakePromptQueue()
        # 2 batches × 2 engines × 2 prompts each = 4 non-empty shards total
        dataloader = [
            [{"p": 0}, {"p": 1}, {"p": 2}, {"p": 3}],
            [{"p": 4}, {"p": 5}, {"p": 6}, {"p": 7}],
        ]
        num_engines = 2
        engines = [MagicMock() for _ in range(num_engines)]
        logger = MagicMock()
        total = fill_prompt_queue(dataloader, engines, q, logger)
        snap = q.snapshot()
        # total_shards counts only real shards, NOT the trailing pills
        assert total > 0
        assert len(snap) == total + num_engines
        # Pills are at the end, one per engine
        pill_count = sum(1 for x in snap if x == POISON_PILL)
        assert pill_count == num_engines
        assert all(x == POISON_PILL for x in snap[-num_engines:])

    def test_empty_dataloader_yields_zero_shards_but_still_pushes_pills(self):
        '''
            Even with no real work, the pills must be pushed so engines
            (if any are running against this queue) can exit cleanly.
        '''
        q = FakePromptQueue()
        num_engines = 3
        engines = [MagicMock() for _ in range(num_engines)]
        logger = MagicMock()
        total = fill_prompt_queue([], engines, q, logger)
        assert total == 0
        snap = q.snapshot()
        assert len(snap) == num_engines
        assert all(x == POISON_PILL for x in snap)


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
