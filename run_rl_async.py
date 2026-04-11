import os
import atexit
import numpy as np
import importlib
import ray
import time
import shutil
import threading

# imports local methods, classes, etc.
from misc.utils import load_algorithm, ray_get_with_timeout, set_random_seeds
from ray.util.queue import Queue as RayQueue, Empty as RayQueueEmpty, Full as RayQueueFull
from rollouts.replay_buffer import ReplayBuffer
from misc.logging import setup_logging, setup_tracker
from misc.setup_rl import load_tokenizer, save_checkpoint, load_checkpoint_for_resume, setup_ray
from misc.nccl_utils import is_nccl_fatal_error
import misc.rollout_stats as rollout_stats

from core.rl_engines import (Algorithm_Registry,
                            create_training_engines,
                            create_rollout_engines,
                            create_rollout_dataloader,
                            shard_batch_for_engines,
                            merge_rollout_with_stats,
                            prepare_training_batches,
                            shard_and_put,
                            run_training_step,
                            sync_weights_direct,
                            refresh_rollout_engine,
                            reinit_nccl_weight_sync_group,
                            start_nccl_gather,
                            complete_nccl_gather,
                            broadcast_and_finalize_nccl)


# Sentinel value pushed into prompt_queue to signal an engine to stop. Must
# match the value in VLLMRolloutEngineAsync.run_pull_loop. One poison pill per
# engine causes exactly one engine to exit.
POISON_PILL = "__STOP__"

def drain_prompt_queue(prompt_queue):
    '''
        Drain and discard all items from prompt_queue, i.e., real shards and stale
        pills. Used before pushing fresh poison pills so the pills land at
        the head of the queue. Real shards are dropped because the infinite
        shard iterator will replay equivalent prompts on producer restart.
        Returns the number of items drained, for logging only.
        Only RayQueueEmpty terminates the drain, other exceptions propagate.
    '''
    drained = 0
    while True:
        try:
            prompt_queue.get(block=False)
        except RayQueueEmpty:
            break

        drained += 1
    return drained

def stop_engines_and_drain(prompt_queue, num_rollout_engines, logger):
    '''
        Drain prompt_queue, then push one poison pill per engine so each
        pull loop exits cleanly. Stop latency = max in-flight generate()
        duration, which is unavoidable.
    '''
    drained = drain_prompt_queue(prompt_queue)
    for _ in range(num_rollout_engines):
        prompt_queue.put(POISON_PILL)

    logger.info(f"[stop_engines_and_drain] drained {drained} items, "
                f"pushed {num_rollout_engines} poison pills")

def wait_for_pull_loops_with_drain(pull_refs, results_queue, replay_buffer, rollout_acc, timeout, logger):
    '''
        Wait for pull loops to exit, draining results_queue continuously so engines blocked
        on results_queue.put() can unblock and reach their next prompt_queue.get() where
        they will see a poison pill.
        Returns (success, num_drained). On success, all pull_refs are resolved.
        On timeout, returns (False, ...) and the caller must handle the stuck
        engine state.
    '''
    drained_total = 0
    deadline = time.time() + timeout
    pending = list(pull_refs)
    while pending:
        # Drain whatever results are available so engines blocked on put() unblock.
        d, drain_acc = drain_results(results_queue, replay_buffer)
        drained_total += d
        rollout_stats.accumulate(rollout_acc, drain_acc)

        time_left = deadline - time.time()
        if time_left <= 0:
            logger.error(f"[wait_for_pull_loops_with_drain] Timeout after {timeout}s "
                         f"with {len(pending)} pull loops still running")
            return False, drained_total

        # Cap at 0.5s so we loop back to drain results_queue so engines
        # stuck on put(), full queue, can't exit until we drain.
        ready, pending = ray.wait(pending, num_returns=len(pending),
                                   timeout=min(time_left, 0.5))

    return True, drained_total

def stop_pull_loops_and_check(pull_refs, prompt_queue, results_queue, replay_buffer,
                              rollout_acc, num_rollout_engines, timeout, logger,
                              push_pills):
    '''
        Push poison pills, wait for pull loops to exit (draining results_queue continuously), 
        then surface any exceptions from the resolved refs.
        Returns (success, num_drained).
          success=True   -> all pull loops resolved cleanly
          success=False  -> wait timed out OR a pull loop raised
        Caller decides how to recover (continue with stale weights, raise, etc).
    '''
    if push_pills:
        for _ in range(num_rollout_engines):
            prompt_queue.put(POISON_PILL)
    ok, drained = wait_for_pull_loops_with_drain(pull_refs=pull_refs,
                                                 results_queue=results_queue,
                                                 replay_buffer=replay_buffer,
                                                 rollout_acc=rollout_acc,
                                                 timeout=timeout,
                                                 logger=logger)
    if not ok:
        return False, drained

    try:
        # 10s is a sanity-bound, it would be better than using ray.get(pull_refs)
        ray_get_with_timeout(refs=pull_refs, timeout=10, description="pull loop final check", logger=logger)
    
    except Exception as e:
        logger.error(f"[stop_pull_loops_and_check] Pull loop raised: {e}")
        return False, drained

    return True, drained

class InfiniteShardIterator:
    '''
        Wraps the rollout dataloader as an endless source of shards. When the
        dataloader is exhausted, advances an internal epoch counter, calls
        set_epoch() so the sampler reshuffles, and starts a fresh pass.
        next_shards() always returns a non-empty list of shards (one batch
        worth) ready to push into prompt_queue.
        epoch is bookkeeping only, the training loop in main() doesn't
        drive epochs from this counter, the sampler reshuffle uses it.
    '''
    def __init__(self, dataloader, num_rollout_engines, start_epoch=0):
        if len(dataloader) == 0:
            raise ValueError("InfiniteShardIterator: dataloader has zero batches; "
                             "rollout would never produce data. Check data files / "
                             "rollout_samples_per_epoch / rollout_batch_size_per_gpu.")
        self.dataloader = dataloader
        self.num_rollout_engines = num_rollout_engines
        self.epoch = start_epoch
        self.batch_iter = None
        self.reset_for_new_epoch()

    def reset_for_new_epoch(self):
        '''
            Advance the sampler to self.epoch and rebuild the batch iterator.
            Called on construction and on each StopIteration.
        '''
        if hasattr(self.dataloader, 'batch_sampler') and hasattr(self.dataloader.batch_sampler, 'set_epoch'):
            self.dataloader.batch_sampler.set_epoch(self.epoch)
        self.batch_iter = iter(self.dataloader)

    def next_shards(self):
        '''
            Pull one batch and shard it across engines. On exhaustion, bumps
            epoch and reshuffles. Always returns a non-empty list of shards.
            If two consecutive passes yield zero usable shards, raises rather
            than spinning forever.
        '''
        empty_passes = 0
        while True:
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                empty_passes += 1
                if empty_passes >= 2:
                    raise RuntimeError("InfiniteShardIterator: two consecutive empty passes "
                                       "over the dataloader produced zero shards. "
                                       "shard_batch_for_engines is dropping every batch.")
                self.epoch += 1
                self.reset_for_new_epoch()
                continue

            shards = shard_batch_for_engines(batch, self.num_rollout_engines)
            if shards:
                return shards

class ShardProducer:
    '''
        Daemon thread that keeps prompt_queue topped up from an
        InfiniteShardIterator. Rollout engines never run out of work as long
        as the producer thread is running.
        Lifecycle is start / stop only, no pause. Weight sync calls stop()
        to park the producer within a sec, runs its drain + broadcast + relaunch
        sequence, then calls start() to spawn a fresh thread. Threads are
        cheap and stop/restart avoids the race analysis a pause/idle protocol
        would require.
        Safety:
          - put() uses a 1s timeout so stop() takes effect within a sec even
            when prompt_queue is full (training is slow).
          - Exceptions in the thread land in self.error and are re-raised
            on the main thread via check_error().
          - daemon=True so the thread cannot survive driver exit.
    '''
    # Bounded put timeout in the producer thread. 1s seems working and gives
    # a worst-case latency for stop() call.
    PUT_TIMEOUT_S = 1.0
    # interval between heartbeat log lines from the thread
    HEARTBEAT_S = 30.0

    def __init__(self, prompt_queue, shard_iter, logger):
        self.prompt_queue = prompt_queue
        self.shard_iter   = shard_iter
        self.logger       = logger
        self.stop_event   = threading.Event()
        self.thread       = None
        self.error        = None
        self.shards_produced = 0

    def start(self):
        '''
            Spawn the daemon thread. No-op if already running. Clears any
            stale error from a previous run so check_error() doesn't surface
            an old crash on the freshly-started thread.
        '''
        if self.thread is not None and self.thread.is_alive():
            return
        self.error = None
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, name="ShardProducer", daemon=True)
        self.thread.start()
        self.logger.info(f"[ShardProducer] started (shards_produced_total={self.shards_produced})")

    def maybe_log_heartbeat(self, state, backpressured):
        '''
            Emit a heartbeat log line if HEARTBEAT_S has elapsed since the
            last one. state is a dict {last_time, last_count} mutated in
            place so the caller's bookkeeping survives across calls.
        '''
        now = time.time()
        if now - state['last_time'] < self.HEARTBEAT_S:
            return
        rate = (self.shards_produced - state['last_count']) / max(now - state['last_time'], 1e-9)
        status = "BACKPRESSURED (prompt_queue full)" if backpressured else "running"
        self.logger.info(f"[ShardProducer heartbeat] {status}, "
                         f"shards_produced={self.shards_produced}, "
                         f"rate={rate:.2f}/s over last {now-state['last_time']:.0f}s")
        state['last_time']  = now
        state['last_count'] = self.shards_produced

    def run(self):
        '''
            Pull a batch worth of shards, push them one at a time into prompt_queue.
            Drops the unpushed tail of the current batch if stop is requested mid-batch,
            the infinite iterator means the next batch on restart is fresh data, so
            we lose at most one batch worth of prompts per stop/start cycle.
        '''
        hb_state = {'last_time': time.time(), 'last_count': self.shards_produced}
        try:
            while not self.stop_event.is_set():
                shards = self.shard_iter.next_shards()
                for shard in shards:
                    while not self.stop_event.is_set():
                        try:
                            self.prompt_queue.put(shard, block=True, timeout=self.PUT_TIMEOUT_S)
                            self.shards_produced += 1
                            break
                        except RayQueueFull:
                            self.maybe_log_heartbeat(hb_state, backpressured=True)
                            continue

                    if self.stop_event.is_set():
                        break

                self.maybe_log_heartbeat(hb_state, backpressured=False)

        except Exception as e:
            self.logger.exception(f"[ShardProducer] crashed: {e}")
            self.error = e

    def stop(self, timeout=10.0):
        '''
            Signal the thread to exit and join it. Safe from atexit.
            Returns within ~PUT_TIMEOUT_S in the common case.
        '''
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            self.logger.warning(f"[ShardProducer] stop() timeout after {timeout}s — "
                                f"thread still alive (daemon, will die with process)")
        self.thread = None

    def check_error(self):
        '''
            Re-raise any exception captured by the producer thread. Driver
            should call this each train step so a producer crash surfaces
            fast instead of starving engines silently.
        '''
        if self.error is not None:
            err = self.error
            self.error = None
            raise RuntimeError(f"ShardProducer thread crashed: {err}") from err

def drain_results(results_queue, replay_buffer):
    '''
        Non-blocking function that pulls all available results from queue into replay buffer.
        Returns (num_batches_drained, accumulated rollout stats).
        Only RayQueueEmpty terminates the drain — other exceptions propagate.
    '''
    acc = rollout_stats.new_accumulator()
    drained = 0
    while True:
        try:
            result_list = results_queue.get(block=False)
        except RayQueueEmpty:
            break

        merged, stats = merge_rollout_with_stats([result_list])
        replay_buffer.add_batch_seqs(merged)
        rollout_stats.accumulate(acc, stats)
        drained += 1

    return drained, acc

def try_rebuild_shards(replay_buffer, train_batch_size, num_engines, seed,
                       epoch, shard_buffer_size, shard_rebuild_count,
                       min_new_samples=0, force=False, recency_decay=1.0,
                       current_policy_version=None, max_batches=None):
    '''
        Rebuild training shards if the replay buffer grew enough.
        force=True always rebuilds which is used at loop boundaries.
        min_new_samples: minimum number of new samples since last rebuild to trigger.
                        Use train_batch_size * num_engines so each engine gets at least
                        one new micro-batch from the rebuild.
        recency_decay / current_policy_version / max_batches: passed through
            to prepare_training_batches. max_batches=None builds the full buffer's worth.
        Returns (shard_refs, new_shard_buffer_size, new_shard_rebuild_count, batches, rebuild_ms) or None if skipped.
    '''
    buf_len = len(replay_buffer)
    if buf_len < train_batch_size:
        return None

    if not force and (buf_len - shard_buffer_size) < min_new_samples:
        return None

    rebuild_start = time.time()
    batches = prepare_training_batches(replay_buffer=replay_buffer,
                                       batch_size=train_batch_size,
                                       num_engines=num_engines,
                                       seed=seed,
                                       epoch=epoch + shard_rebuild_count,
                                       recency_decay=recency_decay,
                                       current_policy_version=current_policy_version,
                                       max_batches=max_batches)

    refs = shard_and_put(batches, num_engines=num_engines)
    rebuild_ms = (time.time() - rebuild_start) * 1000.0
    return refs, buf_len, shard_rebuild_count + 1, batches, rebuild_ms

def check_ess_sync(train_metrics, train_step_count, ess_sync_threshold, fixed_sync_interval, sync_triggered_this_epoch):
    '''
        Check if ESS or fixed_sync_interval triggers a sync.
        Returns (should_sync, ess_value).

        Semantics:
          - ESS-driven sync (only for P3O): gated by sync_triggered_this_epoch so we
            don't oscillate when ESS hovers around the threshold. At most
            one ESS-triggered sync per epoch.
          - fixed_sync_interval (deterministic schedule): NOT gated. Fires
            every N training steps regardless of how many syncs already
            happened this epoch. Set fixed_sync_interval >= steps_per_epoch
            for one sync per epoch.
    '''
    ess = train_metrics.get('ess_factor', None)

    # ESS-driven (P3O): one shot per epoch
    if not sync_triggered_this_epoch and ess is not None and ess < ess_sync_threshold:
        return True, ess

    # Fixed interval: deterministic, no gating
    if fixed_sync_interval and train_step_count > 0 and train_step_count % fixed_sync_interval == 0:
        return True, ess

    return False, ess

def compute_replay_buffer_size(rollout_samples_per_epoch, n_samples, max_lag,
                               train_batch_size_per_gpu, training_gpus,
                               gradient_accumulation_steps, steps_per_epoch):
    '''
        Derive the replay buffer hard cap from user-facing config knobs.
        Returns (replay_buffer_size, items_per_epoch, items_per_opt_step,
                 off_policy_bound, consumption_bound) so the caller can log
                 the breakdown without recomputing.
    '''
    items_per_epoch    = rollout_samples_per_epoch * n_samples
    items_per_opt_step = train_batch_size_per_gpu * training_gpus * gradient_accumulation_steps
    # epochs of old data tolerated before eviction must drop it
    off_policy_bound   = items_per_epoch * max(2, max_lag)
    # items the trainer will consume per epoch
    consumption_bound  = items_per_opt_step * steps_per_epoch
    replay_buffer_size = off_policy_bound + consumption_bound
    return replay_buffer_size, items_per_epoch, items_per_opt_step, off_policy_bound, consumption_bound

def compute_pipeline_queue_sizes(num_rollout_engines, max_lag,
                                  rollout_batch_size_per_gpu, n_samples,
                                  replay_buffer_size):
    '''
        Derive prompt_queue and results_queue capacities.
        Returns (prompt_queue_maxsize, results_queue_maxsize).
    '''
    # one shard per engine per burst
    shards_per_burst      = num_rollout_engines
    items_per_shard       = rollout_batch_size_per_gpu * n_samples
    prompt_queue_maxsize  = shards_per_burst * max(2, max_lag)
    # it should be large enough to absorb one buffer's worth without immediate backpressure
    results_queue_maxsize = max(prompt_queue_maxsize, replay_buffer_size // items_per_shard)
    return prompt_queue_maxsize, results_queue_maxsize

def log_driver_heartbeat(epoch, train_step_count, steps_per_epoch,
                         prompt_queue, prompt_queue_maxsize,
                         results_queue, results_queue_maxsize,
                         pull_refs, producer, replay_buffer,
                         since_last_step_s, logger):
    '''
        Snapshot of the live training-loop state. Cheap (only Ray queue
        qsize() RPCs and a non-blocking ray.wait), so safe to call on a
        wall-clock rhythm like 30s from inside the training loop. Catches
        "rollout went idle and nothing was logged" cases.
    '''
    try:
        pq_size = prompt_queue.qsize()
    except Exception:
        pq_size = -1

    try:
        rq_size = results_queue.qsize()
    except Exception:
        rq_size = -1

    ready_refs, _  = ray.wait(pull_refs, num_returns=len(pull_refs), timeout=0)
    alive_pulls    = len(pull_refs) - len(ready_refs)
    producer_alive = (producer.thread is not None and producer.thread.is_alive())
    since_str      = f"{since_last_step_s:.0f}s" if since_last_step_s is not None else "never"
    logger.info(f"[Epoch {epoch+1}] HEARTBEAT step={train_step_count}/{steps_per_epoch}, "
                f"buffer={len(replay_buffer)}, "
                f"prompt_q={pq_size}/{prompt_queue_maxsize}, "
                f"results_q={rq_size}/{results_queue_maxsize}, "
                f"producer_alive={producer_alive} (shards={producer.shards_produced}), "
                f"pull_loops_alive={alive_pulls}/{len(pull_refs)}, "
                f"since_last_train_step={since_str}")

def check_rollout_engines_health(rollout_engines):
    '''
        Two-stage ping: process alive (health) + pull thread responsive
        (pull concurrency_group, queues behind run_pull_loop). Returns list
        of (idx, reason) for dead engines.
        ping_mailbox timeout must exceed worst-case shard processing time
        such as reward function wall-clock to avoid false positives.
    '''

    dead = []
    for i, eng in enumerate(rollout_engines):
        try:
            ray.get(eng.ping.remote(), timeout=10)
        except Exception as e:
            dead.append((i, f"health: {e}"))
            continue
        try:
            ray.get(eng.ping_mailbox.remote(), timeout=120)
        except Exception as e:
            dead.append((i, f"mailbox wedged: {e}"))

    return dead

def perform_inline_sync(epoch, train_step_count, ess,
                         training_engines, rollout_engines,
                         prompt_queue, results_queue, pull_refs, producer,
                         replay_buffer, rollout_acc,
                         policy_version, rollout_policy_version, version_bumped_early,
                         rollout_timeout, sync_timeout, logger):
    '''
        Perform an inline mid-epoch nccl weight sync. Sequence:
          1. Stop the shard producer so it can't race with the drain.
          2. Push poison pills and fire ZeRO-3 gather on training engines. The gather is a 
             training-side-only collective and it does not involve rollout engines, so it runs in parallel with step 3.
          3. Wait for pull loops to exit, draining results continuously so engines blocked on results_queue.put()
             can unblock. The gather runs concurrently on training GPUs during this wait.
          4. On drain failure: wait for the in-flight gather to land, clear the pending state dict to avoid a CPU memory 
             leak, health-check engines, restart producer + pull loops with OLD version, return.
          5. On drain success: wait for gather (likely already done), bump policy_version, broadcast + finalize, relaunch 
             pull loops with NEW version, restart producer.
          6. On sync exception: fatal NCCL errors re-raise (job exits) and recoverable errors leave engines on stale weights for retry.

        Note, the infinite shard iterator means we don't need to preserve leftover shards, any in-flight prompts
        that didn't complete are simply dropped and the producer will pull fresh prompts on restart.
    '''
    num_engines = len(rollout_engines)
    sync_start = time.time()
    logger.info(f"[Epoch {epoch+1}][Step {train_step_count}] Sync triggered "
                f"(ESS={ess}), stopping producer + engines for NCCL weight sync")

    # Stop producer first so no new shards land in prompt_queue while we drain.
    t0 = time.time()
    producer.stop()
    logger.info(f"[Epoch {epoch+1}] Producer stopped in {time.time()-t0:.2f}s "
                f"(shards_produced_total={producer.shards_produced})")

    # Drain prompt_queue, push one pill per engine. Leftover shards are
    # dropped — the infinite iterator replays equivalent prompts on restart.
    stop_engines_and_drain(prompt_queue=prompt_queue,
                           num_rollout_engines=num_engines,
                           logger=logger)

    # Fire the ZeRO-3 gather on training engines BEFORE waiting for pull loops.
    # The gather is a training-side-only collective, no rollout participation,
    # so it runs concurrently with the pull-loop drain below.
    t_gather_start = time.time()
    gather_futures = start_nccl_gather(training_engines)
    logger.info(f"[Epoch {epoch+1}] Fired ZeRO-3 gather on {len(training_engines)} "
                f"training engines (overlapped with pull-loop drain)")

    # Wait for pull loops to exit. On timeout an engine is stuck in generate(),
    # so we abort sync to avoid wedging the broadcast collective.
    t0 = time.time()
    drain_ok, drained_during_wait = stop_pull_loops_and_check(pull_refs=pull_refs,
                                                              prompt_queue=prompt_queue,
                                                              results_queue=results_queue,
                                                              replay_buffer=replay_buffer,
                                                              rollout_acc=rollout_acc,
                                                              num_rollout_engines=num_engines,
                                                              timeout=rollout_timeout,
                                                              logger=logger,
                                                              push_pills=False)
    logger.info(f"[Epoch {epoch+1}] Pull loops drained in {time.time()-t0:.2f}s "
                f"(ok={drain_ok}, drained {drained_during_wait} results during wait)")

    if not drain_ok:
        logger.error(f"[Epoch {epoch+1}] Skipping inline sync to avoid weight "
                     f"broadcast deadlock; end-of-epoch sync will retry.")

        # The gather is still in flight on training engines. Let it finish.
        # Since it's a zero-3 collective, we can't cancel it without risking a
        # NCCL hang on the remaining ranks, so we need to let is finish.
        # Then clear the pending state dict to avoid a CPU memory leak.
        try:
            ray.get(gather_futures, timeout=sync_timeout)
        except Exception as gather_err:
            logger.warning(f"[Epoch {epoch+1}] In-flight gather failed during "
                           f"drain-failure cleanup (continuing): {gather_err}")
        try:
            ray.get(training_engines[0].clear_pending_nccl_state_dict.remote(), timeout=10)
        except Exception:
            pass

        # we don't have reinit_nccl_weight_sync_group fallback for now, so we make sure that
        # any missing rank fails fast and would wedge next broadcast otherwise.
        dead_engines = check_rollout_engines_health(rollout_engines)
        if dead_engines:
            raise RuntimeError(f"[Epoch {epoch+1}] Rollout engine health check "
                               f"failed after pull-loop drain failure. Dead "
                               f"engines: {dead_engines}. NCCL world_size is "
                               f"fixed; restart job.")

        # Force-cancel the wedged pull-loop tasks before relaunching. Without this, Ray keeps the old tasks
        # queued on the actor and the new run_pull_loop calls below stack up behind them, every subsequent
        # failed sync leaks another wedged task and eventually starves the actor's concurrency slots. force=True
        # interrupts the running task; recursive=True cancels any child tasks the pull loop spawned.
        for ref in pull_refs:
            try:
                ray.cancel(ref, force=True, recursive=True)
            except Exception as cancel_err:
                logger.warning(f"[Epoch {epoch+1}] ray.cancel on wedged pull_ref "
                               f"failed (continuing): {cancel_err}")

        # Drain leftover poison pills (and any stale shards) before relaunching.
        # stop_engines_and_drain pushed N pills but only K engines consumed theirs
        # before the timeout. The remaining N-K pills would kill the new pull
        # loops immediately, causing a fatal RuntimeError on the next ray.wait.
        drain_prompt_queue(prompt_queue)

        # Relaunch pull loops with OLD version, restart producer, return.
        new_pull_refs = [eng.run_pull_loop.remote(prompt_queue=prompt_queue,
                                                  results_queue=results_queue,
                                                  epoch=epoch,
                                                  policy_version=rollout_policy_version)
                         for eng in rollout_engines]
        producer.start()
        logger.warning(f"[Epoch {epoch+1}] Inline sync ABORTED (drain failed) in "
                       f"{time.time()-sync_start:.2f}s; engines relaunched with old version")

        return {'pull_refs': new_pull_refs,
                'policy_version': policy_version,
                'rollout_policy_version': rollout_policy_version,
                'version_bumped_early': version_bumped_early,
                'sync_triggered_this_epoch': False,}

    # Drain any results that landed between the wait and now.
    drained, drain_acc = drain_results(results_queue=results_queue, replay_buffer=replay_buffer)
    rollout_stats.accumulate(rollout_acc, drain_acc)

    # Bump version and sync. Training never lags rollout — assert it.
    assert policy_version >= rollout_policy_version, (f"Policy version invariant violated: policy_version={policy_version} "
                                                      f"< rollout_policy_version={rollout_policy_version}")
    if policy_version == rollout_policy_version:
        policy_version += 1
        version_bumped_early = True

    sync_triggered_this_epoch = False
    try:
        logger.info(f"[Epoch {epoch+1}] NCCL broadcast START (v{rollout_policy_version} -> v{policy_version})")
        t0 = time.time()

        # Complete the gather
        param_metadata = complete_nccl_gather(gather_futures=gather_futures,
                                              version=policy_version,
                                              logger=logger,
                                              sync_timeout=sync_timeout)
        gather_elapsed = time.time() - t_gather_start
        logger.info(f"[Epoch {epoch+1}] ZeRO-3 gather complete in {gather_elapsed:.2f}s "
                    f"({len(param_metadata)} params, overlapped with drain)")

        # Broadcast gathered weights to rollout engines and finalize.
        broadcast_and_finalize_nccl(training_engines=training_engines,
                                    rollout_engines=rollout_engines,
                                    param_metadata=param_metadata,
                                    version=policy_version,
                                    logger=logger,
                                    sync_timeout=sync_timeout)

        rollout_policy_version = policy_version
        sync_triggered_this_epoch = True
        logger.info(f"[Epoch {epoch+1}] NCCL broadcast DONE in {time.time()-t0:.2f}s, "
                    f"rollout_policy_version={rollout_policy_version}")

    except Exception as e:
        # Roll back the speculative version bump. The broadcast did not happen,
        # so policy_version should not have advanced.
        if version_bumped_early:
            policy_version -= 1
            version_bumped_early = False

        # Free rank-0 cached state dict to avoid CPU memory leak across retries.
        try:
            ray.get(training_engines[0].clear_pending_nccl_state_dict.remote(), timeout=10)

        except Exception:
            pass

        # Fail fast on nccl communicator destruction (watchdog, hardware, etc).
        if is_nccl_fatal_error(e):
            logger.error(f"[Epoch {epoch+1}] Inline NCCL sync failed with "
                        f"FATAL communicator error: {e}. No runtime reinit "
                        f"path in async mode — aborting.")
            raise
        # Non-fatal sync error: verify engines responsive before continuing.
        # A wedged engine would otherwise queue the next run_pull_loop behind
        # the broken broadcast and hang until NCCL_TIMEOUT.
        dead_engines = check_rollout_engines_health(rollout_engines)
        if dead_engines:
            raise RuntimeError(f"[Epoch {epoch+1}] Inline NCCL sync failed "
                               f"AND engines unresponsive: {dead_engines}. "
                               f"Aborting to avoid silent hang on the next "
                               f"sync attempt.")

        logger.warning(f"[Epoch {epoch+1}] Inline NCCL sync failed: {e}. "
                    f"Engines will resume with stale weights; end-of-epoch "
                    f"sync will retry.")

    # Relaunch pull loops with the (possibly updated) rollout_policy_version,
    # then restart the producer so generation resumes.
    new_pull_refs = [eng.run_pull_loop.remote(prompt_queue, results_queue, epoch, rollout_policy_version)
                     for eng in rollout_engines]
    producer.start()
    logger.info(f"[Epoch {epoch+1}] Inline sync DONE in {time.time()-sync_start:.2f}s "
                f"(triggered={sync_triggered_this_epoch}, rollout_v={rollout_policy_version})")

    return {'pull_refs': new_pull_refs,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'version_bumped_early': version_bumped_early,
            'sync_triggered_this_epoch': sync_triggered_this_epoch,}

def run_epoch_overlap(epoch, training_engines, rollout_engines,
                      prompt_queue, prompt_queue_maxsize,
                      results_queue, results_queue_maxsize,
                      pull_refs, producer,
                      replay_buffer, policy_version, rollout_policy_version, global_step,
                      train_batch_size, steps_per_epoch, seed, max_lag,
                      gradient_accumulation_steps,
                      ess_sync_threshold, fixed_sync_interval, recency_decay,
                      rollout_timeout, train_step_timeout, sync_timeout,
                      tracker, logger):
    '''
        One epoch of the queue-driven training loop. Generation runs continuously in the
        background: the shard producer keeps prompt_queue topped up from an infinite
        iterator, rollout engines pull shards via run_pull_loop and push results into
        results_queue. The driver drains results between training steps.
        Inline weight sync stops the producer, drains, broadcasts, relaunches the pull loops
        with a new version, and restarts the producer and the updated pull_refs flow back to main()
        via the return value.
        Since async only supports nccl sync, a failed sync grows lag and the next end-of-epoch
        sync attempt retries.
    '''
    epoch_start_time = time.time()
    num_train_engines = len(training_engines)
    generation_start_time = time.time()

    # Bound prepare_training_batches to one optimizer step's worth of micro-batches per call.
    # Without this, prepare_training_batches builds one micro-batch per train_batch_size items
    # in the entire buffer and ray.put's them all, which is O(buffer) per call. Capping makes per-call work constant.
    rebuild_max_batches = num_train_engines * gradient_accumulation_steps

    epoch_metrics    = {}
    train_step_count = 0
    sync_triggered_this_epoch = False

    # train_step_count when the latest inline sync fired, none if no sync this epoch.
    # this is used to detect post-sync drift and if more steps ran after the inline sync,
    # end-of-epoch sync is still needed.
    last_sync_step = None
    version_bumped_early = False
    shard_refs           = None
    shard_buffer_size    = 0
    shard_rebuild_count  = 0
    rollout_acc          = rollout_stats.new_accumulator()
    train_start_time     = time.time()
    cold_start_deadline  = time.time() + rollout_timeout

    # last_step_time stays None until the first successful training step;
    # the heartbeat reports "never" instead of a misleading "Xs since last step" when no step has happened yet.
    last_step_time       = None
    last_heartbeat       = time.time()
    HEARTBEAT_S          = 30.0

    while train_step_count < steps_per_epoch:
        # Surface any exception from the producer thread on the main thread
        # so a producer crash fails fast instead of starving engines silently.
        producer.check_error()

        # Driver heartbeat on a wall-clock cadence so a stuck queue or a long training
        # step never hides a stalled rollout side.
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_S:
            log_driver_heartbeat(epoch=epoch,
                                 train_step_count=train_step_count,
                                 steps_per_epoch=steps_per_epoch,
                                 prompt_queue=prompt_queue,
                                 prompt_queue_maxsize=prompt_queue_maxsize,
                                 results_queue=results_queue,
                                 results_queue_maxsize=results_queue_maxsize,
                                 pull_refs=pull_refs,
                                 producer=producer,
                                 replay_buffer=replay_buffer,
                                 since_last_step_s=(now - last_step_time) if last_step_time is not None else None,
                                 logger=logger)
            last_heartbeat = now

        # Detect dead pull loops fast: if any pull_ref has resolved, the engine has exited
        # and we MUST surface it before the next weight sync hits a stuck NCCL world.
        # ray.get raises the captured exception on failure.
        ready, _ = ray.wait(pull_refs, num_returns=len(pull_refs), timeout=0)
        if ready:
            ray.get(ready)
            raise RuntimeError(f"[Epoch {epoch+1}] {len(ready)}/{len(pull_refs)} "
                               f"rollout pull loop(s) exited unexpectedly mid-training. "
                               f"Cannot continue without restarting them; aborting.")

        # 1. Drain available results non-blocking into the replay buffer.
        drained, drain_acc = drain_results(results_queue, replay_buffer)
        rollout_stats.accumulate(rollout_acc, drain_acc)

        # 2. Rebuild training shards if buffer grew enough or first time after
        # data arrives. try_rebuild_shards uses shard_rebuild_count internally
        # as the per-rebuild offset.
        if drained > 0 or shard_refs is None:
            # Use the training-loop epoch for the rebuild seed, NOT shard_iter.epoch,
            # which is decoupled and can advance much faster. This keeps shuffles deterministic per training step.
            result = try_rebuild_shards(replay_buffer=replay_buffer,
                                        train_batch_size=train_batch_size,
                                        num_engines=num_train_engines,
                                        seed=seed,
                                        epoch=epoch * 1_000_000,
                                        shard_buffer_size=shard_buffer_size,
                                        shard_rebuild_count=shard_rebuild_count,
                                        min_new_samples=train_batch_size * num_train_engines,
                                        force=(shard_refs is None and len(replay_buffer) >= train_batch_size),
                                        recency_decay=recency_decay,
                                        current_policy_version=policy_version,
                                        max_batches=rebuild_max_batches)
            if result:
                shard_refs, shard_buffer_size, shard_rebuild_count, _, rebuild_ms = result
                if tracker:
                    # If shard rebuild cost grows above ~5% of train step time, we need to consider raising
                    # min_new_samples or moving rebuild to a background thread.
                    tracker.log_metrics({"train/rebuild_ms":  rebuild_ms,
                                         "train/rebuild_buf": shard_buffer_size,}, step=global_step)

        # 3. If shards aren't built yet, the buffer is below train_batch_size. Block briefly on results_queue (event-driven)
        # until more data arrives, then loop back to step 1. The producer keeps generation running in
        # the background, so this is a short wait at startup, not an idle gap. Bounded by cold_start_deadline
        # so a stuck rollout side fails fast instead of looping forever printing "still waiting".
        if shard_refs is None:
            if time.time() > cold_start_deadline:
                raise TimeoutError(f"[Epoch {epoch+1}] Cold start exceeded "
                                   f"rollout_timeout={rollout_timeout}s with "
                                   f"buffer={len(replay_buffer)} < batch={train_batch_size}. "
                                   f"Producer may be stuck or rollout engines are not generating.")
            try:
                first = results_queue.get(block=True, timeout=30.0)
                merged, stats = merge_rollout_with_stats([first])
                replay_buffer.add_batch_seqs(merged)
                rollout_stats.accumulate(rollout_acc, stats)
            except RayQueueEmpty:
                time_left = int(cold_start_deadline - time.time())
                logger.info(f"[Epoch {epoch+1}] still waiting for first rollouts "
                            f"(buffer={len(replay_buffer)} < batch={train_batch_size}, "
                            f"{time_left}s of {rollout_timeout}s budget left)")
            continue

        # 4. Run one training step.
        step_start = time.time()
        train_metrics = run_training_step(engines=training_engines,
                                          shard_refs=shard_refs,
                                          logger=logger,
                                          train_step_timeout=train_step_timeout)
        step_time = time.time() - step_start
        for k, v in train_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)
        global_step += 1
        train_step_count += 1

        # cold_start_deadline guards step 3 (the wait-for-first-rollouts
        # branch). After a successful train step we know generation is
        # healthy, so reset the budget for the next time shard_refs becomes
        # None (e.g., after a failed inline sync set it back to None at the
        # bottom of the loop).
        cold_start_deadline = time.time() + rollout_timeout
        # Track wall-clock of last successful step for the heartbeat log.
        last_step_time = time.time()

        if train_step_count % 10 == 0 or train_step_count == 1:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            logger.info(f"[Epoch {epoch+1}][Step {train_step_count}/{steps_per_epoch}] "
                        f"{metric_str}, step_time={step_time:.2f}s, "
                        f"buffer={len(replay_buffer)}, "
                        f"shards_produced={producer.shards_produced}")

        # Warn when results_queue saturates (engines backpressured on put).
        # Indicates training is the bottleneck and rollout GPUs are stalling.
        try:
            qsize = results_queue.qsize()
            if qsize >= int(0.9 * results_queue_maxsize):
                logger.warning(f"[Epoch {epoch+1}][Step {train_step_count}] "
                               f"results_queue near capacity ({qsize}/{results_queue_maxsize}); "
                               f"rollout engines are backpressured on put().")
        except Exception:
            qsize = None

        if tracker:
            tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()},
                               step=global_step)
            tracker.log_metrics({"train/step_time_sec": step_time}, step=global_step)

            # At epoch boundaries, the accumulator resets but the replay buffer carries over,
            # hence training can run multiple steps from buffered shards before the non-blocking
            # drain catches new results. Logging zeros during this window creates spurious dips in the tracker.
            if rollout_acc['total_samples_generated'] > 0:
                rollout_snapshot = rollout_stats.summarize(rollout_acc, rollout_time=time.time() - generation_start_time)
                rollout_log = {f"rollout/{k}": v for k, v in rollout_snapshot.items()}
                rollout_log["rollout/replay_buffer_size"] = len(replay_buffer)
                rollout_log["rollout/policy_lag"] = policy_version - rollout_policy_version
                rollout_log["rollout/shards_produced"] = producer.shards_produced
                if qsize is not None:
                    rollout_log["rollout/results_queue_qsize"] = qsize
                tracker.log_metrics(rollout_log, step=global_step)

        # 5. Check ESS or fixed_sync_interval for inline weight sync.
        should_sync, ess = check_ess_sync(train_metrics=train_metrics,
                                          train_step_count=train_step_count,
                                          ess_sync_threshold=ess_sync_threshold,
                                          fixed_sync_interval=fixed_sync_interval,
                                          sync_triggered_this_epoch=sync_triggered_this_epoch)

        if tracker and ess is not None:
            tracker.log_metrics({"nccl/ess_factor": ess,
                                 "nccl/sync_triggered": 1 if (sync_triggered_this_epoch or should_sync) else 0,
                                }, step=global_step)

        # 6. Inline nccl weight sync in mid-epoch. Stops the producer, drains
        # pull loops, broadcasts weights, relaunches pull loops, restarts the
        # producer. Inline sync is allowed on the last epoch since it benefits the
        # remaining training steps.
        if should_sync:
            inline_sync_start = time.time()
            sync_state = perform_inline_sync(epoch=epoch,
                                             train_step_count=train_step_count,
                                             ess=ess,
                                             training_engines=training_engines,
                                             rollout_engines=rollout_engines,
                                             prompt_queue=prompt_queue,
                                             results_queue=results_queue,
                                             pull_refs=pull_refs,
                                             producer=producer,
                                             replay_buffer=replay_buffer,
                                             rollout_acc=rollout_acc,
                                             policy_version=policy_version,
                                             rollout_policy_version=rollout_policy_version,
                                             version_bumped_early=version_bumped_early,
                                             rollout_timeout=rollout_timeout,
                                             sync_timeout=sync_timeout,
                                             logger=logger)
            pull_refs              = sync_state['pull_refs']
            policy_version         = sync_state['policy_version']
            rollout_policy_version = sync_state['rollout_policy_version']
            version_bumped_early   = sync_state['version_bumped_early']

            if sync_state['sync_triggered_this_epoch']:
                sync_triggered_this_epoch = True
                last_sync_step = train_step_count

            inline_sync_ms = (time.time() - inline_sync_start) * 1000.0
            if tracker:
                # Total inline sync wall time (drain + broadcast + relaunch).
                # Sum across epoch / total epoch wall time = bubble fraction.
                tracker.log_metrics({"nccl/inline_sync_ms": inline_sync_ms,}, step=global_step)

            # If sync didn't actually fire (drain failure path), skip the rest of this iter so
            # we don't loop on stale shard_refs built before the relaunch. The next iter starts fresh.
            if not sync_state['sync_triggered_this_epoch']:
                shard_refs = None
                continue

    # 7. Post-training bookkeeping. Bump policy_version once per epoch to reflect the
    # optimizer updates:
    #   - No inline sync this epoch --> bump (version_bumped_early=False)
    #   - Inline sync at the LAST step --> already bumped during sync, skip
    #   - Inline sync mid-epoch with more steps after --> bump again so the
    #     end-of-epoch sync gate sees the post-sync drift via lag.
    sync_covers_final_weights = (last_sync_step is not None and last_sync_step >= train_step_count)

    if train_step_count > 0 and (not version_bumped_early or not sync_covers_final_weights):
        policy_version += 1

    # Only evict if we trained. A no-train epoch (cold-start abort, etc.) leaves the buffer
    # untouched for the next epoch's retry.
    if train_step_count > 0:
        evicted = replay_buffer.evict_stale(policy_version - max_lag)
        if evicted > 0:
            logger.info(f"[Epoch {epoch+1}] Post-training eviction: {evicted} stale samples removed, "
                        f"{len(replay_buffer)} retained for next epoch")

    # 8. Aggregate rollout stats.
    generation_time = time.time() - generation_start_time
    rollout_metrics = rollout_stats.summarize(rollout_acc, rollout_time=generation_time)
    rollout_metrics["rollout_time_with_overlap"] = time.time() - epoch_start_time

    train_time = time.time() - train_start_time

    return {'rollout_metrics': rollout_metrics,
            'epoch_metrics': epoch_metrics,
            'global_step': global_step,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'pull_refs': pull_refs,
            'train_step_count': train_step_count,
            'train_time': train_time,
            'sync_performed': sync_covers_final_weights}

def main(args, config):
    '''
        This is the main entry point for the rl_async training process.
    '''
    ########
    # 1. Miscellaneous setups
    ########
    # remember that main_rl.py is an orchestrator script,
    # not a distributed worker, so rank is always 0 here.
    rank = 0

    # Setup logging
    logger = setup_logging(rank=rank, log_level=args.log_level)
    logger.info(f"Starting RL training...")

    set_random_seeds(seed=config.run.seed)

    # setup remote experiment tracker
    tracker = setup_tracker(config=config, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    # number of gpus for training which is used by deepspeed
    training_gpus = config.run.training_gpus
    # number of gpus for rollout generation which is used by vllm
    rollout_gpus  = config.run.rollout_gpus

    ########
    # 2. initialize ray
    ########
    logger.info(f"Initializing Ray ...")
    master_addr = setup_ray(ray_address=config.run.ray_address)
    logger.info(f"Ray initialized. Master address: {master_addr}")
    # registers ray.shutdown as a function to be called
    # automatically when the python process exits. Without it, orphaned ray
    # processes can linger and hold onto gpu memory after the script dies.
    atexit.register(ray.shutdown)

    cluster_gpus = ray.cluster_resources().get("GPU", 0)
    needed_gpus = training_gpus + rollout_gpus
    if needed_gpus > cluster_gpus:
        raise ValueError(f"Need {needed_gpus} GPUs (training={training_gpus} + rollout={rollout_gpus}) "
                         f"but Ray cluster only has {int(cluster_gpus)} GPUs")

    ########
    # 3. Initialize training engine
    ########
    logger.info(f"Loading training algorithm: {config.train.alg_name}")
    alg_class = load_algorithm(config.train.alg_name, Algorithm_Registry)

    training_engines = create_training_engines(params=config,
                                               alg=alg_class,
                                               world_size=training_gpus,
                                               master_addr=master_addr,
                                               master_port=config.run.ray_master_port)

    assert len(training_engines) == training_gpus, "Number of training engines does not match number of training gpus"
    logger.info(f"Created {len(training_engines)} training engine runners")

    # Synchronization barrier to prevent deepspeed rendezvous hang
    # wait for all training actors to finish initialization before proceeding
    logger.info("Waiting for all training engines to initialize...")

    init_timeout = config.run.init_timeout
    ready_checks = [engine.is_ready.remote() for engine in training_engines]
    ray_get_with_timeout(refs=ready_checks,
                         timeout=init_timeout,
                         description="training engine initialization",
                         logger=logger)
    logger.info("All training engines ready!")

    ########
    # 4. load tokenizer
    ########
    logger.info(f"Loading tokenizer from {config.model.name}")
    tokenizer = load_tokenizer(model_name=config.model.name,
                               trust_remote_code=config.model.trust_remote_code,
                               rank=rank)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token ID: {tokenizer.pad_token_id}")

    ########
    # 5. Initialize rollout engines
    ########
    logger.info("Setting up rollout engines...")
    reward_func_name = config.reward.reward_func if config.reward.reward_func else None
    if reward_func_name:
        reward_module = importlib.import_module("rewards." + reward_func_name)
        reward_fnc    = getattr(reward_module, "compute_score")
        logger.info(f"Using reward function: {reward_func_name}")

    else:
        raise ValueError("Reward function not specified")

    rollout_engines = create_rollout_engines(params=config,
                                             reward_fnc=reward_fnc,
                                             eos_id=tokenizer.eos_token_id)
    num_rollout_engines = len(rollout_engines)
    logger.info(f"Created {num_rollout_engines} rollout engines with TP={config.rollout.tensor_parallel_size}")

    # Wait for vllm AsyncLLM init to finish on every rollout engine. If there is an error,
    # init fails fast with a clear error.
    logger.info(f"Waiting for {num_rollout_engines} rollout engines to finish vLLM init...")
    rollout_ready_refs = [eng.ping.remote() for eng in rollout_engines]
    ray_get_with_timeout(refs=rollout_ready_refs,
                         timeout=init_timeout,
                         description="rollout engine vLLM initialization",
                         logger=logger)
    logger.info(f"All {num_rollout_engines} rollout engines ready!")

    ########
    # 6. NCCL weight sync setup.
    ########
    nccl_port = config.run.nccl_sync_port if config.run.nccl_sync_port else config.run.ray_master_port + 100
    nccl_sync_backend = config.run.nccl_sync_backend

    ########
    # 7. load the rollout dataloader
    ########
    logger.info(f"Loading rollout dataloader from {config.data.train_files_path}")
    rollout_dataloader = create_rollout_dataloader(params=config,
                                                  tokenizer=tokenizer,
                                                  num_rollout_engines=num_rollout_engines,
                                                  samples_per_epoch=config.rollout.rollout_samples_per_epoch)

    logger.info(f"Rollout dataloader with {len(rollout_dataloader)} batches/machine, "
                f"n_samples={config.rollout.n_samples} per prompt")

    ########
    # 8. Some variables initialization
    ########
    number_of_epochs = config.train.total_number_of_epochs
    steps_per_epoch  = config.train.train_steps_per_epoch
    checkpoint_save_interval = config.run.checkpoint_save_interval if config.run.checkpoint_save_interval is not None else 1

    # Timeout settings (seconds) for ray.get() calls
    rollout_timeout    = config.run.rollout_timeout
    train_step_timeout = config.run.train_step_timeout
    save_timeout = config.run.save_timeout
    sync_timeout = config.run.sync_timeout

    # Overlap settings
    overlap_max_lag = config.overlap.max_lag
    ess_sync_threshold = config.overlap.ess_sync_threshold
    fixed_sync_interval = config.overlap.fixed_sync_interval
    # Recency-weighted replay sampling. 1.0 = uniform; <1.0 biases sampling
    # toward fresher policy versions via decay**(current_v - item_v).
    recency_decay = config.overlap.recency_decay

    # Hard cap the replay buffer with a deque so per-iter work in prepare_training_batches
    # stays bounded as training runs.
    (replay_buffer_size, items_per_epoch, items_per_opt_step,
     off_policy_bound, consumption_bound) = compute_replay_buffer_size(rollout_samples_per_epoch=config.rollout.rollout_samples_per_epoch,
                                                                       n_samples=config.rollout.n_samples,
                                                                       max_lag=overlap_max_lag,
                                                                       train_batch_size_per_gpu=config.train.train_batch_size_per_gpu,
                                                                       training_gpus=training_gpus,
                                                                       gradient_accumulation_steps=config.train.gradient_accumulation_steps,
                                                                       steps_per_epoch=steps_per_epoch,
                                                                       )
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 max_size=replay_buffer_size,
                                 )
    logger.info(f"Replay buffer: deque(maxlen={replay_buffer_size}) "
                f"[off_policy={off_policy_bound} (items_per_epoch={items_per_epoch} x lag={max(2, overlap_max_lag)}) "
                f"+ consumption={consumption_bound} (items_per_opt_step={items_per_opt_step} x "
                f"train_passes={steps_per_epoch})], max_seq_len={config.data.max_seq_len}, "
                f"oldest evicted on insert.")

    ########
    # 9. Resume from checkpoint if requested and clean up incomplete checkpoint directories
    ########
    start_epoch = 0
    global_step = 0
    policy_version = 0
    rollout_policy_version = 0

    if args.resume_from:
        # Resume uses "direct" sync (Ray object store) as nccl group not yet initialized.
        start_epoch, policy_version, global_step = load_checkpoint_for_resume(resume_path=args.resume_from,
                                                                              training_engines=training_engines,
                                                                              rollout_engines=rollout_engines,
                                                                              weight_sync_method="direct",
                                                                              logger=logger,
                                                                              sync_timeout=sync_timeout,
                                                                              save_timeout=save_timeout,
                                                                              sync_fn=sync_weights_direct,
                                                                              refresh_fn=refresh_rollout_engine)
        rollout_policy_version = policy_version
        logger.info(f"Resuming from epoch {start_epoch+1}, policy_version={policy_version}, global_step={global_step}")

    ########
    # 9b. Initialize NCCL weight sync group, after resume, so engines are fresh
    ########
    nccl_world_size, nccl_gname = reinit_nccl_weight_sync_group(training_engines=training_engines,
                                                                rollout_engines=rollout_engines,
                                                                master_addr=master_addr,
                                                                nccl_port=nccl_port,
                                                                tp_size=int(config.rollout.tensor_parallel_size),
                                                                logger=logger,
                                                                init_timeout=config.run.init_timeout,
                                                                backend=nccl_sync_backend)
    logger.info(f"Weight sync: NCCL (port={nccl_port}, world_size={nccl_world_size}) with NCCL group name {nccl_gname}")

    # Clean up incomplete checkpoint directories from previous crashed runs.
    # Only directories missing the CHECKPOINT_COMPLETE marker are removed.
    experiment_dir = os.path.join(config.run.checkpoint_dir, config.run.experiment_id)
    if os.path.isdir(experiment_dir):
        for entry in os.listdir(experiment_dir):
            ckpt_path = os.path.join(experiment_dir, entry)

            if os.path.isdir(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "CHECKPOINT_COMPLETE")):
                logger.warning(f"Removing incomplete checkpoint: {ckpt_path}")
                shutil.rmtree(ckpt_path, ignore_errors=True)

    ########
    # 10. General logging printout before training-loop
    ########
    model_info       = ray.get(training_engines[0].get_model_info.remote())
    total_params     = model_info['total_params']
    trainable_params = model_info['trainable_params']
    frozen_params    = model_info['frozen_params']

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    if model_info['peft_enabled']:
        logger.info(f"Model: {config.model.name} | PEFT: {model_info['peft_type']} | "
                    f"params: {total_params:,} total, {trainable_params:,} peft ({100*trainable_params/total_params:.2f}%), "
                    f"{frozen_params:,} frozen")

    else:
        logger.info(f"Model: {config.model.name} | PEFT: off | "
                    f"params: {total_params:,} total, {trainable_params:,} trainable")

    if 'value_total_params' in model_info:
        logger.info(f"Value model: {config.model.value_model or config.model.name} | "
                    f"params: {model_info['value_total_params']:,} total, {model_info['value_trainable_params']:,} trainable")

    logger.info(f"Weight sync method: nccl (backend={nccl_sync_backend})")
    logger.info(f"Overlap mode: max_lag={overlap_max_lag}, "
                f"ess_sync_threshold={ess_sync_threshold}, fixed_sync_interval={fixed_sync_interval}")

    logger.info(f"checkpoint_save_interval: {checkpoint_save_interval}")
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from} (epoch {start_epoch+1}/{number_of_epochs})")

    logger.info("=" * 50)

    ########
    # 11. Persistent rollout queues, shard producer, pull loops.
    ########
    prompt_queue_maxsize, results_queue_maxsize = compute_pipeline_queue_sizes(num_rollout_engines=num_rollout_engines,
                                                                                max_lag=overlap_max_lag,
                                                                                rollout_batch_size_per_gpu=config.rollout.rollout_batch_size_per_gpu,
                                                                                n_samples=config.rollout.n_samples,
                                                                                replay_buffer_size=replay_buffer_size)
    prompt_queue  = RayQueue(maxsize=prompt_queue_maxsize)
    results_queue = RayQueue(maxsize=results_queue_maxsize)
    logger.info(f"Queues: prompt_queue maxsize={prompt_queue_maxsize}, "
                f"results_queue maxsize={results_queue_maxsize} (both in shard units)")

    # Built once and reused across all epochs so engines never go idle at epoch boundaries.
    # The shard producer runs as a daemon thread and keeps prompt_queue topped up from an infinite iterator
    # over the dataloader. Weight sync stops the producer, drains, broadcasts, relaunches pull loops, and
    # restarts the producer.
    shard_iter = InfiniteShardIterator(dataloader=rollout_dataloader, num_rollout_engines=num_rollout_engines, start_epoch=start_epoch)
    producer   = ShardProducer(prompt_queue=prompt_queue, shard_iter=shard_iter, logger=logger)
    producer.start()

    # Register producer.stop BEFORE ray.shutdown so on any abnormal exit, the daemon thread is joined cleanly
    # before Ray tears down the queue actors it depends on. atexit runs handlers in LIFO order.
    atexit.register(producer.stop)

    # Launch pull loops once. They survive across epochs and are only restarted by weight sync (drain -> broadcast -> relaunch).
    # The epoch arg is the virtual epoch tracked by shard_iter; run_pull_loop only uses it for logging and seed math.
    pull_refs = [eng.run_pull_loop.remote(prompt_queue, results_queue, shard_iter.epoch, rollout_policy_version)
                 for eng in rollout_engines]

    assert len(pull_refs) == num_rollout_engines, f"pull_refs dispatch returned {len(pull_refs)} refs, expected {num_rollout_engines}"
    logger.info(f"Launched {len(pull_refs)} pull loops at policy_version={rollout_policy_version}")

    ########
    # 12. Training and rollout loop
    ########
    entire_training_start_time = time.time()

    for epoch in range(start_epoch, number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == number_of_epochs - 1)

        result = run_epoch_overlap(epoch=epoch,
                                   training_engines=training_engines,
                                   rollout_engines=rollout_engines,
                                   prompt_queue=prompt_queue,
                                   prompt_queue_maxsize=prompt_queue_maxsize,
                                   results_queue=results_queue,
                                   results_queue_maxsize=results_queue_maxsize,
                                   pull_refs=pull_refs,
                                   producer=producer,
                                   replay_buffer=replay_buffer,
                                   policy_version=policy_version,
                                   rollout_policy_version=rollout_policy_version,
                                   global_step=global_step,
                                   train_batch_size=config.train.train_batch_size_per_gpu,
                                   steps_per_epoch=steps_per_epoch,
                                   seed=config.run.seed,
                                   max_lag=overlap_max_lag,
                                   gradient_accumulation_steps=config.train.gradient_accumulation_steps,
                                   ess_sync_threshold=ess_sync_threshold,
                                   fixed_sync_interval=fixed_sync_interval,
                                   recency_decay=recency_decay,
                                   rollout_timeout=rollout_timeout,
                                   train_step_timeout=train_step_timeout,
                                   sync_timeout=sync_timeout,
                                   tracker=tracker,
                                   logger=logger)

        # Unpack result. pull_refs may have been relaunched by inline sync.
        global_step            = result['global_step']
        policy_version         = result['policy_version']
        rollout_metrics        = result['rollout_metrics']
        rollout_policy_version = result['rollout_policy_version']
        pull_refs              = result['pull_refs']

        # Log rollout metrics
        time_str = f"time={rollout_metrics['rollout_time']:.2f}s"
        if 'rollout_time_with_overlap' in rollout_metrics:
            time_str += f" (wall_time={rollout_metrics['rollout_time_with_overlap']:.2f}s)"

        # rollout metrics are streamed to the tracker per training step, here we just print them out.
        logger.info(f"[Epoch {epoch + 1}] Rollout complete: {rollout_metrics['total_samples_generated']} samples, "
                    f"avg_reward={rollout_metrics['avg_reward']:.4f}, reward_std={rollout_metrics['reward_std']:.4f}, "
                    f"reward_min={rollout_metrics['reward_min']:.4f}, reward_max={rollout_metrics['reward_max']:.4f}, "
                    f"frac_positive_reward={rollout_metrics['frac_positive_reward']:.4f}, "
                    f"avg_response_len={rollout_metrics['avg_response_len']:.1f}, "
                    f"response_len_std={rollout_metrics['response_len_std']:.1f}, "
                    f"min_response_len={rollout_metrics['min_response_len']:.1f}, "
                    f"max_response_len={rollout_metrics['max_response_len']:.1f}, "
                    f"truncated_ratio={rollout_metrics['truncated_ratio']:.4f}, "
                    f"seq_truncated_ratio={rollout_metrics['seq_truncated_ratio']:.4f}, "
                    f"eos_ratio={rollout_metrics['eos_ratio']:.4f}, "
                    f"mean_logprob={rollout_metrics['mean_logprob']:.4f}, "
                    f"unique_response_ratio={rollout_metrics['unique_response_ratio']:.4f}, "
                    f"{time_str}, tps={rollout_metrics['tokens_per_sec']:.2f}")

        # Log training summary
        epoch_avg = {k: np.mean(v) for k, v in result['epoch_metrics'].items()}

        train_stats = ray.get(training_engines[0].get_training_stats.remote())
        current_lr  = train_stats.get('lr', 0.0)
        gpu_mem_gb  = train_stats.get('gpu_peak_mem_gb', 0.0)

        logger.info(f"[Epoch {epoch+1}] Training complete: {result['train_step_count']} steps, "
                    f"time={result['train_time']:.2f}s, "
                    f"avg_loss={epoch_avg.get('loss_total', 0.0):.4f}, "
                    f"avg_kl_ref={epoch_avg.get('kl_ref', 0.0):.4f}, "
                    f"avg_approx_kl={epoch_avg.get('approx_kl', 0.0):.6f}, "
                    f"lr={current_lr:.2e}, gpu_peak_mem={gpu_mem_gb:.2f}GB")

        if tracker:
            tracker.log_metrics({"train/epoch_time_sec": result['train_time'],
                                 "train/lr": current_lr,
                                 "train/gpu_peak_mem_gb": gpu_mem_gb,
                                }, step=global_step)

        # If an inline sync already fired and covers the latest weights, skip the end-of-epoch sync.
        # Otherwise gate on lag and sync if it grew past max_lag. perform_inline_sync handles the
        # producer-stop / drain / broadcast / relaunch / producer-start dance end-to-end.
        sync_success = result['sync_performed']
        if not sync_success and not is_last_epoch:
            lag = policy_version - rollout_policy_version
            if lag >= overlap_max_lag:
                logger.info(f"[Epoch {epoch+1}] End-of-epoch NCCL sync "
                            f"(v{rollout_policy_version} -> v{policy_version})...")
                # version_bumped_early is False here because the EoE path always has
                # policy_version > rollout_policy_version (if lag >= overlap_max_lag gate above guarantees this),
                # so perform_inline_sync's "equal version" branch never fires.
                sync_state = perform_inline_sync(epoch=epoch,
                                                 train_step_count=result['train_step_count'],
                                                 ess=None,
                                                 training_engines=training_engines,
                                                 rollout_engines=rollout_engines,
                                                 prompt_queue=prompt_queue,
                                                 results_queue=results_queue,
                                                 pull_refs=pull_refs,
                                                 producer=producer,
                                                 replay_buffer=replay_buffer,
                                                 rollout_acc=rollout_stats.new_accumulator(),
                                                 policy_version=policy_version,
                                                 rollout_policy_version=rollout_policy_version,
                                                 version_bumped_early=False,
                                                 rollout_timeout=rollout_timeout,
                                                 sync_timeout=sync_timeout,
                                                 logger=logger)
                pull_refs              = sync_state['pull_refs']
                policy_version         = sync_state['policy_version']
                rollout_policy_version = sync_state['rollout_policy_version']
                sync_success           = sync_state['sync_triggered_this_epoch']
            else:
                logger.info(f"[Epoch {epoch+1}] Skipping weight sync (lag={lag}, max_lag={overlap_max_lag})")

        # Periodic or final checkpoint save. Save runs in the foreground while pull loops keep generating
        # in the background, no pre-launch needed. checkpoint_save_interval == 0 means "never save periodically";
        # the final epoch is still saved so the run produces at least one ckpt.
        should_save_disk = (checkpoint_save_interval > 0 and (epoch + 1) % checkpoint_save_interval == 0)

        if should_save_disk or is_last_epoch:
            model_path = save_checkpoint(epoch=epoch,
                                         version=policy_version,
                                         global_step=global_step,
                                         tokenizer=tokenizer,
                                         training_engines=training_engines,
                                         checkpoint_dir=config.run.checkpoint_dir,
                                         experiment_id=config.run.experiment_id,
                                         rank=rank,
                                         logger=logger,
                                         save_timeout=save_timeout)
            logger.info(f"[Epoch {epoch+1}] Saved disk checkpoint at {model_path}")

        # NCCL sync metrics
        if tracker:
            tracker.log_metrics({"nccl/policy_version": policy_version,
                                 "nccl/rollout_policy_version": rollout_policy_version,
                                 "nccl/policy_lag": policy_version - rollout_policy_version,
                                 "nccl/sync_success": 1 if sync_success else 0,
                                }, step=global_step)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
        logger.info("=" * 50)

    ########
    # 13. Cleanup
    ########
    if tracker:
        tracker.finish()

    entire_training_time = time.time() - entire_training_start_time
    logger.info(f"Training completed successfully! Total time: {entire_training_time:.2f}s ({entire_training_time/3600:.2f}h)")

    # Stop the shard producer first so it can't push fresh shards while we
    # drain the pull loops. Producer.stop() returns within ~1s.
    logger.info("[Cleanup] Stopping shard producer...")
    producer.stop()

    # Drain pull loops cleanly: drain leftover real shards from prompt_queue,
    # push one poison pill per engine, wait for the loops to exit. Bounded by
    # rollout_timeout so a stuck engine can't hang shutdown indefinitely.
    logger.info("[Cleanup] Draining rollout pull loops...")
    stop_engines_and_drain(prompt_queue=prompt_queue,
                           num_rollout_engines=num_rollout_engines,
                           logger=logger)
    try:
        stop_pull_loops_and_check(pull_refs=pull_refs,
                                  prompt_queue=prompt_queue,
                                  results_queue=results_queue,
                                  replay_buffer=replay_buffer,
                                  rollout_acc=rollout_stats.new_accumulator(),
                                  num_rollout_engines=num_rollout_engines,
                                  timeout=rollout_timeout,
                                  logger=logger,
                                  push_pills=False)
    except Exception as e:
        logger.warning(f"[Cleanup] Pull loop drain raised: {e}")

    # Tear down NCCL weight sync groups (always initialized in async mode).
    logger.info("[Cleanup] Closing NCCL weight sync groups...")
    try:
        ray.get(training_engines[0].close_weight_nccl_group.remote())

    except Exception as e:
        logger.warning(f"[Cleanup] Failed to close training NCCL group: {e}")

    for eng in rollout_engines:
        try:
            ray.get(eng.close_nccl_group.remote())
        except Exception:
            pass

    # Clean up process groups before ray tears down actors.
    shutdown_futures = [engine.shutdown.remote() for engine in training_engines]
    try:
        ray.get(shutdown_futures, timeout=30)

    except Exception:
        pass

    ray.shutdown()
    logger.info("Done!")