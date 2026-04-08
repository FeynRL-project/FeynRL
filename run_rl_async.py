import os
import atexit
import numpy as np
import importlib
import ray
import time
import shutil

# imports local methods, classes, etc.
from misc.utils import load_algorithm, ray_get_with_timeout, set_random_seeds
from ray.util.queue import Queue as RayQueue, Empty as RayQueueEmpty
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
                            sync_weights_nccl)


# Sentinel value pushed into prompt_queue to signal an engine to stop. Must
# match the value in VLLMRolloutEngineAsync.run_pull_loop. One poison pill per
# engine causes exactly one engine to exit.
POISON_PILL = "__STOP__"

def drain_prompt_queue(prompt_queue):
    '''
        Drain all items from prompt_queue (real shards and stale pills),
        returning the real shards in FIFO order. Stale pills are discarded.
        Only RayQueueEmpty terminates the drain, other exceptions propagate
        so the caller can detect a real Ray failure rather than silently
        leaving items in the queue.
    '''
    leftover = []
    while True:
        try:
            item = prompt_queue.get(block=False)

        except RayQueueEmpty:
            break

        if isinstance(item, str) and item == POISON_PILL:
            continue
        leftover.append(item)
    return leftover

def stop_engines_and_drain(prompt_queue, num_rollout_engines):
    '''
        Drains all unconsumed real shards from prompt_queue, then pushes 
        one poison pill per engine. Returns the drained shards in FIFO order 
        so the caller can re-enqueue them after sync. Discards any stale 
        poison pills found in the queue.
        After this returns, every engine's next prompt_queue.get() will yield
        a poison pill — either immediately (if it was waiting on get()) or
        after its current generate() finishes and pushes a result.
        Stop latency = max in-flight generate() duration, which is unavoidable.
    '''
    leftover = drain_prompt_queue(prompt_queue)
    for _ in range(num_rollout_engines):
        prompt_queue.put(POISON_PILL)

    return leftover

def wait_for_pull_loops_with_drain(pull_refs, results_queue, replay_buffer, rollout_acc, timeout, logger):
    '''
        Wait for pull loops to exit, draining results_queue continuously so
        engines blocked on results_queue.put() (bounded queue backpressure)
        can unblock and reach their next prompt_queue.get() where they will
        see a poison pill.
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

def fill_prompt_queue(dataloader, rollout_engines, prompt_queue, logger):
    '''
        Shard every dataloader batch across engines and enqueue each shard.
        Engines pull shards from the queue via run_pull_loop (self-scheduling).
        Returns the total number of shards enqueued.
        After all shards are enqueued, push exactly num_engines POISON_PILLs
        as explicit "epoch done" sentinels. With this, run_pull_loop never
        needs a polling timeout — engines block on get() until they receive
        either a real shard, an inline-sync pill, or this end-of-epoch pill.
        This eliminates the 5-second false-exit window where a slow driver
        rebuild could starve the queue and cause an engine to exit early.
    '''
    num_engines  = len(rollout_engines)
    total_shards = 0
    for rollout_batch in dataloader:
        # example: 10 prompts, 3 engines, batch size 5
        # batch 1: shard_size = ceil(5/3) = 2 -> shards [2, 2, 1]
        # batch 2: same as above
        # results: 6 shards in the queue, varying sizes (2, 2, 1, 2, 2, 1)
        # it doesn't assign shard to an engine though
        shards = shard_batch_for_engines(rollout_batch, num_engines)
        for shard in shards:
            prompt_queue.put(shard)
            total_shards += 1

    for _ in range(num_engines):
        prompt_queue.put(POISON_PILL)

    return total_shards

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

def drain_results_blocking(results_queue, replay_buffer, remaining, logger, timeout):
    '''
        Blocking drain: wait for remaining results after pull loops finish.
        Uses a single wall-clock deadline (not a per-item timeout) so a stuck
        queue fails fast instead of waiting timeout * remaining seconds.
        Returns (drained_count, accumulated stats). drained_count <= remaining
        on timeout.
    '''
    acc = rollout_stats.new_accumulator()
    deadline = time.time() + timeout
    last_log = time.time()
    drained_here = 0
    while drained_here < remaining:
        time_left = deadline - time.time()
        if time_left <= 0:
            logger.warning(f"[drain_results_blocking] Wall-clock timeout after "
                           f"{timeout}s with {remaining - drained_here}/{remaining} "
                           f"results still missing")
            break
        # Cap each get at 30s so we wake periodically to log progress —
        # otherwise long waits look like silent hangs to the user.
        try:
            result_list = results_queue.get(block=True, timeout=min(time_left, 30.0))
        except RayQueueEmpty:
            if time.time() - last_log >= 30.0:
                logger.info(f"[drain_results_blocking] still waiting: "
                            f"{drained_here}/{remaining} drained, "
                            f"{int(time_left)}s of {timeout}s budget left")
                last_log = time.time()
            continue
        merged, stats = merge_rollout_with_stats([result_list])
        replay_buffer.add_batch_seqs(merged)
        rollout_stats.accumulate(acc, stats)
        drained_here += 1
    return drained_here, acc

def wait_for_first_rollouts(results_queue, replay_buffer, rollout_acc,
                             generation_start_time, rollout_timeout, epoch,
                             logger, pull_refs):
    '''
        Cold-start: block on results_queue until first result arrives,
        then drain any extras. Bounded by remaining rollout_timeout.
        Returns total drained count.
    '''
    # rollout_timeout: absolute budget for the entire wait phase
    # generation_start_time: wall clock when the epoch started generation
    deadline = generation_start_time + rollout_timeout
    last_log = time.time()
    first = None
    # Cap each get at 30s so we wake periodically to log progress and
    # check pull_refs for early engine deaths, otherwise a cold-start
    # crash would silently wait the full rollout_timeout.
    while first is None:
        time_left = deadline - time.time()
        if time_left <= 0:
            raise TimeoutError(f"[Epoch {epoch+1}] No training data after "
                               f"{rollout_timeout}s — empty dataloader, "
                               f"buffer below train_batch_size, or stuck engines")
        # Early-detect dead engines: if any pull_ref has resolved before
        # producing a result, ray.get propagates RayActorError immediately.
        # If it returned cleanly, the engine exited without producing
        # data (empty dataloader edge case), we keep waiting for the
        # rollout_timeout to fire with the proper error.
        if pull_refs:
            ready, _ = ray.wait(pull_refs, num_returns=len(pull_refs), timeout=0)
            if ready:
                ray.get(ready)
        try:
            first = results_queue.get(block=True, timeout=min(time_left, 30.0))
        except RayQueueEmpty:
            if time.time() - last_log >= 30.0:
                logger.info(f"[Epoch {epoch+1}] still waiting for first rollouts "
                            f"({int(time_left)}s of {rollout_timeout}s budget left)")
                last_log = time.time()

    # Process the first result like drain_results would.
    merged, stats = merge_rollout_with_stats([first])
    replay_buffer.add_batch_seqs(merged)
    rollout_stats.accumulate(rollout_acc, stats)

    # Drain anything else that arrived during the block (no extra wait).
    extra, extra_acc = drain_results(results_queue, replay_buffer)
    rollout_stats.accumulate(rollout_acc, extra_acc)
    return 1 + extra

def try_rebuild_shards(replay_buffer, train_batch_size, num_engines, seed,
                       epoch, shard_buffer_size, shard_rebuild_count,
                       min_new_samples=0, force=False):
    '''
        Rebuild training shards if the replay buffer grew enough.
        force=True always rebuilds which is used at loop boundaries.
        min_new_samples: minimum number of new samples since last rebuild to trigger.
                        Use train_batch_size * num_engines so each engine gets at least
                        one new micro-batch from the rebuild.
        Returns (shard_refs, new_shard_buffer_size, new_shard_rebuild_count, batches) or None if skipped.
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
                                       epoch=epoch + shard_rebuild_count)
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

def compute_results_queue_maxsize(num_rollout_engines, max_lag):
    '''
        Bounded results_queue size: roughly max_lag epochs of unread results
        across all engines. When full, engines back-pressure on put().
    '''
    # 32 is arbitrary at this moment. Will examine this further, but not sure if i need to add
    # as config param at this moment. 32 means each engine has enough room for one full complete_generation
    # worth of results plus several more in flight.
    return max(num_rollout_engines * max(2, max_lag) * 32, num_rollout_engines * 32)

def prelaunch_next_epoch(epoch, number_of_epochs, num_rollout_engines, max_lag,
                          rollout_dataloader, rollout_engines,
                          rollout_policy_version, logger):
    '''
        Pre-launch the next epoch's queues + pull loops BEFORE save_checkpoint
        runs in main(), so rollout engines generate the next epoch's data while
        the driver writes the checkpoint to disk. This hides the 5-60s
        checkpoint save bubble behind continuous generation.

        The async engine never destroys EngineCore workers at runtime (no
        disk-refresh fallback), so pre-launch always fires when there's a next
        epoch.

        Returns a dict suitable for splatting into run_epoch_overlap as
        prefilled_* kwargs, or None if pre-launch is not applicable
        (last epoch) or fails. Failures are non-fatal: the next epoch will
        cold-start as usual.
    '''
    if epoch + 1 >= number_of_epochs:
        return None

    # Track partial state so a failure mid-setup can be cleanly torn down.
    # Without this, an exception in fill_prompt_queue or run_pull_loop.remote
    # would leak the queue actors and any already-dispatched pull loops,
    # leaving engines blocked on get() forever.
    next_prompt_queue = None
    next_results_queue = None
    next_pull_refs = []
    try:
        next_results_queue_maxsize = compute_results_queue_maxsize(num_rollout_engines, max_lag)
        next_prompt_queue = RayQueue(maxsize=0)
        next_results_queue = RayQueue(maxsize=next_results_queue_maxsize)
        rollout_dataloader.batch_sampler.set_epoch(epoch + 1)

        next_total_shards = fill_prompt_queue(rollout_dataloader, rollout_engines, next_prompt_queue, logger)
        next_pull_refs = [eng.run_pull_loop.remote(next_prompt_queue, next_results_queue, epoch + 1, rollout_policy_version)
                                                   for eng in rollout_engines]

        logger.info(f"[Epoch {epoch+1}] Pre-launched epoch {epoch+2}: "
                    f"{next_total_shards} shards enqueued, "
                    f"{num_rollout_engines} pull loops running in background "
                    f"(hides checkpoint save)")

        return {'prefilled_prompt_queue':  next_prompt_queue,
                'prefilled_results_queue': next_results_queue,
                'prefilled_pull_refs':     next_pull_refs,
                'prefilled_total_shards':  next_total_shards}

    except Exception as e:
        logger.warning(f"[Epoch {epoch+1}] Pre-launch of epoch {epoch+2} failed: {e}. Next epoch will cold-start as usual.")
        # Tear down any partially-created state so we don't leak queue
        # actors or orphan pull loops blocked on the queue forever.
        if next_prompt_queue is not None or next_pull_refs:
            partial_state = {'prefilled_prompt_queue':  next_prompt_queue,
                             'prefilled_results_queue': next_results_queue,
                             'prefilled_pull_refs':     next_pull_refs}
            teardown_prelaunched(partial_state, num_rollout_engines, logger)
        return None

def teardown_prelaunched(prelaunched_state, num_rollout_engines, logger):
    '''
        Best-effort cleanup of pre-launched queues + pull loops when the next
        epoch will not consume them (driver exception, KeyboardInterrupt, etc).
        Pushes one poison pill per engine so each pull loop exits cleanly,
        waits briefly, then drops references so Ray can collect the queue actors.
    '''
    if not prelaunched_state:
        return

    try:
        prompt_q  = prelaunched_state.get('prefilled_prompt_queue')
        pull_refs = prelaunched_state.get('prefilled_pull_refs') or []

        if prompt_q is not None:
            # Drain unread shards FIRST so the pills land at the head of the
            # queue. Otherwise pull loops would process every leftover shard
            # (potentially thousands) before reaching their pill, which defeats
            # the point of a fast teardown.
            try:
                drain_prompt_queue(prompt_q)
            except Exception:
                pass

            for _ in range(num_rollout_engines):
                try:
                    prompt_q.put(POISON_PILL)
                except Exception:
                    break
        if pull_refs:
            try:
                ray.wait(pull_refs, num_returns=len(pull_refs), timeout=10)
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[teardown_prelaunched] Best-effort cleanup failed: {e}")

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

def requeue_and_relaunch(prompt_queue, results_queue, leftover_shards,
                        rollout_engines, epoch, rollout_policy_version,
                        train_step_count, steps_per_epoch):
    '''
        Drain stale pills, requeue leftover (FIFO), push fresh end-of-epoch
        sentinels, then relaunch pull loops unless training is done AND no
        leftover work (which would spawn loops we'd immediately kill).
        Returns the new pull_refs list (possibly empty).
    '''
    num_engines = len(rollout_engines)

    # Drain stale pills first, relaunched loops would otherwise pull one
    # and exit immediately, poisoning all subsequent syncs.
    extra_real = drain_prompt_queue(prompt_queue)
    if extra_real:
        leftover_shards = leftover_shards + extra_real

    for shard in leftover_shards:
        prompt_queue.put(shard)

    for _ in range(num_engines):
        prompt_queue.put(POISON_PILL)

    if train_step_count < steps_per_epoch or leftover_shards:
        return [eng.run_pull_loop.remote(prompt_queue=prompt_queue,
                                         results_queue=results_queue,
                                         epoch=epoch,
                                         policy_version=rollout_policy_version)
                                for eng in rollout_engines]
    return []

def perform_inline_sync(epoch, train_step_count, steps_per_epoch, ess,
                         training_engines, rollout_engines,
                         prompt_queue, results_queue, pull_refs,
                         replay_buffer, rollout_acc, total_drained,
                         policy_version, rollout_policy_version, version_bumped_early,
                         rollout_timeout, sync_timeout, logger):
    '''
        Perform an inline (mid-epoch) NCCL weight sync.
          1. stop_engines_and_drain → push pills, save leftover shards.
          2. Wait for pull loops to exit (drains results during the wait).
          3. On drain failure: health-check engines, requeue, relaunch with
             OLD policy version, return should_continue=True.
          4. On drain success: bump policy_version, sync_weights_nccl,
             requeue + relaunch with NEW policy version.
          5. On sync exception: free rank-0 state dict; fatal NCCL errors
             re-raise (job exits); recoverable errors leave engines on
             stale weights for end-of-epoch retry.
    '''
    num_engines = len(rollout_engines)
    logger.info(f"[Epoch {epoch+1}][Step {train_step_count}] Sync triggered "
                f"(ESS={ess}), stopping engines for NCCL weight sync")

    leftover_shards = stop_engines_and_drain(prompt_queue=prompt_queue,
                                             num_rollout_engines=num_engines)
    logger.info(f"[Epoch {epoch+1}] Drained {len(leftover_shards)} shards for "
                f"requeue, waiting for {num_engines} engines to exit")

    # Wait for pull loops to exit. On timeout an engine is stuck in
    # generate(), so we abort sync to avoid wedging the broadcast collective.
    drain_ok, drained_during_wait = stop_pull_loops_and_check(pull_refs=pull_refs, 
                                                             prompt_queue=prompt_queue, 
                                                             results_queue=results_queue, 
                                                             replay_buffer=replay_buffer, 
                                                             rollout_acc=rollout_acc, 
                                                             num_rollout_engines=num_engines, 
                                                             timeout=rollout_timeout, 
                                                             logger=logger, 
                                                             push_pills=False)
    total_drained += drained_during_wait

    if not drain_ok:
        logger.error(f"[Epoch {epoch+1}] Skipping inline sync to avoid weight "
                     f"broadcast deadlock; end-of-epoch sync will retry.")

        # Async mode has no reinit_nccl_weight_sync_group fallback, so any
        # missing rank → fail-fast (would wedge next broadcast otherwise).
        dead_engines = check_rollout_engines_health(rollout_engines)
        if dead_engines:
            raise RuntimeError(f"[Epoch {epoch+1}] Rollout engine health check "
                               f"failed after pull-loop drain failure. Dead "
                               f"engines: {dead_engines}. NCCL world_size is "
                               f"fixed; restart job.")

        new_pull_refs = requeue_and_relaunch(prompt_queue=prompt_queue,
                                            results_queue=results_queue,
                                            leftover_shards=leftover_shards,
                                            rollout_engines=rollout_engines,
                                            epoch=epoch,
                                            rollout_policy_version=rollout_policy_version,
                                            train_step_count=train_step_count,
                                            steps_per_epoch=steps_per_epoch)
        sync_triggered_this_epoch = False
        should_continue = True

    else:
        # Drain any results that landed between the wait and now.
        drained, drain_acc = drain_results(results_queue=results_queue, replay_buffer=replay_buffer)
        total_drained += drained
        rollout_stats.accumulate(rollout_acc, drain_acc)

        # Bump version and sync.
        # training never lags rollout, so if this fails, it's a problem!
        assert policy_version >= rollout_policy_version, (f"Policy version invariant violated: policy_version={policy_version} "
                                                          f"< rollout_policy_version={rollout_policy_version}")
        if policy_version == rollout_policy_version:
            policy_version += 1
            version_bumped_early = True

        sync_triggered_this_epoch = False

        try:
            logger.info(f"[Epoch {epoch+1}] NCCL sync (v{rollout_policy_version} -> v{policy_version})")
            sync_weights_nccl(training_engines=training_engines,
                              rollout_engines=rollout_engines,
                              version=policy_version,
                              logger=logger,
                              sync_timeout=sync_timeout)

            rollout_policy_version = policy_version
            sync_triggered_this_epoch = True
            logger.info(f"[Epoch {epoch+1}] Weight sync complete, rollout_policy_version={rollout_policy_version}")

        except Exception as e:
            # Free rank-0 cached state dict to avoid CPU memory leak across retries.
            try:
                ray.get(training_engines[0].clear_pending_nccl_state_dict.remote(), timeout=10)
            except Exception:
                pass

            # Fail fast on NCCL communicator destruction (watchdog, hardware, etc).
            if is_nccl_fatal_error(e):
                logger.error(f"[Epoch {epoch+1}] Inline NCCL sync failed with "
                            f"FATAL communicator error: {e}. No runtime reinit "
                            f"path in async mode — aborting.")
                raise
            # Non-fatal sync error to verify engines responsive before continuing.
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

        new_pull_refs = requeue_and_relaunch(prompt_queue=prompt_queue,
                                            results_queue=results_queue,
                                            leftover_shards=leftover_shards,
                                            rollout_engines=rollout_engines,
                                            epoch=epoch,
                                            rollout_policy_version=rollout_policy_version,
                                            train_step_count=train_step_count,
                                            steps_per_epoch=steps_per_epoch)
        should_continue = False

    return {'pull_refs': new_pull_refs,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'version_bumped_early': version_bumped_early,
            'sync_triggered_this_epoch': sync_triggered_this_epoch,
            'total_drained': total_drained,
            'should_continue': should_continue}

def run_epoch_overlap(epoch, training_engines, rollout_engines, rollout_dataloader,
                      replay_buffer, policy_version, rollout_policy_version, global_step,
                      train_batch_size, steps_per_epoch, seed, max_lag,
                      ess_sync_threshold, fixed_sync_interval,
                      rollout_timeout, train_step_timeout, sync_timeout,
                      tracker, logger, prefilled_prompt_queue=None, prefilled_results_queue=None,
                      prefilled_pull_refs=None, prefilled_total_shards=0):
    '''
        Queue-driven overlap epoch: training and generation run concurrently.
        Engines pull prompts from a shared queue and push results to a results queue.
        The driver drains results between training steps. Inline weight sync
        pauses engines by draining the prompt_queue and pushing one poison pill
        per engine , stop_engines_and_drain, broadcasts weights, requeues leftover
        shards, and relaunches the pull loops.

        Since async only supports nccl sync, a failure causes the lag to accumulate
        and the next end-of-epoch attempt retries.

        prefilled_*: optional pre-launched state from the previous epoch's
        end-of-epoch hook. When provided, this epoch reuses queues that were
        already filled and pull loops that have been generating in the
        background while the previous epoch was saving its checkpoint. This
        eliminates the per-epoch cold-start bubble.
    '''
    epoch_start_time = time.time()
    num_train_engines = len(training_engines)
    num_rollout_engines = len(rollout_engines)

    # 1. Setup queues. Coordination uses poison-pill on prompt_queue
    # one pill per engine causes exactly one engine to exit. results_queue is
    # bounded so engines back-pressure when training is slow.
    if prefilled_prompt_queue is not None:
        # Reuse state from previous epoch's checkpoint-save window.
        # The dataloader was already advanced by the pre-launch path (set_epoch
        # + fill_prompt_queue) and pull loops were launched there as well.
        results_queue = prefilled_results_queue
        prompt_queue  = prefilled_prompt_queue
        total_shards  = prefilled_total_shards
        pull_refs     = prefilled_pull_refs
        logger.info(f"[Epoch {epoch+1}] Reusing pre-launched queue: {total_shards} shards "
                    f"already enqueued, {len(pull_refs)} pull loops already running")

    else:
        results_queue_maxsize = compute_results_queue_maxsize(num_rollout_engines, max_lag)
        results_queue         = RayQueue(maxsize=results_queue_maxsize)
        # unbounded queue, the total size is bounded by the dataloader output though.
        prompt_queue          = RayQueue(maxsize=0)

        # 2. Fill prompt queue with all sharded batches for this epoch
        rollout_dataloader.batch_sampler.set_epoch(epoch)
        
        # note shard_batch_for_engines inside fill_prompt_queue splits a batch into roughly 
        # num_engines chunks. The chunks are then pushed into a shared queue, NOT routed to specific engine.
        total_shards = fill_prompt_queue(rollout_dataloader, rollout_engines, prompt_queue, logger)
        logger.info(f"[Epoch {epoch+1}] Queue-driven generation: {total_shards} shards enqueued "
                    f"for {num_rollout_engines} engines, policy v{rollout_policy_version}")

        # 3. Launch pull loops on all engines (non-blocking)
        pull_refs = [eng.run_pull_loop.remote(prompt_queue, results_queue, epoch, rollout_policy_version)
                     for eng in rollout_engines]

    generation_start_time = time.time()

    # 4. Training loop with interleaved result draining
    epoch_metrics    = {}
    train_step_count = 0
    sync_triggered_this_epoch = False
    # train_step_count when the latest inline sync fired, none if no sync
    # this epoch. This is to detect post-sync drift: if more steps
    # ran after the inline sync, end-of-epoch sync is still needed.
    last_sync_step = None
    version_bumped_early = False
    shard_refs           = None
    shard_buffer_size    = 0
    shard_rebuild_count  = 0
    total_drained        = 0
    rollout_acc          = rollout_stats.new_accumulator()
    train_start_time     = time.time()

    while train_step_count < steps_per_epoch:
        # 4a. Drain available results non-blocking
        logger.info(f"[DEBUG][Epoch {epoch+1}] iter start: train_step={train_step_count}/"
                    f"{steps_per_epoch}, buffer={len(replay_buffer)}, "
                    f"shard_refs={'set' if shard_refs is not None else 'None'}")
        t0 = time.time()
        drained, drain_acc = drain_results(results_queue, replay_buffer)
        total_drained     += drained
        rollout_stats.accumulate(rollout_acc, drain_acc)
        logger.info(f"[DEBUG][Epoch {epoch+1}] 4a drain_results done in {time.time()-t0:.2f}s, "
                    f"drained={drained}, total_drained={total_drained}")

        # 4b. Rebuild shards if buffer grew enough or first time after data arrives.
        # try_rebuild_shards adds shard_rebuild_count internally as the per-rebuild offset.
        if drained > 0 or shard_refs is None:
            t0 = time.time()
            logger.info(f"[DEBUG][Epoch {epoch+1}] 4b try_rebuild_shards START "
                        f"(buffer={len(replay_buffer)})")
            # this is training shards
            result = try_rebuild_shards(replay_buffer=replay_buffer,
                                        train_batch_size=train_batch_size,
                                        num_engines=num_train_engines,
                                        seed=seed,
                                        epoch=epoch * 1_000_000,
                                        shard_buffer_size=shard_buffer_size,
                                        shard_rebuild_count=shard_rebuild_count,
                                        min_new_samples=train_batch_size * num_train_engines,
                                        force=(shard_refs is None and len(replay_buffer) >= train_batch_size))
            logger.info(f"[DEBUG][Epoch {epoch+1}] 4b try_rebuild_shards done in "
                        f"{time.time()-t0:.2f}s, result={'shards built' if result else 'None'}")
            if result:
                shard_refs, shard_buffer_size, shard_rebuild_count, _, rebuild_ms = result
                if tracker:
                    # If shard rebuild cost grows above ~5% of train step time, we should consider raising min_new_samples or
                    # moving the rebuild to a background thread. Need some experiments to confirm this.
                    tracker.log_metrics({"train/rebuild_ms":  rebuild_ms, "train/rebuild_buf": shard_buffer_size,}, step=global_step)

        # 4c. Cold-start wait: if there are no shards yet, block on results_queue (event-driven)
        # instead of polling. The driver has nothing else to do here. Rollout engines
        # run as independent Ray actors so this block does NOT slow down generation.
        if shard_refs is None:
            t0 = time.time()
            logger.info(f"[DEBUG][Epoch {epoch+1}] 4c wait_for_first_rollouts START "
                        f"(buffer={len(replay_buffer)} < batch={train_batch_size})")
            total_drained += wait_for_first_rollouts(results_queue=results_queue,
                                                    replay_buffer=replay_buffer,
                                                    rollout_acc=rollout_acc,
                                                    generation_start_time=generation_start_time,
                                                    rollout_timeout=rollout_timeout,
                                                    epoch=epoch,
                                                    logger=logger,
                                                    pull_refs=pull_refs)
            logger.info(f"[DEBUG][Epoch {epoch+1}] 4c wait_for_first_rollouts done in "
                        f"{time.time()-t0:.2f}s, buffer={len(replay_buffer)}")
            continue

        # 4d. Run one training step
        t0 = time.time()
        logger.info(f"[DEBUG][Epoch {epoch+1}] 4d run_training_step START "
                    f"(shards built, train_step={train_step_count+1}/{steps_per_epoch})")
        train_metrics = run_training_step(engines=training_engines,
                                          shard_refs=shard_refs,
                                          logger=logger,
                                          train_step_timeout=train_step_timeout)
        logger.info(f"[DEBUG][Epoch {epoch+1}] 4d run_training_step done in "
                    f"{time.time()-t0:.2f}s")
        for k, v in train_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)
        global_step += 1
        train_step_count += 1

        if train_step_count % 10 == 0 or train_step_count == 1:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            logger.info(f"[Epoch {epoch+1}][Step {train_step_count}/{steps_per_epoch}] "
                        f"{metric_str}")

        if tracker:
            tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()},
                               step=global_step)

            rollout_snapshot = rollout_stats.summarize(rollout_acc, rollout_time=time.time() - generation_start_time)
            rollout_log = {f"rollout/{k}": v for k, v in rollout_snapshot.items()}
            rollout_log["rollout/replay_buffer_size"] = len(replay_buffer)
            rollout_log["rollout/policy_lag"] = policy_version - rollout_policy_version
            
            # results_queue saturation tells us whether engines are back-pressured
            # on put() during training steps.
            try:
                rollout_log["rollout/results_queue_qsize"] = results_queue.qsize()

            except Exception:
                pass
            tracker.log_metrics(rollout_log, step=global_step)

        # 4e. Check ESS / fixed_sync_interval for inline weight sync
        should_sync, ess = check_ess_sync(train_metrics=train_metrics,
                                          train_step_count=train_step_count,
                                          ess_sync_threshold=ess_sync_threshold,
                                          fixed_sync_interval=fixed_sync_interval,
                                          sync_triggered_this_epoch=sync_triggered_this_epoch)

        if tracker and ess is not None:
            tracker.log_metrics({"nccl/ess_factor": ess,
                                 "nccl/sync_triggered": 1 if (sync_triggered_this_epoch or should_sync) else 0,
                                }, step=global_step)

        # Inline nccl weight sync (mid-epoch) which handles drain -> sync -> requeue -> relaunch end-to-end and
        # returns the updated state. Inline sync IS allowed on the last epoch as it benefits the remaining
        # training steps in this epoch (fresher rollouts -> tighter importance ratios -> bounded KL).
        # Only the end-of-epoch sync is skipped on the last epoch, since there is no next epoch to consume the new weights.
        if should_sync:
            inline_sync_start = time.time()
            logger.info(f"[DEBUG][Epoch {epoch+1}] 4e perform_inline_sync START "
                        f"(ess={ess}, train_step={train_step_count}/{steps_per_epoch})")
            sync_state = perform_inline_sync(epoch=epoch,
                                             train_step_count=train_step_count,
                                             steps_per_epoch=steps_per_epoch,
                                             ess=ess,
                                             training_engines=training_engines,
                                             rollout_engines=rollout_engines,
                                             prompt_queue=prompt_queue,
                                             results_queue=results_queue,
                                             pull_refs=pull_refs,
                                             replay_buffer=replay_buffer,
                                             rollout_acc=rollout_acc,
                                             total_drained=total_drained,
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
            total_drained          = sync_state['total_drained']

            if sync_state['sync_triggered_this_epoch']:
                sync_triggered_this_epoch = True
                # Record sync step so main can detect post-sync drift and still
                # fire EoE sync when more training steps ran after this inline sync.
                last_sync_step = train_step_count

            inline_sync_ms = (time.time() - inline_sync_start) * 1000.0
            logger.info(f"[DEBUG][Epoch {epoch+1}] 4e perform_inline_sync done in "
                        f"{inline_sync_ms/1000:.2f}s, "
                        f"should_continue={sync_state['should_continue']}, "
                        f"sync_triggered={sync_state['sync_triggered_this_epoch']}")

            if tracker:
                # Total inline sync wall time (drain + broadcast + relaunch).
                # Sum across an epoch / total epoch wall time = the bubble
                # fraction. >5% means consider raising ess_sync_threshold.
                # >15% means inflight weight resume becomes worth the work.
                tracker.log_metrics({"nccl/inline_sync_ms": inline_sync_ms,
                                    }, step=global_step)
            if sync_state['should_continue']:
                continue  # drain failed: skip the rest of this iteration

    # 5. If pull loops are still running, push poison pills to make them exit once remaining real
    # shards are consumed, drain results while waiting, and surface any exceptions. When pull_refs
    # is empty (e.g. inline sync drained on the final training step), there's nothing to wait for.
    logger.info(f"[DEBUG][Epoch {epoch+1}] STEP 5 START (training loop exited at "
                f"train_step={train_step_count}/{steps_per_epoch}, "
                f"pull_refs={'set' if pull_refs else 'empty'})")
    pull_loop_ok = True
    if pull_refs:
        t0 = time.time()
        pull_loop_ok, drained_during_wait = stop_pull_loops_and_check(pull_refs=pull_refs,
                                                                      prompt_queue=prompt_queue,
                                                                      results_queue=results_queue,
                                                                      replay_buffer=replay_buffer,
                                                                      rollout_acc=rollout_acc,
                                                                      num_rollout_engines=num_rollout_engines,
                                                                      timeout=rollout_timeout,
                                                                      logger=logger,
                                                                      push_pills=True)
        logger.info(f"[DEBUG][Epoch {epoch+1}] STEP 5 stop_pull_loops_and_check done in "
                    f"{time.time()-t0:.2f}s, ok={pull_loop_ok}, drained_during_wait={drained_during_wait}")
        total_drained += drained_during_wait
        if not pull_loop_ok:
            logger.error(f"[Epoch {epoch+1}] End-of-epoch pull-loop drain FAILED. "
                         f"At least one rollout engine is stuck in generate() — its "
                         f"Ray mailbox has an in-flight call ahead of any new RPC. "
                         f"Caller MUST skip end-of-epoch sync_weights_nccl: a "
                         f"receive_all_weights_nccl RPC would queue behind the "
                         f"stuck call and the NCCL broadcast on training rank 0 "
                         f"would wedge the communicator, deadlocking the next "
                         f"training step.")

    # 6. Drain any remaining results. Skip the blocking drain when pull_loop_ok is False: a stuck/dead engine
    # means the missing results will never arrive, and waiting the full rollout_timeout (e.g. 3600s) before main()
    # raises just stalls the failure for an hour. main() will raise on the dead engine anyway.
    logger.info(f"[DEBUG][Epoch {epoch+1}] STEP 6 START (drain remaining results, "
                f"total_drained={total_drained}/{total_shards})")
    t0 = time.time()
    drained, drain_acc = drain_results(results_queue, replay_buffer)
    total_drained += drained
    rollout_stats.accumulate(rollout_acc, drain_acc)
    logger.info(f"[DEBUG][Epoch {epoch+1}] STEP 6 non-blocking drain done in "
                f"{time.time()-t0:.2f}s, +{drained} → total_drained={total_drained}/{total_shards}")

    remaining = total_shards - total_drained
    if remaining > 0 and pull_loop_ok:
        logger.info(f"[Epoch {epoch+1}] Blocking drain for {remaining} remaining results")
        blocking_drained, blocking_acc = drain_results_blocking(results_queue=results_queue,
                                                                replay_buffer=replay_buffer,
                                                                remaining=remaining,
                                                                logger=logger,
                                                                timeout=rollout_timeout)

        rollout_stats.accumulate(rollout_acc, blocking_acc)
        total_drained += blocking_drained
        if blocking_drained < remaining:
            logger.warning(f"[Epoch {epoch+1}] Blocking drain only got "
                           f"{blocking_drained}/{remaining} missing results "
                           f"before {rollout_timeout}s timeout. "
                           f"total_drained={total_drained}/{total_shards}.")

    elif remaining > 0:
        logger.warning(f"[Epoch {epoch+1}] Skipping blocking drain for {remaining} "
                       f"missing results because pull_loop_ok=False (stuck/dead "
                       f"engine — results will never arrive). main() will raise.")

    logger.info(f"[Epoch {epoch+1}] Generation complete: {total_drained}/{total_shards} shards drained, "
                f"replay buffer: {len(replay_buffer)} samples")

    # 7. Post-training bookkeeping. Bump policy_version once per epoch
    # to reflect the optimizer updates done in this epoch:
    #   - No inline sync this epoch --> bump (version_bumped_early=False)
    #   - Inline sync at the LAST step --> already bumped during sync, skip
    #   - Inline sync mid-epoch with more steps after --> bump again so the
    #     end-of-epoch sync gate sees the post-sync drift via lag.
    sync_covers_final_weights = (last_sync_step is not None and
                                 last_sync_step >= train_step_count)

    if train_step_count > 0 and (not version_bumped_early or not sync_covers_final_weights):
        policy_version += 1

    evicted = replay_buffer.evict_stale(policy_version - max_lag)
    if evicted > 0:
        logger.info(f"[Epoch {epoch+1}] Post-training eviction: {evicted} stale samples removed, "
                    f"{len(replay_buffer)} retained for next epoch")

    # 8. Aggregate rollout stats
    generation_time = time.time() - generation_start_time
    rollout_metrics = rollout_stats.summarize(rollout_acc, rollout_time=generation_time)
    rollout_metrics["rollout_time_with_overlap"] = time.time() - epoch_start_time

    # 9. Overlap efficiency: with queue-pull, gen_wait is approximated as the
    # tail of generation that finished after training (or zero if training was the bottleneck).
    train_time = time.time() - train_start_time
    overlap_gen_wait = max(0.0, generation_time - train_time)
    interleaved_total = train_time + overlap_gen_wait
    overlap_ratio = train_time / interleaved_total if interleaved_total > 0 else 1.0

    return {'rollout_metrics': rollout_metrics,
            'epoch_metrics': epoch_metrics,
            'global_step': global_step,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'train_step_count': train_step_count,
            'train_time': train_time,
            'sync_performed': sync_covers_final_weights,
            'overlap_interleaved_sec': train_time,
            'overlap_gen_wait_sec': overlap_gen_wait,
            'overlap_ratio': overlap_ratio,
            'pull_loop_ok': pull_loop_ok}

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

    # replay buffer size = rollout_samples_per_epoch (prompts) * n_samples (completions per prompt)
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 )
    logger.info(f"Replay buffer initialized (max_seq_len={config.data.max_seq_len})")

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
    # 11. Training and rollout loop
    ########
    entire_training_start_time = time.time()

    # When prelaunched_state, the next iteration's run_epoch_overlap reuses these
    # queues and pull_refs instead of creating fresh ones, eliminating the
    # cold-start bubble at epoch boundaries.
    prelaunched_state = None

    for epoch in range(start_epoch, number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == number_of_epochs - 1)

        # When prelaunched_state, it fills the
        # prefilled_* kwargs (queues + pull_refs from previous epoch's
        # checkpoint-save hook). When None, the function creates fresh queues.
        result = run_epoch_overlap(epoch=epoch,
                                   training_engines=training_engines,
                                   rollout_engines=rollout_engines,
                                   rollout_dataloader=rollout_dataloader,
                                   replay_buffer=replay_buffer,
                                   policy_version=policy_version,
                                   rollout_policy_version=rollout_policy_version,
                                   global_step=global_step,
                                   train_batch_size=config.train.train_batch_size_per_gpu,
                                   steps_per_epoch=steps_per_epoch,
                                   seed=config.run.seed,
                                   max_lag=overlap_max_lag,
                                   ess_sync_threshold=ess_sync_threshold,
                                   fixed_sync_interval=fixed_sync_interval,
                                   rollout_timeout=rollout_timeout,
                                   train_step_timeout=train_step_timeout,
                                   sync_timeout=sync_timeout,
                                   tracker=tracker,
                                   logger=logger,
                                   **(prelaunched_state or {}))
        prelaunched_state = None

        # Unpack result
        global_step            = result['global_step']
        policy_version         = result['policy_version']
        rollout_metrics        = result['rollout_metrics']
        rollout_policy_version = result['rollout_policy_version']

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

        # Log overlap efficiency metrics
        o_interleaved = result['overlap_interleaved_sec']
        o_wait        = result['overlap_gen_wait_sec']
        o_ratio       = result['overlap_ratio']
        logger.info(f"[Epoch {epoch+1}] Overlap: interleaved={o_interleaved:.2f}s, "
                    f"gen_wait={o_wait:.2f}s, ratio={o_ratio:.2%}")
        if tracker:
            tracker.log_metrics({"overlap/interleaved_sec": o_interleaved,
                                 "overlap/gen_wait_sec": o_wait,
                                 "overlap/ratio": o_ratio,
                                }, step=global_step)

        # If nccl sync was already performed inline, sync_success is True
        # from the result and we skip. Otherwise we check the lag against
        # max_lag and either sync or skip.
        sync_success = result['sync_performed']
        pull_loop_ok = result.get('pull_loop_ok', True)

        # If any rollout engine is stuck in generate() at end of epoch, its
        # ray mailbox has an in-flight call ahead of any new RPC. We MUST NOT
        # issue receive_all_weights_nccl on it as the receive would queue
        # behind the stuck call, training rank 0 would enter the NCCL broadcast
        # collective and wedge the weight-sync communicator, and the next
        # training step's collective would deadlock the entire DeepSpeed group.
        # Skip sync, skip pre-launch, then health-check all engines and fail
        # fast if any are dead.
        if not pull_loop_ok:
            logger.error(f"[Epoch {epoch+1}] Pull-loop drain failed; bypassing "
                         f"end-of-epoch sync_weights_nccl and pre-launch to "
                         f"avoid wedging the NCCL communicator.")
            # check if any rollout engines are dead
            dead_engines = check_rollout_engines_health(rollout_engines)

            if dead_engines:
                raise RuntimeError(f"[Epoch {epoch+1}] Rollout engine health check failed "
                                   f"after pull-loop drain failure. Dead engines: {dead_engines}. "
                                   f"Cannot continue — a stuck actor would deadlock the next "
                                   f"epoch's NCCL collectives. Restart the job.")

            logger.warning(f"[Epoch {epoch+1}] All engines responded to BOTH "
                           f"pings (process alive AND mailbox responsive); "
                           f"continuing into next epoch with stale weights. "
                           f"Lag will be retried at next end-of-epoch sync.")

        if pull_loop_ok and not sync_success and not is_last_epoch:
            lag = policy_version - rollout_policy_version
            if lag >= overlap_max_lag:
                logger.info(f"[Epoch {epoch+1}] End-of-epoch NCCL sync "
                            f"(v{rollout_policy_version} -> v{policy_version})...")
                try:
                    # sync_weights_nccl waits on finalize internally and raises
                    # if any engine reports a partial load.
                    sync_weights_nccl(training_engines=training_engines,
                                      rollout_engines=rollout_engines,
                                      version=policy_version,
                                      logger=logger,
                                      sync_timeout=sync_timeout)
                    rollout_policy_version = policy_version
                    sync_success = True

                except Exception as e:
                    # Free the gathered state dict on rank 0 so it doesn't leak across the failed-then-retry path.
                    try:
                        ray.get(training_engines[0].clear_pending_nccl_state_dict.remote(), timeout=10)
                    
                    except Exception:
                        pass
                    
                    # since async mode has no runtime reinit path, a destroyed comm means every 
                    # subsequent sync attempt will instantly re-raise, the replay buffer will drain 
                    # via stale-version eviction, and the job will eventually die with a misleading 
                    # no training data error hours later.
                    if is_nccl_fatal_error(e):
                        logger.error(f"[Epoch {epoch+1}] End-of-epoch NCCL sync "
                                     f"failed with FATAL communicator error: {e}. "
                                     f"The weight-sync pg cannot be reused. Aborting.")
                        raise
                    # Non-fatal: verify engines responsive before next epoch.
                    # Catches wedged engines that would hang the next epoch's sync.
                    dead_engines = check_rollout_engines_health(rollout_engines)
                    if dead_engines:
                        raise RuntimeError(f"[Epoch {epoch+1}] End-of-epoch NCCL sync failed "
                                           f"AND engines unresponsive: {dead_engines}. Aborting "
                                           f"to avoid silent hang on the next sync attempt.")

                    logger.error(f"[Epoch {epoch+1}] NCCL sync failed: {e}. "
                                 f"Engines will keep stale weights; lag will grow "
                                 f"until next end-of-epoch sync attempt.")
            else:
                # skip intentionally as lag is small enough to tolerate.
                logger.info(f"[Epoch {epoch+1}] Skipping weight sync (lag={lag}, max_lag={overlap_max_lag})")

        # Periodic / final checkpoint save (independent of sync state).
        should_save_disk = (checkpoint_save_interval > 0 and
                           ((epoch + 1) % checkpoint_save_interval == 0 or is_last_epoch))

        # Pre-launch the next epoch, queues + pull loops, so rollout engines
        # generate while save_checkpoint writes to disk. Returns None on the
        # last epoch or on any failure as next epoch will cold-start in that case.
        # Skip pre-launch if pull-loop drain failed: pinging passed but mailbox
        # state is uncertain so let the next epoch cold-start cleanly instead.
        if not pull_loop_ok:
            prelaunched_state = None

        else:
            prelaunched_state = prelaunch_next_epoch(epoch=epoch,
                                                     number_of_epochs=number_of_epochs,
                                                     num_rollout_engines=num_rollout_engines,
                                                     max_lag=overlap_max_lag,
                                                     rollout_dataloader=rollout_dataloader,
                                                     rollout_engines=rollout_engines,
                                                     rollout_policy_version=rollout_policy_version,
                                                     logger=logger)

        if should_save_disk or is_last_epoch:
            try:
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
            except Exception:
                # since this runs AFTER prelaunch_next_epoch, a failure here would leak
                # the pre-launched queues + pull loops. Best-effort tear them down
                # before re-raising.
                teardown_prelaunched(prelaunched_state, num_rollout_engines, logger)
                prelaunched_state = None
                raise

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
    # 12. Cleanup
    ########
    if tracker:
        tracker.finish()

    entire_training_time = time.time() - entire_training_start_time
    logger.info(f"Training completed successfully! Total time: {entire_training_time:.2f}s ({entire_training_time/3600:.2f}h)")

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