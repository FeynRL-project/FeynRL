import os
import atexit
import numpy as np
import importlib
import ray
import time
import shutil
import threading
import math
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
                            broadcast_and_finalize_nccl,
                            check_rollout_engines_health,
                            clear_pending_nccl_state_dict,
                            log_driver_heartbeat)

# Sentinel value pushed into prompt_queue to signal an engine to stop same as
# VLLMRolloutEngineAsync.run_pull_loop. One poison pill per engine causes exactly
# one engine to exit.
POISON_PILL = "__STOP__"

class InfiniteShardIterator:
    '''
        Wraps the rollout dataloader as an endless source of shards. When the dataloader
        is exhausted, advances an internal epoch counter, calls set_epoch() so the sampler
        reshuffles, and starts a fresh pass.
        next_shards() always returns a non-empty list of shards (one batch worth) ready to
        push into prompt_queue. epoch is bookkeeping only, the training loop in main() doesn't
        drive epochs from this counter, the sampler reshuffle uses it.
        Shards-per-pass bookkeeping:
          shards_this_pass           : running count in current pass.
          last_completed_pass_shards : frozen count after last StopIteration (None until first pass finishes).
        Driver uses last_completed_pass_shards as the round's shard target once set, the static upper bound
        (len(dataloader)×num_engines) can over-count when shard_batch_for_engines drops empty shards, which would turn
        wait_for_round_completion's fatal timeout into a false alarm.
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
        self.shards_this_pass = 0
        self.last_completed_pass_shards = None
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
            Pull one batch and shard it across engines. On StopIteration, freezes shards_this_pass into
            last_completed_pass_shards, bumps epoch, reshuffles. Always returns a non-empty list. Two empty
            passes in a row raise rather than spin forever.
        '''
        empty_passes = 0
        while True:
            try:
                batch = next(self.batch_iter)
            except StopIteration:
                # Freeze the pass count before resetting, this is the authoritative number of non-empty
                # shards the pipeline emitted in the pass that just ended.
                self.last_completed_pass_shards = self.shards_this_pass
                self.shards_this_pass = 0
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
                self.shards_this_pass += len(shards)
                return shards

class ShardProducer:
    '''
        Daemon thread that keeps prompt_queue topped up from an InfiniteShardIterator.
        Lifecycle is start/stop only (no pause), weight sync calls stop(), runs its drain+broadcast+relaunch,
        then start() spawns a fresh thread. bounded put (1s timeout) so stop() is responsive; thread
        exceptions surface via self.error-> check_error(); daemon=True so
        the thread dies with the process.
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

def compute_pipeline_capacities(rollout_samples_per_epoch, rollout_batch_size_per_gpu,
                               num_rollout_engines, n_samples, max_lag):
    '''
        Derive replay buffer hard cap and queue capacities from config.
        Returns (replay_buffer_size, results_queue_maxsize, prompt_queue_maxsize,
                 items_per_round_theoretical).
        items_per_round_theoretical shows one dataloader pass worth of items.
        Used by run_round diagnostics to compare actual items-per-cycle against
        this theoretical round size.
    '''
    # Actual number of prompts per dataloader pass, rounded up to batch boundary.
    # Matches core/rl_engines.create_rollout_dataloader. This is an UPPER bound, actual
    # fill may be lower due to sequences dropped at max_seq_len or failed generations
    # At policy version T with max_lag=N, the buffer holds data from versions {T-N+1, ..., T}
    bsz_rollout        = num_rollout_engines * rollout_batch_size_per_gpu
    prompt_per_pass    = math.ceil(rollout_samples_per_epoch / bsz_rollout) * bsz_rollout
    replay_buffer_size = prompt_per_pass * n_samples * max_lag
    items_per_round_theoretical = prompt_per_pass * n_samples

    # prompt_queue holds shards (lists of prompts). num_rollout_engines * 2
    # gives each engine at least one shard queued plus one in-flight so the
    # pipeline never starves at shard boundaries. Independent of max_lag.
    prompt_queue_maxsize = num_rollout_engines * max(2, max_lag)
    # results_queue sized to absorb one full buffer's worth of shards so the
    # rollout engines can put without backpressure during the brief drain
    # window at round start.
    items_per_shard       = rollout_batch_size_per_gpu * n_samples
    results_queue_maxsize = max(prompt_queue_maxsize, replay_buffer_size // max(1, items_per_shard))

    return replay_buffer_size, results_queue_maxsize, prompt_queue_maxsize, items_per_round_theoretical

def drain_prompt_queue(prompt_queue):
    '''
        Discard everything in prompt_queue so poison pills land at the head.
        Dropped shards are replayed by the infinite iterator on producer
        restart. Returns drain count. Non-Empty exceptions propagate.
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
        Drain prompt_queue, then push one poison pill per engine. Stop
        latency is max in-flight generate() duration whcih is unavoidable.
        10s put timeout is defensive for a producer-race edge case;
        on full queue, warn and continue rather than hang.
    '''
    drained = drain_prompt_queue(prompt_queue)
    pushed = 0
    for _ in range(num_rollout_engines):
        try:
            prompt_queue.put(POISON_PILL, block=True, timeout=10.0)
            pushed += 1
        except RayQueueFull:
            logger.warning(f"[stop_engines_and_drain] prompt_queue full after drain, "
                           f"pushed {pushed}/{num_rollout_engines} pills. "
                           f"Remaining engines will hit pills later via queue drain "
                           f"in wait_for_pull_loops.")
            break

    logger.info(f"[stop_engines_and_drain] drained {drained} items, "
                f"pushed {pushed}/{num_rollout_engines} poison pills")

def drain_results(results_queue, replay_buffer):
    '''
        Non-blocking function that pulls all available results from queue into replay buffer.
        Returns (num_batches_drained, accumulated rollout stats).
        Only RayQueueEmpty terminates the drain, other exceptions propagate.
    '''
    acc = rollout_stats.new_accumulator()
    drained = 0
    while True:
        try:
            result_list = results_queue.get(block=False)
        except RayQueueEmpty:
            break

        merged, stats = merge_rollout_with_stats([result_list])
        # Accumulate stats first so a raise from add_batch_seqs (e.g. malformed
        # shard) doesn't drop the shard's stats.
        rollout_stats.accumulate(acc, stats)
        replay_buffer.add_batch_seqs(merged)
        drained += 1

    return drained, acc

def wait_for_pull_loops(pull_refs, prompt_queue, results_queue, replay_buffer,
                        rollout_acc, num_rollout_engines, timeout, logger, push_pills):
    '''
        Shut down pull loops cleanly:
          1. Push one POISON_PILL per engine if push_pills.
          2. Wait for pull_refs while continuously draining results_queue. Note
             engines blocked on put() can't exit until we free queue space.
          3. Surface any pull-loop exceptions via ray_get_with_timeout.
        Returns (ok, total_drained_shards). ok=True on clean shutdown, False on
        timeout or pull-loop raise. total_drained_shards is the cumulative count of
        result-lists pulled from results_queue during the shutdown which is used as
        carryover into the next round's wait target.
    '''
    # Track undelivered pills so we can retry as prompt_queue frees up
    # during the drain loop below. drain_results frees results_queue →
    # engines unblock from put() → engines loop back to prompt_queue.get
    # → prompt_queue frees space → retry-push lands.
    pills_remaining = num_rollout_engines if push_pills else 0
    if push_pills:
        for _ in range(num_rollout_engines):
            try:
                prompt_queue.put(POISON_PILL, block=True, timeout=10.0)
                pills_remaining -= 1
            except RayQueueFull:
                logger.warning(f"[wait_for_pull_loops] prompt_queue full, "
                               f"pushed {num_rollout_engines - pills_remaining}/"
                               f"{num_rollout_engines} pills. Retrying remaining "
                               f"during drain loop.")
                break

    deadline = time.time() + timeout
    pending  = list(pull_refs)
    total_drained_shards = 0
    while pending:
        # Retry undelivered pills non-blocking. Each successful put lets one
        # more engine see its pill and exit.
        while pills_remaining > 0:
            try:
                prompt_queue.put(POISON_PILL, block=False)
                pills_remaining -= 1
            except RayQueueFull:
                break

        drained, drain_acc = drain_results(results_queue, replay_buffer)
        total_drained_shards += drained
        rollout_stats.accumulate(rollout_acc, drain_acc)

        time_left = deadline - time.time()
        if time_left <= 0:
            logger.error(f"[wait_for_pull_loops] Timeout after {timeout}s "
                         f"with {len(pending)} pull loops still running")
            return False, total_drained_shards

        # 0.5s cap so we loop back to drain as engines stuck on results_queue.put
        # (full queue) can't exit until we free space.
        _, pending = ray.wait(pending, num_returns=len(pending), timeout=min(time_left, 0.5))

    try:
        # 10s is a sanity bound.
        ray_get_with_timeout(refs=pull_refs, timeout=10, description="pull loop final check", logger=logger)
    except Exception as e:
        logger.error(f"[wait_for_pull_loops] Pull loop raised: {e}")
        return False, total_drained_shards

    return True, total_drained_shards

def wait_for_round_completion(results_queue, replay_buffer, rollout_acc, target_shards, timeout, pull_refs):
    '''
        Blocking drain into replay_buffer until target_shards arrive. Shard count is used
        because sequences dropped at max_seq_len make item counts lossy, while shard count
        reliably tracks rollout work done.
        Since partial rounds would desync the dataloader from training rounds and mask a real
        rollout bottleneck, it raises TimeoutError. Raise run.rollout_timeout if slow.
        Also raises RuntimeError if a pull loop exits mid-wait (dead engine -> shards will never arrive).
        Returns target_shards on success.
    '''
    deadline = time.time() + timeout
    shards = 0
    while shards < target_shards:
        # Surface dead pull loops fast, a crashed engine will never produce more shards.
        ready, _ = ray.wait(pull_refs, num_returns=len(pull_refs), timeout=0)
        if ready:
            ray.get(ready)
            raise RuntimeError(f"[wait_for_round_completion] Pull loop(s) exited after "
                               f"{shards}/{target_shards} shards drained")

        time_left = deadline - time.time()
        if time_left <= 0:
            raise TimeoutError(f"[wait_for_round_completion] Timeout after {timeout}s: "
                               f"got only {shards}/{target_shards} shards, "
                               f"buffer={len(replay_buffer)}. Rollout is too slow or stuck. "
                               f"Options: (1) increase run.rollout_timeout, "
                               f"(2) add more rollout GPUs or reduce rollout.tensor_parallel_size, "
                               f"(3) reduce rollout.rollout_samples_per_epoch, "
                               f"(4) reduce rollout.max_tokens.")

        try:
            # Bound per-get wait so we periodically re-check pull_refs and deadline.
            result_list = results_queue.get(block=True, timeout=min(time_left, 5.0))
        except RayQueueEmpty:
            continue

        merged, stats = merge_rollout_with_stats([result_list])
        # Accumulate stats first so a raise from add_batch_seqs doesn't drop
        # the shard's stats.
        rollout_stats.accumulate(rollout_acc, stats)
        replay_buffer.add_batch_seqs(merged)
        shards += 1

    return shards

def perform_inline_sync(epoch, training_engines, rollout_engines,
                        prompt_queue, results_queue, pull_refs, producer,
                        replay_buffer, rollout_acc,
                        policy_version, rollout_policy_version,
                        rollout_timeout, sync_timeout, logger):
    '''
        End-of-round nccl weight sync.
        1. Stop producer, drain prompt_queue, push one poison pill per engine.
        2. Fire zero-3 gather on training engines (collective which runs concurrently with step 3.
        3. Wait for pull loops to exit, continuously draining results_queue so engines blocked
           on put() can unblock and reach the pill.
        4. On drain timeout: engines stuck inside complete_generation, i.e. vllm hang. No safe recovery,
           force-cancel would kill the actor worker and break the nccl world (actors have max_restarts=0).
           Hence, we let the in-flight gather land, clear pending state dict, and raise.
        5. On drain success: complete gather, broadcast, finalize. On broadcast success, bump rollout_policy_version.
        On exception, fatal nccl re-raises; recoverable errors leave engines on stale
           weights as lag grows, next round retries.
        6. Relaunch pull loops at the rollout_policy_version and restart producer.
        Note leftover in-flight prompts are dropped and the infinite shard iterator replays equivalent work
        on producer restart.
    '''
    assert policy_version > rollout_policy_version, (f"perform_inline_sync requires training drift: got "
                                                     f"policy_version={policy_version}, rollout_policy_version={rollout_policy_version}. "
                                                     f"Caller must bump policy_version before calling.")

    num_engines = len(rollout_engines)
    sync_start  = time.time()
    logger.info(f"[Epoch {epoch+1}] Sync START (v{rollout_policy_version} -> v{policy_version})")

    # step 1: stop production, push pills, fire gather concurrently with drain.
    producer.stop()
    stop_engines_and_drain(prompt_queue=prompt_queue,
                           num_rollout_engines=num_engines,
                           logger=logger)
    t_gather_start = time.time()
    gather_futures = start_nccl_gather(training_engines)

    # snapshot counters to report how many items are drained into the buffer
    # during sync via wait_for_pull_loops + post-gather drain_results.
    items_added_pre_sync = replay_buffer.total_items_added
    qsize_pre_sync       = results_queue.qsize()

    # step 2: wait for pull loops to exit and drain results_queue concurrently.
    drain_ok, shards_from_pull_loops = wait_for_pull_loops(pull_refs=pull_refs,
                                                           prompt_queue=prompt_queue,
                                                           results_queue=results_queue,
                                                           replay_buffer=replay_buffer,
                                                           rollout_acc=rollout_acc,
                                                           num_rollout_engines=num_engines,
                                                           timeout=rollout_timeout,
                                                           logger=logger,
                                                           push_pills=False)

    if not drain_ok:
        # Engines stuck in complete_generation. Clean up training side, raise.
        try:
            ray.get(gather_futures, timeout=sync_timeout)
        except Exception as e:
            logger.warning(f"[Epoch {epoch+1}] Gather cleanup raised: {e}")

        clear_pending_nccl_state_dict(rank0_engine=training_engines[0], logger=logger)
        raise RuntimeError(f"[Epoch {epoch+1}] drain failed after rollout_timeout={rollout_timeout}s. "
                           f"Rollout engines stuck in complete_generation (likely vLLM hang). "
                           f"Restart job; if recurring, reduce rollout.max_tokens or "
                           f"rollout.rollout_batch_size_per_gpu.")

    # step 3: complete gather, broadcast, finalize.
    sync_triggered_this_epoch  = False
    shards_from_residual_drain = 0
    try:
        # drain residual results that arrived between wait exit and now.
        # catches any failure (malformed shard, corrupted accumulator state)
        # and triggers the state_dict cleanup path.
        shards_from_residual_drain, drain_acc = drain_results(results_queue=results_queue, replay_buffer=replay_buffer)
        rollout_stats.accumulate(rollout_acc, drain_acc)

        param_metadata = complete_nccl_gather(gather_futures=gather_futures,
                                              version=policy_version,
                                              logger=logger,
                                              sync_timeout=sync_timeout)
        logger.info(f"[Epoch {epoch+1}] Gather done in {time.time()-t_gather_start:.2f}s "
                    f"({len(param_metadata)} params)")

        broadcast_and_finalize_nccl(training_engines=training_engines,
                                    rollout_engines=rollout_engines,
                                    param_metadata=param_metadata,
                                    version=policy_version,
                                    logger=logger,
                                    sync_timeout=sync_timeout)
        rollout_policy_version    = policy_version
        sync_triggered_this_epoch = True

    except Exception as e:
        clear_pending_nccl_state_dict(rank0_engine=training_engines[0], logger=logger)
        if is_nccl_fatal_error(e):
            logger.error(f"[Epoch {epoch+1}] Fatal NCCL error, aborting: {e}")
            raise
        # A wedged engine would queue the next run_pull_loop behind the broken
        # broadcast and hang until timeout.
        dead = check_rollout_engines_health(rollout_engines=rollout_engines,
                                            rollout_timeout=rollout_timeout)
        if dead:
            raise RuntimeError(f"[Epoch {epoch+1}] Broadcast failed AND engines unresponsive: {dead}. "
                               f"Aborting to avoid silent hang on next sync.")
        logger.warning(f"[Epoch {epoch+1}] Non-fatal broadcast error: {e}. "
                       f"Engines resume with stale weights; next round retries.")

    # snapshot items drained into buffer during sync + results_queue state
    # before/after. At max_lag=1 these sync-drained items will be fifo-evicted
    # by the next round's wait_for_round_completion, never trained on.
    items_added_during_sync = replay_buffer.total_items_added - items_added_pre_sync
    qsize_post_sync         = results_queue.qsize()
    sync_drained_shards     = shards_from_pull_loops + shards_from_residual_drain
    logger.info(f"[Epoch {epoch+1}] Sync drain: +{items_added_during_sync} items "
                f"({sync_drained_shards} shards) into buffer, "
                f"results_queue {qsize_pre_sync} -> {qsize_post_sync}, "
                f"buffer={len(replay_buffer)}")

    # step 4: relaunch pull loops, restart producer.
    new_pull_refs = [eng.run_pull_loop.remote(prompt_queue, results_queue, epoch, rollout_policy_version)
                                             for eng in rollout_engines]
    producer.start()
    logger.info(f"[Epoch {epoch+1}] Sync DONE in {time.time()-sync_start:.2f}s "
                f"(triggered={sync_triggered_this_epoch}, rollout_v={rollout_policy_version})")

    return {'pull_refs': new_pull_refs,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'sync_triggered_this_epoch': sync_triggered_this_epoch,
            'sync_drained_shards': sync_drained_shards,}

def run_round(epoch, training_engines, rollout_engines,
              prompt_queue, prompt_queue_maxsize,
              results_queue, results_queue_maxsize,
              pull_refs, producer,
              replay_buffer, policy_version, rollout_policy_version, global_step,
              train_batch_size, steps_per_epoch, seed,
              target_shards_per_round,
              rollout_timeout, train_step_timeout, sync_timeout,
              is_last_epoch,
              tracker, logger,
              carryover_shards, overlap_max_lag,
              items_per_round_theoretical):
    '''
        One round of round-based overlap training (matching sync mode):
          1. Wait for target_shards_per_round shards to arrive in results_queue ,blocks until rollout
             has produced one round's worth, or timeout. These shards were generated concurrently
             with the previous round's training, so at steady state the wait is short or zero.
          2. Build training shards once from the full replay buffer, prepare_training_batches + shard_and_put.
          3. Call run_training_step exactly steps_per_epoch times on the same shard_refs. Each call does one
             full pass over the shard with internal GA.
          4. Bump policy_version, then perform_inline_sync (unless last epoch) to broadcast new weights to
             rollout engines so the next round generates with fresh weights.
        No mid-epoch sync as one sync per round, steps_per_epoch training passes per round.
    '''
    round_start_time  = time.time()
    num_train_engines = len(training_engines)
    rollout_acc       = rollout_stats.new_accumulator()

    # If results_queue is 0 right after a sync, no overlap data is waiting and,
    # hence, wait_for_round_completion will block for a full fresh round which
    # translates to no wall-clock speedup from overlap.
    qsize_round_start     = results_queue.qsize()
    items_added_pre_round = replay_buffer.total_items_added
    logger.info(f"[Epoch {epoch+1}] Round start: results_queue_qsize={qsize_round_start}, "
                f"buffer={len(replay_buffer)}")

    # Surface producer thread errors and dead pull loops before doing anything.
    producer.check_error()
    ready, _ = ray.wait(pull_refs, num_returns=len(pull_refs), timeout=0)
    if ready:
        ray.get(ready)
        raise RuntimeError(f"[Epoch {epoch+1}] {len(ready)}/{len(pull_refs)} "
                           f"rollout pull loop(s) exited before round start.")

    # Determine round's shard target
    # target_shards_per_round = len(dataloader) * num_rollout_engines is an upper bound.
    observed_pass_shards = producer.shard_iter.last_completed_pass_shards
    round_target_shards  = observed_pass_shards if observed_pass_shards is not None else target_shards_per_round
    if observed_pass_shards is not None and observed_pass_shards != target_shards_per_round:
        logger.info(f"[Epoch {epoch+1}] Round target adjusted from upper bound "
                    f"{target_shards_per_round} to observed per-pass count "
                    f"{observed_pass_shards} (shard_batch_for_engines dropped "
                    f"{target_shards_per_round - observed_pass_shards} empty shards).")

    # subtract shards already drained into the buffer during the previous sync
    # from this round's wait target. Those items are already trained-eligible,
    # so we only need to block for the remainder. carryover_shards=0 on the first epoch.
    wait_target = max(0, round_target_shards - carryover_shards)
    if carryover_shards > 0:
        logger.info(f"[Epoch {epoch+1}] carryover_shards={carryover_shards}, "
                    f"wait_target={wait_target}/{round_target_shards}")

    # Step 1: Drain one round's rollout output into the buffer
    drain_start     = time.time()
    shards_received = wait_for_round_completion(results_queue=results_queue,
                                                replay_buffer=replay_buffer,
                                                rollout_acc=rollout_acc,
                                                target_shards=wait_target,
                                                timeout=rollout_timeout,
                                                pull_refs=pull_refs)
    drain_time = time.time() - drain_start
    logger.info(f"[Epoch {epoch+1}] Drain: {shards_received}/{wait_target} "
                f"shards in {drain_time:.2f}s, buffer={len(replay_buffer)}")

    # enforce max_lag by versions, not FIFO. Keep items whose policy_version >= policy_version - overlap_max_lag.
    # For example, max_lag=1 tolerates 1 version of drift (the sync-drained overlap items are kept).
    # max_lag=2 keeps two prior versions plus current, etc.
    if overlap_max_lag is not None and overlap_max_lag > 0:
        min_v   = policy_version - overlap_max_lag
        evicted = replay_buffer.evict_stale(min_version=min_v)
        if evicted > 0:
            logger.info(f"[Epoch {epoch+1}] evict_stale(min_v={min_v}): -{evicted} items, "
                        f"buffer={len(replay_buffer)}")

    # snapshot version diversity in the training buffer. At max_lag=K, we expect unique_versions
    # up to K+1 (current and K stale). items_added_this_round counts new items pulled from results_queue this round
    # Note it can be 0 if wait_target=0 due to carryover saturation.
    items_added_this_round = replay_buffer.total_items_added - items_added_pre_round
    buf_versions           = [it["policy_version"] for it in replay_buffer.items]
    if buf_versions:
        lag_max  = policy_version - min(buf_versions)
        lag_mean = policy_version - (sum(buf_versions) / len(buf_versions))
        unique_v = len(set(buf_versions))
        logger.info(f"[Epoch {epoch+1}] Buffer at train: items_added_this_round={items_added_this_round}, "
                    f"lag_max={lag_max}, lag_mean={lag_mean:.2f}, unique_versions={unique_v}")

    # need at least one micro-batch per engine to avoid deepspeed hang on empty shard.
    min_buffer_items = train_batch_size * num_train_engines
    if len(replay_buffer) < min_buffer_items:
        raise RuntimeError(f"[Epoch {epoch+1}] Buffer underfilled after drain: "
                           f"{len(replay_buffer)} < {min_buffer_items} (train_batch_size × num_engines). "
                           f"Drained {shards_received}/{wait_target} shards (round_target={round_target_shards}, "
                           f"carryover={carryover_shards}). "
                           f"Likely rollout stall, or too many sequences exceeded max_seq_len. "
                           f"Check rollout engine health and data.max_seq_len vs rollout.max_tokens.")

    # Step 2: Build training shards once from the full buffer snapshot
    build_start   = time.time()
    train_batches = prepare_training_batches(replay_buffer=replay_buffer,
                                             batch_size=train_batch_size,
                                             num_engines=num_train_engines,
                                             seed=seed,
                                             epoch=epoch)
    shard_refs = shard_and_put(train_batches, num_engines=num_train_engines)
    build_ms   = (time.time() - build_start) * 1000.0
    logger.info(f"[Epoch {epoch+1}] Shard build: {len(train_batches)} micro-batches "
                f"in {build_ms:.0f}ms, buffer={len(replay_buffer)}")
    if tracker:
        tracker.log_metrics({"train/rebuild_ms":  build_ms,
                             "train/rebuild_buf": len(replay_buffer),}, step=global_step)

    # Step 3: Train steps_per_epoch times on the same shard_refs like sync-mode pattern
    epoch_metrics    = {}
    train_start_time = time.time()
    last_heartbeat   = time.time()
    last_step_time   = None
    HEARTBEAT_S      = 30.0

    for step in range(steps_per_epoch):
        # Heartbeat so a long training step doesn't hide a stalled state.
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_S:
            log_driver_heartbeat(epoch=epoch,
                                 train_step_count=step,
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

        # Surface producer crashes fast.
        producer.check_error()

        # Surface dead pull loops fast as any exit mid-round is fatal.
        ready, _ = ray.wait(pull_refs, num_returns=len(pull_refs), timeout=0)
        if ready:
            ray.get(ready)
            raise RuntimeError(f"[Epoch {epoch+1}][Step {step+1}/{steps_per_epoch}] "
                               f"{len(ready)}/{len(pull_refs)} pull loop(s) exited mid-training.")

        # Run one training step: one full pass over all micro-batches + internal GA
        step_start    = time.time()
        train_metrics = run_training_step(engines=training_engines,
                                          shard_refs=shard_refs,
                                          logger=logger,
                                          train_step_timeout=train_step_timeout)
        step_time = time.time() - step_start
        for k, v in train_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)
        global_step += 1
        last_step_time = time.time()

        # Warn when results_queue saturates when engines backpressured on put.
        try:
            qsize = results_queue.qsize()
            if qsize >= int(0.9 * results_queue_maxsize):
                logger.warning(f"[Epoch {epoch+1}][Step {step+1}] "
                               f"results_queue near capacity ({qsize}/{results_queue_maxsize}); "
                               f"rollout engines backpressured (training is the bottleneck).")
        except Exception:
            qsize = None

        if (step + 1) % 10 == 0 or step == 0:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            logger.info(f"[Epoch {epoch+1}][Step {step+1}/{steps_per_epoch}] "
                        f"{metric_str}, step_time={step_time:.2f}s, "
                        f"buffer={len(replay_buffer)}, "
                        f"shards_produced={producer.shards_produced}")

        if tracker:
            tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()},
                               step=global_step)
            tracker.log_metrics({"train/step_time_sec": step_time}, step=global_step)

            # rollout/* fires once at end-of-epoch, so this is to record the intermediate states.
            # Since at step=K-1 this snapshot is a strict subset of the rollout/* row written
            # shortly after, we skip the last step.
            if rollout_acc['total_samples_generated'] > 0 and step < steps_per_epoch - 1:
                rollout_snapshot = rollout_stats.summarize(rollout_acc,
                                                           rollout_time=time.time() - round_start_time)
                rollout_log = {f"rollout_inprogress/{k}": v for k, v in rollout_snapshot.items()}
                rollout_log["rollout_inprogress/replay_buffer_size"] = len(replay_buffer)
                rollout_log["rollout_inprogress/policy_lag"] = policy_version - rollout_policy_version
                rollout_log["rollout_inprogress/shards_produced"] = producer.shards_produced
                if qsize is not None:
                    rollout_log["rollout_inprogress/results_queue_qsize"] = qsize

                # Actual staleness of items currently in the buffer, policy_lag above is always 0 under inline sync.
                buf_versions = [it["policy_version"] for it in replay_buffer.items]
                if buf_versions:
                    rollout_log["rollout_inprogress/buffer_lag_max"]         = policy_version - min(buf_versions)
                    rollout_log["rollout_inprogress/buffer_lag_mean"]        = policy_version - (sum(buf_versions) / len(buf_versions))
                    rollout_log["rollout_inprogress/buffer_unique_versions"] = len(set(buf_versions))

                tracker.log_metrics(rollout_log, step=global_step)

    train_time = time.time() - train_start_time

    # Step 4: Bump policy_version to reflect training drift, then sync.
    policy_version += 1

    sync_success        = False
    sync_drained_shards = 0
    if not is_last_epoch:
        sync_start = time.time()
        sync_state = perform_inline_sync(epoch=epoch,
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
                                         rollout_timeout=rollout_timeout,
                                         sync_timeout=sync_timeout,
                                         logger=logger)
        pull_refs              = sync_state['pull_refs']
        policy_version         = sync_state['policy_version']
        rollout_policy_version = sync_state['rollout_policy_version']
        sync_success           = sync_state['sync_triggered_this_epoch']
        sync_drained_shards    = sync_state['sync_drained_shards']

        sync_ms = (time.time() - sync_start) * 1000.0
        if tracker:
            tracker.log_metrics({"nccl/inline_sync_ms": sync_ms}, step=global_step)

    else:
        # Since last epoch skips perform_inline_sync, we need to drain in-flight
        # rollouts here so rollout/* doesn't under-count the tail.
        logger.info(f"[Epoch {epoch+1}] Final-epoch drain START "
                    f"(stopping producer, draining in-flight rollouts)")
        try:
            producer.stop()
            stop_engines_and_drain(prompt_queue=prompt_queue,
                                   num_rollout_engines=len(rollout_engines),
                                   logger=logger)
            ok, _ = wait_for_pull_loops(pull_refs=pull_refs,
                                        prompt_queue=prompt_queue,
                                        results_queue=results_queue,
                                        replay_buffer=replay_buffer,
                                        rollout_acc=rollout_acc,
                                        num_rollout_engines=len(rollout_engines),
                                        timeout=rollout_timeout,
                                        logger=logger,
                                        push_pills=False)

            if not ok:
                logger.warning(f"[Epoch {epoch+1}] Pull loop drain timed out; "
                               f"rollout stats may be incomplete due to wedged engine.")

            final_drained, final_acc = drain_results(results_queue, replay_buffer)
            if final_drained > 0:
                rollout_stats.accumulate(rollout_acc, final_acc)

        except Exception as e:
            logger.warning(f"[Epoch {epoch+1}] Final-epoch rollout drain raised: "
                           f"{e}; rollout stats for this epoch may be incomplete.")

    # Step 5: Summarize rollout stats for the round
    generation_time = time.time() - round_start_time
    rollout_metrics = rollout_stats.summarize(rollout_acc, rollout_time=generation_time)
    rollout_metrics["rollout_time_with_overlap"] = generation_time

    # Since max_lag is an upper bound on the actual lag and it is a function of
    # rollout and trainign speed to hit that limit, here provide some estimate how are things are going.
    # ratio ~ 1.0  -> balanced pipeline: max_lag enforced as configured.
    # ratio > 1.0  -> rollout faster than training: buffer saturates in fewer than max_lag cycles, FIFO beats evict_stale, effective lag < max_lag.
    # ratio < 1.0  -> training faster than rollout: wait_for_round_completion blocks for fresh items, no async speedup, but max_lag enforced.
    items_per_cycle = replay_buffer.total_items_added - items_added_pre_round
    if items_per_round_theoretical > 0:
        rollout_speed_ratio = items_per_cycle / items_per_round_theoretical
    else:
        rollout_speed_ratio = float('nan')

    logger.info(f"[Epoch {epoch+1}] Pipeline regime: items_per_cycle={items_per_cycle}, "
                f"theoretical_items_per_round={items_per_round_theoretical}, "
                f"rollout_speed_ratio={rollout_speed_ratio:.2f}")

    if tracker and rollout_acc['total_samples_generated'] > 0:
        rollout_log = {f"rollout/{k}": v for k, v in rollout_metrics.items()}
        rollout_log["rollout/replay_buffer_size"] = len(replay_buffer)
        rollout_log["rollout/policy_lag"] = policy_version - rollout_policy_version
        rollout_log["rollout/shards_produced"] = producer.shards_produced
        rollout_log["rollout/items_per_cycle"] = items_per_cycle
        rollout_log["rollout/speed_ratio"]     = rollout_speed_ratio
        buf_versions = [it["policy_version"] for it in replay_buffer.items]
        if buf_versions:
            rollout_log["rollout/buffer_lag_max"]         = policy_version - min(buf_versions)
            rollout_log["rollout/buffer_lag_mean"]        = policy_version - (sum(buf_versions) / len(buf_versions))
            rollout_log["rollout/buffer_unique_versions"] = len(set(buf_versions))
        tracker.log_metrics(rollout_log, step=global_step)

    return {'rollout_metrics': rollout_metrics,
            'epoch_metrics': epoch_metrics,
            'global_step': global_step,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'pull_refs': pull_refs,
            'train_step_count': steps_per_epoch,
            'train_time': train_time,
            'sync_performed': sync_success,
            'sync_drained_shards': sync_drained_shards}

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

    # Overlap setting: max_lag bounds how many distinct policy versions stay
    # in the replay buffer (FIFO-evicted on insert past capacity). At policy
    # version T with max_lag=N, buffer holds versions {T-N+1, ..., T}.
    overlap_max_lag = config.overlap.max_lag

    # Buffer sizing: items_per_round × max_lag. Upper bound — actual fill may
    # be lower due to sequences dropped at max_seq_len or failed generations.
    (replay_buffer_size, results_queue_maxsize, prompt_queue_maxsize,
                         items_per_round_theoretical) = compute_pipeline_capacities(rollout_samples_per_epoch=config.rollout.rollout_samples_per_epoch,
                                                                                    rollout_batch_size_per_gpu=config.rollout.rollout_batch_size_per_gpu,
                                                                                    num_rollout_engines=num_rollout_engines,
                                                                                    n_samples=config.rollout.n_samples,
                                                                                    max_lag=overlap_max_lag,
                                                                                    )

    # One round = one pass over the rollout dataloader. Each dataloader batch
    # produces up to num_rollout_engines shards (one per engine). Used as the
    # target for wait_for_round_completion's blocking drain.
    target_shards_per_round = len(rollout_dataloader) * num_rollout_engines
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 max_size=replay_buffer_size,
                                 )
    logger.info(f"Pipeline capacities: replay_buffer={replay_buffer_size}, "
                f"results_queue={results_queue_maxsize}, prompt_queue={prompt_queue_maxsize}, "
                f"max_seq_len={config.data.max_seq_len}")

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
                f"target_shards_per_round={target_shards_per_round}, "
                f"replay_buffer_size={replay_buffer_size}")

    logger.info(f"checkpoint_save_interval: {checkpoint_save_interval}")
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from} (epoch {start_epoch+1}/{number_of_epochs})")

    logger.info("=" * 50)

    ########
    # 11. Persistent rollout queues, shard producer, pull loops.
    ########
    # Queue sizes already computed by compute_pipeline_capacities above.
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

    # Carryover is the count of shards drained into the replay buffer during the previous epoch's sync.
    # This is used to skip redundant blocking on rollout output that was already produced in the background.
    carryover_shards = 0

    for epoch in range(start_epoch, number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == number_of_epochs - 1)
        result = run_round(epoch=epoch,
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
                           target_shards_per_round=target_shards_per_round,
                           rollout_timeout=rollout_timeout,
                           train_step_timeout=train_step_timeout,
                           sync_timeout=sync_timeout,
                           is_last_epoch=is_last_epoch,
                           tracker=tracker,
                           logger=logger,
                           carryover_shards=carryover_shards,
                           overlap_max_lag=overlap_max_lag,
                           items_per_round_theoretical=items_per_round_theoretical)

        # Unpack result. pull_refs may have been relaunched by the end-of-round sync.
        global_step            = result['global_step']
        policy_version         = result['policy_version']
        rollout_metrics        = result['rollout_metrics']
        rollout_policy_version = result['rollout_policy_version']
        pull_refs              = result['pull_refs']
        carryover_shards       = result['sync_drained_shards']

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

        # End-of-round sync happens inside run_round (unless is_last_epoch).
        # Training drift is always synced to rollout engines for the next round,
        # so there is no EoE lag-gated retry path here.
        sync_success = result['sync_performed']

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
        wait_for_pull_loops(pull_refs=pull_refs,
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