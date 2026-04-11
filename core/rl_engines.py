import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import ray
import time

# imports local methods, classes, etc.
from data_feeds.prompts import PromptsFeed # our custom pytorch dataset
from data_feeds.mixed_sampler import create_prompt_dataset_and_sampler
from misc.utils import safe_string_to_torch_dtype, ray_get_with_timeout, set_random_seeds, get_determinism_env_vars
from misc.nccl_env import nccl_watchdog_env_vars
from rollouts.vllm_engine import VLLMRolloutEngine
from rollouts.vllm_engine_async import VLLMRolloutEngineAsync
import misc.rollout_stats as rollout_stats

Algorithm_Registry = {# supported algorithms
                      'grpo':  ('algs.GRPO.grpo', 'GRPO'),
                      'cispo': ('algs.CISPO.cispo', 'CISPO'),
                      'p3o':   ('algs.P3O.p3o', 'P3O'),
                      'ppo':   ('algs.PPO.ppo', 'PPO'),
                     }

def create_training_engines(params, alg, world_size, master_addr, master_port):
    '''
        This function is responsible for running the training engine.
    '''
    kwargs = { # model related arguments
               'model_path':params.model.name,
               'ref_model_path':params.model.ref_model,
               'model_dtype':safe_string_to_torch_dtype(params.model.dtype),
               'trust_remote_code':params.model.trust_remote_code,
               'attn_impl':params.model.attn_implementation,
               'seed':params.run.seed,

               # training related arguments
               'kl_coeff':params.train.kl_coeff,
               'clip_low':params.train.clip_low,
               'clip_high':params.train.clip_high,
               'entropy_coeff':params.train.entropy_coeff,
               'micro_batch_size_per_gpu':params.train.train_batch_size_per_gpu,
               'update_after_full_replay':params.train.update_after_full_replay,
               'normalize_loss':params.train.normalize_loss,

               # deepspeed related arguments
               'deepspeed_config':params.deepspeed,
               'deepspeed_ref_config':params.deepspeed_ref,

               # gradient checkpointing
               'gradient_checkpointing':params.model.gradient_checkpointing,

               # peft
               'peft_config':params.peft,

               # decoupled loss when async overlap engine is used
               'use_decoupled_loss': params.overlap.enabled if params.overlap else False,
               'alpha': params.overlap.alpha if params.overlap else None,
               'behave_imp_weight_cap': params.overlap.behave_imp_weight_cap if params.overlap else None,
    }

    # ppo arguments
    alg_name = params.train.alg_name.lower()
    if alg_name == 'ppo':
        kwargs['value_model_path'] = params.model.value_model or params.model.name
        kwargs['tau'] = params.train.tau
        kwargs['gamma'] = params.train.gamma
        kwargs['deepspeed_value_config'] = params.deepspeed_value
    # setup ray runners
    ray_runners = []
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", get_determinism_env_vars())
    for rank in range(world_size):
        # Since NCCL identifies gpus by their actual PCIe/NVLink topology,
        # not LOCAL_RANK, we keep LOCAL_RANK as 0 for all actors.
        ray_vars = {"MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": "0",
                    "PYTHONPATH": os.getcwd(), # Ensure current directory is in path for all workers
                    "CUBLAS_WORKSPACE_CONFIG": cublas_workspace, # deterministic cuBLAS
                    "PYTHONHASHSEED": str(params.run.seed),
                    # Disable nccl's internal CUDA memory allocator so all GPU memory
                    # is managed by pytorch's caching allocator. Prevents allocator
                    # conflicts that cause cache flushes and param buffer corruption.
                    "NCCL_CUMEM_ENABLE": "0",
                    # NCCL watchdog: abort wedged collectives after timeout
                    # so the job fails fast instead of hanging the GPU stream.
                    **nccl_watchdog_env_vars(),
                    }

        # NCCL env vars
        if params.run.nccl_socket_ifname:
            ray_vars["NCCL_SOCKET_IFNAME"] = params.run.nccl_socket_ifname
        if params.run.nccl_ib_hca:
            ray_vars["NCCL_IB_HCA"] = params.run.nccl_ib_hca

        runner = alg.options(num_gpus=1, runtime_env={"env_vars": ray_vars}
                            ).remote(**kwargs)
        ray_runners.append(runner)

    return ray_runners

def create_rollout_engines(params, reward_fnc, eos_id):
    '''
        This function is responsible for setting up distributed
        inference/rollout/generation engine.
    '''
    tp = int(params.rollout.tensor_parallel_size)
    rollout_gpus = int(params.run.rollout_gpus)

    kwargs = { # model related arguments
              "model_path":params.model.name,
              "trust_remote_code":params.model.trust_remote_code,

              # experiment setup related arguments
              "seed":params.run.seed,

              # rollout generation related arguments
              "temperature":params.rollout.temperature,
              "max_tokens":params.rollout.max_tokens,
              "n_samples":params.rollout.n_samples,
              "top_p":params.rollout.top_p,
              "top_k":params.rollout.top_k,
              "ignore_eos":params.rollout.ignore_eos,
              "stop":params.rollout.stop,
              "stop_token_ids":params.rollout.stop_token_ids,
              "prompt_logprobs":params.rollout.prompt_logprobs,
              "gpu_memory_utilization":params.rollout.gpu_memory_utilization,
              "force_strict_on_policy":params.rollout.force_strict_on_policy,
              "eos_id":eos_id,
              "tensor_parallel_size":tp,
              "model_dtype":params.model.dtype,
              "max_seq_len":params.data.max_seq_len,
              "max_model_len":params.rollout.max_model_len,

              # reward related arguments
              "reward_func":reward_fnc,
              "reward_broadcast":params.reward.broadcast,
              "batch_invariant":params.rollout.batch_invariant,
            }

    # if model doesn't fit in one gpu, tp can be > 1
    num_engines = max(1, rollout_gpus // tp)
    engines = []
    cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG", get_determinism_env_vars())
    for i in range(num_engines):
        kwargs['engine_id'] = i
        rollout_env_vars = {"PYTHONPATH": os.getcwd(),
                            "CUBLAS_WORKSPACE_CONFIG": cublas_workspace,
                            "PYTHONHASHSEED": str(params.run.seed),
                            "NCCL_CUMEM_ENABLE": "0",
                            # NCCL watchdog: abort wedged collectives after
                            # timeout so the job fails fast instead of hanging.
                            **nccl_watchdog_env_vars(),
                           }
        # The goal of batch_invariant is topology-invariance. it means that
        # same prompt → same output regardless of engine count
        if params.rollout.batch_invariant:
            rollout_env_vars["VLLM_BATCH_INVARIANT"] = "1"

        if params.run.nccl_socket_ifname:
            rollout_env_vars["NCCL_SOCKET_IFNAME"] = params.run.nccl_socket_ifname
        if params.run.nccl_ib_hca:
            rollout_env_vars["NCCL_IB_HCA"] = params.run.nccl_ib_hca

        if params.overlap and params.overlap.enabled:
            engines.append(VLLMRolloutEngineAsync.options(num_gpus=tp,
                                                          runtime_env={"env_vars": rollout_env_vars}
                                                         ).remote(**kwargs))

        else:
            engines.append(VLLMRolloutEngine.options(num_gpus=tp,
                                                    runtime_env={"env_vars": rollout_env_vars}
                                                    ).remote(**kwargs))

    return engines

def create_rollout_dataloader(params, tokenizer, num_rollout_engines, samples_per_epoch):
    '''
       This dataloader is used for rollout generation which
       would be used to train the policy.
       Uses MixedDatasetSampler for mixed sampling across datasets.
    '''
    if samples_per_epoch <= 0:
        raise ValueError(f"samples_per_epoch must be > 0, got {samples_per_epoch}")

    # we need to multiply by num_rollout_engines because we shard data across rollout engines
    bsz = num_rollout_engines * params.rollout.rollout_batch_size_per_gpu
    # Calculate number of batches from total samples
    num_batches = (samples_per_epoch + bsz - 1) // bsz

    dataset, sampler, collate_fn = create_prompt_dataset_and_sampler(
                                                data_paths=params.data.train_files_path,
                                                prompt_key=params.data.prompt_key,
                                                solution_key=params.data.solution_key,
                                                max_seq_len=params.data.max_seq_len,
                                                tokenizer=tokenizer,
                                                train_ratios=params.data.train_ratios,
                                                seed=params.run.seed,
                                                local_batch_size=bsz,
                                                dataset_cls=PromptsFeed,
                                                steps_per_epoch=num_batches,
                                                shuffle_within_batch=True,
                                                dynamic_ratio_every_step=params.train.dynamic_ratio_every_step,
                                                )
    # Seed each DataLoader worker deterministically so any randomness
    # inside __getitem__ / collate_fn is reproducible across runs.
    # This DataLoader runs on the driver only, single process, no rank.
    def worker_init_fn(worker_id):
        worker_seed = params.run.seed + worker_id
        set_random_seeds(worker_seed)

    # MixedDatasetSampler is a batch sampler (yields batches of indices)
    # pin_memory=False: the collate_fn returns plain Python lists/dicts (no tensors to pin)
    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=sampler,
                            num_workers=params.data.num_workers,
                            pin_memory=False,
                            collate_fn=collate_fn,
                            worker_init_fn=worker_init_fn,
                            )

    return dataloader

def shard_batch_for_engines(rollout_batch, num_rollout_engines):
    '''
        Shard a batch of prompts across rollout engines.
    '''
    if not rollout_batch:
        return []

    # recall: num_rollout_engines  = max(1, int(rollout_gpus) // tensor_parallel_size)
    # and rollout_batch is a list of dictionaries.
    # it is not necessary to have equal number of samples per engine, though they can't be empty.
    shard_size = (len(rollout_batch) + num_rollout_engines - 1) // num_rollout_engines
    rollout_shards = [rollout_batch[i * shard_size:(i + 1) * shard_size] for i in range(num_rollout_engines)]
    rollout_shards = [shard for shard in rollout_shards if len(shard) > 0]
    return rollout_shards

def merge_rollout_with_stats(rollout_lists):
    '''
        Calculate rollout stats while merging them
    '''
    # rollout engines retrun the followings:
    # policy_version, loaded_version, input_ids, token_rewards, token_zscores
    # token_masks, token_dones, token_old_logprobs, pred_rewards, pred_masks
    # pred_dones, pred_old_logprobs, pred_zscores, finish_reason, finish_reason
    # finish_reason, stop_reason, ended_on_eos, response_ids, prompt_ids, response_text,
    # response_len, truncated
    total_samples_generated = 0
    # rewards
    all_rewards = []
    all_zscores = []
    # response
    all_response_lens = []
    min_response_len = float('inf')
    max_response_len = float('-inf')
    # logprobs
    total_logprob_sum = 0.0
    total_logprob_tokens = 0
    # tokens
    total_tokens = 0
    total_truncated = 0
    total_seq_truncated = 0
    total_eos = 0
    total_finish_stop = 0
    # prompts
    total_prompt_len = 0
    prompt_response_groups = {}

    rollout_merged = []
    for rl in rollout_lists:
        rollout_merged.extend(rl)
        for sample in rl:
            total_samples_generated += 1
            # reward stats
            all_rewards.append(sample['pred_rewards'].sum().item())
            all_zscores.append(sample['pred_zscores'].sum().item())
            # response stats
            all_response_lens.append(sample['response_len'])
            min_response_len = min(min_response_len, sample['response_len'])
            max_response_len = max(max_response_len, sample['response_len'])
            # pred_old_logprobs only contains logprob for response
            resp_logprobs = sample['pred_old_logprobs'] * sample['pred_masks']
            total_logprob_sum += resp_logprobs.sum().item()
            total_logprob_tokens += (sample['pred_masks'] > 0.5).sum().item()
            # other stats
            if sample.get('ended_on_eos', False):
                total_eos += 1
            if sample.get('finish_reason') == 'stop':
                total_finish_stop += 1

            # prompt_ids tuple -> [total_count, set of unique response texts]
            total_prompt_len += len(sample['prompt_ids'])
            prompt_key = tuple(sample['prompt_ids'])
            if prompt_key not in prompt_response_groups:
                prompt_response_groups[prompt_key] = [0, set()]
            prompt_response_groups[prompt_key][0] += 1
            prompt_response_groups[prompt_key][1].add(sample.get('response_text', ''))

            # token stats
            total_tokens += len(sample['prompt_ids']) + len(sample['response_ids'])
            total_truncated += sample['truncated']
            total_seq_truncated += sample['seq_truncated']

    stats = {'total_samples_generated': total_samples_generated,
            'all_rewards': all_rewards,
            'all_zscores': all_zscores,
            'all_response_lens': all_response_lens,
            'min_response_len': min_response_len,
            'max_response_len': max_response_len,
            'total_tokens': total_tokens,
            'total_truncated': total_truncated,
            'total_seq_truncated': total_seq_truncated,
            'total_eos': total_eos,
            'total_finish_stop': total_finish_stop,
            'total_prompt_len': total_prompt_len,
            'prompt_response_groups': prompt_response_groups,
            'total_logprob_sum': total_logprob_sum,
            'total_logprob_tokens': total_logprob_tokens,}

    return rollout_merged, stats

def collect_rollouts(dataloader,
                     rollout_engines,
                     epoch,
                     policy_version,
                     replay_buffer,
                     n_samples,
                     logger,
                     rollout_timeout):

    '''
        This function is used to run rollout engine and generate rollouts/samples.
    '''
    num_rollout_engines = len(rollout_engines)
    rollout_start_time = time.time()
    acc = rollout_stats.new_accumulator()

    # rollout_samples_per_epoch is the number of PROMPTS, not total completions.
    # example: rollout_gpus=2, rollout_batch_size_per_gpu=12, n_samples=3, rollout_samples_per_epoch = 25
    # local_batch_size = num_rollout_engines * rollout_batch_size_per_gpu = 2 * 12 = 24
    # Batches needed = ceil(25 / 24) = 2 batches
    # Total Prompts = 2 * 24 = 48 prompts (rounded up to batch boundary)
    # Total completions in replay buffer = 48 prompts * 3 n_samples = 144
    batch_size = dataloader.batch_sampler.local_batch_size
    num_batches_per_epoch = len(dataloader)
    total_prompts = num_batches_per_epoch * batch_size
    prompts_per_engine = batch_size // num_rollout_engines

    logger.info(f"[Rollout] {total_prompts} prompts ({num_batches_per_epoch} batches x {batch_size} prompts/batch), "
                f"{num_rollout_engines} engines ({prompts_per_engine} prompts/engine/batch), "
                f"{n_samples} samples/prompt, "
                f"~{total_prompts * n_samples} expected samples in replay buffer")

    for rollout_batch in dataloader:
        # 1. split data across rollout engines
        rollout_shards = shard_batch_for_engines(rollout_batch, num_rollout_engines)
        if not rollout_shards:
            continue

        # 2. schedule rollout generation
        rollout_samples = []
        for i, shard in enumerate(rollout_shards):
            rollout_samples.append(rollout_engines[i].generate.remote(prompts=shard,
                                                                      current_iter=epoch,
                                                                      policy_version=policy_version))

        # 3. gather rollouts. This is a blocking call means all engines must
        # finish generating rollouts before we can proceed.
        rollout_lists = ray_get_with_timeout(refs=rollout_samples,
                                             timeout=rollout_timeout,
                                             description=f"rollout generation (epoch {epoch+1})",
                                             logger=logger)

        # 4. merge rollouts across all engines and collect stats
        rollout_merged, stats = merge_rollout_with_stats(rollout_lists)
        rollout_stats.accumulate(acc, stats)

        # 5. now add them to replay buffer
        replay_buffer.add_batch_seqs(rollout_merged)

    if len(replay_buffer) == 0:
        raise ValueError("Replay buffer is empty")

    if acc['total_samples_generated'] == 0:
        logger.warning("No samples generated during rollout phase!")

    return rollout_stats.summarize(acc, rollout_time=time.time() - rollout_start_time)

def weighted_sampler_by_recency(replay_buffer,
                                recency_decay: float,
                                current_policy_version: int | None,
                                generator) -> 'WeightedRandomSampler | None':
    '''
        Build a WeightedRandomSampler (with replacement) that biases sampling
        toward items produced by more recent policy versions.
        recency_decay: (0.0, 1.0]. higher = retain more old data (closer to uniform), lower
        = bias more aggressively toward recent data.
            recency_decay = 1.0  -> uniform: all weights = 1.0, no bias.
            recency_decay = 0.9  -> mild: 1 step old has 0.9x weight of fresh
            recency_decay = 0.8  -> moderate
            recency_decay = 0.5  -> strong: 1 step old has half weight, old
                                            items rarely sampled
            recency_decay -> 0   -> essentially on-policy: only fresh sampled
    '''
    if recency_decay >= 1.0 or len(replay_buffer) == 0:
        return None

    if current_policy_version is None:
        current_v = max(s["policy_version"] for s in replay_buffer.items)

    else:
        current_v = current_policy_version

    weights = torch.tensor([recency_decay ** (current_v - s["policy_version"])
                            for s in replay_buffer.items], dtype=torch.float32)

    return WeightedRandomSampler(weights=weights,
                                 num_samples=len(replay_buffer),
                                 replacement=True,
                                 generator=generator)

def prepare_training_batches(replay_buffer,
                             batch_size: int,
                             num_engines: int,
                             seed: int = 0,
                             epoch: int = 0,
                             recency_decay: float = 1.0,
                             current_policy_version: int | None = None,
                             max_batches: int | None = None) -> list:
    '''
        Create and pad training batches for distributed training.
        current_policy_version: trainer's current policy version, defaults
            to None for sync-mode callers that don't track versions.
        max_batches: cap the number of micro-batches built before padding.
            Used by async mode to bound per-call work to one optimizer step
            regardless of replay buffer size. without this, prepare_training_batches
            scales O(buffer) which becomes seconds at large scale.
    '''
    # Create dataloader from replay buffer. Use a seeded generator for
    # deterministic order across runs.
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    sampler = weighted_sampler_by_recency(replay_buffer, recency_decay,
                                          current_policy_version, g)
    if sampler is not None:
        loader = DataLoader(dataset=replay_buffer,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=replay_buffer.collate_fn)
    else:
        loader = DataLoader(dataset=replay_buffer,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False,
                            collate_fn=replay_buffer.collate_fn,
                            generator=g)
    # We materialize lazily via a for-loop so max_batches can short-circuit
    # before the entire buffer is read.
    train_batches = []
    for b in loader:
        train_batches.append(b)
        if max_batches is not None and len(train_batches) >= max_batches:
            break

    # Pad to ensure equal batches per engine (prevents DeepSpeed hang)
    num_batches = len(train_batches)
    batches_per_engine = (num_batches + num_engines - 1) // num_engines
    total_needed = batches_per_engine * num_engines

    if total_needed > num_batches:
        # Pad by repeating the last batch
        padding = [train_batches[-1]] * (total_needed - num_batches)
        batches_padded = train_batches + padding

    else:
        batches_padded = train_batches

    return batches_padded

def shard_and_put(batches, num_engines):
    '''
       Pre-shard batches across engines and store in Ray object store.
       Returns a list of ObjectRefs, one per engine.
    '''
    shard_refs = []
    shard_sizes = []
    for eid in range(num_engines):
        # engine 0 gets [0, 2, 4, ...], engine 1 gets [1, 3, 5, ...]
        shard = batches[eid::num_engines]
        assert len(shard) > 0, f"Engine {eid} has empty shard. This will cause DeepSpeed hang"
        shard_sizes.append(len(shard))
        shard_refs.append(ray.put(shard))

    if len(set(shard_sizes)) > 1:
        print(f"[shard_and_put] SHARD SIZE MISMATCH: {shard_sizes}. "
              f"This WILL cause a ZeRO-3 collective deadlock!", flush=True)
    else:
        print(f"[shard_and_put] {num_engines} shards, {shard_sizes[0]} micro-batches each "
              f"(total={len(batches)})", flush=True)

    return shard_refs

def run_training_step(engines, shard_refs, logger, train_step_timeout):
    '''
       Execute one training step across all engines.
       shard_refs: list of Ray ObjectRefs (one per engine), created by shard_and_put().
       Ray auto-resolves ObjectRefs passed to .remote(), so the engine receives the actual data.
    '''
    step_start = time.time()

    futures = []
    for eid, engine in enumerate(engines):
        futures.append(engine.train_step.remote(engine_id=eid, micro_batches=shard_refs[eid]))

    logger.info(f"[run_training_step] Dispatched to {len(engines)} engines, waiting for results...")
    # Gather training metrics from all engines
    metrics_list = ray_get_with_timeout(refs=futures,
                                        timeout=train_step_timeout,
                                        description="training step",
                                        logger=logger)
    logger.info(f"[run_training_step] All engines returned in {time.time() - step_start:.1f}s")

    # Dynamically aggregate all metric keys across engines.
    # metrics_list: clipfrac, approx_kl, loss_ent, loss_pi, loss_total, kl_ref
    # if value network, add: v_loss
    # loss_total includes: loss_pi + ent_coef * loss_ent
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    return {k: np.mean([m.get(k, 0.0) for m in metrics_list]) for k in all_keys}

def sync_weights_direct(training_engines, rollout_engines, version, logger, sync_timeout):
    '''
        Transfer weights directly from deepspeed training engines to vllm rollout
        engines via ray object store. No disk I/O.
    '''
    state_dict_ref, _ = gather_training_weights(training_engines, logger, sync_timeout=sync_timeout)
    if state_dict_ref is None:
        return False

    return push_weights_to_rollout(rollout_engines, state_dict_ref, version, logger, sync_timeout=sync_timeout)

def gather_training_weights(training_engines, logger, sync_timeout):
    '''
        Gather state_dict from training engines and store in ray object store.
    '''
    start_time = time.time()
    logger.info(f"[WeightSync] Gathering state_dict from training engines...")

    # All training engines must participate in gather_state_dict(). see common.py.
    gather_futures = [engine.gather_state_dict.remote() for engine in training_engines]
    gather_results = ray_get_with_timeout(refs=gather_futures,
                                          timeout=sync_timeout,
                                          description="gather_state_dict from training engines",
                                          logger=logger)

    state_dict = gather_results[0]
    if not state_dict:
        logger.error("[WeightSync] Rank 0 returned empty state_dict")
        return None, 0

    end_time = time.time() - start_time
    logger.info(f"[WeightSync] Gathered the parameters in {end_time:.2f}s")

    # Takes a local state_dict, serializes it, and stores it in
    # the ray object store which is distributed shared memory.
    # it is mostly non-blocking (upload to shared memory)
    state_dict_ref = ray.put(state_dict)
    del state_dict

    return state_dict_ref, end_time

def push_weights_to_rollout(rollout_engines, state_dict_ref, version, logger, sync_timeout):
    '''
        Push pre-gathered weights to rollout engines. Blocks until all engines updated.
        Returns True if all engines updated successfully.
    '''
    start_time = time.time()
    update_futures = [eng.update_weights_direct.remote(state_dict_ref, version) for eng in rollout_engines]
    results = ray_get_with_timeout(refs=update_futures,
                                   timeout=sync_timeout,
                                   description=f"push weights v{version} to rollout engines",
                                   logger=logger)

    end_time = time.time() - start_time
    success = all(results)

    if success:
        logger.info(f"[WeightSync] Pushed weights v{version} to rollout engines in {end_time:.2f}s")

    else:
        logger.warning(f"[WeightSync] Some rollout engines failed to update to v{version}")

    return success

def refresh_rollout_engine(rollout_engines, updated_policy_path, version, logger, sync_timeout):
    '''
        Refresh rollout engine with the latest policy using disk-based fallback.
    '''
    refresh_futures = []
    for eng in rollout_engines:
        refresh_futures.append(eng.refresh_model.remote(updated_policy_path, version))

    ray_get_with_timeout(refs=refresh_futures,
                         timeout=sync_timeout,
                         description=f"refresh rollout engines from disk (v{version})",
                         logger=logger)

def reinit_nccl_weight_sync_group(training_engines, rollout_engines, master_addr, nccl_port, tp_size, logger, init_timeout, backend):
    '''
        Close existing NCCL weight sync groups and re-initialize.
        Safe to call even if no group exists yet. Used after resume or
        disk-fallback refresh that may destroy EngineCore workers.
    '''
    # Close old groups (no-ops if not initialized)
    try:
        ray.get(training_engines[0].close_weight_nccl_group.remote())
    except Exception:
        pass
    for eng in rollout_engines:
        try:
            ray.get(eng.close_nccl_group.remote())
        except Exception:
            pass

    return init_nccl_weight_sync(training_engines=training_engines,
                                 rollout_engines=rollout_engines,
                                 master_addr=master_addr,
                                 nccl_port=nccl_port,
                                 tp_size=tp_size,
                                 logger=logger,
                                 init_timeout=init_timeout,
                                 backend=backend)

def init_nccl_weight_sync(training_engines, rollout_engines, master_addr, nccl_port, tp_size, logger, init_timeout, backend):
    '''
        Initialize the NCCL weight sync group across training rank 0 and
        all vllm tp workers. All participants must call into the NCCL
        rendezvous concurrently.
        Rank assignment:
          rank 0: training engine rank 0
          rank 1..tp: rollout engine 0, TP workers 0..tp-1
          rank tp+1..2*tp: rollout engine 1, TP workers 0..tp-1
          ....
        world_size = 1 + num_rollout_engines * tp_size
        backend: "nccl" for gpu-to-gpu broadcast, gloo for cpu-based.
    '''
    num_rollout_engines = len(rollout_engines)
    world_size = 1 + num_rollout_engines * tp_size
    group_name = "feynrl_weight_sync"

    logger.info(f"[init_nccl_weight_sync - main] Initializing weight sync group: world_size={world_size}, "
                f"port={nccl_port}, training_rank=0, "
                f"{num_rollout_engines} rollout engines x TP={tp_size}, backend={backend}")

    # All participants must call rendezvous concurrently.
    # Training rank 0 gets rank=0.
    futures = []
    futures.append(training_engines[0].init_weight_nccl_group.remote(master_addr=master_addr,
                                                                    master_port=nccl_port,
                                                                    rank=0,
                                                                    world_size=world_size,
                                                                    group_name=group_name,
                                                                    timeout_seconds=init_timeout,
                                                                    backend=backend))

    # Each rollout engine gets rank_offset = 1 + engine_idx * tp_size,
    # and its TP workers compute their own ranks internally.
    for engine_idx, engine in enumerate(rollout_engines):
        rank_offset = 1 + engine_idx * tp_size
        futures.append(engine.init_nccl_group.remote(master_addr=master_addr,
                                                    master_port=nccl_port,
                                                    rank_offset=rank_offset,
                                                    world_size=world_size,
                                                    group_name=group_name,
                                                    timeout_seconds=init_timeout,
                                                    backend=backend))
    # no need to return results, just waiting for all to finish
    ray_get_with_timeout(refs=futures,
                        timeout=init_timeout,
                        description="NCCL weight sync group initialization at main",
                        logger=logger)
    logger.info(f"[WeightSync] Weight sync group initialized (world_size={world_size}, backend={backend})")

    return world_size, group_name

def start_nccl_gather(training_engines):
    '''
        Fire the ZeRO-3 gather on all training ranks (non-blocking).
        Returns the list of Ray ObjectRefs for the gather futures. The caller
        must eventually ray.get() them to obtain param_metadata.

        This is the first phase of NCCL weight sync, separated so callers can
        overlap the gather with other work (e.g. draining rollout pull loops).
    '''
    return [engine.gather_weights_for_nccl.remote() for engine in training_engines]

def complete_nccl_gather(gather_futures, version, logger, sync_timeout):
    '''
        Wait for gather futures and extract param_metadata.
        Returns param_metadata (list of (name, dtype, shape) tuples), or empty
        list if no parameters were gathered.
    '''
    gather_results = ray_get_with_timeout(refs=gather_futures,
                                          timeout=sync_timeout,
                                          description=f"NCCL weight gather v{version}",
                                          logger=logger)

    param_metadata = []
    for result in gather_results:
        if isinstance(result, list) and len(result) > 0:
            param_metadata = result
            break

    return param_metadata

def broadcast_and_finalize_nccl(training_engines, rollout_engines, param_metadata, version, logger, sync_timeout):
    '''
        Phases 2+3 of NCCL weight sync: broadcast gathered weights from training rank 0 to all rollout engines, then finalize.
        Expects gather to have already completed and param_metadata extracted via complete_nccl_gather. 
        The gathered state dict sits in training_engines[0].pending_nccl_state_dict.
        Returns True on full success, raises on any failure.
    '''
    if not param_metadata:
        logger.warning("[broadcast_and_finalize_nccl] No param_metadata; skipping broadcast")
        return True

    num_params = len(param_metadata)
    num_engines = len(rollout_engines)

    # Phase 2: Fire receive on all engines, then training broadcast.
    # No barrier needed — NCCL broadcast is the implicit synchronization point.
    # Receive RPCs queue behind any in-flight generate calls in the actor mailbox.
    logger.info(f"[broadcast_and_finalize_nccl] Dispatching receive ({num_params} params) to {num_engines} engines...")
    rollout_refs = [eng.receive_all_weights_nccl.remote(param_metadata)
                    for eng in rollout_engines]

    broadcast_ref = training_engines[0].nccl_broadcast_gathered.remote()
    all_refs = rollout_refs + [broadcast_ref]

    try:
        ray_get_with_timeout(refs=all_refs,
                             timeout=sync_timeout,
                             description=f"NCCL broadcast v{version}",
                             logger=logger)
    except Exception as e:
        logger.error(f"[broadcast_and_finalize_nccl] NCCL broadcast v{version} failed or timed out: {e}. "
                     f"Rollout engines may be stuck in NCCL collective. "
                     f"If training hangs after this, restart the job.")
        raise

    # Phase 3: Finalize and verify every engine reports a full load.
    # finalize_weight_nccl returns False if a partial load was detected
    # (received < expected params). A single False means we cannot safely
    # advance rollout_policy_version because at least one engine still
    # holds the old weights — silently advancing would mislabel future
    # samples and corrupt importance-sampling statistics.
    finalize_refs = [eng.finalize_weight_nccl.remote(version, num_params) for eng in rollout_engines]
    finalize_results = ray_get_with_timeout(refs=finalize_refs,
                                            timeout=sync_timeout,
                                            description=f"finalize weight sync v{version}",
                                            logger=logger)

    if not all(bool(r) for r in finalize_results):
        failed = [i for i, r in enumerate(finalize_results) if not r]
        raise RuntimeError(f"[broadcast_and_finalize_nccl] Partial weight load on engines {failed}: "
                           f"finalize_weight_nccl returned False. At least one engine "
                           f"received fewer params than expected ({num_params}). Aborting "
                           f"sync to prevent silent version mismatch.")

    return True

def sync_weights_nccl(training_engines, rollout_engines, version, logger, sync_timeout):
    '''
        Broadcast weights from training engines to rollout engines via nccl.
        Three phases:
          1. Gather: all training ranks participate in zero-3 collective gather.
          2. Broadcast: fire receive on all rollout engines, then fire training broadcast.
             Receive RPCs queue in Ray's actor mailbox behind any in-flight generate
             calls. NCCL broadcast blocks until all participants enter.
          3. Finalize: rollout engines load received weights into vLLM. We wait on
             every engine and check the boolean return values — a False from any
             engine indicates a partial load (the engine refused to advance its
             loaded_version) and is treated as a sync failure.
        Must only be called when ALL training engines are idle.
        Returns True on full success, raises on any failure (gather/broadcast/
        partial finalize). On failure, the caller is responsible for calling
        clear_pending_nccl_state_dict on training rank 0 to free the gathere CPU buffer.
    '''
    start_time = time.time()
    logger.info(f"[sync_weights_nccl] Starting weight sync v{version} to {len(rollout_engines)} rollout engines...")

    gather_futures = start_nccl_gather(training_engines)
    param_metadata = complete_nccl_gather(gather_futures, version, logger, sync_timeout)

    if not param_metadata:
        logger.warning("[sync_weights_nccl] Phase 1: no parameters gathered; skipping broadcast")
        return True

    broadcast_and_finalize_nccl(training_engines, rollout_engines, param_metadata, version, logger, sync_timeout)

    elapsed = time.time() - start_time
    logger.info(f"[sync_weights_nccl] Sync v{version} complete in {elapsed:.2f}s "
                f"({len(param_metadata)} params, {len(rollout_engines)} engines)")
    return True