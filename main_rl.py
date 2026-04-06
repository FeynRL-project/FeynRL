import os
import json
import copy
import atexit
import numpy as np
import argparse
import importlib
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import ray
import time
import shutil
from dataclasses import dataclass
import concurrent.futures

# imports local methods, classes, etc.
import configs.load as cfg # all config arguments
from data_feeds.prompts import PromptsFeed # our custom pytorch dataset
from data_feeds.mixed_sampler import create_prompt_dataset_and_sampler
from misc.utils import safe_string_to_torch_dtype, get_experiment_dir_name, load_algorithm, ray_get_with_timeout, set_random_seeds, get_determinism_env_vars
from misc.nccl_utils import NCCLBarrier
from rollouts.vllm_engine import VLLMRolloutEngine
from rollouts.vllm_engine_async import VLLMRolloutEngineAsync
from rollouts.replay_buffer import ReplayBuffer
from misc.logging import setup_logging, setup_tracker
from misc.setup_rl import load_tokenizer, save_checkpoint, load_checkpoint_for_resume, setup_ray
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

    stats = {'total_samples_generated': total_samples_generated,
            'all_rewards': all_rewards,
            'all_zscores': all_zscores,
            'all_response_lens': all_response_lens,
            'min_response_len': min_response_len,
            'max_response_len': max_response_len,
            'total_tokens': total_tokens,
            'total_truncated': total_truncated,
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

def prepare_training_batches(replay_buffer, batch_size: int, num_engines: int, seed: int = 0, epoch: int = 0) -> list:
    '''
        Create and pad training batches for distributed training.
    '''
    # Create dataloader from replay buffer and convert
    # to list as ray needs serializable data.
    # Use a seeded generator for deterministic shuffle order across runs.
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    train_batches = list(DataLoader(dataset=replay_buffer,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=False,
                                    collate_fn=replay_buffer.collate_fn,
                                    generator=g,
                                    ))
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

def sync_weights_nccl(training_engines, rollout_engines, version, logger, sync_timeout, use_barrier=True):
    '''
        Broadcast weights from training engines to rollout engines via NCCL.
        Three phases:
          1. Gather: all training ranks participate in zero-3 collective gather.
          2. Broadcast: fire receive on all rollout engines, then fire training broadcast.
             When use_barrier=True: poll NCCLBarrier to ensure all engines are
             ready before broadcasting. Use when engines are known to be idle.
             When use_barrier=False: skip barrier, rely on NCCL's implicit synchronization.
             Use when engines may be busy generating (non-blocking sync). The receive RPCs
             queue in Ray's actor mailbox behind any in-flight generate calls, and NCCL
             broadcast blocks until all participants enter.
          3. Finalize: rollout engines load received weights into vLLM.
        Must only be called when ALL training engines are idle.
    '''
    start_time = time.time()
    mode = "blocking" if use_barrier else "non-blocking"
    logger.info(f"[sync_weights_nccl] Starting {mode} weight sync v{version} to {len(rollout_engines)} rollout engines...")

    # Phase 1: Gather state dict on all training ranks.
    gather_futures = [engine.gather_weights_for_nccl.remote() for engine in training_engines]
    gather_results = ray_get_with_timeout(refs=gather_futures,
                                          timeout=sync_timeout,
                                          description=f"NCCL weight gather v{version}",
                                          logger=logger)

    param_metadata = []
    for result in gather_results:
        if isinstance(result, list) and len(result) > 0:
            param_metadata = result
            break

    if not param_metadata:
        logger.warning("[sync_weights_nccl] Phase 1: no parameters gathered; skipping broadcast")
        return True, []

    num_params = len(param_metadata)
    num_engines = len(rollout_engines)

    # Phase 2a: Fire receive on all engines.
    if use_barrier:
        barrier = NCCLBarrier.remote(expected=num_engines)
        logger.info(f"[sync_weights_nccl] Phase 2a: dispatching receive ({num_params} params) to {num_engines} engines (with barrier)...")
        rollout_refs = [eng.receive_all_weights_nccl.remote(param_metadata, barrier)
                        for eng in rollout_engines]

        # Phase 2b: Poll barrier until ALL engines have signaled ready.
        barrier_timeout = min(sync_timeout, 300)
        barrier_poll_timeout = 10
        barrier_deadline = time.time() + barrier_timeout
        # Poll the barrier actor until all engines have signaled ready.
        # 0.5s sleep prevents busy-spinning on the remote call while adding
        # negligible latency (at most one extra poll interval after the last
        # engine checks in). This scales to multi-GPU / multi-node clusters
        # because we're only querying a single counter on one Ray actor and
        # cluster size affects how long we wait, not how often we poll.
        while time.time() < barrier_deadline:
            try:
                count = ray.get(barrier.get_count.remote(), timeout=barrier_poll_timeout)
            except Exception:
                count = 0
            if count >= num_engines:
                break
            time.sleep(0.5)

        else:
            try:
                count = ray.get(barrier.get_count.remote(), timeout=barrier_poll_timeout)
            except Exception:
                count = 0

            if count < num_engines:
                raise TimeoutError(f"[sync_weights_nccl] Phase 2b: only {count}/{num_engines} engines signaled ready "
                                   f"within {barrier_timeout}s. Check engine logs for errors.")

        logger.info(f"[sync_weights_nccl] Phase 2b: all {num_engines} engines ready, firing training broadcast...")

    else:
        # Non-blocking: no barrier. RPCs queue behind any in-flight generate calls.
        # NCCL broadcast is the implicit synchronization point.
        logger.info(f"[sync_weights_nccl] Phase 2a: dispatching receive ({num_params} params) to {num_engines} engines (no barrier)...")
        rollout_refs = [eng.receive_all_weights_nccl.remote(param_metadata, None)
                        for eng in rollout_engines]

    # Phase 2c: Fire training broadcast and wait for all broadcasts to complete.
    broadcast_ref = training_engines[0].nccl_broadcast_gathered.remote()
    all_refs = rollout_refs + [broadcast_ref]

    try:
        ray_get_with_timeout(refs=all_refs,
                             timeout=sync_timeout,
                             description=f"NCCL broadcast v{version}",
                             logger=logger)
    except Exception as e:
        # If broadcast times out or fails, rollout engines may be stuck in
        # torch.distributed.broadcast / PyNcclCommunicator.broadcast with no
        # cancellation API. The engines will remain blocked until all participants
        # enter the collective (or NCCL's internal watchdog kills the process).
        # Subsequent Ray calls to these engines will queue behind the stuck broadcast.
        # Recovery requires restarting the stuck actors.
        logger.error(f"[sync_weights_nccl] NCCL broadcast v{version} failed or timed out: {e}. "
                     f"Rollout engines may be stuck in NCCL collective. "
                     f"If training hangs after this, restart the job.")
        raise

    # Phase 3: Finalize — fire but don't block.
    finalize_refs = [eng.finalize_weight_nccl.remote(version, num_params) for eng in rollout_engines]

    elapsed = time.time() - start_time
    logger.info(f"[sync_weights_nccl] Broadcast complete in {elapsed:.2f}s, finalize dispatched ({mode})")
    return True, finalize_refs

@dataclass
class ChunkFuture:
    '''
        A group of per-engine Ray ObjectRefs covering chunk_size dataloader batches.
    '''
    # list of [per-engine ObjectRef lists], one per batch in chunk
    futures: list
    dispatch_time: float
    chunk_idx: int

def chunk_is_ready(chunk):
    '''
        Non-blocking check to see if all futures chunks completed
        we use ray.wait with timeout=0 so this is essentially free.
    '''
    all_refs = [ref for batch_futures in chunk.futures for ref in batch_futures]
    ready, _ = ray.wait(all_refs, num_returns=len(all_refs), timeout=0)
    return len(ready) == len(all_refs)

def dispatch_one_chunk(dataloader_iter, rollout_engines, epoch, policy_version, chunk_size, chunk_idx, logger):
    '''
        Dispatch exactly one chunk (given chunk_size) of generation to rollout engines.
        Only ONE chunk should be in-flight at a time. This keeps each actor's
        fifo mailbox empty between chunks, creating idle windows where nccl weight sync can execute immediately.
    '''
    num_engines = len(rollout_engines)
    batch_futures = []

    for _ in range(chunk_size):
        try:
            # Pull up to chunk_size batches from dataloader_iter and call generate on each engine.
            # it returns a ChunkFuture, or none when is exhausted.
            # since dataloader_iter is an iterator, it remembers its position.
            # So the next call to dispatch_one_chunk picks up where the last one left off.
            rollout_batch = next(dataloader_iter)

        except StopIteration as e:
            logger.info(f"[Main - Dispatcher] Dataloader exhausted after {chunk_idx} chunks.")
            break
        # shard batch for rollout engines
        shards = shard_batch_for_engines(rollout_batch, num_engines)
        # now generate responses for each shard
        futures = [rollout_engines[i].generate.remote(prompts=shard,
                                                      current_iter=epoch,
                                                      policy_version=policy_version)
                   for i, shard in enumerate(shards)]
        batch_futures.append(futures)

    if not batch_futures:
        return None

    return ChunkFuture(futures=batch_futures, dispatch_time=time.time(), chunk_idx=chunk_idx)

def finalize_chunk(chunk, replay_buffer, logger, rollout_timeout):
    '''
        Block until a single chunk's futures resolve, merge results into
        the replay buffer, and return accumulated per-chunk stats.
    '''
    acc = rollout_stats.new_accumulator()
    for batch_futures in chunk.futures:
        results = ray_get_with_timeout(refs=batch_futures, timeout=rollout_timeout, description=f"chunk {chunk.chunk_idx}", logger=logger)
        merged, stats = merge_rollout_with_stats(results)
        replay_buffer.add_batch_seqs(merged)
        rollout_stats.accumulate(acc, stats)

    return acc

def aggregate_chunk_stats(chunk_stats_list, generation_time, wall_time):
    '''
        Combine partial stats from multiple finalize_chunk calls.
    '''
    acc = rollout_stats.new_accumulator()
    for cs in chunk_stats_list:
        rollout_stats.accumulate(acc, cs)

    result = rollout_stats.summarize(acc, rollout_time=generation_time)
    result["rollout_time_with_overlap"] = wall_time
    return result

def run_epoch_sync(epoch, training_engines, rollout_engines, rollout_dataloader,
                   replay_buffer, policy_version, rollout_policy_version, global_step,
                   n_samples, train_batch_size, steps_per_epoch, seed,
                   rollout_timeout, train_step_timeout, tracker, logger):
    '''
        Sequential epoch: [collect_rollouts] -> [prepare_training_batches] -> [train]]
    '''
    # 1. Reset replay buffer
    replay_buffer.reset()

    # 2. all engines must finish before we proceed. collect_rollouts is blocking call.
    logger.info(f"[Epoch {epoch+1}] Starting rollout generation...")
    rollout_dataloader.batch_sampler.set_epoch(epoch)
    rollout_metrics = collect_rollouts(dataloader=rollout_dataloader,
                                       rollout_engines=rollout_engines,
                                       epoch=epoch,
                                       policy_version=policy_version,
                                       replay_buffer=replay_buffer,
                                       n_samples=n_samples,
                                       logger=logger,
                                       rollout_timeout=rollout_timeout)

    # 3. Prepare training batches
    logger.info(f"[Epoch {epoch+1}] Replay buffer has {len(replay_buffer)} samples")
    train_start_time = time.time()
    num_engines      = len(training_engines)
    # shuffles the replay buffer globally and creates training batches
    train_batches_padded = prepare_training_batches(replay_buffer=replay_buffer,
                                                    batch_size=train_batch_size,
                                                    num_engines=num_engines,
                                                    seed=seed,
                                                    epoch=epoch)
    samples_per_engine = len(replay_buffer) // num_engines
    micro_per_engine   = len(train_batches_padded) // num_engines
    logger.info(f"[Epoch {epoch+1}] Training: "
                f"{len(replay_buffer)} replay samples / {num_engines} training engines "
                f"= {samples_per_engine} samples/engine / bsz={train_batch_size} "
                f"= {micro_per_engine} micro-batches/engine, "
                f"{steps_per_epoch} pass(es) over replay buffer")

    # while each engine gets same micro-batches per step, we shuffle them inside train_step of each engine
    shard_refs = shard_and_put(train_batches_padded, num_engines=num_engines)

    # 4. Training loop
    epoch_metrics = {}
    for step in range(steps_per_epoch):
        train_metrics = run_training_step(engines=training_engines,
                                          shard_refs=shard_refs,
                                          logger=logger,
                                          train_step_timeout=train_step_timeout)
        # collect the metrics
        for k, v in train_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)

        global_step += 1
        if step % 10 == 0:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            logger.info(f"[Epoch {epoch+1}][Step {step+1}/{steps_per_epoch}] {metric_str}")

        if tracker:
            tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)

    # 5. Post-training
    policy_version += 1

    return {'rollout_metrics': rollout_metrics,
            'epoch_metrics': epoch_metrics,
            'global_step': global_step,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'train_step_count': steps_per_epoch,
            'train_time': time.time() - train_start_time,
            'sync_performed': False}

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

    batches = prepare_training_batches(replay_buffer=replay_buffer,
                                       batch_size=train_batch_size,
                                       num_engines=num_engines,
                                       seed=seed,
                                       epoch=epoch + shard_rebuild_count)
    refs = shard_and_put(batches, num_engines=num_engines)
    return refs, buf_len, shard_rebuild_count + 1, batches

def cycle_chunk(pending_chunk, replay_buffer, dataloader_iter, rollout_engines,
                epoch, rollout_policy_version, chunk_size, chunk_idx,
                chunk_stats_list, logger, rollout_timeout):
    '''
        Finalize pending chunk, append stats, dispatch next chunk.
        Returns (new_pending_chunk, new_chunk_idx).
    '''
    cs = finalize_chunk(chunk=pending_chunk,
                        replay_buffer=replay_buffer,
                        logger=logger,
                        rollout_timeout=rollout_timeout)
    chunk_stats_list.append(cs)

    next_chunk = dispatch_one_chunk(dataloader_iter=dataloader_iter,
                                    rollout_engines=rollout_engines,
                                    epoch=epoch,
                                    policy_version=rollout_policy_version,
                                    chunk_size=chunk_size,
                                    chunk_idx=chunk_idx,
                                    logger=logger)
    new_idx = chunk_idx + 1 if next_chunk is not None else chunk_idx
    return next_chunk, new_idx

def check_ess_sync(train_metrics, train_step_count, ess_sync_threshold,
                    fixed_sync_interval, sync_triggered_this_epoch):
    '''
        Check if ESS or fixed_sync_interval triggers a sync.
        Returns (should_stop, ess_value).
    '''
    ess = train_metrics.get('ess_factor', None)
    if not sync_triggered_this_epoch:
        if ess is not None and ess < ess_sync_threshold:
            return True, ess

        if fixed_sync_interval and train_step_count % fixed_sync_interval == 0:
            return True, ess

    return False, ess

def run_epoch_overlap(epoch, training_engines, rollout_engines, rollout_dataloader,
                      replay_buffer, policy_version, rollout_policy_version, global_step,
                      train_batch_size, steps_per_epoch, seed, chunk_size, max_lag,
                      ess_sync_threshold, fixed_sync_interval, is_last_epoch,
                      rollout_timeout, train_step_timeout, sync_timeout,
                      tracker, logger, pre_dispatched_chunk=None):
    '''
        Double-buffered overlap epoch: training and generation run concurrently.
        Training mode: training runs while the pending chunk generates on rollout engines.
                       Training is not interrupted by chunk readiness (no chunk_is_ready exit).
        Drain mode:    when training is done but chunks remain, the NEXT chunk is pre-dispatched
                       before finalizing the current one, so rollout engines generate during
                       the finalize wait (true double-buffering).
        NCCL sync:     when ESS or fixed_sync_interval triggers, no lookahead is dispatched,
                       engines are idle after finalize, and sync executes immediately.
    '''
    epoch_start_time = time.time()
    num_engines = len(training_engines)

    # 1. Buffer is already clean and no need to reset here..
    # 2. Setup chunked generation
    rollout_dataloader.batch_sampler.set_epoch(epoch)
    generation_start_time = time.time()
    data_policy_version   = rollout_policy_version
    dataloader_iter       = iter(rollout_dataloader)

    total_batches = len(rollout_dataloader)
    total_chunks  = (total_batches + chunk_size - 1) // chunk_size

    chunk_stats_list = []
    chunk_idx = 0
    epoch_metrics = {}
    sync_triggered_this_epoch = False
    train_step_count = 0
    training_done = False
    ess_break = False
    shard_refs = None
    shard_buffer_size = 0
    shard_rebuild_count = 0
    # Minimum new samples before rebuilding shards mid-training/drain.
    # One batch per engine ensures each engine gets at least one new micro-batch
    # from the rebuild. Set to 0 to rebuild on every chunk.
    shard_rebuild_min_samples = train_batch_size * num_engines
    # Drain training counters. drain_epoch_limit caps total drain steps across the
    # entire epoch (not per outer-loop iteration) to prevent excessive off-policy training.
    drain_steps_this_epoch = 0
    drain_epoch_limit = steps_per_epoch * 2
    train_start_time = time.time()

    # Dispatch first chunk, training can't overlap yet as there is no data.
    # If the previous epoch pre-dispatched a chunk during checkpoint save,
    # reuse it instead of dispatching a new one.
    if pre_dispatched_chunk is not None:
        pending_chunk = pre_dispatched_chunk
        chunk_idx += 1
        # Advance the dataloader past the batch(es) already consumed by early dispatch.
        # The pre-dispatched chunk used chunk_size batches from a separate iterator
        # on the same epoch, so we skip the same number here to avoid duplicates.
        for _ in range(chunk_size):
            try:
                next(dataloader_iter)

            except StopIteration:
                break
        logger.info(f"[Epoch {epoch+1}] Reusing pre-dispatched chunk from previous epoch")

    else:
        pending_chunk = dispatch_one_chunk(dataloader_iter=dataloader_iter,
                                           rollout_engines=rollout_engines,
                                           epoch=epoch,
                                           policy_version=rollout_policy_version,
                                           chunk_size=chunk_size,
                                           chunk_idx=chunk_idx,
                                           logger=logger)
        chunk_idx += 1

    logger.info(f"[Epoch {epoch+1}] Double-buffered generation: "
                f"~{total_chunks} chunks (chunk_size={chunk_size}), "
                f"policy v{rollout_policy_version}")

    # 3. Double-buffered generate-train loop. Each iteration:
    #   3.1.  Run training steps while the pending chunk generates (no chunk_is_ready exit).
    #   3.1b. If training is done, pre-dispatch the NEXT chunk before finalizing the current
    #         one, drain mode double-buffering: engines generate during finalize wait.
    #   3.2.  Finalize the chunk. Blocks until engines finish, merges results into buffer.
    #   3.3.  Rebuild training shards if the buffer grew from finalized chunk data.
    #   3.4.  If ESS triggered, NCCL sync at chunk boundary, engines are idle since no
    #         lookahead was dispatched when ess_break is True.
    #   3.5.  Advance pending: use lookahead (drain) or dispatch new chunk (training mode).
    # Overlap efficiency metrics. interleaved_sec includes training + drain + overhead
    # (shard rebuild, chunk cycling). gen_wait_sec is time blocked at finalize in step 3.2.
    # ratio = interleaved / (interleaved + wait): closer to 1.0 = better overlap.
    total_overlap_interleaved_sec = 0.0
    total_overlap_gen_wait_sec = 0.0
    version_bumped_early = False
    while pending_chunk is not None:
        # 3.1 Run training steps while chunks generate. When the in-flight chunk
        # finishes mid-training, finalize it immediately and dispatch the next one
        # so rollout engines stay busy throughout training (continuous overlap).
        iter_train_start = time.time()
        if not training_done and shard_refs is not None:
            while train_step_count < steps_per_epoch:
                # dict[str, float]: averaged metrics across engines
                train_metrics = run_training_step(engines=training_engines,
                                                  shard_refs=shard_refs,
                                                  logger=logger,
                                                  train_step_timeout=train_step_timeout)

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

                # ESS-driven early stop
                # (bool, float): ess_value can be None for non-P3O algorithms
                should_stop, ess = check_ess_sync(train_metrics, train_step_count,
                                                  ess_sync_threshold, fixed_sync_interval,
                                                  sync_triggered_this_epoch)
                if should_stop:
                    logger.info(f"[Epoch {epoch+1}][Step {train_step_count}] "
                                f"Sync triggered (ESS={ess}), ending training early "
                                f"({steps_per_epoch - train_step_count} steps skipped)")

                if should_stop:
                    training_done = True
                    ess_break     = True

                if tracker and ess is not None:
                    tracker.log_metrics({"nccl/ess_factor": ess,
                                         "nccl/sync_triggered": 1 if (sync_triggered_this_epoch or should_stop) else 0,
                                        }, step=global_step)

                if train_step_count >= steps_per_epoch:
                    training_done = True

                if training_done:
                    logger.info(f"[Epoch {epoch+1}] Training done after step {train_step_count}, "
                                f"pending_chunk={'yes' if pending_chunk is not None else 'no'}, "
                                f"ess_break={ess_break}")
                    break

                # Mid-training chunk cycling: if the in-flight chunk finished
                # while we were training, finalize it now and dispatch the next
                # one so rollout engines stay busy instead of idling.
                # chunk_is_ready: True if all futures in chunk are resolved
                if pending_chunk is not None and chunk_is_ready(pending_chunk):
                    logger.info(f"[Epoch {epoch+1}] Chunk {pending_chunk.chunk_idx + 1}/{total_chunks} "
                                f"finalized mid-training, replay buffer: {len(replay_buffer)} samples")

                    # (ChunkFuture|None, int): (next pending chunk, updated chunk_idx)
                    pending_chunk, chunk_idx = cycle_chunk(pending_chunk=pending_chunk,
                                                           replay_buffer=replay_buffer,
                                                           dataloader_iter=dataloader_iter,
                                                           rollout_engines=rollout_engines,
                                                           epoch=epoch,
                                                           rollout_policy_version=rollout_policy_version,
                                                           chunk_size=chunk_size,
                                                           chunk_idx=chunk_idx,
                                                           chunk_stats_list=chunk_stats_list,
                                                           logger=logger,
                                                           rollout_timeout=rollout_timeout)

                    # (shard_refs, buf_len, rebuild_count, batches) or None if skipped
                    result = try_rebuild_shards(replay_buffer=replay_buffer,
                                                train_batch_size=train_batch_size,
                                                num_engines=num_engines,
                                                seed=seed,
                                                epoch=epoch * total_chunks,
                                                shard_buffer_size=shard_buffer_size,
                                                shard_rebuild_count=shard_rebuild_count,
                                                min_new_samples=shard_rebuild_min_samples)
                    if result:
                        shard_refs, shard_buffer_size, shard_rebuild_count, _ = result

        # 3.1a Drain-mode training: when steps_per_epoch is exhausted but chunks
        # remain, keep training on the growing replay buffer instead of idling.
        logger.info(f"[Epoch {epoch+1}] Post-training: training_done={training_done}, ess_break={ess_break}, "
                    f"shard_refs={'yes' if shard_refs is not None else 'no'}, "
                    f"pending_chunk={'yes (idx=' + str(pending_chunk.chunk_idx) + ')' if pending_chunk is not None else 'no'}")
        # Runs whether the chunk is ready or not — trains on existing shards while
        # waiting, and cycles chunks as they finish (same pattern as main training).
        # ESS/fixed_sync can still trigger ess_break to stop and force a sync.
        # Drain steps are capped per epoch (not per outer-loop iteration) to prevent
        # excessive training on increasingly off-policy data.
        if training_done and not ess_break and shard_refs is not None and pending_chunk is not None:
            drain_step_limit = min(steps_per_epoch, drain_epoch_limit - drain_steps_this_epoch)
            if drain_step_limit > 0:
                drain_steps_taken = 0
                logger.info(f"[Epoch {epoch+1}] Drain training: up to {drain_step_limit} steps "
                            f"(epoch total: {drain_steps_this_epoch}/{drain_epoch_limit}) "
                            f"on {len(replay_buffer)} replay samples")

                while drain_steps_taken < drain_step_limit:
                    # dict[str, float]: averaged metrics across engines
                    train_metrics = run_training_step(engines=training_engines,
                                                      shard_refs=shard_refs,
                                                      logger=logger,
                                                      train_step_timeout=train_step_timeout)
                    for k, v in train_metrics.items():
                        epoch_metrics.setdefault(k, []).append(v)

                    global_step += 1
                    train_step_count += 1
                    drain_steps_taken += 1
                    drain_steps_this_epoch += 1

                    if drain_steps_taken % 10 == 0 or drain_steps_taken == 1:
                        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                        logger.info(f"[Epoch {epoch+1}][Drain Step {drain_steps_taken}/{drain_step_limit}] {metric_str}")

                    if tracker:
                        tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()},
                                           step=global_step)

                    # ESS/fixed_sync check
                    # (bool, float): ess_value can be None for non-P3O algorithms
                    should_stop, ess = check_ess_sync(train_metrics,
                                                      train_step_count,
                                                      ess_sync_threshold,
                                                      fixed_sync_interval,
                                                      sync_triggered_this_epoch)
                    if should_stop:
                        logger.info(f"[Epoch {epoch+1}][Drain Step {drain_steps_taken}] "
                                    f"Sync triggered (ESS={ess}), ending drain training")
                        ess_break = True

                    if tracker and ess is not None:
                        tracker.log_metrics({"nccl/ess_factor": ess,
                                             "nccl/sync_triggered": 1 if (sync_triggered_this_epoch or should_stop) else 0,
                                            }, step=global_step)

                    if should_stop:
                        break

                    # Mid-drain chunk cycling: finalize ready chunks and dispatch next
                    # chunk_is_ready returns bool, cycle_chunk returns (ChunkFuture|None, int)
                    if pending_chunk is not None and chunk_is_ready(pending_chunk):
                        logger.info(f"[Epoch {epoch+1}] Chunk finalized mid-drain, "
                                    f"replay buffer: {len(replay_buffer)} samples")
                        pending_chunk, chunk_idx = cycle_chunk(pending_chunk=pending_chunk,
                                                               replay_buffer=replay_buffer,
                                                               dataloader_iter=dataloader_iter,
                                                               rollout_engines=rollout_engines,
                                                               epoch=epoch,
                                                               rollout_policy_version=rollout_policy_version,
                                                               chunk_size=chunk_size,
                                                               chunk_idx=chunk_idx,
                                                               chunk_stats_list=chunk_stats_list,
                                                               logger=logger,
                                                               rollout_timeout=rollout_timeout)

                        # (shard_refs, buf_len, rebuild_count, batches) or None if skipped
                        result = try_rebuild_shards(replay_buffer=replay_buffer,
                                                    train_batch_size=train_batch_size,
                                                    num_engines=num_engines,
                                                    seed=seed,
                                                    epoch=epoch * total_chunks,
                                                    shard_buffer_size=shard_buffer_size,
                                                    shard_rebuild_count=shard_rebuild_count,
                                                    min_new_samples=shard_rebuild_min_samples)
                        if result:
                            shard_refs, shard_buffer_size, shard_rebuild_count, _ = result

                    if pending_chunk is None:
                        break

        # 3.1b Pre-dispatch lookahead in drain mode.
        # When training is done and there are more chunks, dispatch the next
        # chunk BEFORE finalizing the current one. This way rollout engines
        # generate the next chunk while finalize_chunk blocks on the current one.
        # In training mode (training_done=False), we dispatch at the bottom
        # of the loop (step 3.5) so training overlaps in the next iteration.
        if training_done and not ess_break:
            # ChunkFuture or None if dataloader exhausted
            lookahead = dispatch_one_chunk(dataloader_iter=dataloader_iter,
                                           rollout_engines=rollout_engines,
                                           epoch=epoch,
                                           policy_version=rollout_policy_version,
                                           chunk_size=chunk_size,
                                           chunk_idx=chunk_idx,
                                           logger=logger)
            if lookahead is not None:
                chunk_idx += 1

        else:
            lookahead = None

        # 3.2 Finalize current chunk (may already be finalized mid-training)
        if pending_chunk is not None:
            iter_train_sec = time.time() - iter_train_start
            total_overlap_interleaved_sec += iter_train_sec
            gen_wait_start = time.time()
            is_ready = chunk_is_ready(pending_chunk)
            logger.info(f"[Epoch {epoch+1}] Step 3.2: finalizing chunk {pending_chunk.chunk_idx}, "
                        f"ready={is_ready}, dispatched {time.time() - pending_chunk.dispatch_time:.1f}s ago")
            # dict: accumulated rollout stats for this chunk
            cs = finalize_chunk(chunk=pending_chunk,
                                replay_buffer=replay_buffer,
                                logger=logger,
                                rollout_timeout=rollout_timeout)
            logger.info(f"[Epoch {epoch+1}] Step 3.2: finalize_chunk returned after "
                        f"{time.time() - gen_wait_start:.1f}s")

            total_overlap_gen_wait_sec += time.time() - gen_wait_start
            chunk_stats_list.append(cs)

            logger.info(f"[Epoch {epoch+1}] Chunk {pending_chunk.chunk_idx + 1}/{total_chunks} finalized, "
                        f"replay buffer: {len(replay_buffer)} samples")

            # Rebuild training shards when new data arrives (force=True at loop boundary).
            # (shard_refs, buf_len, rebuild_count, batches) or None if skipped
            result = try_rebuild_shards(replay_buffer=replay_buffer,
                                        train_batch_size=train_batch_size,
                                        num_engines=num_engines,
                                        seed=seed,
                                        epoch=epoch * total_chunks,
                                        shard_buffer_size=shard_buffer_size,
                                        shard_rebuild_count=shard_rebuild_count,
                                        force=True)
            if result:
                shard_refs, shard_buffer_size, shard_rebuild_count, train_batches_padded = result
                samples_per_engine = len(replay_buffer) // num_engines
                micro_per_engine   = len(train_batches_padded) // num_engines

                logger.info(f"[Epoch {epoch+1}] Training shards {'rebuilt' if shard_rebuild_count > 1 else 'prepared'}: "
                            f"{len(replay_buffer)} replay samples / {num_engines} engines "
                            f"= {samples_per_engine} samples/engine / bs={train_batch_size} "
                            f"= {micro_per_engine} micro-batches/engine, "
                            f"{steps_per_epoch} pass(es)")
        else:
            # pending_chunk was already finalized mid-training and no more chunks remain
            iter_train_sec = time.time() - iter_train_start
            total_overlap_interleaved_sec += iter_train_sec

        # 3.4 NCCL sync when ESS/fixed_sync triggers. Bump policy_version first
        # so the sync broadcasts the updated weights. Without this, policy_version
        # equals rollout_policy_version within an epoch and the sync would be skipped,
        # causing repeated ESS breaks with stale rollout weights across epochs.
        if ess_break and train_step_count > 0 and policy_version == rollout_policy_version:
            policy_version       += 1
            version_bumped_early = True

        if ess_break and not sync_triggered_this_epoch and not is_last_epoch and policy_version != rollout_policy_version:
            try:
                logger.info(f"[Epoch {epoch+1}] NCCL sync at chunk boundary "
                            f"(v{rollout_policy_version} -> v{policy_version})")
                # (bool, finalize futures which is list[ObjectRef])
                _, finalize_refs = sync_weights_nccl(training_engines=training_engines,
                                                     rollout_engines=rollout_engines,
                                                     version=policy_version,
                                                     logger=logger,
                                                     sync_timeout=sync_timeout)
                # Wait for finalize here since ess_break stops generation —
                # engines must finish loading before the next epoch dispatches.
                ray_get_with_timeout(refs=finalize_refs, timeout=sync_timeout,
                                     description=f"finalize weight sync v{policy_version}", logger=logger)
                rollout_policy_version = policy_version
                sync_triggered_this_epoch = True

            except Exception as e:
                # The fallback chain only kicks in at the end-of-epoch boundary.
                logger.warning(f"[Epoch {epoch+1}] NCCL sync at chunk boundary failed: {e}")

        # 3.5 Dispatch next chunk or stop.
        if ess_break:
            # Training ended early due to ESS or fixed interval. Stop generating
            # as continuing would fill the buffer with data that won't be trained on
            # this epoch and would be evicted or reset in post-training cleanup.
            pending_chunk = None

        elif lookahead is not None:
            # Drain mode: lookahead was pre-dispatched in step 3.1b.
            # It's already generating while we finalized + rebuilt above.
            pending_chunk = lookahead

        elif not training_done:
            # Training mode: dispatch now. Training will overlap with this
            # chunk's generation in the next loop iteration.
            # ChunkFuture or None if dataloader exhausted
            pending_chunk = dispatch_one_chunk(dataloader_iter=dataloader_iter,
                                               rollout_engines=rollout_engines,
                                               epoch=epoch,
                                               policy_version=rollout_policy_version,
                                               chunk_size=chunk_size,
                                               chunk_idx=chunk_idx,
                                               logger=logger)
            if pending_chunk is not None:
                chunk_idx += 1

        else:
            # Drain mode but dataloader exhausted: dispatch in step 3.1b
            # already returned None, so no more chunks to generate.
            pending_chunk = None

    # 4. Bulk training: finish remaining steps after all chunks are finalized.
    # This runs when generation completes faster than training (e.g., small
    # prompts, few chunks). The interleaved loop above consumed as many steps
    # as it could; this block handles the remainder up to steps_per_epoch.
    if not training_done and len(replay_buffer) >= train_batch_size:
        # (shard_refs, buf_len, rebuild_count, batches) or None if skipped
        result = try_rebuild_shards(replay_buffer, train_batch_size, num_engines, seed,
                                    epoch * total_chunks, shard_buffer_size, shard_rebuild_count,
                                    force=(shard_refs is None))
        if result:
            shard_refs, shard_buffer_size, shard_rebuild_count, train_batches_padded = result
            samples_per_engine = len(replay_buffer) // num_engines
            micro_per_engine   = len(train_batches_padded) // num_engines
            logger.info(f"[Epoch {epoch+1}] Bulk training shards {'rebuilt' if shard_rebuild_count > 1 else 'prepared'}: "
                        f"{len(replay_buffer)} replay samples / {num_engines} engines "
                        f"= {samples_per_engine} samples/engine / bs={train_batch_size} "
                        f"= {micro_per_engine} micro-batches/engine, "
                        f"{steps_per_epoch} pass(es)")

        # run training steps
        while train_step_count < steps_per_epoch:
            # dict[str, float]: averaged metrics across engines
            train_metrics = run_training_step(training_engines, shard_refs,
                                               logger=logger,
                                               train_step_timeout=train_step_timeout)
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

            # ESS check in bulk mode, as all chunks finalized, engines idle
            # (bool, float): ess_value can be None for non-P3O algorithms
            should_break, ess = check_ess_sync(train_metrics, train_step_count,
                                                ess_sync_threshold, fixed_sync_interval,
                                                sync_triggered_this_epoch)
            if should_break:
                # Bump version before sync so broadcast uses updated version
                if policy_version == rollout_policy_version:
                    policy_version += 1
                    version_bumped_early = True

                try:
                    # (bool, list[ObjectRef])
                    _, finalize_refs = sync_weights_nccl(training_engines=training_engines,
                                                         rollout_engines=rollout_engines,
                                                         version=policy_version,
                                                         logger=logger,
                                                         sync_timeout=sync_timeout)

                    ray_get_with_timeout(refs=finalize_refs,
                                         timeout=sync_timeout,
                                         description=f"finalize weight sync v{policy_version}",
                                         logger=logger)

                    rollout_policy_version = policy_version
                    sync_triggered_this_epoch = True

                except Exception as e:
                    logger.warning(f"[Epoch {epoch+1}] Inline NCCL sync failed: {e}")

            if tracker and ess is not None:
                tracker.log_metrics({"nccl/ess_factor": ess,
                                     "nccl/sync_triggered": 1 if sync_triggered_this_epoch else 0,
                                    }, step=global_step)

            if should_break:
                break

    # 5. Post-training bookkeeping.
    # Only bump policy_version if at least one training step ran AND the version
    # wasn't already bumped at step 3.4 (ESS-triggered mid-epoch sync).
    if train_step_count > 0 and not version_bumped_early:
        policy_version += 1

    # max_lag >= 1 is enforced by config validation for overlap mode.
    evicted = replay_buffer.evict_stale(policy_version - max_lag)
    if evicted > 0:
        logger.info(f"[Epoch {epoch+1}] Post-training eviction: {evicted} stale samples removed, "
                    f"{len(replay_buffer)} retained for next epoch")

    # 6. Aggregate rollout stats
    generation_time = time.time() - generation_start_time
    if chunk_stats_list:
        rollout_metrics = aggregate_chunk_stats(chunk_stats_list=chunk_stats_list,
                                                generation_time=generation_time,
                                                wall_time=time.time() - epoch_start_time)
    else:
        logger.warning(f"[Epoch {epoch+1}] No chunks were finalized — dataloader may be empty")
        rollout_metrics = rollout_stats.summarize(rollout_stats.new_accumulator(), rollout_time=generation_time)
        rollout_metrics["rollout_time_with_overlap"] = time.time() - epoch_start_time

    # 7. Overlap efficiency metrics
    interleaved_total = total_overlap_interleaved_sec + total_overlap_gen_wait_sec
    overlap_ratio = total_overlap_interleaved_sec / interleaved_total if interleaved_total > 0 else 0.0

    return {'rollout_metrics': rollout_metrics,
            'epoch_metrics': epoch_metrics,
            'global_step': global_step,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'train_step_count': train_step_count,
            'train_time': time.time() - train_start_time,
            'sync_performed': sync_triggered_this_epoch,
            'data_policy_version': data_policy_version,
            'overlap_interleaved_sec': total_overlap_interleaved_sec,
            'overlap_gen_wait_sec': total_overlap_gen_wait_sec,
            'overlap_ratio': overlap_ratio}

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./config/rl_args.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging level")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a ckp to resume training. It must contain a CHECKPOINT_COMPLETE marker.")
    args = parser.parse_args()

    ########
    # 1. Miscellaneous setups
    ########
    # remember that main_rl.py is an orchestrator script,
    # not a distributed worker, so rank is always 0 here.
    rank = 0

    # Setup logging
    logger = setup_logging(rank=rank, log_level=args.log_level)
    logger.info(f"Starting RL training...")

    config = cfg.load_and_verify(method="rl",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 rank=rank)
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
        reward_fnc = getattr(reward_module, "compute_score")
        logger.info(f"Using reward function: {reward_func_name}")

    else:
        raise ValueError("Reward function not specified")

    rollout_engines = create_rollout_engines(params=config,
                                             reward_fnc=reward_fnc,
                                             eos_id=tokenizer.eos_token_id)
    num_rollout_engines = len(rollout_engines)
    logger.info(f"Created {num_rollout_engines} rollout engines with TP={config.rollout.tensor_parallel_size}")

    ########
    # 6. Weight sync method (NCCL group initialized after resume in section 9b)
    ########
    weight_sync_method = config.run.weight_sync_method
    nccl_port = None
    nccl_sync_backend = None

    if weight_sync_method == "nccl":
        nccl_port = config.run.nccl_sync_port if config.run.nccl_sync_port else config.run.ray_master_port + 100
        nccl_sync_backend = config.run.nccl_sync_backend

    else:
        logger.info(f"Weight sync method is {weight_sync_method}")

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
    overlap_enabled = config.overlap.enabled
    overlap_max_lag = config.overlap.max_lag
    ess_sync_threshold = config.overlap.ess_sync_threshold
    fixed_sync_interval = config.overlap.fixed_sync_interval
    chunk_size = config.overlap.chunk_size

    ########
    # 9. Resume from checkpoint if requested and clean up incomplete checkpoint directories
    ########
    start_epoch = 0
    global_step = 0
    policy_version = 0
    rollout_policy_version = 0

    if args.resume_from:
        start_epoch, policy_version, global_step = load_checkpoint_for_resume(resume_path=args.resume_from,
                                                                              training_engines=training_engines,
                                                                              rollout_engines=rollout_engines,
                                                                              weight_sync_method=weight_sync_method,
                                                                              logger=logger,
                                                                              sync_timeout=sync_timeout,
                                                                              save_timeout=save_timeout,
                                                                              sync_fn=sync_weights_direct,
                                                                              refresh_fn=refresh_rollout_engine)
        rollout_policy_version = policy_version
        logger.info(f"Resuming from epoch {start_epoch+1}, policy_version={policy_version}, global_step={global_step}")

    ########
    # 9b. Initialize NCCL weight sync group (after resume, so engines are fresh)
    ########
    if weight_sync_method == "nccl":
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

    logger.info(f"Weight sync method: {weight_sync_method}")
    if overlap_enabled:
        logger.info(f"Overlap mode: max_lag={overlap_max_lag}, chunk_size={chunk_size}, "
                    f"ess_sync_threshold={ess_sync_threshold}, fixed_sync_interval={fixed_sync_interval}")

    logger.info(f"checkpoint_save_interval: {checkpoint_save_interval}")
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from} (epoch {start_epoch+1}/{number_of_epochs})")

    logger.info("=" * 50)

    ########
    # 11. Training and rollout loop
    ########
    entire_training_start_time = time.time()
    pre_dispatched_chunk = None  # early dispatch from previous epoch's checkpoint save

    for epoch in range(start_epoch, number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == number_of_epochs - 1)

        # Run epoch
        if overlap_enabled:
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
                                       chunk_size=chunk_size,
                                       max_lag=overlap_max_lag,
                                       ess_sync_threshold=ess_sync_threshold,
                                       fixed_sync_interval=fixed_sync_interval,
                                       is_last_epoch=is_last_epoch,
                                       rollout_timeout=rollout_timeout,
                                       train_step_timeout=train_step_timeout,
                                       sync_timeout=sync_timeout,
                                       tracker=tracker,
                                       logger=logger,
                                       pre_dispatched_chunk=pre_dispatched_chunk)
            pre_dispatched_chunk = None  # consumed

        else:
            result = run_epoch_sync(epoch=epoch,
                                    training_engines=training_engines,
                                    rollout_engines=rollout_engines,
                                    rollout_dataloader=rollout_dataloader,
                                    replay_buffer=replay_buffer,
                                    policy_version=policy_version,
                                    rollout_policy_version=rollout_policy_version,
                                    global_step=global_step,
                                    n_samples=config.rollout.n_samples,
                                    train_batch_size=config.train.train_batch_size_per_gpu,
                                    steps_per_epoch=steps_per_epoch,
                                    seed=config.run.seed,
                                    rollout_timeout=rollout_timeout,
                                    train_step_timeout=train_step_timeout,
                                    tracker=tracker,
                                    logger=logger)

        # Unpack result
        global_step            = result['global_step']
        policy_version         = result['policy_version']
        rollout_metrics        = result['rollout_metrics']
        rollout_policy_version = result['rollout_policy_version']

        # Log rollout metrics
        time_str = f"time={rollout_metrics['rollout_time']:.2f}s"
        if 'rollout_time_with_overlap' in rollout_metrics:
            time_str += f" (wall_time={rollout_metrics['rollout_time_with_overlap']:.2f}s)"

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

        if tracker:
            rollout_log = {"rollout/" + k: v for k, v in rollout_metrics.items()}
            if overlap_enabled:
                rollout_log["rollout/policy_lag"] = policy_version - result.get('data_policy_version', policy_version)
            tracker.log_metrics(rollout_log, step=global_step)

        # Log training summary
        epoch_avg = {k: np.mean(v) for k, v in result['epoch_metrics'].items()}

        train_stats = ray.get(training_engines[0].get_training_stats.remote())
        current_lr = train_stats.get('lr', 0.0)
        gpu_mem_gb = train_stats.get('gpu_peak_mem_gb', 0.0)

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
        if overlap_enabled:
            o_interleaved = result['overlap_interleaved_sec']
            o_wait = result['overlap_gen_wait_sec']
            o_ratio = result['overlap_ratio']
            logger.info(f"[Epoch {epoch+1}] Overlap: interleaved={o_interleaved:.2f}s, "
                        f"gen_wait={o_wait:.2f}s, ratio={o_ratio:.2%}")
            if tracker:
                tracker.log_metrics({"overlap/interleaved_sec": o_interleaved,
                                     "overlap/gen_wait_sec": o_wait,
                                     "overlap/ratio": o_ratio,
                                    }, step=global_step)

        # End-of-epoch weight sync
        sync_success = result['sync_performed']
        nccl_finalize_refs = None
        pre_dispatched_chunk = None  # may be set by NCCL async path below

        if not sync_success and not is_last_epoch:
            if weight_sync_method == "nccl":
                # NCCL sync if lag exceeds max_lag, with direct fallback
                lag = policy_version - rollout_policy_version
                if lag >= overlap_max_lag:
                    logger.info(f"[Epoch {epoch+1}] End-of-epoch NCCL sync "
                                f"(v{rollout_policy_version} -> v{policy_version})...")
                    try:
                        # Concurrent generation during an NCCL sync deadlocks the GPUs:
                        # training engines wait inside the broadcast while rollout engines generate for minutes.
                        # Thus, we execute NCCL synchronously across all engines before dispatching.
                        _, nccl_finalize_refs = sync_weights_nccl(training_engines=training_engines,
                                                                  rollout_engines=rollout_engines,
                                                                  version=policy_version,
                                                                  logger=logger,
                                                                  sync_timeout=sync_timeout)

                        ray_get_with_timeout(refs=nccl_finalize_refs,
                                             timeout=sync_timeout,
                                             description=f"finalize weight sync v{policy_version}",
                                             logger=logger)
                        # already waited, prevent double-wait at line below
                        nccl_finalize_refs = None

                        rollout_policy_version = policy_version
                        sync_success = True

                        # Dispatch NEXT epoch chunk with *updated* weights now that sync is clean.
                        if overlap_enabled and not is_last_epoch and overlap_max_lag > 0:
                            rollout_dataloader.batch_sampler.set_epoch(epoch + 1)
                            early_iter = iter(rollout_dataloader)
                            pre_dispatched_chunk = dispatch_one_chunk(dataloader_iter=early_iter,
                                                                      rollout_engines=rollout_engines,
                                                                      epoch=epoch + 1,
                                                                      policy_version=rollout_policy_version,
                                                                      chunk_size=chunk_size,
                                                                      chunk_idx=0,
                                                                      logger=logger)
                            if pre_dispatched_chunk is not None:
                                logger.info(f"[Epoch {epoch+1}] Early dispatch: next epoch's first chunk started")

                    except Exception as e:
                        nccl_finalize_refs = None
                        logger.warning(f"[Epoch {epoch+1}] NCCL sync failed: {e}, falling back to direct")
                        try:
                            sync_success = sync_weights_direct(training_engines=training_engines,
                                                               rollout_engines=rollout_engines,
                                                               version=policy_version,
                                                               logger=logger,
                                                               sync_timeout=sync_timeout)
                            if sync_success:
                                rollout_policy_version = policy_version

                        except Exception as e2:
                            logger.error(f"[Epoch {epoch+1}] Direct fallback also failed: {e2}")
                else:
                    logger.info(f"[Epoch {epoch+1}] Skipping weight sync (lag={lag}, max_lag={overlap_max_lag})")

            elif weight_sync_method == "direct":
                logger.info(f"[Epoch {epoch+1}] Syncing weights directly to rollout engines "
                            f"(v{rollout_policy_version} -> v{policy_version})...")

                try:
                    sync_success = sync_weights_direct(training_engines=training_engines,
                                                       rollout_engines=rollout_engines,
                                                       version=policy_version,
                                                       logger=logger,
                                                       sync_timeout=sync_timeout)
                except Exception as e:
                    logger.warning(f"[Epoch {epoch+1}] Direct sync raised {e}, falling back to disk")
                    sync_success = False

                if sync_success:
                    rollout_policy_version = policy_version
                    logger.info(f"[Epoch {epoch+1}] Direct sync successful")

        # Wait for NCCL finalize if it was dispatched non-blocking above.
        if sync_success and weight_sync_method == "nccl" and nccl_finalize_refs is not None:
            ray_get_with_timeout(refs=nccl_finalize_refs,
                                 timeout=sync_timeout,
                                 description=f"finalize weight sync v{policy_version}",
                                 logger=logger)
            nccl_finalize_refs = None

        # Early dispatch for non-NCCL sync methods (NCCL path dispatches above, concurrently with sync).
        if pre_dispatched_chunk is None and overlap_enabled and not is_last_epoch and sync_success:
            rollout_dataloader.batch_sampler.set_epoch(epoch + 1)
            early_iter = iter(rollout_dataloader)
            pre_dispatched_chunk = dispatch_one_chunk(dataloader_iter=early_iter,
                                                      rollout_engines=rollout_engines,
                                                      epoch=epoch + 1,
                                                      policy_version=rollout_policy_version,
                                                      chunk_size=chunk_size,
                                                      chunk_idx=0,
                                                      logger=logger)
            if pre_dispatched_chunk is not None:
                logger.info(f"[Epoch {epoch+1}] Early dispatch: next epoch's first chunk generating during checkpoint save")

        # Save checkpoint
        should_save_disk = (checkpoint_save_interval > 0 and
                           ((epoch + 1) % checkpoint_save_interval == 0 or is_last_epoch))

        # save to disk when:
        # 1. using disk-based sync (always need disk save for rollout refresh).
        # 2. all sync methods failed (need disk as last resort for rollout refresh).
        # 3. periodic/final checkpoint save.
        no_sync_succeeded = not result['sync_performed'] and not sync_success
        need_disk_for_rollout = (weight_sync_method == "disk" and not is_last_epoch) or \
                                (no_sync_succeeded and weight_sync_method in ("nccl", "direct", "disk") and not is_last_epoch)

        if need_disk_for_rollout or should_save_disk or is_last_epoch:
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

        # Disk-based rollout refresh
        if need_disk_for_rollout and not is_last_epoch:
            logger.info(f"[Epoch {epoch+1}] Refreshing rollout engines with new policy (version {policy_version})...")
            refresh_rollout_engine(rollout_engines=rollout_engines,
                                   updated_policy_path=model_path,
                                   version=policy_version,
                                   logger=logger,
                                   sync_timeout=sync_timeout)
            rollout_policy_version = policy_version
            logger.info(f"[Epoch {epoch+1}] Rollout engines refreshed")

            # Refresh destroys EngineCore workers, invalidating NCCL groups.
            # Re-initialize so subsequent NCCL syncs don't hang.
            if weight_sync_method == "nccl":
                logger.info(f"[Epoch {epoch+1}] Re-initializing NCCL weight sync group after engine refresh...")
                reinit_nccl_weight_sync_group(training_engines=training_engines,
                                              rollout_engines=rollout_engines,
                                              master_addr=master_addr,
                                              nccl_port=nccl_port,
                                              tp_size=int(config.rollout.tensor_parallel_size),
                                              logger=logger,
                                              init_timeout=config.run.init_timeout,
                                              backend=nccl_sync_backend)

        # NCCL sync metrics
        if tracker and overlap_enabled:
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

    # Tear down NCCL weight sync groups if they were initialized
    if weight_sync_method == "nccl":
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