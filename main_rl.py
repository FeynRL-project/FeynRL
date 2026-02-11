import os
import random
import atexit
import numpy as np
import argparse
import importlib
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import ray
import time
import mlflow

# imports local methods, classes, etc.
import configs.load as cfg # all config arguments
from data_feeds.prompts import PromptsFeed # our custom pytorch dataset
from data_feeds.mixed_sampler import create_prompt_dataset_and_sampler
from misc.utils import safe_string_to_torch_dtype, get_experiment_dir_name, load_algorithm, ray_get_with_timeout
from rollouts.vllm_engine import VLLMRolloutEngine
from rollouts.replay_buffer import ReplayBuffer
from misc.logging import setup_logging, setup_tracker

Algorithm_Registry = {
    # supported algorithms
    'sgrpo': ('algs.SGRPO.sgrpo', 'SGRPO'),
    'cispo': ('algs.CISPO.cispo', 'CISPO'),
    'ppo':   ('algs.PPO.ppo', 'PPO'),
}

def set_random_seeds(seed):
    '''
        Set random seeds to make runs more reproducible (still not guaranteed). With distributed training,
        floating-point math and non-deterministic ops (e.g., torch.Tensor.index_add_) can still cause differences,
        seeding just reduces the variance a bit.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_ray(ray_address):
    '''
       Initialize the Ray cluster and retrieve the driver node's IP address.
    '''
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)

    else:
        ray.init(ignore_reinit_error=True)

    try:
        # The IP is used as master_addr for deepspeed/pytorch distributed rendezvous.
        # rank 0 listens on this address and all other ranks connect to it to form the process group.
        master_addr = ray.util.get_node_ip_address()
    except Exception:
        # Fallback to localhost if we cannot get the IP (e.g. single node without network)
        print("Warning: Could not get master address, using localhost. This is fine for single-node but will fail for multi-node.")
        master_addr = "127.0.0.1"

    return master_addr

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

               # training related arguments
               'kl_coeff':params.train.kl_coeff,
               'clip_low':params.train.clip_low,
               'clip_high':params.train.clip_high,
               'entropy_coeff':params.train.entropy_coeff,
               'micro_batch_size_per_gpu':params.train.train_batch_size_per_gpu,
               'update_after_full_replay':params.train.update_after_full_replay,

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
    for rank in range(world_size):
        # Since NCCL identifies gpus by their actual PCIe/NVLink topology,
        # not LOCAL_RANK, we keep LOCAL_RANK as 0 for all actors.
        ray_vars = {"MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": "0",
                    "PYTHONPATH": os.getcwd(), # Ensure current directory is in path for all workers
                    }

        # NCCL env vars for multi-node clusters
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

              # reward related arguments
              "reward_func":reward_fnc,
              "reward_broadcast":params.reward.broadcast,
              "eps_reward_norm":params.reward.eps_reward_norm,
            }

    # if model doesn't fit in one gpu, tp can be > 1
    num_engines = max(1, rollout_gpus // tp)
    engines = []
    for i in range(num_engines):
        kwargs['engine_id'] = i
        engines.append(VLLMRolloutEngine.options(num_gpus=tp,
                                                 runtime_env={"env_vars": {"PYTHONPATH": os.getcwd()}}
                                                ).remote(**kwargs))

    return engines

def load_tokenizer(model_name, trust_remote_code=False, rank=0):
    '''
       Load tokenizer from huggingface.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    if tokenizer.pad_token_id is None:
        print("Warning: Pad token is not present, using eos token as pad token")
        if getattr(tokenizer, 'eos_token', None) is not None:
            # prefer explicit token if available
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else:
            # fallback to eos token id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

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
    # MixedDatasetSampler is a batch sampler (yields batches of indices)
    # pin_memory=False: the collate_fn returns plain Python lists/dicts (no tensors to pin)
    dataloader = DataLoader(dataset=dataset,
                            batch_sampler=sampler,
                            num_workers=params.data.num_workers,
                            pin_memory=False,
                            collate_fn=collate_fn,
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
    total_samples_generated = 0
    total_reward_sum = 0.0
    total_response_len = 0
    total_tokens = 0
    total_prompts = 0
    total_pass_at_k = 0.0
    total_prompt_mean_rewards = []

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
        rollout_merged = []
        for rl in rollout_lists:
            rollout_merged.extend(rl)
            
            # group samples by prompt
            prompts_seen = set()
            prompt_rewards_acc = []
            
            for sample in rl:
                total_samples_generated += 1
                total_reward_sum += sample['pred_rewards'].sum().item()
                total_response_len += sample['response_len']
                total_tokens += len(sample['prompt_ids']) + len(sample['response_ids'])

                prompt_id = tuple(sample['prompt_ids'])  # unique key per prompt
                if prompt_id not in prompts_seen:
                    prompts_seen.add(prompt_id)
                    
                    # pass@k
                    if "pass_at_k" in sample:
                        total_prompts += 1
                        total_pass_at_k += sample["pass_at_k"]

                    # mean reward per prompt (group_mean_reward was added in generate)
                    if "group_mean_reward" in sample:
                        prompt_rewards_acc.append(sample["group_mean_reward"])
            
            total_prompt_mean_rewards.extend(prompt_rewards_acc)

        # 5. now add them to replay buffer
        replay_buffer.add_batch_seqs(rollout_merged)

    rollout_time = time.time() - rollout_start_time
    avg_reward = total_reward_sum / max(1, total_samples_generated)
    avg_response_len = total_response_len / max(1, total_samples_generated)
    avg_pass_at_k = total_pass_at_k / max(1, total_prompts)

    avg_reward_per_prompt = float(
        sum(total_prompt_mean_rewards) / max(1, len(total_prompt_mean_rewards))
    )

    if len(replay_buffer) <= 1:
        raise ValueError("Replay buffer is empty")

    rollout_time = time.time() - rollout_start_time
    if total_samples_generated == 0:
        logger.warning("No samples generated during rollout phase!")
        avg_reward = 0.0
        avg_response_len = 0.0

    else:
        avg_reward = total_reward_sum / total_samples_generated
        avg_response_len = total_response_len / total_samples_generated

    tps = total_tokens / max(1e-6, rollout_time)

    return {"total_samples_generated": total_samples_generated,
            "avg_reward": avg_reward,
            "avg_reward_per_prompt": avg_reward_per_prompt,
            "total_reward": total_reward_sum,
            "avg_response_len": avg_response_len,
            "rollout_time": rollout_time,
            "tokens_per_sec": tps,
            "avg_pass_at_k": avg_pass_at_k}

def collect_rollouts_async(dataloader, rollout_engines, epoch, policy_version):
    '''
        Non-blocking rollout collection: iterates the dataloader and schedules
        all rollout generation, then returns control immediately without waiting for results.
    '''
    num_rollout_engines = len(rollout_engines)
    all_futures = []

    for rollout_batch in dataloader:
        rollout_shards = shard_batch_for_engines(rollout_batch, num_rollout_engines)
        if not rollout_shards:
            continue

        batch_futures = []
        for i, shard in enumerate(rollout_shards):
            batch_futures.append(rollout_engines[i].generate.remote(prompts=shard,
                                                                    current_iter=epoch,
                                                                    policy_version=policy_version))
        all_futures.append(batch_futures)

    return all_futures

def finalize_rollouts(all_futures, replay_buffer, logger, rollout_timeout, start_time=None):
    '''
        Blocking call to gather all previously scheduled rollouts from
        collect_rollouts_async and add them to the replay buffer.
    '''
    # wall_start_time: when the async rollout was first scheduled (includes training overlap time)
    # finalize_start_time: when we actually started waiting for results (generation-only time)
    wall_start_time = start_time if start_time is not None else time.time()
    finalize_start_time = time.time()

    total_samples_generated = 0
    total_reward_sum = 0.0
    total_response_len = 0
    total_tokens = 0

    for batch_futures in all_futures:
        rollout_lists = ray_get_with_timeout(refs=batch_futures,
                                             timeout=rollout_timeout,
                                             description="finalize prefetched rollouts",
                                             logger=logger)

        rollout_merged = []
        for rl in rollout_lists:
            rollout_merged.extend(rl)
            for sample in rl:
                total_samples_generated += 1
                total_reward_sum += sample['pred_rewards'].sum().item()
                total_response_len += sample['response_len']
                total_tokens += len(sample['prompt_ids']) + len(sample['response_ids'])

        replay_buffer.add_batch_seqs(rollout_merged)

    if len(replay_buffer) == 0:
        raise ValueError("Replay buffer is empty after finalize_rollouts")

    # wall_time: total time since scheduling (includes overlapped training time)
    # finalize_time: time spent in this function waiting/processing (generation-only perspective)
    wall_time = time.time() - wall_start_time
    finalize_time = time.time() - finalize_start_time

    if total_samples_generated == 0:
        avg_reward = 0.0
        avg_response_len = 0.0
    else:
        avg_reward = total_reward_sum / total_samples_generated
        avg_response_len = total_response_len / total_samples_generated

    tps = total_tokens / max(1e-6, wall_time)

    return {"total_samples_generated": total_samples_generated,
            "avg_reward": avg_reward,
            "total_reward": total_reward_sum,
            "avg_response_len": avg_response_len,
            "rollout_time": wall_time,
            "finalize_wait_time": finalize_time,
            "tokens_per_sec": tps}

def prepare_training_batches(replay_buffer, batch_size: int, num_engines: int) -> list:
    '''
        Create and pad training batches for distributed training.
    '''
    # Create dataloader from replay buffer and convert
    # to list as ray needs serializable data
    train_batches = list(DataLoader(dataset=replay_buffer,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    pin_memory=False,
                                    collate_fn=replay_buffer.collate_fn,
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
    for eid in range(num_engines):
        # engine 0 gets [0, 2, 4, ...], engine 1 gets [1, 3, 5, ...]
        shard = batches[eid::num_engines]
        assert len(shard) > 0, f"Engine {eid} has empty shard. This will cause DeepSpeed hang"
        shard_refs.append(ray.put(shard))

    return shard_refs

def run_training_step(engines, shard_refs, logger, train_step_timeout):
    '''
       Execute one training step across all engines.
       shard_refs: list of Ray ObjectRefs (one per engine), created by shard_and_put().
       Ray auto-resolves ObjectRefs passed to .remote(), so the engine receives the actual data.
    '''
    futures = []
    for eid, engine in enumerate(engines):
        futures.append(engine.train_step.remote(engine_id=eid, micro_batches=shard_refs[eid]))

    # Gather training metrics from all engines
    metrics_list = ray_get_with_timeout(refs=futures,
                                        timeout=train_step_timeout,
                                        description="training step",
                                        logger=logger)

    # Dynamically aggregate all metric keys across engines.
    # This works for both sgrpo (policy-only) and ppo (policy + value metrics).
    # metrics_list: clipfrac, approx_kl, loss_ent, loss_pi, loss_total, kl_ref
    # if value network, add: v_loss
    # loss_total includes: loss_pi + ent_coef * loss_ent

    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    return {k: np.mean([m.get(k, 0.0) for m in metrics_list]) for k in all_keys}

def save_checkpoint(epoch, version, tokenizer, training_engines, checkpoint_dir, experiment_id, rank, logger, save_timeout):
    '''
       Save model checkpoint (must run on all ranks for ZeRO-3)
    '''
    tag = f"iter{epoch+1:06d}_v{version:06d}"
    model_path = get_experiment_dir_name(output_dir=checkpoint_dir, tag=tag, experiment_id=experiment_id)
    logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")

    # save tokenizer so it's ready when vllm loads the model
    if rank == 0:
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)

    # save must run on *all ranks* for zero-3 correctness.
    save_futures = []
    for engine in training_engines:
        save_futures.append(engine.save_checkpoint.remote(output_dir=model_path, tag=tag))

    # Wait for all saves to complete
    ray_get_with_timeout(refs=save_futures,
                         timeout=save_timeout,
                         description=f"checkpoint save (epoch {epoch+1})",
                         logger=logger)

    logger.info(f"[Epoch {epoch+1}] Checkpoint saved: {model_path}")
    return model_path

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

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/rl_args.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
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
                                 rank=rank,
                                 )
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
        raise ValueError(
            f"Need {needed_gpus} GPUs (training={training_gpus} + rollout={rollout_gpus}) "
            f"but Ray cluster only has {int(cluster_gpus)} GPUs"
        )

    ########
    # 3. initialize training engine
    ########
    logger.info(f"Loading training algorithm: {config.train.alg_name}")
    alg_class = load_algorithm(config.train.alg_name, Algorithm_Registry)

    training_engine = create_training_engines(params=config,
                                              alg=alg_class,
                                              world_size=training_gpus,
                                              master_addr=master_addr,
                                              master_port=config.run.ray_master_port)

    assert len(training_engine) == training_gpus, "Number of training engines does not match number of training gpus"
    logger.info(f"Created {len(training_engine)} training engine runners")

    # Synchronization barrier to prevent deepspeed rendezvous hang
    # wait for all training actors to finish initialization before proceeding
    logger.info("Waiting for all training engines to initialize...")

    init_timeout = config.run.init_timeout
    ready_checks = [engine.is_ready.remote() for engine in training_engine]
    ready = ray_get_with_timeout(refs=ready_checks,
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
    # 5. initialize inference engine
    ########
    logger.info("Setting up inference/rollout engines...")
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
    # 6. load the rollout dataloader
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
    # 7. Training and evaluation loop
    ########
    global_step = 0
    number_of_epochs  = config.train.total_number_of_epochs
    steps_per_epoch = config.train.train_steps_per_epoch

    # Weight sync settings
    weight_sync_method = config.run.weight_sync_method
    checkpoint_save_interval = config.run.checkpoint_save_interval if config.run.checkpoint_save_interval is not None else 1

    # Timeout settings (seconds) for ray.get() calls
    rollout_timeout = config.run.rollout_timeout
    train_step_timeout = config.run.train_step_timeout
    save_timeout = config.run.save_timeout
    # when overlap is enabled, sync_timeout must be large enough to cover the remaining in-flight rollout generation time PLUS the
    # actual weight transfer time. This is because push_weights_to_rollout dispatches update_weights_direct to rollout actors that may 
    # still be running generate tasks. Ray actors are single-threaded, so the update queues behind generation. 
    # Set sync_timeout >= rollout_timeout + expected weight transfer time to avoid spurious timeouts.
    sync_timeout = config.run.sync_timeout

    # Overlap mode settings
    overlap_enabled = config.run.overlap_enabled
    # Max training steps ahead of rollout policy version. With overlap_max_lag=N, weights sync every N epochs
    # since policy_version increments by 1 each epoch, lag reaches N every N epochs.
    overlap_max_lag = config.run.overlap_max_lag

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    logger.info(f"Weight sync: {weight_sync_method}, Checkpoint save interval: {checkpoint_save_interval}")

    if overlap_enabled:
        logger.info(f"Overlap mode ENABLED: max_lag={overlap_max_lag}")

    logger.info("=" * 50)
    entire_training_start_time = time.time()

    policy_version = 0
    # rollout_policy_version tracks which policy version the rollout engines
    # currently have loaded. When overlap is enabled, we allow the rollout
    # engine to lag behind the training policy by up to overlap_max_lag versions.
    pending_rollout_futures = None
    pending_rollout_buffer = None
    # version used to generate the prefetched data
    prefetch_start_time = None
    prefetch_policy_version = None
    rollout_policy_version = 0

    for epoch in range(number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == number_of_epochs - 1)

        ################
        # 1. Collect rollouts and finalize prefetch (if overlap is enabled)
        ################
        if overlap_enabled and pending_rollout_futures is not None:
            # Finalize prefetched rollouts scheduled in the previous epoch.
            # This blocks until all in-flight generation completes.
            data_policy_version = prefetch_policy_version
            logger.info(f"[Epoch {epoch+1}/{number_of_epochs}] Finalizing prefetched rollouts "
                        f"(generated with policy v{data_policy_version})...")
            rollout_stats = finalize_rollouts(all_futures=pending_rollout_futures,
                                              replay_buffer=pending_rollout_buffer,
                                              logger=logger,
                                              rollout_timeout=rollout_timeout,
                                              start_time=prefetch_start_time)
            replay_buffer = pending_rollout_buffer
            pending_rollout_futures = None
            pending_rollout_buffer = None
            prefetch_start_time = None
            prefetch_policy_version = None

        else:
            # First epoch or overlap disabled, we generate synchronously
            # and it is considered on-policy (lag = 0)
            data_policy_version = policy_version
            logger.info(f"[Epoch {epoch+1}/{number_of_epochs}] Starting rollout generation...")
            rollout_dataloader.batch_sampler.set_epoch(epoch)
            rollout_stats = collect_rollouts(dataloader=rollout_dataloader,
                                             rollout_engines=rollout_engines,
                                             epoch=epoch,
                                             policy_version=policy_version,
                                             replay_buffer=replay_buffer,
                                             n_samples=config.rollout.n_samples,
                                             logger=logger,
                                             rollout_timeout=rollout_timeout)

        finalize_wait = rollout_stats.get('finalize_wait_time', None)
        time_str = f"time={rollout_stats['rollout_time']:.2f}s"
        if finalize_wait is not None:
            time_str += f" (finalize_wait={finalize_wait:.2f}s)"

        logger.info(f"[Epoch {epoch + 1}] Rollout complete: {rollout_stats['total_samples_generated']} samples, "
                    f"avg_reward={rollout_stats['avg_reward']:.4f}, total_reward={rollout_stats['total_reward']:.4f}, "
                    f"avg_response_len={rollout_stats['avg_response_len']:.1f}, "
                    f"{time_str}, tps={rollout_stats['tokens_per_sec']:.2f}")

        # Log rollout metrics immediately so they appear in WandB/MLflow
        # before training starts (avoids visual lag in dashboards)
        if tracker:
            rollout_log = {
                "rollout/avg_reward": rollout_stats['avg_reward'],
                "rollout/total_reward": rollout_stats['total_reward'],
                "rollout/avg_response_len": rollout_stats['avg_response_len'],
                "rollout/total_samples": rollout_stats['total_samples_generated'],
                "rollout/rollout_time_sec": rollout_stats['rollout_time'],
                "rollout/tokens_per_sec": rollout_stats['tokens_per_sec'],
            }
            if 'finalize_wait_time' in rollout_stats:
                rollout_log["rollout/finalize_wait_sec"] = rollout_stats['finalize_wait_time']
            if overlap_enabled:
                rollout_log["rollout/policy_lag"] = policy_version - data_policy_version
            tracker.log_metrics(rollout_log, step=global_step)

        ################
        # 2. Schedule next rollout BEFORE training which would be the maximum overlap
        ################
        # When overlap is enabled, we try to schedule the next epoch's rollout
        # NOW so it runs concurrently with training on different GPUs.
        # The constraint is staleness: the prefetched data will be trained on
        # next epoch when policy_version = current + 1. The lag at that point is: (policy_version + 1) - rollout_policy_version
        # If this lag <= overlap_max_lag, we can schedule now and before training.
        # Otherwise, we must wait for training + weight sync to reduce the lag, and schedule AFTER sync with less overlap.
        prefetch_scheduled = False
        if overlap_enabled and not is_last_epoch:
            future_lag = (policy_version + 1) - rollout_policy_version
            if future_lag <= overlap_max_lag:
                logger.info(f"[Epoch {epoch + 1}] Scheduling async prefetch for epoch {epoch + 2} "
                            f"(rollout policy v{rollout_policy_version}, "
                            f"future lag={future_lag}/{overlap_max_lag})")
                # create a new buffer for pending_rollout_buffer
                pending_rollout_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                                      max_seq_len=config.data.max_seq_len)

                rollout_dataloader.batch_sampler.set_epoch(epoch + 1)
                prefetch_start_time = time.time()
                prefetch_policy_version = rollout_policy_version
                pending_rollout_futures = collect_rollouts_async(dataloader=rollout_dataloader,
                                                                rollout_engines=rollout_engines,
                                                                epoch=epoch + 1,
                                                                policy_version=rollout_policy_version)
                prefetch_scheduled = True

        ################
        # 3. Prepare training batches
        ################
        logger.info(f"[Epoch {epoch+1}] Replay buffer has {len(replay_buffer)} samples")
        train_start_time = time.time()

        # All ranks/gpus should get EQUAL number of batches
        # to prevent deepspeed hang so we pad the batches which are not
        # divisible by num_train_engines
        train_batches_padded = prepare_training_batches(replay_buffer=replay_buffer,
                                                        batch_size=config.train.train_batch_size_per_gpu,
                                                        num_engines=len(training_engine))
        samples_per_engine = len(replay_buffer) // len(training_engine)
        micro_per_engine = len(train_batches_padded) // len(training_engine)
        logger.info(f"[Epoch {epoch+1}] Training: "
                    f"{len(replay_buffer)} replay samples / {len(training_engine)} training engines "
                    f"= {samples_per_engine} samples/engine / bs={config.train.train_batch_size_per_gpu} "
                    f"= {micro_per_engine} micro-batches/engine, "
                    f"{steps_per_epoch} pass(es) over replay buffer")

        # Pre-shard and store in Ray object store once.
        # Avoids re-serializing the same data on every step (batches are
        # reused unchanged across all steps_per_epoch iterations).
        shard_refs = shard_and_put(train_batches_padded, num_engines=len(training_engine))

        ################
        # 4. Training loop
        ################
        # training loop runs concurrently with prefetched rollout if scheduled
        epoch_metrics = {}
        for step in range(steps_per_epoch):
            train_metrics = run_training_step(training_engine, shard_refs,
                                                logger=logger,
                                                train_step_timeout=train_step_timeout)

            # Epoch average of average across all training engines (dynamic keys)
            for k, v in train_metrics.items():
                epoch_metrics.setdefault(k, []).append(v)
            global_step += 1

            # Log progress
            if step % 10 == 0:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                logger.info(f"[Epoch {epoch+1}][Step {step+1}/{steps_per_epoch}] {metric_str}")

            # Log to experiment tracker every step
            if tracker:
                tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()},
                                   step=global_step)

        policy_version += 1
        if config.train.alg_name.lower() in Algorithm_Registry.keys():
            replay_buffer.reset()

        ################
        # 5. Log epoch summary
        ################
        train_time = time.time() - train_start_time
        epoch_time = time.time() - epoch_start_time

        epoch_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}

        logger.info(f"[Epoch {epoch+1}] Training complete: time={train_time:.2f}s, "
                    f"avg_loss={epoch_avg.get('loss_total', 0.0):.4f}, "
                    f"avg_kl_ref={epoch_avg.get('kl_ref', 0.0):.4f}, "
                    f"avg_kl_old={epoch_avg.get('kl_old', 0.0):.6f}")

        # Log epoch metrics to MLflow
        if rank == 0 and mlflow_run:
            mlflow.log_metrics({
                    "epoch/avg_loss": epoch_avg_loss,
                    "epoch/avg_kl_old": epoch_avg_kl_old,
                    "epoch/avg_kl_ref": epoch_avg_kl_ref,
                    "epoch/avg_clipfrac": epoch_avg_clipfrac,
                    "epoch/avg_reward": rollout_stats['avg_reward'],
                    "epoch/avg_reward_per_prompt": rollout_stats['avg_reward_per_prompt'],
                    "epoch/total_reward": rollout_stats['total_reward'],
                    "epoch/avg_response_len": rollout_stats['avg_response_len'],
                    "epoch/total_samples": rollout_stats['total_samples_generated'],
                    "epoch/avg_pass_at_k": rollout_stats["avg_pass_at_k"],
                    "epoch/rollout_time_sec": rollout_stats['rollout_time'],
                    "epoch/tokens_per_sec": rollout_stats['tokens_per_sec'],
                    "epoch/train_time_sec": train_time,
                    }, step=epoch + 1)

        ################
        # 6. Sync weights to rollout engines
        ################
        # When overlap is disabled, we always sync.
        # When overlap is enabled, only sync when the lag reaches overlap_max_lag.
        # This lets us skip syncs and keep rollout engines busy generating.
        # Note: if a prefetch is in-flight on the rollout actors, the sync's
        # update_weights_direct call queues behind the generation as ray actors
        # execute sequentially, so there is no interference.
        sync_attempted = False
        sync_success = False
        lag = policy_version - rollout_policy_version
        need_sync = (not overlap_enabled) or (lag >= overlap_max_lag)

        if weight_sync_method == "direct" and not is_last_epoch and need_sync:
            sync_attempted = True

            if overlap_enabled:
                lag_str=f"lag={lag}"

            else:
                lag_str=""

            logger.info(f"[Epoch {epoch+1}] Syncing weights directly to rollout engines "
                        f"(v{rollout_policy_version} -> v{policy_version}, {lag_str})...")

            try:
                sync_success = sync_weights_direct(training_engines=training_engine,
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

        elif not need_sync and not is_last_epoch:
            logger.info(f"[Epoch {epoch+1}] Skipping weight sync (lag={lag}, max_lag={overlap_max_lag})")

        # Log overlap-specific metrics after sync decision is made
        if tracker and overlap_enabled:
            tracker.log_metrics({
                "overlap/policy_version": policy_version,
                "overlap/rollout_policy_version": rollout_policy_version,
                "overlap/policy_lag": policy_version - rollout_policy_version,
                "overlap/prefetch_before_training": 1 if prefetch_scheduled else 0,
                "overlap/sync_skipped": 0 if need_sync else 1,
            }, step=global_step)

        ################
        # 7. Save checkpoint
        ################
        should_save_disk = (checkpoint_save_interval > 0 and
                           ((epoch + 1) % checkpoint_save_interval == 0 or is_last_epoch))

        # save to disk when:
        # 1. using disk-based sync and sync is needed.
        # 2. direct sync was attempted but failed.
        # 3. periodic/final checkpoint save.
        need_disk_for_rollout = (weight_sync_method == "disk" and need_sync) or (sync_attempted and not sync_success)
        if need_disk_for_rollout or should_save_disk or is_last_epoch:
            model_path = save_checkpoint(epoch=epoch,
                                         version=policy_version,
                                         tokenizer=tokenizer,
                                         training_engines=training_engine,
                                         checkpoint_dir=config.run.checkpoint_dir,
                                         experiment_id=config.run.experiment_id,
                                         rank=rank,
                                         logger=logger,
                                         save_timeout=save_timeout)
            logger.info(f"[Epoch {epoch+1}] Saved disk checkpoint at {model_path}")

        ################
        # 8. Disk-based rollout refresh
        ################
        if need_disk_for_rollout and not is_last_epoch:
            logger.info(f"[Epoch {epoch+1}] Refreshing rollout engines with new policy (version {policy_version})...")
            refresh_rollout_engine(rollout_engines=rollout_engines,
                                   updated_policy_path=model_path,
                                   version=policy_version,
                                   logger=logger,
                                   sync_timeout=sync_timeout)
            rollout_policy_version = policy_version
            logger.info(f"[Epoch {epoch+1}] Rollout engines refreshed")

        ################
        # 9. Schedule prefetch after sync if not scheduled before training
        ################
        # If we couldn't schedule before training as lag would have exceeded max_lag,
        # we schedule now after weights have been synced.
        # This still provides some overlap: the rollout runs concurrently
        # with checkpoint save and the next epoch's finalize wait.
        if overlap_enabled and not is_last_epoch and not prefetch_scheduled:
            logger.info(f"[Epoch {epoch + 1}] Scheduling async prefetch for epoch {epoch + 2} "
                        f"(rollout policy v{rollout_policy_version}, post-sync)")
            pending_rollout_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                                  max_seq_len=config.data.max_seq_len)
            rollout_dataloader.batch_sampler.set_epoch(epoch + 1)
            prefetch_start_time = time.time()
            prefetch_policy_version = rollout_policy_version
            pending_rollout_futures = collect_rollouts_async(dataloader=rollout_dataloader,
                                                            rollout_engines=rollout_engines,
                                                            epoch=epoch + 1,
                                                            policy_version=rollout_policy_version)

        logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
        logger.info("=" * 50)

    # End experiment tracker run
    if tracker:
        tracker.finish()

    logger.info("Training completed successfully!")
    entire_training_time = time.time() - entire_training_start_time
    logger.info(f"Total training time from start to finish: {entire_training_time:.2f}s")
    ray.shutdown()
    logger.info("Done!")