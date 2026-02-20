import os
import random
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
from misc.utils import safe_string_to_torch_dtype, get_experiment_dir_name, load_algorithm
from rollouts.vllm_engine import VLLMRolloutEngine
from rollouts.replay_buffer import ReplayBuffer
from misc.logging import setup_logging, setup_viz

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
       Initialize ray cluster and setup master address.
    '''
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)

    else:
        ray.init(ignore_reinit_error=True)

    try:
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

        # NCCL env vars for multi-node clusters with multiple NICs
        # if params.run.nccl_socket_ifname:
        #     ray_vars["NCCL_SOCKET_IFNAME"] = params.run.nccl_socket_ifname
        # if params.run.nccl_ib_hca:
        #     ray_vars["NCCL_IB_HCA"] = params.run.nccl_ib_hca

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
                     logger):

    '''
        This function is used to run rollout engine and generate rollouts/samples.
    '''
    num_rollout_engines = len(rollout_engines)
    rollout_start_time = time.time()
    total_samples_generated = 0
    total_reward_sum = 0.0
    total_response_len = 0
    total_tokens = 0

    # example: rollout_gpus=2, rollout_batch_size_per_gpu=12, n_samples=3, rollout_samples_per_epoch = 25
    # local_batch_size = num_rollout_engines * rollout_batch_size_per_gpu = 2 * 12 = 24
    # Batches needed = ceil(25 / 24) = 2 batches
    # Total Prompts = 2 * 24 = 48 prompts
    # Total Samples in Buffer = 48 prompts * n_samples (e.g., 3) = 144 samples

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
        rollout_lists = ray.get(rollout_samples)

        # 4. merge rollouts across all engines and collect stats
        rollout_merged = []
        for rl in rollout_lists:
            rollout_merged.extend(rl)
            for sample in rl:
                total_samples_generated += 1
                total_reward_sum += sample['pred_rewards'].sum().item()
                total_response_len += sample['response_len']
                total_tokens += len(sample['prompt_ids']) + len(sample['response_ids'])

        # 5. now add them to replay buffer
        replay_buffer.add_batch_seqs(rollout_merged)

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
            "total_reward": total_reward_sum,
            "avg_response_len": avg_response_len,
            "rollout_time": rollout_time,
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

def run_training_step(engines, shard_refs):
    '''
       Execute one training step across all engines.
       shard_refs: list of Ray ObjectRefs (one per engine), created by shard_and_put().
       Ray auto-resolves ObjectRefs passed to .remote(), so the engine receives the actual data.
    '''
    futures = []
    for eid, engine in enumerate(engines):
        futures.append(engine.train_step.remote(engine_id=eid, micro_batches=shard_refs[eid]))

    # Gather training metrics from all engines
    metrics_list = ray.get(futures)

    # Dynamically aggregate all metric keys across engines.
    # This works for both sgrpo (policy-only) and ppo (policy + value metrics).
    # metrics_list: clipfrac, approx_kl, ent_loss, pi_loss, pi_loss_total, kl_ref
    # if value network, add: v_loss
    # pi_loss_total includes: pi_loss + ent_coef * ent_loss

    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())

    return {k: np.mean([m.get(k, 0.0) for m in metrics_list]) for k in all_keys}

def save_checkpoint(epoch, version, tokenizer, training_engines, checkpoint_dir, experiment_id, rank, logger):
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
    ray.get(save_futures)

    # Flush filesystem buffers to ensure checkpoint is fully written
    if rank == 0:
        os.sync()

    logger.info(f"[Epoch {epoch+1}] Checkpoint saved: {model_path}")
    return model_path

def sync_weights_direct(training_engines, rollout_engines, version, logger):
    '''
        Transfer weights directly from deepspeed training engines to vllm rollout
        engines via ray object store. No disk I/O.
    '''
    state_dict_ref, _ = gather_training_weights(training_engines, logger)
    if state_dict_ref is None:
        return False

    return push_weights_to_rollout(rollout_engines, state_dict_ref, version, logger)

def gather_training_weights(training_engines, logger):
    '''
        Gather state_dict from training engines and store in ray object store.
    '''
    start_time = time.time()
    logger.info(f"[WeightSync] Gathering state_dict from training engines...")

    # All training engines must participate in gather_state_dict(). see common.py.
    gather_futures = [engine.gather_state_dict.remote() for engine in training_engines]
    gather_results = ray.get(gather_futures)

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

def push_weights_to_rollout(rollout_engines, state_dict_ref, version, logger):
    '''
        Push pre-gathered weights to rollout engines. Blocks until all engines updated.
        Returns True if all engines updated successfully.
    '''
    start_time = time.time()
    update_futures = [eng.update_weights_direct.remote(state_dict_ref, version) for eng in rollout_engines]
    results = ray.get(update_futures)

    end_time = time.time() - start_time
    success = all(results)

    if success:
        logger.info(f"[WeightSync] Pushed weights v{version} to rollout engines in {end_time:.2f}s")

    else:
        logger.warning(f"[WeightSync] Some rollout engines failed to update to v{version}")

    return success

def refresh_rollout_engine(rollout_engines, updated_policy_path, version):
    '''
        Refresh rollout engine with the latest policy using disk-based fallback.
    '''
    refresh_futures = []
    for eng in rollout_engines:
        refresh_futures.append(eng.refresh_model.remote(updated_policy_path, version))

    ray.get(refresh_futures)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
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

    mlflow_run = setup_viz(config=config, tracking_uri=config.run.tracking_uri, rank=rank)
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

    ready_checks = [engine.is_ready.remote() for engine in training_engine]
    ready = ray.get(ready_checks)
    logger.info("All training engines ready!")

    ########
    # 5. load tokenizer
    ########
    logger.info(f"Loading tokenizer from {config.model.name}")
    tokenizer = load_tokenizer(model_name=config.model.name,
                               trust_remote_code=config.model.trust_remote_code,
                               rank=rank)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token ID: {tokenizer.pad_token_id}")

    ########
    # 6. initialize inference engine
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
    # 6. Load the rollout dataloader
    ########
    logger.info(f"Loading rollout dataloader from {config.data.train_files_path}")
    rollout_dataloader = create_rollout_dataloader(params=config,
                                                  tokenizer=tokenizer,
                                                  num_rollout_engines=num_rollout_engines,
                                                  samples_per_epoch=config.rollout.rollout_samples_per_epoch)

    logger.info(f"Rollout dataloader with {len(rollout_dataloader)} batches/machine, "
                f"n_samples={config.rollout.n_samples} per prompt")

    # replay buffer size would be rollout_samples_per_epoch * n_samples per prompt
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 )
    logger.info(f"Replay buffer initialized (max_seq_len={config.data.max_seq_len})")
    ########
    # 7. Training and evaluation loop
    ########
    policy_version = 0
    global_step = 0
    number_of_epochs  = config.train.total_number_of_epochs
    steps_per_epoch = config.train.train_steps_per_epoch

    # Weight sync settings
    weight_sync_method = config.run.weight_sync_method or "direct"
    checkpoint_save_interval = config.run.checkpoint_save_interval if config.run.checkpoint_save_interval is not None else 1

    # Overlap mode settings
    overlap_enabled = config.run.overlap_enabled
    # Max training steps ahead of rollout policy version
    overlap_max_lag = config.run.overlap_max_lag
    # Update rollout weights every N training steps
    overlap_weight_update_interval = config.run.overlap_weight_update_interval

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    logger.info(f"Weight sync: {weight_sync_method}, Checkpoint save interval: {checkpoint_save_interval}")

    if overlap_enabled:
        logger.info(f"Overlap mode ENABLED: max_lag={overlap_max_lag}, weight_update_interval={overlap_weight_update_interval}")

    logger.info("=" * 50)
    entire_training_start_time = time.time()
    for epoch in range(number_of_epochs):
        epoch_start_time = time.time()
        logger.info(f"[Epoch {epoch+1}/{number_of_epochs}] Starting rollout generation...")

        # Set epoch on sampler to reshuffle data
        rollout_dataloader.batch_sampler.set_epoch(epoch)

        ################
        # 1. Collect rollouts
        ################
        # data is collected in parallel across all rollout engines and saved to replay buffer
        rollout_stats = collect_rollouts(dataloader=rollout_dataloader,
                                         rollout_engines=rollout_engines,
                                         epoch=epoch,
                                         policy_version=policy_version,
                                         replay_buffer=replay_buffer,
                                         n_samples=config.rollout.n_samples,
                                         logger=logger)

        logger.info(f"[Epoch {epoch+1}] Rollout complete: {rollout_stats['total_samples_generated']} samples, "
                    f"avg_reward={rollout_stats['avg_reward']:.4f}, total_reward={rollout_stats['total_reward']:.4f}, "
                    f"avg_response_len={rollout_stats['avg_response_len']:.1f}, "
                    f"time={rollout_stats['rollout_time']:.2f}s, tps={rollout_stats['tokens_per_sec']:.2f}")

        ################
        # 2. Prepare training batches
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
        # 3. Training loop
        ################
        epoch_metrics = {}
        for step in range(steps_per_epoch):
            train_metrics = run_training_step(training_engine, shard_refs)

            # Epoch average of average across all training engines (dynamic keys)
            for k, v in train_metrics.items():
                epoch_metrics.setdefault(k, []).append(v)
            global_step += 1

            # Log progress
            if step % 10 == 0:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
                logger.info(f"[Epoch {epoch+1}][Step {step+1}/{steps_per_epoch}] {metric_str}")

            # Log to MLflow every step (only rank 0)
            if rank == 0 and mlflow_run:
                mlflow.log_metrics({f"train/{k}": v for k, v in train_metrics.items()},
                                   step=global_step)

        policy_version += 1
        if config.train.alg_name.lower() in Algorithm_Registry.keys():
            replay_buffer.reset()
        ################
        # 4. Log epoch summary
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
            epoch_mlflow = {f"epoch/avg_{k}": v for k, v in epoch_avg.items()}
            epoch_mlflow.update({
                    "epoch/avg_reward": rollout_stats['avg_reward'],
                    "epoch/total_reward": rollout_stats['total_reward'],
                    "epoch/avg_response_len": rollout_stats['avg_response_len'],
                    "epoch/total_samples": rollout_stats['total_samples_generated'],
                    "epoch/rollout_time_sec": rollout_stats['rollout_time'],
                    "epoch/tokens_per_sec": rollout_stats['tokens_per_sec'],
                    "epoch/train_time_sec": train_time,
                    })
            mlflow.log_metrics(epoch_mlflow, step=epoch + 1)

        ################
        # 5. Refresh rollout policy via direct weight sync
        ################
        sync_success = False
        is_last_epoch = (epoch == number_of_epochs - 1)
        if weight_sync_method == "direct" and not is_last_epoch:
            logger.info(f"[Epoch {epoch+1}] Syncing weights directly to rollout engines (version {policy_version})...")
            sync_success = sync_weights_direct(training_engines=training_engine,
                                               rollout_engines=rollout_engines,
                                               version=policy_version,
                                               logger=logger)
            if sync_success:
                logger.info(f"[Epoch {epoch+1}] Direct sync successful")

            else:
                logger.warning(f"[Epoch {epoch+1}] Direct sync failed, falling back to disk-based refresh")
        ################
        # 5. save checkpoint
        ################
        should_save_disk = (checkpoint_save_interval > 0 and
                           ((epoch + 1) % checkpoint_save_interval == 0 or is_last_epoch))

        if weight_sync_method == "disk" or sync_success == False or should_save_disk == True:
            model_path = save_checkpoint(epoch=epoch,
                                         version=policy_version,
                                         tokenizer=tokenizer,
                                         training_engines=training_engine,
                                         checkpoint_dir=config.run.checkpoint_dir,
                                         experiment_id=config.run.experiment_id,
                                         rank=rank,
                                         logger=logger)
            logger.info(f"[Epoch {epoch+1}] Saved disk checkpoint at {model_path}")
        ################
        # 6. Refresh rollout policy via disk-based fallback
        ################
        if sync_success == False and not is_last_epoch:
            logger.info(f"[Epoch {epoch+1}] Refreshing rollout engines with new policy (version {policy_version})...")
            refresh_rollout_engine(rollout_engines=rollout_engines,
                                   updated_policy_path=model_path,
                                   version=policy_version)
            logger.info(f"[Epoch {epoch+1}] Rollout engines refreshed")

        logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
        logger.info("=" * 50)

    # End MLflow run
    if rank == 0 and mlflow_run:
        mlflow.end_run()

    logger.info("Training completed successfully!")
    entire_training_time = time.time() - entire_training_start_time
    logger.info(f"Total training time from start to finish: {entire_training_time:.2f}s")
    ray.shutdown()
    logger.info("Done!")