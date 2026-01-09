import os
import random
import numpy as np
import argparse
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import ray
import time
import mlflow

# imports local methods, classes, etc.
import configs.load as cfg # all config arguments
from custom_datasets.prompt_only import PromptOnlyDataset # our custom pytorch dataset
from misc.utils import safe_string_to_torch_dtype, get_experiment_dir_name
from rollouts.vllm_engine import VLLMRolloutEngine
from rollouts.replay_buffer import ReplayBuffer
import rewards as reward_fns
from misc.logging import setup_logging, setup_mlflow

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

def rank_setup():
    '''
        Detect rank from environment variables.
    '''
    # Unique id of gpu in the ENTIRE WORLD. It ranges from 0 to world_size - 1
    rank = int(os.environ.get('RANK', 0))

    # Unique id of gpu in the LOCAL node (or simply one node). It ranges from 0 to local_node_size - 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # add some checks to make sure number of gpus and local rank are correct.
    if not torch.cuda.is_available():
        if rank == 0:
            print("Warning: CUDA is not available, running on CPU. Sorry!")
    else:
        num_local_gpus = torch.cuda.device_count()
        if local_rank >= num_local_gpus:
            raise RuntimeError(f"LOCAL_RANK {local_rank} >= available GPUs {num_local_gpus}")

        torch.cuda.set_device(local_rank)

    return rank, local_rank

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
        print("Warning: Could not get master address, using localhost")
        master_addr = "127.0.0.1"

    return ray, master_addr

def training_engine_setup(params, alg, world_size, master_addr, master_port):
    '''
        This function is responsible for running the training engine.
    '''
    kwargs = { # model relataed arguments
               'model_path':params.model.name,
               'ref_model_path':params.model.ref_model,
               'model_dtype':safe_string_to_torch_dtype(params.model.dtype),
               'trust_remote_code':params.model.trust_remote_code,
               'attn_impl':params.model.attn_implementation,
               'use_cache':params.model.use_cache,

               # training related arguments
               'kl_coeff':params.train.kl_coeff,
               'clip_low':params.train.clip_low,
               'clip_high':params.train.clip_high,
               'entropy_coeff':params.train.entropy_coeff,
               'micro_batch_size_per_gpu':params.train.train_batch_size_per_gpu,
               'update_after_full_replay':params.train.update_after_full_replay,

               # deepspeed related arguments
               'deepspeed_config':params.deepspeed
    }
    # setup ray runners
    ray_runners = []
    for rank in range(world_size):
        ray_vars = {"MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": "0",}
        runner = alg.options(num_gpus=1, runtime_env={"env_vars": ray_vars}).remote(**kwargs)
        ray_runners.append(runner)

    return ray_runners

def rollout_engine_setup(params, reward_fnc, eos_id):
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

              # reward related arguments
              "reward_func":reward_fnc,
              "reward_broadcast":params.reward.broadcast,
              "eps_reward_norm":params.reward.eps_reward_norm,
            }

    num_rollout_engines = max(1, rollout_gpus // tp)
    rollout_engines = []
    for i in range(num_rollout_engines):
        kwargs['engine_id'] = i
        rollout_engines.append(VLLMRolloutEngine.options(num_gpus=tp).remote(**kwargs))

    return num_rollout_engines, rollout_engines

def load_tokenizer(model_name, trust_remote_code=False, rank=0):
    '''
       Load tokenizer from huggingface.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    # log warning if pad token is not present.
    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Warning: Pad token is not present, using eos token as pad token")
        if getattr(tokenizer, 'eos_token', None) is not None:
            # prefer explicit token if available
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else:
            # fallback to eos token id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def rollout_dataloader_setup(params, tokenizer, num_rollout_engines):
    '''
       This dataloader is used for rollout generation which would be used to train the policy.
    '''
    # 1. Initialize our custom datasets
    prompt_ds = PromptOnlyDataset(prompt_key=params.data.prompt_key,
                                  max_seq_len=params.data.max_seq_len,
                                  tokenizer=tokenizer,
                                  data_path=params.data.train_files_path,
                                  return_text=False)

    # since we split the data across the rollout engines
    bsz = num_rollout_engines * params.rollout.rollout_batch_size_per_gpu
    dataloader = DataLoader(dataset=prompt_ds,
                            batch_size=bsz,
                            num_workers=params.data.num_workers,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=prompt_ds.collate_fn,
                            )

    return dataloader

def collect_rollouts(rollout_dataloader,
                     num_rollout_engines,
                     rollout_engines,
                     epoch,
                     policy_version,
                     replay_buffer,
                     ray_agent):
    '''
        This function is used to run rollout engine and generate rollouts/samples.
    '''
    assert num_rollout_engines == len(rollout_engines), 'Number of rollout engines does not match with the number of rollout engines'

    rollout_start_time = time.time()
    total_samples_generated = 0
    total_reward_sum = 0.0
    total_response_len = 0

    # Note dataLoader's batch_size is already num_rollout_engines * rollout_batch_size,
    batch_size   = rollout_dataloader.batch_size
    dataset_size = len(rollout_dataloader.dataset)
    num_steps_to_generate_all = (dataset_size + batch_size - 1) // batch_size

    print(
        f"[Rollout Stats] Dataset size: {dataset_size} | "
        f"Batch size: {batch_size} "
        f"({num_rollout_engines} engines × {batch_size // num_rollout_engines} per engine), "
        f"Steps to generate all samples: {num_steps_to_generate_all}"
    )

    for rollout_batch in rollout_dataloader:
        # 1. split data across rollout engines
        # recall: num_rollout_engines  = max(1, int(rollout_gpus) // tensor_parallel_size)
        # and rollout_batch is a list of dictionaries.
        shard_size = (len(rollout_batch) + num_rollout_engines - 1) // num_rollout_engines
        # it is not necessary to have equal number of samples per engine, though they can't be empty.
        rollout_shards = [rollout_batch[i * shard_size:(i + 1) * shard_size] for i in range(num_rollout_engines)]
        rollout_shards = [shard for shard in rollout_shards if len(shard) > 0]

        # 2. schedule rollout generation
        rollout_samples = []
        for i, shard in enumerate(rollout_shards):
            rollout_samples.append(rollout_engines[i].generate.remote(prompts=shard,
                                                                      current_iter=epoch,
                                                                      policy_version=policy_version))

        # 3. gather rollouts
        rollout_lists = ray_agent.get(rollout_samples)

        # 4. merge rollouts across all engines and collect stats
        rollout_merged = []
        for rl in rollout_lists:
            rollout_merged.extend(rl)
            for sample in rl:
                total_samples_generated += 1
                total_reward_sum += sample['rewards'].sum().item()
                total_response_len += sample['response_len']

        # 5. now add them to replay buffer
        replay_buffer.add_batch_seqs(rollout_merged)

    rollout_time = time.time() - rollout_start_time
    avg_reward = total_reward_sum / max(1, total_samples_generated)
    avg_response_len = total_response_len / max(1, total_samples_generated)

    if len(replay_buffer) <= 1:
        raise ValueError("Replay buffer is empty")

    return {"total_samples_generated": total_samples_generated,
            "avg_reward": avg_reward,
            "avg_response_len": avg_response_len,
            "rollout_time": rollout_time}

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
    rank, local_rank = rank_setup()

    # Setup logging
    logger = setup_logging(rank=rank, log_level=args.log_level)
    logger.info(f"Starting RL training...")

    config = cfg.load_and_verify(method="rl",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 )
    set_random_seeds(seed=config.run.seed)

    # Setup MLflow (only on rank 0)
    mlflow_run = setup_mlflow(config=config, tracking_uri=config.run.tracking_uri, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    # number of gpus for training which is used by deepspeed
    training_gpus = config.run.training_gpus
    # number of gpus for rollout generation which is used by vllm
    rollout_gpus  = config.run.rollout_gpus

    ########
    # 2. initialize ray
    ########
    logger.info(f"Initializing Ray cluster...")
    ray_engine, master_addr = setup_ray(ray_address=config.run.ray_address)
    logger.info(f"Ray initialized. Master address: {master_addr}")

    ########
    # 3. initialize training engine
    ########
    logger.info(f"Setting up training algorithm: {config.train.alg_name}")
    alg_name = str.lower(config.train.alg_name)
    if alg_name in {'pg', 'ppo', 'grpo', 'cispo'}:
        if alg_name == 'pg':
            import algs.PG.pg as calg
            alg = calg.PG
        elif alg_name == 'ppo':
            import algs.PPO.ppo as calg
            alg = calg.PPO
        elif alg_name == 'grpo':
            import algs.GRPO.grpo as calg
            alg = calg.GRPO
        elif alg_name == 'cispo':
            import algs.CISPO.cispo as calg
            alg = calg.CISPO
    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")

    training_engine_runners = training_engine_setup(params=config,
                                                    alg=alg,
                                                    world_size=training_gpus,
                                                    master_addr=master_addr,
                                                    master_port=config.run.ray_master_port)

    assert len(training_engine_runners) == training_gpus, "Number of training engines does not match number of training gpus"
    logger.info(f"Created {len(training_engine_runners)} training engine runners")

    # Synchronization barrier to prevent deepspeed rendezvous hang
    # wait for all training actors to finish initialization before proceeding
    logger.info("Waiting for all training engines to initialize...")

    ready_checks = [engine.is_ready.remote() for engine in training_engine_runners]
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
    if config.reward.reward_func:
        reward_fnc = getattr(reward_fns, config.reward.reward_func)
        logger.info(f"Using reward function: {config.reward.reward_func}")

    else:
        raise ValueError("Reward function not specified")

    num_rollout_engines, rollout_engines = rollout_engine_setup(params=config, reward_fnc=reward_fnc, eos_id=tokenizer.eos_token_id)
    logger.info(f"Created {num_rollout_engines} rollout engines")

    ########
    # 6. Load the rollout dataloader
    ########
    logger.info(f"Created {num_rollout_engines} rollout engines with tensor_parallel_size={config.rollout.tensor_parallel_size}")
    logger.info(f"Loading rollout dataloader from {config.data.train_files_path}")
    rollout_dataloader = rollout_dataloader_setup(params=config,
                                                  tokenizer=tokenizer,
                                                  num_rollout_engines=num_rollout_engines)
    logger.info(f"Rollout dataloader ready. Total batches per epoch: {len(rollout_dataloader)}")

    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 )
    logger.info("Replay buffer initialized")
    ########
    # 7. Training and evaluation loop
    ########
    policy_version = 0
    number_of_epochs  = config.train.total_number_of_epochs
    number_of_training_steps_per_epoch = config.train.train_steps_per_epoch
    global_step = 0

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {number_of_training_steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    logger.info("=" * 50)

    for epoch in range(number_of_epochs):
        epoch_start_time = time.time()
        logger.info(f"[Epoch {epoch+1}/{number_of_epochs}] Starting rollout generation...")

        ################
        # 1. Rollout generation and Sample collection
        ################
        rollout_stats = collect_rollouts(rollout_dataloader=rollout_dataloader,
                                         num_rollout_engines=num_rollout_engines,
                                         rollout_engines=rollout_engines,
                                         epoch=epoch,
                                         policy_version=policy_version,
                                         replay_buffer=replay_buffer,
                                         ray_agent=ray)

        logger.info(f"[Epoch {epoch+1}] Rollout complete: {rollout_stats['total_samples_generated']} samples, "
                    f"avg_reward={rollout_stats['avg_reward']:.4f}, avg_response_len={rollout_stats['avg_response_len']:.1f}, "
                    f"time={rollout_stats['rollout_time']:.2f}s")

        if len(replay_buffer) <= 1:
            raise ValueError("Replay buffer is empty")

        ################
        # Data and batch preperation for training
        ################
        logger.info(f"[Epoch {epoch+1}] Starting training on {len(replay_buffer)} replay buffer samples...")
        train_start_time = time.time()

        # Create dataloader from replay buffer and convert to list as ray needs serializable data
        train_batches = list(DataLoader(dataset=replay_buffer,
                                        batch_size=config.train.train_batch_size_per_gpu,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=False,
                                        collate_fn=replay_buffer.collate_fn,
                                        ))
        logger.info(f"[Epoch {epoch+1}] Created {len(train_batches)} training batches")

        # We need to ensure all ranks/gpus get EQUAL number of batches to prevent deepspeed hang so we pad the batches
        # which are not divisible by num_train_engines
        num_train_engines = len(training_engine_runners)
        num_batches = len(train_batches)
        batches_per_engine = (num_batches + num_train_engines - 1) // num_train_engines
        total_batches_needed = batches_per_engine * num_train_engines

        if total_batches_needed > num_batches:
            # Pad by repeating the last batch
            padding = [train_batches[-1]] * (total_batches_needed - num_batches)
            train_batches_padded = train_batches + padding

        else:
            train_batches_padded = train_batches

        ################
        # Policy learning and training
        ################
        epoch_metrics = {'loss_total': [], 'loss_pi': [], 'loss_ent': [], 'approx_kl': [], 'clipfrac': []}
        for tidx in range(number_of_training_steps_per_epoch):

            # Schedule training engines to run update step
            train_futures = []
            for eid, engine in enumerate(training_engine_runners):
                # Send equal number of batches to each training engine
                # [eid::step]: num_train_engines = 2 and 6 batches: [B0, B1, B2, B3, B4, B5]
                # [0::2] -> [B0, B2, B4]
                # [1::2] -> [B1, B3, B5]
                shard = train_batches_padded[eid::num_train_engines]

                # All ranks MUST participate in training, hence no empty shards
                assert len(shard) > 0, f"Engine {eid} has empty shard - this will cause DeepSpeed hang"
                train_futures.append(engine.train_step.remote(engine_id=eid, micro_batches=shard))

            # Gather training metrics from all engines
            # train_metrics: clipfrac, approx_kl, loss_ent, loss_pi, loss_total
            train_metrics = ray.get(train_futures)

            # Aggregate metrics across all training engines
            avg_loss     = np.mean([m.get('loss_total', 0.0) for m in train_metrics])
            avg_loss_pi  = np.mean([m.get('loss_pi', 0.0) for m in train_metrics])
            avg_loss_ent = np.mean([m.get('loss_ent', 0.0) for m in train_metrics])
            avg_kl       = np.mean([m.get('approx_kl', 0.0) for m in train_metrics])
            avg_clipfrac = np.mean([m.get('clipfrac', 0.0) for m in train_metrics])

            # Epoch average of average across all training engines
            epoch_metrics['loss_total'].append(avg_loss)
            epoch_metrics['loss_pi'].append(avg_loss_pi)
            epoch_metrics['loss_ent'].append(avg_loss_ent)
            epoch_metrics['approx_kl'].append(avg_kl)
            epoch_metrics['clipfrac'].append(avg_clipfrac)

            global_step += 1

            # Log to console every 10 steps
            if tidx % 10 == 0:
                logger.info(f"[Epoch {epoch+1}][Step {tidx+1}/{number_of_training_steps_per_epoch}] "
                           f"loss={avg_loss:.4f}, pi_loss={avg_loss_pi:.4f}, ent_loss={avg_loss_ent:.4f}, "
                           f"kl={avg_kl:.6f}, clipfrac={avg_clipfrac:.4f}")

            # Log to MLflow every step (only rank 0)
            if rank == 0 and mlflow_run:
                mlflow.log_metrics({
                    "train/loss_total": avg_loss,
                    "train/loss_pi": avg_loss_pi,
                    "train/loss_ent": avg_loss_ent,
                    "train/approx_kl": avg_kl,
                    "train/clipfrac": avg_clipfrac,
                }, step=global_step)

        # 4. update policy version and reset replay buffer
        policy_version += 1
        if alg_name.lower() in {"pg", "ppo", "grpo", "cispo"}:
            replay_buffer.__reset__()

        # Log epoch summary
        train_time = time.time() - train_start_time
        epoch_time = time.time() - epoch_start_time

        epoch_avg_loss = np.mean(epoch_metrics['loss_total'])
        epoch_avg_kl   = np.mean(epoch_metrics['approx_kl'])
        epoch_avg_clipfrac = np.mean(epoch_metrics['clipfrac'])

        logger.info(f"[Epoch {epoch+1}] Training complete: time={train_time:.2f}s, "
                    f"avg_loss={epoch_avg_loss:.4f}, avg_kl={epoch_avg_kl:.6f}")

        # Log epoch metrics to MLflow
        if rank == 0 and mlflow_run:
            mlflow.log_metrics({
                    "epoch/avg_loss": epoch_avg_loss,
                    "epoch/avg_kl": epoch_avg_kl,
                    "epoch/avg_clipfrac": epoch_avg_clipfrac,
                    "epoch/avg_reward": rollout_stats['avg_reward'],
                    "epoch/avg_response_len": rollout_stats['avg_response_len'],
                    "epoch/total_samples": rollout_stats['total_samples_generated'],
                    "epoch/rollout_time_sec": rollout_stats['rollout_time'],
                    "epoch/train_time_sec": train_time,
                    "epoch/total_time_sec": epoch_time,
                    }, step=epoch + 1)

        ################
        # Save current policy
        ################
        tag = f"iter{epoch+1:06d}_v{policy_version:06d}"
        model_path = get_experiment_dir_name(output_dir=config.run.checkpoint_dir, tag=tag, experiment_id=config.run.experiment_id)
        logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")

        # save tokenizer so it's ready when vllm loads the model
        if rank == 0:
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(model_path)

        # save must run on *all ranks* for zero-3 correctness.
        save_futures = []
        for engine in training_engine_runners:
            save_futures.append(engine.save_checkpoint.remote(output_dir=model_path, tag=tag))

        # Wait for all saves to complete
        ray.get(save_futures)

        # Flush filesystem buffers to ensure checkpoint is fully written
        if rank == 0:
            os.sync()

        logger.info(f"[Epoch {epoch+1}] Checkpoint saved: {model_path}")

        ################
        # Refresh rollout policy
        ################
        logger.info(f"[Epoch {epoch+1}] Refreshing rollout engines with new policy (version {policy_version})...")
        refresh_futures = []
        for eng in rollout_engines:
            refresh_futures.append(eng.refresh_model.remote(model_path, policy_version))

        ray.get(refresh_futures)
        logger.info(f"[Epoch {epoch+1}] Rollout engines refreshed")

        logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
        logger.info("=" * 50)

    # End MLflow run
    if rank == 0 and mlflow_run:
        mlflow.end_run()

    logger.info("Training completed successfully!")
    ray.shutdown()