import os
import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import ray
import time

# imports local methods, classes, etc.
import configs.load as cfg # all config arguments
from custom_datasets.prompt_only import PromptOnlyDataset # our custom pytorch dataset
from misc.utils import safe_string_to_torch_dtype
from rollouts.vllm_engine import VLLMRolloutEngine
from rollouts.replay_buffer import ReplayBuffer
import rewards as reward_fns

def set_random_seeds(seed):
    '''
        Set random seeds, etc., to make it easier to reproduce results eventhough it is not 100% guaranteed.
        In particualr, since we do distributed training, floating-point arithmetic, non-deterministic operations (e.g., torch.Tensor.index_add_),
        setting the seed is not enough, just make things a bit "predictable".
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

def training_runner_setup(model_path,
                          ref_model_path,
                          model_dtype,
                          trust_remote_code,
                          attn_impl,
                          world_size,
                          master_addr,
                          master_port,
                          alg,
                          kl_coeff,
                          clip_low,
                          clip_high,
                          entropy_coeff,
                          use_cache,
                          micro_batch_size_per_gpu,
                          update_after_full_replay,
                          deepspeed_config):
    '''
        This function is responsible for running the training engine.
    '''
    ray_runners = []
    for rank in range(world_size):
        ray_vars = {
                    "MASTER_ADDR": master_addr,
                    "MASTER_PORT": str(master_port),
                    "RANK": str(rank),
                    "WORLD_SIZE": str(world_size),
                    "LOCAL_RANK": "0",
                   }

        runner = alg.options(num_gpus=1,
                            runtime_env={"env_vars": ray_vars},
                            ).remote(
                                     model_path=model_path,
                                     ref_model_path=ref_model_path,
                                     model_dtype=safe_string_to_torch_dtype(model_dtype),
                                     trust_remote_code=trust_remote_code,
                                     attn_impl=attn_impl,
                                     kl_coeff=kl_coeff,
                                     clip_low=clip_low,
                                     clip_high=clip_high,
                                     entropy_coeff=entropy_coeff,
                                     use_cache=use_cache,
                                     micro_batch_size_per_gpu=micro_batch_size_per_gpu,
                                     update_after_full_replay=update_after_full_replay,
                                     deepspeed_config=deepspeed_config,
                                     )
        ray_runners.append(runner)

    return ray_runners

def inference_engine_setup(model_path,
                           trust_remote_code,
                           temperature,
                           max_tokens,
                           n_samples,
                           top_p,
                           top_k,
                           seed,
                           ignore_eos,
                           stop,
                           stop_token_ids,
                           prompt_logprobs,
                           force_strict_on_policy,
                           reward_func,
                           tensor_parallel_size,
                           eos_id,
                           reward_broadcast,
                           eps_reward_norm,
                           rollout_gpus,
                           ):
    '''
        This function is responsible for setting up distributed inference engine.
    '''

    kwargs = { "model_path": model_path,
               "trust_remote_code": trust_remote_code,
               "temperature": temperature,
               "max_tokens": max_tokens,
               "n_samples": n_samples,
               "top_p": top_p,
               "top_k": top_k,
               "seed": seed,
               "ignore_eos": ignore_eos,
               "stop": stop,
               "stop_token_ids": stop_token_ids,
               "prompt_logprobs": prompt_logprobs,
               "force_strict_on_policy": force_strict_on_policy,
               "reward_func": reward_func,
               "tensor_parallel_size": tensor_parallel_size,
               "eos_id": eos_id,
               "reward_broadcast": reward_broadcast,
               "eps_reward_norm": eps_reward_norm,
            }

    tp = int(tensor_parallel_size)
    num_rollout_engines = max(1, int(rollout_gpus) // tp)

    rollout_engines = []
    for _ in range(num_rollout_engines):
        rollout_engines.append(VLLMRolloutEngine.options(num_gpus=tp).remote(**kwargs))

    return num_rollout_engines, rollout_engines

def load_tokenizer(model_name,
                   trust_remote_code=False,
                   rank=0):
    '''
       Load tokenizer from huggingface.
    '''
    # 1. Tokenizer initialization
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

def rollout_dataloader_setup(files_path,
                             num_workers,
                             max_seq_len,
                             prompt_key,
                             rollout_batch_size,
                             tokenizer,
                             num_rollout_engines,
                             split='train',
                             ):
    '''
       This dataloader is used for rollout generation.
    '''
    # 1. Initialize our custom datasets
    prompt_ds = PromptOnlyDataset(prompt_key=prompt_key,
                                  max_seq_len=max_seq_len,
                                  tokenizer=tokenizer,
                                  data_path=files_path,
                                  return_text=False)

    # since we split the data across the rollout engines
    bsz = num_rollout_engines * rollout_batch_size
    dataloader = DataLoader(dataset=prompt_ds,
                            batch_size=bsz,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=prompt_ds.collate_fn,
                            )

    return dataloader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    args = parser.parse_args()

    ########
    # 1. Miscellaneous setups
    ########
    rank, local_rank = rank_setup()
    config = cfg.load_and_verify(method="rl",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 )
    set_random_seeds(seed=config.run.seed)

    # number of gpus for training which is used by deepspeed
    training_gpus = config.run.training_gpus
    # number of gpus for rollout generation which is used by vllm
    rollout_gpus  = config.run.rollout_gpus

    ########
    # 2. initialize ray
    ########
    ray_engine, master_addr = setup_ray(ray_address=config.run.ray_address)

    ########
    # 3. initialize training engine
    ########
    if str.lower(config.train.alg_name) in {'pg', 'ppo', 'grpo', 'cispo'}:
        if str.lower(config.train.alg_name) == 'pg':
            import algs.PG.pg as calg
            alg = calg.PG
        elif str.lower(config.train.alg_name) == 'ppo':
            import algs.PPO.ppo as calg
            alg = calg.PPO
        elif str.lower(config.train.alg_name) == 'grpo':
            import algs.GRPO.grpo as calg
            alg = calg.GRPO
        elif str.lower(config.train.alg_name) == 'cispo':
            import algs.CISPO.cispo as calg
            alg = calg.CISPO
    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")

    training_engine_runners = training_runner_setup(model_path=config.model.name,
                                                  ref_model_path=config.model.ref_model,
                                                  model_dtype=config.model.dtype,
                                                  trust_remote_code=config.model.trust_remote_code,
                                                  attn_impl=config.model.attn_implementation,
                                                  world_size=training_gpus,
                                                  master_addr=master_addr,
                                                  master_port=config.run.ray_master_port,
                                                  alg=alg,
                                                  kl_coeff=config.train.kl_coeff,
                                                  clip_low=config.train.clip_low,
                                                  clip_high=config.train.clip_high,
                                                  entropy_coeff=config.train.entropy_coeff,
                                                  use_cache=config.model.use_cache,
                                                  micro_batch_size_per_gpu=config.train.train_batch_size_per_gpu,
                                                  update_after_full_replay=config.train.update_after_full_replay,
                                                  deepspeed_config=config.deepspeed)

    assert len(training_engine_runners) == training_gpus, "Number of training engines does not match number of training gpus"

    # Synchronization barrier to prevent deepspeed rendezvous hang
    # wait for all training actors to finish initialization before proceeding
    if rank == 0:
        print("Waiting for all training engines to initialize...")

    ready_checks = [engine.is_ready.remote() for engine in training_engine_runners]
    ready = ray.get(ready_checks)
    if rank == 0:
        print("All training engines ready!")

    ########
    # 5. load tokenizer
    ########
    tokenizer = load_tokenizer(model_name=config.model.name,
                               trust_remote_code=config.model.trust_remote_code,
                               rank=rank)

    ########
    # 6. initialize inference engine
    ########
    if config.reward.reward_func:
        reward_fnc = getattr(reward_fns, config.reward.reward_func)

    else:
        raise ValueError("Reward function not specified")

    num_rollout_engines, rollout_engines = inference_engine_setup(model_path=config.model.name,
                                                                  trust_remote_code=config.model.trust_remote_code,
                                                                  temperature=config.rollout.temperature,
                                                                  max_tokens=config.rollout.max_tokens,
                                                                  n_samples=config.rollout.n_samples,
                                                                  top_p=config.rollout.top_p,
                                                                  top_k=config.rollout.top_k,
                                                                  seed=config.run.seed,
                                                                  ignore_eos=config.rollout.ignore_eos,
                                                                  stop=config.rollout.stop,
                                                                  stop_token_ids=config.rollout.stop_token_ids,
                                                                  prompt_logprobs=config.rollout.prompt_logprobs,
                                                                  force_strict_on_policy=config.rollout.force_strict_on_policy,
                                                                  reward_func=reward_fnc,
                                                                  tensor_parallel_size=config.rollout.tensor_parallel_size,
                                                                  eos_id=tokenizer.eos_token_id,
                                                                  reward_broadcast=config.reward.broadcast,
                                                                  eps_reward_norm=config.reward.eps_reward_norm,
                                                                  rollout_gpus=config.run.rollout_gpus,)

    ########
    # 6. Load the rollout dataloader
    ########
    rollout_dataloader = rollout_dataloader_setup(files_path=config.data.train_files_path,
                                                  num_workers=config.data.num_workers,
                                                  max_seq_len=config.data.max_seq_len,
                                                  prompt_key=config.data.prompt_key,
                                                  rollout_batch_size=config.rollout.rollout_batch_size_per_gpu,
                                                  tokenizer=tokenizer,
                                                  num_rollout_engines=num_rollout_engines,
                                                  split='train',
                                                  )

    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 )
    ########
    # 7. Training and evaluation loop
    ########
    policy_version = 0
    number_of_epochs  = config.train.total_number_of_epochs
    total_training_steps_per_epoch = config.train.train_steps_per_epoch

    for epoch in range(number_of_epochs):
        ################
        # Sample generation step
        ################
        # 1) Generate rollouts
        for rollout_batch in rollout_dataloader:
            # 1.1 split data across rollout engines
            # recall: num_rollout_engines  = max(1, int(rollout_gpus) // tensor_parallel_size)
            # rollout_batch is a List[Dict].
            shard_size = (len(rollout_batch) + num_rollout_engines - 1) // num_rollout_engines
            rollout_shards = [rollout_batch[i * shard_size:(i + 1) * shard_size] for i in range(num_rollout_engines)]

            # 1.2 generate rollouts
            rollout_samples = []
            for eng, shard in zip(rollout_engines, rollout_shards):
                rollout_samples.append(eng.generate.remote(prompts=shard,
                                                           current_iter=epoch,
                                                           policy_version=policy_version))

            # 1.3 gather rollouts
            rollout_lists = ray.get(rollout_samples)

            # 1.4 merge rollouts
            rollout_merged = []
            for rl in rollout_lists:
                rollout_merged.extend(rl)

            # 1.5 add to replay buffer
            replay_buffer.add_batch_seqs(rollout_merged)

        if len(replay_buffer) <= 1:
            raise ValueError("Replay buffer is empty")

        ################
        # Training step
        ################
        # 2. create dataloader from replay buffer and convert to list as ray needs serializable data
        # (pytorch dataloader cannot be serialized and sent to ray workers).
        train_batches = list(DataLoader(dataset=replay_buffer,
                                        batch_size=config.train.train_batch_size_per_gpu,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=False,
                                        collate_fn=replay_buffer.collate_fn,
                                        ))

        # 3. update the policy based on the current replay buffer
        # shard batches across training engines
        num_train_engines = len(training_engine_runners)

        # ensure all ranks get equal number of batches to prevent deepspeed hang
        # pad train_batches to be divisible by num_train_engines
        num_batches = len(train_batches)
        batches_per_engine = (num_batches + num_train_engines - 1) // num_train_engines
        total_batches_needed = batches_per_engine * num_train_engines

        if total_batches_needed > num_batches:
            # Pad by repeating the last batch
            padding = [train_batches[-1]] * (total_batches_needed - num_batches)
            train_batches_padded = train_batches + padding

        else:
            train_batches_padded = train_batches

        for tidx in range(total_training_steps_per_epoch):
            # 3.1 send training task to all training engines
            train_futures = []

            for i, engine in enumerate(training_engine_runners):
                # shard the train_batches which is guaranteed to be equal size
                # example for [i::step]: num_train_engines = 2 and 6 batches: [B0, B1, B2, B3, B4, B5]
                # [0::2] -> [B0, B2, B4]
                # [1::2] -> [B1, B3, B5]
                shard = train_batches_padded[i::num_train_engines]

                # All ranks MUST participate in training
                assert len(shard) > 0, f"Rank {i} has empty shard - this will cause DeepSpeed hang"
                train_futures.append(engine.train_step.remote(shard))

            # 3.2 gather training metrics from all engines
            train_metrics = ray.get(train_futures)

            # 3.3 log training progress
            if rank == 0 and tidx % 10 == 0:
                # aggregate metrics across all training engines
                avg_loss = np.mean([m.get('loss_total', 0.0) for m in train_metrics])
                avg_kl = np.mean([m.get('approx_kl', 0.0) for m in train_metrics])
                print(f"Epoch {epoch+1}/{number_of_epochs}, "
                      f"Step {tidx+1}/{total_training_steps_per_epoch}, "
                      f"Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}")

        # 4. update policy version and reset replay buffer
        policy_version += 1
        replay_buffer.__reset__()

        ################
        # Save current policy
        ################
        tag = f"iter{epoch:06d}_v{policy_version:06d}"
        save_dir = os.path.join("./checkpoints", tag)

        # save tokenizer so it's ready when vllm loads the model
        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            tokenizer.save_pretrained(save_dir)

        # save must run on *all ranks* for zero-3 correctness.
        save_futures = []
        for engine in training_engine_runners:
            save_futures.append(engine.save_checkpoint.remote(output_path="./checkpoints", tag=tag))

        # Wait for all saves to complete
        save_paths = ray.get(save_futures)

        # Use the first saved path (all should be the same)
        model_path = save_paths[0] if save_paths else None

        # barrier to ensure all files are written before vllm refresh
        time.sleep(1)  # small delay to ensure filesystem consistency

        ################
        # Refresh rollout policy
        ################
        if model_path:
            refresh_futures = []
            for eng in rollout_engines:
                refresh_futures.append(eng.refresh_model.remote(model_path, policy_version))
            ray.get(refresh_futures)

    print("Training completed")
    ray.shutdown()



