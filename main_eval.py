import os
import json
import yaml
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
from tqdm import tqdm

# imports local methods, classes, etc.
import configs.load as cfg # all config arguments
from custom_datasets.prompt_only import PromptOnlyDataset # our custom pytorch dataset
from rollouts.vllm_engine import VLLMRolloutEngine
from misc.logging import setup_logging
from rollouts.replay_buffer import ReplayBuffer

def set_random_seeds(seed):
    '''
        Set random seeds to make runs more reproducible (still not guaranteed).
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
        print("Warning: Could not get master address, using localhost")
        master_addr = "127.0.0.1"

    return master_addr

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

def create_rollout_dataloader(params, tokenizer, num_rollout_engines):
    '''
       This dataloader is used for rollout generation which 
       would be used to train the policy.
    '''
    # 1. Initialize our custom datasets
    reward_function_name = params.reward.reward_func
    return_answer = True if reward_function_name in ['gsm8k_reward_func'] else False
    prompt_ds = PromptOnlyDataset(prompt_key=params.data.prompt_key,
                                  max_seq_len=params.data.max_seq_len,
                                  tokenizer=tokenizer,
                                  data_path=params.data.test_files_path,
                                  return_text=False,
                                  return_answer=return_answer,
                                  )

    # since we split the data across the rollout engines
    bsz = num_rollout_engines * params.rollout.rollout_batch_size_per_gpu
    dataloader = DataLoader(dataset=prompt_ds,
                            batch_size=bsz,
                            num_workers=params.data.num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=prompt_ds.collate_fn,
                            )


    return dataloader

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
        engines.append(VLLMRolloutEngine.options(num_gpus=tp).remote(**kwargs))

    return engines

def collect_rollouts(dataloader,
                     rollout_engines,
                     epoch,
                     policy_version,
                     replay_buffer,
                     logger):

    '''
        This function is used to run rollout engine and generate rollouts/samples.
    '''
    num_rollout_engines = len(rollout_engines)
    rollout_start_time = time.time()
    total_samples_generated = 0
    total_reward_sum = 0.0
    total_response_len = 0

    # Note dataLoader's batch_size is already num_rollout_engines * rollout_batch_size,
    batch_size   = dataloader.batch_size
    dataset_size = len(dataloader.dataset)
    num_steps_to_generate_all = (dataset_size + batch_size - 1) // batch_size

    logger.info(f"[Rollout] Dataset: {dataset_size}, Batch: {batch_size} "
                f"({num_rollout_engines} engines × {batch_size // num_rollout_engines} per engine), "
                f"Steps to generate all samples: {num_steps_to_generate_all}")

    tqdm_dataloader = tqdm(dataloader, total=num_steps_to_generate_all)
    for rollout_batch in tqdm_dataloader:
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
        rollout_lists = ray.get(rollout_samples)

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

    logger.info(f"Average reward: {avg_reward}, Average response length: {avg_response_len}")

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
    rank = 0
    # Setup logging
    logger = setup_logging(rank=rank, log_level=args.log_level)
    logger.info(f"Starting Evaluation...")

    config = cfg.load_and_verify(method="eval",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 rank=rank,
                                 )
    set_random_seeds(seed=config.run.seed)

    checkpoint_dir = config.run.checkpoint_dir

    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    # number of gpus for rollout generation which is used by vllm
    rollout_gpus  = config.run.rollout_gpus

    ########
    # 2. initialize ray
    ########
    logger.info(f"Initializing Ray ...")
    master_addr = setup_ray(ray_address=config.run.ray_address)
    logger.info(f"Ray initialized. Master address: {master_addr}")

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
    # 6. Load the test datat
    ########
    logger.info(f"Loading rollout dataloader from {config.data.test_files_path}")
    rollout_dataloader = create_rollout_dataloader(params=config,
                                                  tokenizer=tokenizer,
                                                  num_rollout_engines=num_rollout_engines)
    logger.info(f"Rollout dataloader ready. Total batches per epoch: {len(rollout_dataloader)}")
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 )
    logger.info("Replay buffer initialized")

    rollout_stats = collect_rollouts(dataloader=rollout_dataloader,
                                         rollout_engines=rollout_engines,
                                         epoch=0,
                                         policy_version=0,
                                         replay_buffer=replay_buffer,
                                         logger=logger)

    logger.info(f"Rollout complete: {rollout_stats['total_samples_generated']} samples, "
                f"avg_reward={rollout_stats['avg_reward']:.4f}, avg_response_len={rollout_stats['avg_response_len']:.1f}, "
                f"time={rollout_stats['rollout_time']:.2f}s")

    logger.info("Evaluation completed successfully!")


    os.makedirs(checkpoint_dir, exist_ok=True)
    # save rollout stats
    rollout_stats_path = os.path.join(checkpoint_dir, "rollout_stats.json")
    with open(rollout_stats_path, "w") as f:
        json.dump(rollout_stats, f)
    logger.info(f"Rollout stats saved to {rollout_stats_path}")

    # save experiment config
    experiment_config_path = os.path.join(checkpoint_dir, "experiment_config.yaml")
    with open(experiment_config_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Experiment config saved to {experiment_config_path}")
    
    ray.shutdown()