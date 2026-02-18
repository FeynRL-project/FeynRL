import os
import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
import torch.distributed
from tqdm import tqdm
import gc
import mlflow
import time

# imports local methods, classes, etc.
import configs.load as cfg# all config arguments
from data_feeds.preference import PreferenceFeed
from data_feeds.mixed_sampler import create_dataset_and_sampler
from misc.utils import safe_string_to_torch_dtype, get_experiment_dir_name, load_algorithm
from misc.logging import setup_logging, setup_viz


Algorithm_Registry = {
    # supported algorithms
    'dpo': ('algs.DPO.dpo', 'dpo'),
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

def init_rank_world_size():
    '''
        Detect rank and world size from environment variables.
        we way to run is to use torchrun (torchrun --nnodes=2 --nproc_per_node=4 main_sl.py) where we can specify
        nnodes=2 -> world_size
        nproc_per_node=4 -> local_world_size/num_local_gpus
    '''
    # total number of gpus (e.g, 2 nodes x 4 gpus = 8 gpus in total). world size need to be at least 1
    world_size = int(os.environ.get('WORLD_SIZE', 1))

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

    return rank, world_size, local_rank

def load_models_and_tokenizer(model_name, model_dtype, ref_model_name,trust_remote_code, attn_impl, rank):
    '''
        Load models and tokenizer.
    '''
    assert model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"
    assert attn_impl=='' or attn_impl in ['eager', 'flash_attention_2'], "attn_impl must be one of 'eager', 'flash_attention_2' or empty string"

    # convert string to torch dtype if it is not already
    model_dtype = safe_string_to_torch_dtype(model_dtype)

    # 1. model and its config initialization
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                dtype=model_dtype,
                                                trust_remote_code=trust_remote_code,
                                                config=model_config,
                                                attn_implementation=None if attn_impl == '' else attn_impl)
    # 2. load reference model
    if ref_model_name is None:
        ref_model_name = model_name

    ref_model_config = AutoConfig.from_pretrained(ref_model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name,
                                                        dtype=model_dtype,
                                                        trust_remote_code=trust_remote_code,
                                                        config=ref_model_config,
                                                        attn_implementation=None if attn_impl == '' else attn_impl)

    # 3. Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

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

    return model, ref_model, tokenizer

def create_training_engine(deepspeed_config, deepspeed_ref_config, model, ref_model):
    '''
        This function is responsible for setting up distributed training engine.
        For now, it only supports deepspeed.
    '''
    # Convert pydantic model to python Dict for DeepSpeed
    ds_config_dict = deepspeed_config.model_dump()
    ds_ref_config_dict = deepspeed_ref_config.model_dump()

    # check to avoid re-initializing distributed backend
    if not torch.distributed.is_initialized():
        # 1. Initialize distributed training engine
        deepspeed.init_distributed()

    # 2. Initialize model engine
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model,
                                                         model_parameters=model.parameters(),
                                                         config=ds_config_dict
                                                        )

    ref_model.eval()
    ref_model_engine, _, _, _ = deepspeed.initialize(model=ref_model, config=ds_ref_config_dict)

    return model_engine, ref_model_engine, optimizer

def create_data_loader(params, tokenizer, rank, world_size, batch_size, split):
    '''
       Setup DataLoader for distributed training.
       As a reminder, batch_size is the per-gpu-micro-batch size.
       Hence, global batch size = batch_size * world_size * gradient_accumulation_steps.
    '''
    # 1. Initialize our custom datasets
    data_path = params.data.train_files_path if split == 'train' else params.data.val_files_path

    # steps_per_epoch is only needed for training (MixedDatasetSampler)
    steps_per_epoch = params.train.micro_batches_per_epoch if split == 'train' else None

    dataset, sampler = create_dataset_and_sampler(data_paths=data_path,
                                                  prompt_key=params.data.prompt_key,
                                                  answer_key=params.data.answer_key,
                                                  max_seq_len=params.data.max_seq_len,
                                                  tokenizer=tokenizer,
                                                  train_ratios=params.data.train_ratios,
                                                  split=split,
                                                  rank=rank,
                                                  world_size=world_size,
                                                  seed=params.run.seed,
                                                  local_batch_size=batch_size,
                                                  dataset_cls=PreferenceFeed,
                                                  steps_per_epoch=steps_per_epoch,
                                                  shuffle_within_batch=True,
                                                  dynamic_ratio_every_step=params.train.dynamic_ratio_every_step)

    # 2. Initialize data loader
    def worker_init_fn(worker_id):
        # each worker gets a different seed but deterministic across runs when seed fixed
        worker_seed = params.run.seed + worker_id + (rank * 100000)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if split == 'train':
        # MixedDatasetSampler is a batch sampler (yields batches of indices).
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                num_workers=params.data.num_workers,
                                pin_memory=True,
                                worker_init_fn=worker_init_fn)
    else:
        # DistributedSampler yields individual indices.
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=params.data.num_workers,
                                pin_memory=True,
                                drop_last=False,  # ensure all validation samples are used
                                worker_init_fn=worker_init_fn)

    return dataloader, sampler

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
    args = parser.parse_args()

    ########
    # 1. Setup Environment
    ########
    rank, world_size, local_rank = init_rank_world_size()
    logger = setup_logging(rank=rank, log_level=args.log_level)

    ########
    # 2. Load config and other misc. setup
    ########
    config = cfg.load_and_verify(method="sl",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 world_size=world_size,
                                 rank=rank,
                                 )
    set_random_seeds(seed=config.run.seed)

    # Setup MLflow (only on rank 0)
    mlflow_run = setup_viz(config=config, tracking_uri=config.run.tracking_uri, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    ########
    # 4. load model or previous checkpoints
    ########
    model, ref_model, tokenizer = load_models_and_tokenizer(model_name=config.model.name,
                                                 model_dtype=config.model.dtype,
                                                 ref_model_name=config.model.ref_model,
                                                 trust_remote_code=config.model.trust_remote_code,
                                                 attn_impl=config.model.attn_implementation,
                                                 rank=rank)

    ########
    # 5. Setup trainiing and inference engines
    ########
    if config.model.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled")
        model.gradient_checkpointing_enable()

    model_engine, ref_model_engine, optimizer = create_training_engine(deepspeed_config=config.deepspeed,
                                                                       deepspeed_ref_config=config.deepspeed_ref,
                                                                       model=model,
                                                                       ref_model=ref_model)

    ########
    # 6. Build env or data loader
    ########
    train_dataloader, train_sampler = create_data_loader(params=config,
                                                        tokenizer=tokenizer,
                                                        batch_size=config.train.train_batch_size_per_gpu,
                                                        split='train',
                                                        world_size=world_size,
                                                        rank=rank)

    val_dataloader, _ = create_data_loader(params=config,
                                          tokenizer=tokenizer,
                                          batch_size=config.train.val_batch_size_per_gpu,
                                          split='val',
                                          world_size=world_size,
                                          rank=rank)

    ########
    # 7. Intitate the learning algorithm (e.g., ppo)
    ########
    alg_class = load_algorithm(config.train.alg_name, Algorithm_Registry)
    alg = alg_class(model_engine=model_engine,
                    ref_model_engine=ref_model_engine,
                    optimizer=optimizer,
                    normalize_loss=config.train.normalize_loss,
                    beta=config.train.cl_beta)

    ########
    # 8. Training and evaluation loop
    ########
    if rank == 0:
        print("Starting training...")

    total_number_of_train_samples = len(train_dataloader.dataset)
    micro_batches_per_epoch = config.train.micro_batches_per_epoch
    optimizer_steps_per_epoch = micro_batches_per_epoch // config.train.gradient_accumulation_steps

    # if micro_batches_per_epoch is not divisible by gradient_accumulation_steps
    if micro_batches_per_epoch % config.train.gradient_accumulation_steps != 0:
        remainder = micro_batches_per_epoch % config.train.gradient_accumulation_steps
        # raising error to enforce correctness
        raise ValueError(
            f"micro_batches_per_epoch ({micro_batches_per_epoch}) MUST be divisible by "
            f"gradient_accumulation_steps ({config.train.gradient_accumulation_steps}) to ensure "
            "all gradients are applied within the epoch boundaries. "
            f"Adjust configuration. Remainder: {remainder}"
        )

    logger.info("=" * 50)
    logger.info(f"Starting training: {config.train.total_number_of_epochs} epochs")
    logger.info(
        f"Train set: {len(train_dataloader.dataset)} samples | "
        f"micro_batches/epoch={micro_batches_per_epoch} | "
        f"optimizer_steps/epoch={optimizer_steps_per_epoch} | "
        f"grad_accum={config.train.gradient_accumulation_steps}"
    )
    logger.info(
        f"batch_size_per_gpu={config.train.train_batch_size_per_gpu} | "
        f"global_batch_size={config.train.train_batch_size_per_gpu * config.train.gradient_accumulation_steps * world_size}"
    )
    logger.info("=" * 50)
    global_step = 0
    # Sync before starting
    # Ensure all nodes have loaded the model and data before anyone starts iterating
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    for epoch in range(config.train.total_number_of_epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        # Ensure gradients are zeroed at the start of epoch to prevent any bleeding from previous epoch
        # if accumulation steps were not perfectly aligned (though we enforce alignment above).
        model_engine.optimizer.zero_grad()

        ########
        # 8.1 Training loop
        ########
        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.train.total_number_of_epochs}", disable=(rank != 0))
        else:
            progress_bar = train_dataloader

        # micro_batches_per_epoch = number of micro-batch iterations per epoch.
        # This allows processing a subset of the data per epoch (useful for large datasets).
        # Optimizer steps per epoch = micro_batches_per_epoch // gradient_accumulation_steps

        for step, micro_batch in enumerate(progress_bar):
            # Move batch to gpu (deepspeed engine device)
            micro_batch = {k: v.to(model_engine.device) for k, v in micro_batch.items()}

            # Run one train step for micro-batch.
            metric = alg.train_step(micro_batch)

            # Only increment global_step when ds actually updates weights
            if model_engine.is_gradient_accumulation_boundary():
                global_step += 1

            # logging
            if rank == 0:
                progress_bar.set_postfix({'loss': metric['loss'],
                                           'chosen_rewards': metric['chosen_rewards'],
                                           'rejected_rewards': metric['rejected_rewards'],
                                           'reward_accuracies': metric['reward_accuracies'],
                                           })
                if mlflow_run and model_engine.is_gradient_accumulation_boundary():
                    mlflow.log_metrics({
                        "train/loss": metric['loss'],
                        "train/chosen_rewards": metric['chosen_rewards'],
                        "train/rejected_rewards": metric['rejected_rewards'],
                        "train/reward_accuracies": metric['reward_accuracies'],
                    }, step=global_step)

        # Sync before validation to ensure consistent state
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{config.train.total_number_of_epochs} completed in {epoch_time:.2f} seconds")

        # Clear graph and to reclaim fragmented memory from training ONCE per epoch
        torch.cuda.empty_cache()
        gc.collect()

        ########
        # 8.2 Validation loop
        ########
        # DPO eval_step returns per-batch metrics (loss, chosen_rewards, rejected_rewards, reward_accuracies).
        # We accumulate and average across batches and GPUs.
        local_loss_sum   = torch.tensor(0.0, device=model_engine.device)
        local_batch_count = torch.tensor(0.0, device=model_engine.device)

        for data in val_dataloader:
            val_batch = {k: v.to(model_engine.device) for k, v in data.items()}
            val_metric = alg.eval_step(val_batch)
            local_loss_sum += val_metric['loss']
            local_batch_count += 1.0

        # Aggregate across all ranks.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_batch_count, op=torch.distributed.ReduceOp.SUM)

        # Avoid division by zero
        if local_batch_count.item() == 0:
            global_avg_loss = 0.0
        else:
            global_avg_loss = (local_loss_sum / local_batch_count).item()

        if rank == 0:
            print(f"Epoch {epoch+1}, Validation Loss: {global_avg_loss}")
            if mlflow_run:
                mlflow.log_metrics({
                    "val/loss": global_avg_loss,
                }, step=global_step)

        ########
        # 8.3 Save checkpoint
        ########
        # Sync before saving to ensure no one is still writing
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        tag = f"iter{epoch+1:06d}"
        model_path = get_experiment_dir_name(output_dir=config.run.checkpoint_dir, tag=tag, experiment_id=config.run.experiment_id)
        logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")

        # Save as HuggingFace format
        model_engine.save_16bit_model(model_path)

        # Barrier to ensure all ranks finished writing before rank 0 saves config
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Save model config and tokenizer on rank 0 only
        if rank == 0:
            if hasattr(model_engine.module, 'config'):
                model_engine.module.config.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info(f"[Epoch {epoch+1}] Model config and tokenizer saved")

        # Wait for saving to complete on all ranks
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    logger.info("Training completed successfully!")

    # End MLflow run cleanly
    if rank == 0 and mlflow_run:
        mlflow.end_run()
