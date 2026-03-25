import os
import argparse
import deepspeed
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader
import torch.distributed
from tqdm import tqdm
import gc
import time
from peft import get_peft_model, LoraConfig

# imports local methods, classes, etc.
import configs.load as cfg # all config arguments
from data_feeds.paired import PairedFeed
from data_feeds.mixed_sampler import create_dataset_and_sampler
from misc.utils import safe_string_to_torch_dtype, get_experiment_dir_name, load_algorithm, set_random_seeds, get_determinism_env_vars
from misc.logging import setup_logging, setup_tracker
from misc.checkpoint_utils import resume_from_checkpoint, save_training_checkpoint, cleanup_incomplete_checkpoints


Algorithm_Registry = {# supported algorithms
                     'sft': ('algs.SFT.sft', 'SFT'),
                     }

def init_rank_world_size():
    '''
        Detect rank and world size from environment variables.
        we way to run is to use torchrun (torchrun --nnodes=2 --nproc_per_node=4 main_sl.py) where we can specify
        nnodes=2 -> world_size
        nproc_per_node=4 -> local_world_size/num_local_gpus
    '''
    # Set deterministic cuBLAS workspace before any CUDA context/device setup.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", get_determinism_env_vars())

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

    # Log resolved distributed env on rank 0 for debugging multi-node setups
    if rank == 0:
        master_addr = os.environ.get('MASTER_ADDR', 'not set')
        master_port = os.environ.get('MASTER_PORT', 'not set')
        print(f"[Distributed] rank={rank}, world_size={world_size}, local_rank={local_rank}, "
              f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")

    return rank, world_size, local_rank

def apply_peft_module(model, peft_config, rank=0):
    '''
        Apply PEFT module to the model if it is enabled.
    '''
    if peft_config.peft_type == 'lora':
        lora_config = LoraConfig(r=peft_config.lora_rank,
                                 lora_alpha=peft_config.lora_alpha,
                                 lora_dropout=peft_config.lora_dropout,
                                 target_modules=peft_config.lora_target_modules,
                                 task_type=peft_config.task_type)

        model_peft = get_peft_model(model, lora_config)
        if rank == 0:
            print("LoRA model loaded successfully")
        return model_peft

    else:
        raise ValueError(f"Unsupported PEFT type: {peft_config.peft_type}")

def load_models_and_tokenizer(model_name, model_dtype, trust_remote_code, attn_impl, rank):
    '''
        Load models and tokenizer.
    '''
    assert model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"
    assert attn_impl is None or attn_impl == '' or attn_impl in ['eager', 'flash_attention_2'], \
        "attn_impl must be one of None, '', 'eager', 'flash_attention_2'"

    # convert string to torch dtype if it is not already
    model_dtype = safe_string_to_torch_dtype(model_dtype)

    # 1. model and its config initialization
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                dtype=model_dtype,
                                                trust_remote_code=trust_remote_code,
                                                config=model_config,
                                                attn_implementation=None if attn_impl == '' else attn_impl)

    # 2. Tokenizer initialization
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

    # Sync pad_token_id into model config so exported checkpoints are consistent
    # as vllm and similar read pad_token_id from config.json, not tokenizer
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def create_training_engine(deepspeed_config, model):
    '''
        This function is responsible for setting up distributed training engine.
        For now, it only supports deepspeed.
    '''
    # Convert pydantic model to python Dict for DeepSpeed
    ds_config_dict = deepspeed_config.model_dump()

    # check to avoid re-initializing distributed backend
    if not torch.distributed.is_initialized():
        # 1. Initialize distributed training engine
        deepspeed.init_distributed()

    # 2. Initialize model engine
    # only pass trainable params so ds doesn't waste memory on frozen ones
    trainable_params = [p for p in model.parameters() if p.requires_grad ]
    model_engine, optimizer, _, _ = deepspeed.initialize(
                                                        model=model,
                                                        model_parameters=trainable_params,
                                                        config=ds_config_dict
                                                        )
    return model_engine, optimizer

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
                                                  dataset_cls=PairedFeed,
                                                  steps_per_epoch=steps_per_epoch,
                                                  shuffle_within_batch=True,
                                                  dynamic_ratio_every_step=params.train.dynamic_ratio_every_step)

    # 2. Initialize data loader
    def worker_init_fn(worker_id):
        # each worker gets a different seed but deterministic across runs when seed fixed
        worker_seed = params.run.seed + worker_id + (rank * 100000)
        # we already handled rank differentiation above so we don't need to send it to set_random_seeds
        set_random_seeds(worker_seed)

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
    parser.add_argument("--config_file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging level")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training. It must contain a CHECKPOINT_COMPLETE marker.")
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
    set_random_seeds(seed=config.run.seed, rank=rank)

    # setup remote experiment tracker
    tracker = setup_tracker(config=config, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")
    checkpoint_save_interval = config.run.checkpoint_save_interval if config.run.checkpoint_save_interval is not None else 1

    ########
    # 3. Initialize distributed backend early.
    ########
    # nccl topology discovery on multi-gpu nodes especially with InfiniBand can take time.
    # Initializing here lets nccl establish connections while model loading runs on CPU,
    # preventing hangs inside deepspeed.initialize().
    # Set NCCL env vars before init to control transport selection.
    if config.run.nccl_socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = config.run.nccl_socket_ifname

    if config.run.nccl_ib_hca:
        os.environ["NCCL_IB_HCA"] = config.run.nccl_ib_hca
    deepspeed.init_distributed()

    ########
    # 4. load model or previous checkpoints
    ########
    model, tokenizer = load_models_and_tokenizer(model_name=config.model.name,
                                                 model_dtype=config.model.dtype,
                                                 trust_remote_code=config.model.trust_remote_code,
                                                 attn_impl=config.model.attn_implementation,
                                                 rank=rank)
    # apply PEFT module if enabled
    if config.peft.use_peft:
        model = apply_peft_module(model=model, peft_config=config.peft, rank=rank)

        if rank == 0:
            model.print_trainable_parameters()

        # Catch misconfigured lora_target_modules early, if no params are trainable,
        # ds init will fail late or silently train nothing.
        num_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert num_trainable > 0, "PEFT produced zero trainable parameters. Check peft.lora_target_modules"

    ########
    # 5. Setup training and inference engines
    ########
    if config.model.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled")
        model.gradient_checkpointing_enable()

        # With gradient checkpointing + peft/lora, pytorch may require that at least
        # one input to each checkpointed block has requires_grad=True. When the base model
        # is frozen which is the case in lora (i.e.,requires_grad=False), it causes backward to
        # fail or skip grads. Hence, we need to force the inputs to require grad so lora params
        # inside checkpointed blocks still receive gradients.
        if config.peft.use_peft and config.peft.peft_type == "lora":
            model.enable_input_require_grads()

    model_engine, optimizer = create_training_engine(deepspeed_config=config.deepspeed, model=model)
    is_zero3 = config.deepspeed.zero_optimization.get("stage", 0) == 3
    engine_ga_steps_value = getattr(model_engine, "gradient_accumulation_steps")
    ga_steps = int(engine_ga_steps_value() if callable(engine_ga_steps_value) else engine_ga_steps_value)
    if ga_steps != config.train.gradient_accumulation_steps:
        raise RuntimeError(f"DeepSpeed gradient_accumulation_steps ({ga_steps}) does not match config "
                           f"({config.train.gradient_accumulation_steps}).")

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

    # With ZeRO-3, model parameters are partitioned across GPUs and only gathered temporarily
    # into gpu memory during a forward/backward pass. Peak memory scales with batch size.
    # During training, ds pre-allocates memory for train_batch_size_per_gpu.
    # A larger val batch gathers the same parameters but processes more samples simultaneously,
    # requiring more activation memory which can cause oom.
    if is_zero3 and config.train.val_batch_size_per_gpu > config.train.train_batch_size_per_gpu:
        logger.warning(f"val_batch_size_per_gpu ({config.train.val_batch_size_per_gpu}) > "
                       f"train_batch_size_per_gpu ({config.train.train_batch_size_per_gpu}) with ZeRO stage 3. "
                       f"This may cause OOM during validation. The warning is just a heads-up and this won't always OOM. "
                       f"Consider setting val_batch_size_per_gpu <= train_batch_size_per_gpu.")

    ########
    # 7. Initiate the learning algorithm
    ########
    alg_class = load_algorithm(config.train.alg_name, Algorithm_Registry)
    alg = alg_class(model_engine=model_engine,
                    optimizer=optimizer,
                    normalize_loss=config.train.normalize_loss,
                    world_size=world_size)

    ########
    # 8. Variable initialization
    ########
    if rank == 0:
        print("Starting training...")

    micro_batches_per_epoch = config.train.micro_batches_per_epoch
    optimizer_steps_per_epoch = micro_batches_per_epoch // ga_steps

    # if micro_batches_per_epoch is not divisible by gradient_accumulation_steps
    if micro_batches_per_epoch % ga_steps != 0:
        remainder = micro_batches_per_epoch % ga_steps
        # raising error to enforce correctness
        raise ValueError(f"micro_batches_per_epoch ({micro_batches_per_epoch}) MUST be divisible by "
                         f"gradient_accumulation_steps ({ga_steps}) to ensure "
                         "all gradients are applied within the epoch boundaries. "
                         f"Adjust configuration. Remainder: {remainder}")

    ########
    # 9. Resume from checkpoint
    ########
    start_epoch = 0
    global_step = 0
    if args.resume_from:
        zero_stage = config.deepspeed.zero_optimization.get("stage", 0)
        start_epoch, global_step = resume_from_checkpoint(resume_path=args.resume_from,
                                                          model_engine=model_engine,
                                                          world_size=world_size,
                                                          logger=logger,
                                                          zero_stage=zero_stage,
                                                          model_dtype=config.model.dtype,
                                                          use_peft=config.peft.use_peft)

    experiment_dir = os.path.join(config.run.checkpoint_dir, config.run.experiment_id)
    cleanup_incomplete_checkpoints(experiment_dir=experiment_dir, rank=rank, logger=logger)

    # Use ds_numel for ZeRO-3 partitioned params, numel() for non-ZeRO
    total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in model_engine.module.parameters())
    trainable_params = sum(getattr(p, 'ds_numel', p.numel()) for p in model_engine.module.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info("=" * 50)
    logger.info(f"Starting training: {config.train.total_number_of_epochs} epochs")
    logger.info(f"Train set: {len(train_dataloader.dataset)} samples | micro_batches/epoch={micro_batches_per_epoch} | "
                f"optimizer_steps/epoch={optimizer_steps_per_epoch} | grad_accum={ga_steps}")
    logger.info(f"batch_size_per_gpu={config.train.train_batch_size_per_gpu} | "
                f"global_batch_size={config.train.train_batch_size_per_gpu * ga_steps * world_size}")

    if config.peft.use_peft:
        logger.info(f"Model: {config.model.name} | PEFT: {config.peft.peft_type} | "
                    f"params: {total_params:,} total, {trainable_params:,} peft ({100*trainable_params/total_params:.2f}%), "
                    f"{frozen_params:,} frozen | checkpoint_save_interval: {checkpoint_save_interval}")

    else:
        logger.info(f"Model: {config.model.name} | PEFT: off | "
                    f"params: {total_params:,} total, {trainable_params:,} trainable | "
                    f"checkpoint_save_interval: {checkpoint_save_interval}")

    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from} (epoch {start_epoch + 1}/{config.train.total_number_of_epochs})")
    logger.info("=" * 50)

    ########
    # 10. Training and evaluation loop
    ########
    # Sync before starting
    # Ensure all nodes have loaded the model and data before anyone starts iterating
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    training_start_time = time.time()
    for epoch in range(start_epoch, config.train.total_number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == config.train.total_number_of_epochs - 1)

        ########
        # 11 Training loop
        ########
        train_sampler.set_epoch(epoch)
        # Ensure gradients are zeroed at the start of epoch to prevent any bleeding from previous epoch
        # if accumulation steps were not perfectly aligned (though we enforce alignment above).
        model_engine.train()
        model_engine.zero_grad()

        if rank == 0:
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.train.total_number_of_epochs}", disable=(rank != 0))
        else:
            progress_bar = train_dataloader

        epoch_loss_sum    = torch.tensor(0.0, device=model_engine.device)
        epoch_token_count = torch.tensor(0.0, device=model_engine.device)
        data_iter = iter(progress_bar)

        for window_idx in range(optimizer_steps_per_epoch):
            ########
            # 11.1 Calculate global token count across all gpus for loss normalization.
            ########
            window_cpu = []
            local_token_count = 0.0
            for _ in range(ga_steps):
                # Collect one GA window on cpu
                #  we do not move to gpu yet to avoid holding ga_steps batches in vram.
                mb = next(data_iter)
                local_token_count += mb['loss_mask'].sum().item()
                window_cpu.append(mb)

            # All-reduce to get global token count across all gpus.
            # We need this because different ranks may have different number of tokens.
            # This ensures correct normalization regardless of token distribution across ranks.
            ga_denom_tensor = torch.tensor(local_token_count, device=model_engine.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(ga_denom_tensor, op=torch.distributed.ReduceOp.SUM)

            ga_denom = ga_denom_tensor.item()
            if ga_denom <= 0:
                raise RuntimeError(f"Invalid global GA token denominator: {ga_denom}")

            ########
            # 11.2 Run train step for each micro-batch with correct global normalization.
            ########
            window_loss_sum = torch.tensor(0.0, device=model_engine.device)
            for mb in window_cpu:
                micro_batch = {k: v.to(model_engine.device) for k, v in mb.items()}
                metric      = alg.train_step(micro_batch, ga_denom=ga_denom, ga_steps=ga_steps)
                window_loss_sum   += metric['loss_sum']
                epoch_loss_sum    += metric['loss_sum']
                epoch_token_count += metric['num_tokens']

            global_step += 1

            ########
            # 11.3 Logging at GA boundary and report the global per-token mean over the full GA window.
            ########
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(window_loss_sum, op=torch.distributed.ReduceOp.SUM)

            if rank == 0:
                per_token_loss = window_loss_sum.item() / ga_denom
                progress_bar.set_postfix(loss=per_token_loss)

                if tracker:
                    current_lr = optimizer.param_groups[0]['lr']
                    tracker.log_metrics({"train/loss": per_token_loss,
                                         "train/lr": current_lr,
                                         }, step=global_step)

        ########
        # 12 Sync before validation to ensure consistent state
        ########
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(epoch_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(epoch_token_count, op=torch.distributed.ReduceOp.SUM)

        epoch_time     = time.time() - epoch_start_time
        avg_train_loss = (epoch_loss_sum / epoch_token_count).item() if epoch_token_count.item() > 0 else 0.0
        gpu_mem_gb     = torch.cuda.max_memory_allocated(model_engine.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        logger.info(f"Epoch {epoch+1}/{config.train.total_number_of_epochs} completed in {epoch_time:.2f}s | "
                    f"avg_train_loss: {avg_train_loss:.4f} | global_step: {global_step} | "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e} | gpu_peak_mem: {gpu_mem_gb:.2f}GB")

        # Clear graph and to reclaim fragmented memory from training ONCE per epoch
        gc.collect()
        torch.cuda.empty_cache()

        ########
        # 13. Validation loop
        ########
        # to be safe and caculate loss average across batches and across GPUs correctly, we use
        # the following method: sum total loss and total tokens, then divide.
        local_loss_sum    = torch.tensor(0.0, device=model_engine.device)
        local_token_count = torch.tensor(0.0, device=model_engine.device)

        val_start_time = time.time()
        val_iter = tqdm(val_dataloader, desc="Validation", disable=(rank != 0))
        model_engine.eval()
        with torch.no_grad():
            for data in val_iter:
                val_batch = {k: v.to(model_engine.device) for k, v in data.items()}
                val_metric = alg.eval_step(val_batch)
                local_loss_sum += float(val_metric['loss_sum'])
                local_token_count += float(val_metric['num_tokens'])

        # Aggregate across all ranks.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(local_token_count, op=torch.distributed.ReduceOp.SUM)

        # Avoid division by zero
        if local_token_count.item() == 0:
            global_avg_loss = 0.0

        else:
            global_avg_loss = (local_loss_sum / local_token_count).item()

        val_time = time.time() - val_start_time
        logger.info(f"Epoch {epoch+1}, val_loss: {global_avg_loss:.4f} | val_time: {val_time:.2f}s")
        if rank == 0:
            if tracker:
                tracker.log_metrics({"val/loss": global_avg_loss}, step=global_step)

        ########
        # 14. Save checkpoint
        ########
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        tag = f"iter{epoch+1:06d}"
        model_path = get_experiment_dir_name(output_dir=config.run.checkpoint_dir, tag=tag, experiment_id=config.run.experiment_id)
        should_save = is_last_epoch or (checkpoint_save_interval > 0 and (epoch + 1) % checkpoint_save_interval == 0)
        if should_save:
            logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")
            save_training_checkpoint(epoch=epoch,
                                     global_step=global_step,
                                     model_engine=model_engine,
                                     tokenizer=tokenizer,
                                     model_path=model_path,
                                     peft_config=config.peft,
                                     rank=rank,
                                     world_size=world_size,
                                     logger=logger,
                                     label="sl",
                                     zero_stage=config.deepspeed.zero_optimization.get("stage", 0),
                                     model_dtype=config.model.dtype)

    total_training_time = time.time() - training_start_time
    logger.info(f"Training completed successfully! Total time: {total_training_time:.2f}s ({total_training_time/3600:.2f}h)")

    # End experiment tracker run
    if tracker:
        tracker.finish()

    ########
    # 15. Clean shutdown
    ########
    # release ds engine and cuda resources before tearing down nccl.
    del model_engine
    del optimizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    # use os._exit to skip python atexit handlers that may trigger segv
    # during final garbage collection of remaining cuda/nccl objects.
    os._exit(0)