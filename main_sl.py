import os
import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as torch_dist
from tqdm import tqdm
import gc

# imports local methods, classes, etc.
import config.load as cfg # all config arguments
from custom_datasets.paired_dataset import PairedDataset # our custom pytorch dataset

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

def rank_world_size_setup():
    '''
        Detect rank and world size from environment variables.
    '''
    # total number of gpus (e.g, 2 nodes x 4 gpus = 8 gpus in total). world size need to be at least 1
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Unique id of gpu in the ENTIRE WORLD. It ranges from 0 to world_size - 1
    rank = int(os.environ.get('RANK', 0))

    # Unique id of gpu in the LOCAL node. It ranges from 0 to local_node_size - 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank

def load_models_and_tokenizer(model_name,
                              model_dtype,
                              ref_model_name=None,
                              trust_remote_code=False,
                              attn_impl='',
                              model_class='llm',
                              rank=0):
    '''
        Load models and tokenizer from huggingface.
        It also loads the ref model if provided.
        This fucntion would be resposible to make sure with use correct precision
        and decides how to laod the model if it is a text-only model or multi-modal model.
    '''
    assert model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"
    assert attn_impl=='' or attn_impl in ['eager', 'flash_attention_2'], "attn_impl must be one of 'eager', 'flash_attention_2' or empty string"

    # 1. model and its config initialization
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=model_dtype,
                                                trust_remote_code=trust_remote_code,
                                                config=model_config,
                                                attn_implementation=None if attn_impl == '' else attn_impl)

    # if ref model is provided to use it in kl for example.
    if ref_model_name is not None:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name,
                                                         torch_dtype=model_dtype,
                                                         trust_remote_code=trust_remote_code,
                                                         config=model_config,
                                                         attn_implementation=None if attn_impl == '' else attn_impl)
    else:
        ref_model = None

    # 2. Tokenizer initialization
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    # log warning if pad token is not present.
    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Warning: Pad token is not present, using eos token as pad token")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, ref_model, tokenizer  

def training_engine_setup(deepspeed_config, model, ref_model=None):
    '''
        This function is responsible for setting up distributed training engine.
        For now, it only supports deepspeed.
    '''
    # Convert pydantic model to python Dict for DeepSpeed
    ds_config_dict = deepspeed_config.model_dump()

    # 1. Initialize distributed training engine
    deepspeed.init_distributed()

    # 2. Initialize model engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
                                                        model=model,
                                                        model_parameters=model.parameters(),
                                                        config=ds_config_dict
                                                        )
    ref_model_engine = None
    if ref_model is not None:
        ref_model_engine, _, _, _ = deepspeed.initialize(
                                                    model=ref_model,
                                                    config=ds_config_dict
                                                    )

    return model_engine, ref_model_engine, optimizer

def data_loader_setup(dnames,
                      dataset_ratios,
                      files_path,
                      num_workers,
                      max_seq_len,
                      prompt_key,
                      answer_key,
                      batch_size,
                      tokenizer,
                      seed,
                      split='train',
                      world_size=1,
                      rank=0):
    '''
       This function is responsible for setting up data loader.
       batch_size is an input to handle global or micro batch size.
    '''
    # 1. Initialize our custom datasets
    dataset = PairedDataset(prompt_key=prompt_key,
                            answer_key=answer_key,
                            max_seq_len=max_seq_len,
                            tokenizer=tokenizer,
                            data_path=files_path)
    shuffle = True if split == 'train' else False

    # For validation, it's usually better to be False to see all data, but if using DistributedSampler
    # with drop_last=False, it might pad with duplicates.
    drop_last = True if split == 'train' else False

    # 2. Initialize distributed sampler
    sampler = DistributedSampler(dataset,
                                shuffle=shuffle,
                                num_replicas=world_size,
                                rank=rank,
                                drop_last=drop_last)

    # 3. Initialize data loader
    dataloader = DataLoader(
                            dataset=dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=drop_last,
                            )

    return dataloader, sampler

def inference_engine_setup(deepspeed_config, model):
    '''
        This function is responsible for setting up distributed inference engine.
        For now, it only supports deepspeed.
    '''
    return None

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    args = parser.parse_args()

    ########
    # 1. Setup Environment
    ########
    rank, world_size, local_rank = rank_world_size_setup()

    ########
    # 2. Load config
    ########
    config = cfg.load_and_verify(input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 world_size=world_size,
                                 )
    set_random_seeds(seed=config.run.seed)
    ########
    # 3. Logging and saving (e.g., W&B, results dir, etc.)
    ########
    #TBA

    ########
    # 4. load model or previous checkpoints
    ########
    model, ref_model, tokenizer = load_models_and_tokenizer(model_name=config.model.name,
                                                            model_dtype=config.model.dtype,
                                                            ref_model_name=config.model.ref_model,
                                                            trust_remote_code=config.model.trust_remote_code,
                                                            model_class=config.model.model_class,
                                                            attn_impl=config.model.attn_implementation)

    ########
    # 5. Setup trainiing and inference engines
    ########
    model_engine, ref_model_engine, optimizer = training_engine_setup(deepspeed_config=config.deepspeed,
                                                                      model=model,
                                                                      ref_model=ref_model)

    if config.model.gradient_checkpointing:
        model_engine.gradient_checkpointing_enable()
    ########
    # 6. Build env or data loader
    ########
    train_dataloader, train_sampler = data_loader_setup(dnames=config.data.train_dnames,
                                                        dataset_ratios=config.data.train_ratios,
                                                        files_path=config.data.train_files_path,
                                                        num_workers=config.data.num_workers,
                                                        max_seq_len=config.data.max_seq_len,
                                                        prompt_key=config.data.prompt_key,
                                                        answer_key=config.data.answer_key,
                                                        batch_size=config.train.train_batch_size_per_gpu,
                                                        tokenizer=tokenizer,
                                                        seed=config.run.seed,
                                                        split='train',
                                                        world_size=world_size,
                                                        rank=rank)

    val_dataloader, _ = data_loader_setup(dnames=config.data.train_dnames,
                                          dataset_ratios=config.data.train_ratios,
                                          files_path=config.data.val_files_path,
                                          num_workers=config.data.num_workers,
                                          max_seq_len=config.data.max_seq_len,
                                          prompt_key=config.data.prompt_key,
                                          answer_key=config.data.answer_key,
                                          batch_size=config.train.val_batch_size_per_gpu,
                                          tokenizer=tokenizer,
                                          seed=config.run.seed,
                                          split='val',
                                          world_size=world_size,
                                          rank=rank)

    ########
    # 7. Intitate the learning algorithm (e.g., ppo)
    ########
    if str.lower(config.train.alg_name) in {'sft'}:
        if str.lower(config.train.alg_name) == 'sft':
            import algs.SFT.sft as calg
            alg = calg.SFT(
                           model_engine=model_engine,
                           optimizer=optimizer,
                           device=model_engine.device,
                           use_cache=config.train.use_cache)

    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")

    ########
    # 8. Training and evaluation loop
    ########
    if rank == 0:
        print("Starting training...")

    global_step = 0

    # Sync before starting
    # Ensure all nodes have loaded the model and data before anyone starts iterating
    torch.distributed.barrier()

    for epoch in range(config.train.total_number_of_epochs):
        train_sampler.set_epoch(epoch)

        ########
        # 8.1 Training loop
        ########
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.train.total_number_of_epochs}", disable=(rank != 0))

        for step, micro_batch in enumerate(progress_bar):
            # Move batch to gpu (deepspeed engine device)
            micro_batch = {k: v.to(model_engine.device) for k, v in micro_batch.items()}

            # Run one train step for micro-batch.
            metric = alg.train_step(micro_batch)
            global_step += 1

            # logging
            if rank == 0 and step % config.deepspeed.steps_per_print == 0:
                progress_bar.set_postfix(loss=metric['loss'])

        # Sync before validation to ensure consistent state
        torch.distributed.barrier()

        # Clear graph and to reclaim fragmented memory from training ONCE per epoch
        torch.cuda.empty_cache()
        gc.collect()

        ########
        # 8.2 Validation loop
        ########
        val_loss = 0.0
        num_val_batches = 0
        for data in val_dataloader:
            val_batch = {k: v.to(model_engine.device) for k, v in data.items()}
            val_metric = alg.eval_step(val_batch)
            val_loss += val_metric['loss']
            num_val_batches += 1

        # Average loss across batches and across GPUs
        avg_val_loss = torch.tensor(val_loss / max(1, num_val_batches)).to(model_engine.device)

        # Sync validation loss across GPUs
        torch.distributed.all_reduce(avg_val_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss.item()}")

        ########
        # 8.3 Save checkpoint
        ########
        # Sync before saving to ensure no one is still writing
        torch.distributed.barrier()

        tag = f"epoch_{epoch+1}"
        # DeepSpeed handles the collective saving internally so we don't need to worry about different ranks.
        model_engine.save_checkpoint(config.deepspeed.monitor_config.get("tensorboard", {}).get("output_path", "./checkpoints"), tag=tag)

        # Wait for saving to complete on all ranks
        torch.distributed.barrier()

    if rank == 0:
        print("Training complete.")