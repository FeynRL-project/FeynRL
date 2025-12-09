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

# imports local methods, classes, etc.
import config.load as cfg # all config arguments
from custom_datasets.paired_dataset import PairedDataset # our custom pytorch dataset
from misc.utils import save_checkpoint

def set_random_seeds(seed):
    '''
        Set random seeds, etc., for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rank_world_size_setup(world_size):
    '''
        Set rank and world size for distributed training.
    '''
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def batch_size_setup(train_batch_size_per_gpu, gradient_accumulation, val_batch_size_per_gpu, world_size):
    '''
        Set batch size for each gpu/rank.
        This function helps to avoid having to do this in multiple places especially
        with distributed training engine sometime causes confusion.
        (global) train_batch_size = micro_batch_size x gradient_accumulation x number_of_gpus
    '''
    bsz = {}
    global_train_batch_size = train_batch_size_per_gpu * gradient_accumulation * world_size
    bsz['train_global'] = global_train_batch_size
    bsz['train_local']  = train_batch_size_per_gpu
    global_val_batch_size = val_batch_size_per_gpu * world_size
    bsz['val_global']   = global_val_batch_size
    bsz['val_local']    = val_batch_size_per_gpu

    # update related arguments in deepspeed config
    return bsz

def load_models_and_tokenizer(model_name,
                              model_dtype,
                              ref_model_name=None,
                              trust_remote_code=False,
                              model_class='llm'):
    '''
        Load models and tokenizer from huggingface.
        It also loads the ref model if provided.
        This fucntion would be resposible to make sure with use correct precision
        and decides how to laod the model if it is a text-only model or multi-modal model.
    '''
    assert model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"

    ########
    # 1. model and its config initialization
    ########
    model_config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                torch_dtype=model_dtype,
                                                trust_remote_code=trust_remote_code,
                                                config=model_config)

    # if ref model is provided to use it in kl for example.
    if ref_model_name is not None:
        ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name,
                                                         torch_dtype=model_dtype,
                                                         trust_remote_code=trust_remote_code,
                                                         config=model_config)
    else:
        ref_model = None

    ########
    # 2. Tokenizer initialization
    ########
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    # log warning if pad token is not present.
    if tokenizer.pad_token_id is None:
        print("Warning: Pad token is not present, using eos token as pad token")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, ref_model, tokenizer  

def training_engine_setup(deepspeed_config, model, ref_model=None):
    '''
        This function is responsible for setting up distributed training engine.
        For now, it only supports deepspeed.
    '''
    ########
    # 1. Initialize distributed training engine
    ########
    deepspeed.init_distributed()

    ########
    # 2. Initialize model engine
    ########
    model_engine, optimizer, _, _ = deepspeed.initialize(
                                                        model=model,
                                                        model_parameters=model.parameters(),
                                                        config=deepspeed_config
                                                        )
    ref_model_engine = None
    if ref_model is not None:
        ref_model_engine, *_ = deepspeed.initialize(
                                                    model=ref_model,
                                                    config=deepspeed_config
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
    ########
    # 1. Initialize our custom datasets
    ########
    dataset = PairedDataset(prompt_key=prompt_key,
                            answer_key=answer_key,
                            max_seq_len=max_seq_len,
                            tokenizer=tokenizer,
                            data_path=files_path)
    shuffle = True if split == 'train' else False

    ########
    # 2. Initialize distributed sampler
    ########
    sampler = DistributedSampler(dataset,
                                shuffle=shuffle,
                                num_replicas=world_size,
                                rank=rank,
                                drop_last=True)

    ########
    # 3. Initialize data loader
    ########
    dataloader = DataLoader(
                            dataset=dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=data_config.num_workers,
                            pin_memory=True,
                            drop_last=True,
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
    parser.add_argument("--world_size", type=int, default=1, help="world size")

    args = parser.parse_args()
 
    ########
    # 1. Load config
    ########
    config = cfg.load_and_verify(input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 world_size=args.world_size)

    ########
    # 2. Generic setup (e.g., random seed, device, world size, etc.)
    ########
    set_random_seeds(seed=config.run.seed)
    rank, world_size = rank_world_size_setup(world_size=config.run.world_size)
    batch_sizes = batch_size_setup(train_batch_size_per_gpu=config.train.train_batch_size_per_gpu,
                                   gradient_accumulation=config.train.gradient_accumulation_steps,
                                   val_batch_size_per_gpu=config.train.val_batch_size_per_gpu,
                                   world_size=world_size)

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
                                                            model_class=config.model.model_class)

    ########
    # 5. Setup trainiing and inference engines
    ########
    model_engine, ref_model_engine, optimizer = training_engine_setup(deepspeed_config=config.deepspeed,
                                                                      model=model,
                                                                      ref_model=ref_model)
    inference_engine = inference_engine_setup(config, model)

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
                                                        batch_size=batch_sizes['train_local'],
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
                                          batch_size=batch_sizes['val_local'],
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
                           micro_batch_size_per_gpu=config.train.micro_batch_size_per_gpu,
                           clip_grad_norm=config.train.clip_grad_norm,
                           use_cache=config.train.use_cache,
                           device='cpu')

    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")

    ########
    # 8. Training and evaluation loop
    ########
    for epoch in range(config.train.num_epochs):
        train_sampler.set_epoch(epoch)
        ########
        # 8.1 Training loop
        ########
        total_train_step = 0
        for data in tqdm(train_dataloader,
                        total=config.train.steps_per_epoch,
                        desc=f"Epoch {epoch + 1}/{config.train.num_epochs}"):
            total_train_step += 1
            metric = alg.train_step(data)
            if total_train_step % config.train.log_interval == 0:
                print(f"Global step: {total_train_step}, Loss: {metric['loss']}")

        ########
        # 8.2 Validation loop
        ########
        total_eval_step = 0
        for data in tqdm(val_dataloader,
                        total=config.val.steps_per_epoch,
                        desc=f"Epoch {epoch + 1}/{config.train.num_epochs}"):
            total_eval_step += 1
            metric = alg.eval_step(data)
            if total_eval_step % config.val.log_interval == 0:
                print(f"Global step: {global_step}, Loss: {metric['loss']}")

        ########
        # 8.3 Save checkpoint
        ########
        if epoch % config.train.save_interval == 0:
            save_checkpoint(config, model, optimizer, epoch)

    ########
    # 9. Save final checkpoint
    ########
    save_checkpoint(config, model, optimizer, config.train.num_epochs)

    

