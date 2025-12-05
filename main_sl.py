import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader, DistributedSampler

# local imports
import config.load as cfg # all config arguments
import datasets    as ds  # our custom pytorch dataset

def set_random_seeds(seed):
    '''
        Set random seeds, etc., for reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

def inference_engine_setup(deepspeed_config, model):
    '''
        This function is responsible for setting up distributed inference engine.
        For now, it only supports deepspeed.
    ''' 
    return None

def data_loader_setup(data_config, batch_size, tokenizer, split='train', world_size=1, rank=0):
    '''
       This function is responsible for setting up data loader.
       batch_size is an input to handle global or micro batch size.
    '''
    ########
    # 1. Initialize our custom datasets
    ########
    dataset = ds.PairedDataset(prompt_key=data_config.prompt_key,
                                    answer_key=data_config.answer_key,
                                    max_seq_len=data_config.max_seq_len,
                                    tokenizer=tokenizer,
                                    data_path=data_config.data_path)
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

    return dataloader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    parser.add_argument("--run_id", type=str, default="run_1", help="run id")
    parser.add_argument("--world_size", type=int, default=1, help="world size")

    args = parser.parse_args()
 
    ########
    # 1. Load config
    ########
    config = cfg.load_and_verify(args.config_file)

    ########
    # 2. Generic setup (e.g., random seed, device, world size, etc.)
    ########
    set_random_seeds(config)

    ########
    # 3. Logging and saving (e.g., W&B, results dir, etc.)
    ########
    #TBA

    ########
    # 4. load model or previous checkpoints
    ########
    model, ref_model, tokenizer = load_models_and_tokenizer(config)

    ########
    # 5. Setup trainiing and inference engines
    ########
    model_engine, ref_model_engine, optimizer = training_engine_setup(config, model, ref_model)
    inference_engine = inference_engine_setup(config, model)

    ########
    # 6. Build env or data loader
    ########
    dataloader = data_loader_setup(config, tokenizer)    

    ########
    # 7. Intitate the learning algorithm (e.g., ppo)
    ########
    if str.lower(config.train.alg_name) in {'ppo', 'grpo', 'dummy'}:
        if str.lower(config.train.alg_name) == 'dummy':
            import algs.DUMMY.dummy as al
            alg = al.Dummy()

    else:
        raise ValueError(f"Unknown algorithm: {config.train.alg_name}")
    
    ########
    # 8. Training and evaluation loop
    ########
    alg.train(config, model_engine, ref_model_engine, optimizer, dataloader, inference_engine)

    ########
    # 9. Save final checkpoint
    ########

    

