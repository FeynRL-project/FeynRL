import random
import numpy as np
import argparse
import deepspeed
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import DataLoader, DistributedSampler

# local imports
import config.load as cfg
import datasets as dh

def set_random_seeds(config):
    '''
        Set random seeds, etc., for reproducibility.
    '''
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

def load_models_and_tokenizer(config: cfg.Config):
    '''
        Load models and tokenizer from huggingface.
        It also loads the ref model if provided.
        This fucntion would be resposible to make sure with use correct precision.
    '''
    assert config.model.dtype != 'auto', "dtype must not be auto to avoid any precision issues"
    model = AutoModelForCausalLM.from_pretrained(config.model.name,
                                                torch_dtype=config.model.dtype)
    # if ref model is provided to use it in kl for example.
    if config.model.ref_model is not None:
        ref_model = AutoModelForCausalLM.from_pretrained(config.model.ref_model,
                                                         torch_dtype=config.model.dtype)
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)

    if tokenizer.pad_token_id is None:
        # if pad token is not present, we use eos token as pad token
        print("Warning: Pad token is not present, using eos token as pad token")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, ref_model, tokenizer  

def training_engine_setup(config, model, ref_model=None):
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
                                                        config=config.deepspeed
                                                        )
    ref_model_engine = None
    if ref_model is not None:
        ref_model_engine, *_ = deepspeed.initialize(
                                             model=ref_model,
                                             config=config.deepspeed
                                            )

    return model_engine, ref_model_engine, optimizer

def inference_engine_setup(config, model):
    pass

def data_loader_setup(config, tokenizer):
    '''
       This function is responsible for setting up data loader.
    '''
    dataset = dh.DummyDataset(tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            sampler=sampler)

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

    

