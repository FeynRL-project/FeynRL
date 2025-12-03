import random
import numpy as np
import argparse
from transformers import AutoModel, AutoTokenizer

# local imports
import config.load as cfg
from config.load import load_and_verify


def setup_model(config):
    model = AutoModel.from_pretrained(config.model.name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    return model, tokenizer

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./config/dummy.yaml", help="config file")
    args = parser.parse_args()
 
    ########
    # 1. Generic setup (e.g., random seed, device, world size, etc.)
    ########

    ########
    # 2. Load config
    ########
    config = cfg.load_and_verify(args.config_file)

    ########
    # 3. Logging and saving (e.g., W&B, results dir, etc.)
    ######  ##

    ########
    # 4. Build env or data loader
    ########

    ########
    # 5. load model or previous checkpoints
    ########

    ########
    # 6. Setup trainiing and inference engines
    ########

    ########
    # 7. Intitate the learning algorithm (e.g., ppo)
    ########
    
    ########
    # 8. Initialize optimizer and scheduler
    ########

    ########
    # 9. Training and evaluation loop
    ########

    ########
    # 10. Save final checkpoint
    ########

    

