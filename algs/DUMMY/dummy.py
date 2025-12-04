import torch
import numpy as np

class Dummy:
    def __init__(self,
                model_config,
                model_engine,
                ref_model_engine,
                optimizer,
                dataloader,
                device="cpu"):
        self.model_config = model_config
        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.optimizer = optimizer
        self.device = device

    
    def train(self,
            dataloader,
            optimizer,
            dataloader,
            inference_engine):
        for epoch in range(self.model_config.train.epochs):
            for batch in dataloader:
                