import torch
import pytest
from unittest.mock import MagicMock
from algs.SFT.sft import SFT

def test_sft_init():
    model_engine = MagicMock()
    optimizer = MagicMock()
    
    sft = SFT(
        model_engine=model_engine,
        optimizer=optimizer,
        normalize_loss=True
    )
    
    assert sft.model_engine == model_engine
    assert sft.optimizer == optimizer
    assert sft.normalize_loss is True

def test_sft_train_step():
    model_engine = MagicMock()
    optimizer = MagicMock()
    
    sft = SFT(
        model_engine=model_engine,
        optimizer=optimizer,
    )
    
    # Mock forward and compute_loss
    sft.forward = MagicMock(return_value=(torch.zeros(1, 4, 10), torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4)))
    sft.compute_loss = MagicMock(return_value=(torch.tensor(1.0, requires_grad=True), 1.0, 4.0))
    
    micro_batch = {
        'input_ids': torch.zeros(1, 5, dtype=torch.long),
        'attn_mask': torch.ones(1, 5),
        'loss_mask': torch.ones(1, 4),
    }
    
    metrics = sft.train_step(micro_batch)
    
    assert metrics['loss'] == 1.0
    assert model_engine.train.called
    assert model_engine.backward.called
    assert model_engine.step.called
