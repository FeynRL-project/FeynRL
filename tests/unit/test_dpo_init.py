import torch
import pytest
from unittest.mock import MagicMock
from algs.DPO.dpo import DPO

def test_dpo_init():
    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()
    
    dpo = DPO(
        model_engine=model_engine,
        ref_model_engine=ref_model_engine,
        optimizer=optimizer,
        beta=0.1,
    )
    
    assert dpo.beta == 0.1
    assert dpo.model_engine == model_engine
    assert dpo.ref_model_engine == ref_model_engine
    assert ref_model_engine.eval.called

def test_dpo_train_step():
    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()
    
    dpo = DPO(
        model_engine=model_engine,
        ref_model_engine=ref_model_engine,
        optimizer=optimizer,
        beta=0.1,
    )
    
    # Mock forward and compute_loss
    dpo.forward = MagicMock(return_value=(torch.zeros(2, 4, 10), torch.zeros(2, 4), torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 4)))
    dpo.compute_loss = MagicMock(return_value=(torch.tensor(1.0, requires_grad=True), {'loss': 1.0}))
    
    micro_batch = {
        'input_ids': torch.zeros(1, 2, 5, dtype=torch.long),
        'attn_mask': torch.ones(1, 2, 5),
        'loss_mask': torch.ones(1, 2, 4),
    }
    
    metrics = dpo.train_step(micro_batch)
    
    assert metrics['loss'] == 1.0
    assert model_engine.train.called
    assert model_engine.backward.called
    assert model_engine.step.called
