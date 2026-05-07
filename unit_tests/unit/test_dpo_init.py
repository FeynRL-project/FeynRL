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
    
    # Mock forward to return 3 values: (logprobs [2B, T-1], ref_logprobs [2B, T-1], loss_mask [2B, T-1])
    dpo.forward = MagicMock(return_value=(torch.zeros(2, 4), torch.zeros(2, 4), torch.ones(2, 4)))
    dpo.compute_loss = MagicMock(return_value=(torch.tensor(1.0, requires_grad=True), {'loss': 1.0}))

    micro_batch = {
        'input_ids': torch.zeros(1, 2, 5, dtype=torch.long),
        'attn_mask': torch.ones(1, 2, 5),
        'loss_mask': torch.ones(1, 2, 4),
    }

    metrics = dpo.train_step(micro_batch)

    assert metrics['loss'] == 1.0
    assert model_engine.backward.called
    assert model_engine.step.called

def test_dpo_eval_step():
    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()

    dpo = DPO(
        model_engine=model_engine,
        ref_model_engine=ref_model_engine,
        optimizer=optimizer,
        beta=0.1,
    )

    # Mock forward to return 3 values: (logprobs [2B, T-1], ref_logprobs [2B, T-1], loss_mask [2B, T-1])
    dpo.forward = MagicMock(return_value=(torch.zeros(2, 4), torch.zeros(2, 4), torch.ones(2, 4)))

    # eval_step returns per-sample tensors (dict of [B]) so the val loop can
    # mask DistributedSampler padding duplicates exactly.
    expected = {'loss':              torch.tensor([0.7]),
                'chosen_rewards':    torch.tensor([0.5]),
                'rejected_rewards':  torch.tensor([-0.3]),
                'reward_accuracies': torch.tensor([1.0])}
    dpo.compute_per_sample_loss_and_metrics = MagicMock(return_value=expected)

    micro_batch = {
        'input_ids': torch.zeros(1, 2, 5, dtype=torch.long),
        'attn_mask': torch.ones(1, 2, 5),
        'loss_mask': torch.ones(1, 2, 4),
    }

    per_sample = dpo.eval_step(micro_batch)

    # Per-sample contract: each metric is a [B] tensor, not a Python float.
    assert torch.equal(per_sample['loss'], expected['loss'])
    assert torch.equal(per_sample['reward_accuracies'], expected['reward_accuracies'])
    assert per_sample['loss'].shape == (1,)
    # eval_step should NOT call backward or step
    assert not model_engine.backward.called
    assert not model_engine.step.called
