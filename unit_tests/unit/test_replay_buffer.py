import torch
import pytest
from rollouts.replay_buffer import ReplayBuffer

def test_replay_buffer_add_batch_seqs():
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    
    # Example samples
    sample1 = {
        "response_len": 5,
        "input_ids": torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0]),
        "pred_rewards": torch.randn(10),
        "pred_zscores": torch.randn(10),
        "pred_masks": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pred_dones": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pred_old_logprobs": torch.randn(10),
    }
    
    rb.add_batch_seqs([sample1])
    
    assert len(rb) == 1
    assert rb.total_action_tokens == 5

def test_replay_buffer_collate_fn():
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    
    # Two sequences of different lengths
    x1 = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attn_masks": torch.tensor([1, 1, 1]),
        "old_logps": torch.tensor([0.1, 0.2, 0.3]),
        "masks": torch.tensor([1, 1, 1]),
        "rewards": torch.tensor([0.5, 0.5, 1.0]),
        "dones": torch.tensor([0, 0, 1]),
        "zscores": torch.tensor([0.0, 0.0, 0.0]),
    }
    x2 = {
        "input_ids": torch.tensor([4, 5]),
        "attn_masks": torch.tensor([1, 1]),
        "old_logps": torch.tensor([0.4, 0.5]),
        "masks": torch.tensor([1, 1]),
        "rewards": torch.tensor([0.6, 1.0]),
        "dones": torch.tensor([0, 1]),
        "zscores": torch.tensor([0.0, 0.0]),
    }
    
    batch_data = [x1, x2]
    collated = rb.collate_fn(batch_data)
    
    # Padded target_len should be 3
    assert collated['input_ids'].shape == (2, 3)
    assert collated['input_ids'][0].tolist() == [1, 2, 3]
    assert collated['input_ids'][1].tolist() == [4, 5, 0] # Padded with pad_token_id=0
    
    assert collated['mask'].shape == (2, 3)
    assert collated['mask'][1].tolist() == [1, 1, 0] # Padded with 0
    
    assert collated['done'].shape == (2, 3)
    assert collated['done'][1].tolist() == [0, 1, 0]

def test_replay_buffer_max_seq_len_truncation():
    max_seq_len = 5
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=max_seq_len)
    
    # Test longer sequences in add()
    # add() truncates to max_seq_len
    rb.add(
        input_ids=torch.arange(10),
        rewards=torch.randn(10),
        zscores=torch.randn(10),
        masks=torch.ones(10),
        dones=torch.zeros(10),
        old_logprobs=torch.randn(10)
    )
    
    item = rb[0]
    assert item['input_ids'].shape[0] == max_seq_len

def test_replay_buffer_reset():
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    rb.items.append({})
    rb.total_action_tokens = 100
    
    rb.reset()
    assert len(rb) == 0
    assert rb.total_action_tokens == 0

def test_replay_buffer_empty_batch():
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    with pytest.raises(ValueError, match="collate_fn received an empty batch"):
        rb.collate_fn([])
