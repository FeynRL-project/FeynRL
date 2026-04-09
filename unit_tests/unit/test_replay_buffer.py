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
        "policy_version": 0,
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
        old_logprobs=torch.randn(10),
        policy_version=0,
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


# ---------------------------------------------------------------------------
# max_size: deque-backed hard cap for async mode
# ---------------------------------------------------------------------------

def _mk_sample(version, response_len=3):
    '''Minimal valid sample for add_batch_seqs.'''
    return {
        "response_len": response_len,
        "input_ids": torch.tensor([1, 2, 3, 4, 5, 0, 0, 0, 0, 0]),
        "pred_rewards": torch.zeros(10),
        "pred_zscores": torch.zeros(10),
        "pred_masks": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pred_dones": torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pred_old_logprobs": torch.zeros(10),
        "policy_version": version,
    }


def test_replay_buffer_default_max_size_is_unbounded_list():
    '''Default ctor (no max_size) should be a plain list, unbounded — preserves
    sync mode behavior bit-for-bit.'''
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    assert rb.max_size is None
    # Add many samples; nothing evicts
    for v in range(100):
        rb.add_batch_seqs([_mk_sample(version=v)])
    assert len(rb) == 100


def test_replay_buffer_max_size_caps_at_capacity():
    '''With max_size set, the deque auto-evicts oldest on insert past capacity.'''
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10, max_size=8)
    assert rb.max_size == 8
    for v in range(20):
        rb.add_batch_seqs([_mk_sample(version=v)])
    # Capacity enforced
    assert len(rb) == 8
    # The 8 retained are the 8 most recent (versions 12..19)
    versions = [item["policy_version"] for item in rb.items]
    assert versions == list(range(12, 20))


def test_replay_buffer_max_size_evict_stale_preserves_cap():
    '''After evict_stale, the buffer must remain a deque with the same maxlen.
    Otherwise the next add_batch_seqs would silently grow unbounded.'''
    from collections import deque
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10, max_size=10)
    for v in range(10):
        rb.add_batch_seqs([_mk_sample(version=v)])
    # Evict everything older than version 5
    evicted = rb.evict_stale(min_version=5)
    assert evicted == 5
    assert len(rb) == 5
    # Crucially: the deque maxlen is preserved
    assert isinstance(rb.items, deque)
    assert rb.items.maxlen == 10
    # And new inserts still respect the cap
    for v in range(20, 30):
        rb.add_batch_seqs([_mk_sample(version=v)])
    assert len(rb) == 10
    assert rb.items.maxlen == 10


def test_replay_buffer_max_size_reset_preserves_cap():
    '''reset() on a bounded buffer must rebuild as a deque with the same
    maxlen, not a plain list. (This case isn't hit by run_rl_async today
    but matches the deque-aware evict_stale behavior.)'''
    from collections import deque
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10, max_size=4)
    for v in range(4):
        rb.add_batch_seqs([_mk_sample(version=v)])
    rb.reset()
    assert len(rb) == 0
    assert isinstance(rb.items, deque)
    assert rb.items.maxlen == 4
    # And the cap still works after reset
    for v in range(10):
        rb.add_batch_seqs([_mk_sample(version=v)])
    assert len(rb) == 4


def test_replay_buffer_unbounded_reset_stays_list():
    '''reset() on an unbounded buffer (sync mode) must stay a plain list.'''
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)  # max_size=None
    rb.add_batch_seqs([_mk_sample(version=0)])
    rb.reset()
    assert isinstance(rb.items, list)
    assert len(rb) == 0
