import torch
from algs.RL.common import COMMON


def _common(dtype=torch.bfloat16):
    # COMMON has no __init__ (the alg subclasses define it); extract_mm_kwargs only
    # needs self.model_dtype, so a bare instance with that attribute is enough.
    c = COMMON()
    c.model_dtype = dtype
    return c


def _text_batch():
    return {
        "input_ids": torch.zeros(2, 4, dtype=torch.long),
        "attn_mask": torch.ones(2, 4, dtype=torch.long),
        "mask": torch.ones(2, 4),
        "rewards": torch.zeros(2, 4),
        "zscore": torch.zeros(2, 4),
        "old_logprobs": torch.zeros(2, 4),
        "done": torch.zeros(2, 4),
        "batch_action_tokens": 7,       # scalar -> not a tensor, skipped
        "action_token_weight": 0.5,     # scalar -> not a tensor, skipped
    }


def test_extract_mm_kwargs_vlm_extracts_and_casts():
    '''
        VLM batch: only the vision tensors come out; pixel_values cast to model dtype,
        image_grid_thw (int) left as int64.
    '''
    c = _common(torch.bfloat16)
    mb = _text_batch()
    mb["pixel_values"] = torch.zeros(10, 1176, dtype=torch.float32)
    mb["image_grid_thw"] = torch.tensor([[1, 16, 16]], dtype=torch.long)

    mm = c.extract_mm_kwargs(mb, torch.device("cpu"))

    assert set(mm.keys()) == {"pixel_values", "image_grid_thw"}
    assert mm["pixel_values"].dtype == torch.bfloat16   # float -> compute dtype
    assert mm["image_grid_thw"].dtype == torch.long     # int kept as-is
    # text + scalar keys are excluded
    assert "input_ids" not in mm and "batch_action_tokens" not in mm


def test_extract_mm_kwargs_llm_returns_empty():
    '''
        Text-only batch -> {} so policy/ref/value forwards behave exactly as before.
    '''
    c = _common()
    assert c.extract_mm_kwargs(_text_batch(), torch.device("cpu")) == {}
