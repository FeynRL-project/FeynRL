import torch
import pytest
from misc.utils import safe_string_to_torch_dtype, ensure_1d, pad_1d_to_length

def test_safe_string_to_torch_dtype():
    assert safe_string_to_torch_dtype("fp16") == torch.float16
    assert safe_string_to_torch_dtype("bf16") == torch.bfloat16
    assert safe_string_to_torch_dtype("fp32") == torch.float32
    assert safe_string_to_torch_dtype(torch.float64) == torch.float64
    assert safe_string_to_torch_dtype(None) is None
    with pytest.raises(ValueError, match="Unsupported model_dtype"):
        safe_string_to_torch_dtype("int8")

def test_ensure_1d():
    x = torch.zeros(5)
    assert ensure_1d(x, "x") is x
    
    y = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="Expected y to be 1D"):
        ensure_1d(y, "y")

def test_pad_1d_to_length():
    x = torch.tensor([1.0, 2.0])
    # Pad
    padded = pad_1d_to_length(x, pad_value=0.0, target_len=4)
    assert torch.equal(padded, torch.tensor([1.0, 2.0, 0.0, 0.0]))
    
    # Truncate
    truncated = pad_1d_to_length(x, pad_value=0.0, target_len=1)
    assert torch.equal(truncated, torch.tensor([1.0]))
    
    # Same
    same = pad_1d_to_length(x, pad_value=0.0, target_len=2)
    assert torch.equal(same, x)
