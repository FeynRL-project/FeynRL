import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from algs.PPO.value_net import ValueNetwork

class MockBackbone(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.param = nn.Parameter(torch.randn(1)) # to have a device/dtype
        
    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        B, T = input_ids.shape
        hidden_dim = self.config.hidden_size
        return SimpleNamespace(last_hidden_state=torch.randn(B, T, hidden_dim, device=self.param.device, dtype=self.param.dtype))

def test_value_network_init():
    hidden_size = 128
    base_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size),
        model=MockBackbone(hidden_size)
    )
    
    vn = ValueNetwork(base_model)
    assert isinstance(vn.value_head, nn.Linear)
    assert vn.value_head.in_features == hidden_size
    assert vn.value_head.out_features == 1
    assert torch.all(vn.value_head.weight == 0)

def test_value_network_init_transformer_attr():
    hidden_size = 64
    base_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size),
        transformer=MockBackbone(hidden_size)
    )
    
    vn = ValueNetwork(base_model)
    assert vn.backbone == base_model.transformer

def test_value_network_invalid_init():
    base_model = SimpleNamespace(config=SimpleNamespace(hidden_size=64))
    with pytest.raises(ValueError, match="Cannot find backbone"):
        ValueNetwork(base_model)

def test_value_network_forward():
    hidden_size = 32
    B, T = 2, 8
    backbone = MockBackbone(hidden_size)
    base_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size),
        model=backbone
    )
    
    vn = ValueNetwork(base_model)
    input_ids = torch.zeros(B, T, dtype=torch.long)
    
    output = vn(input_ids)
    assert hasattr(output, 'logits')
    assert output.logits.shape == (B, T, 1)

def test_value_network_delegation():
    hidden_size = 16
    backbone = MockBackbone(hidden_size)
    backbone.gradient_checkpointing_enable = MagicMock()
    backbone.enable_input_require_grads = MagicMock()
    
    base_model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden_size),
        model=backbone
    )
    
    vn = ValueNetwork(base_model)
    vn.gradient_checkpointing_enable()
    backbone.gradient_checkpointing_enable.assert_called_once()
    
    vn.enable_input_require_grads()
    backbone.enable_input_require_grads.assert_called_once()

from unittest.mock import MagicMock
