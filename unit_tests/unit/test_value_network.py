import torch
import torch.nn as nn
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
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


# ---------------------------------------------------------------------------
# VLM: the value model must forward vision tensors to the backbone, and resolve
# hidden_size from text_config when the top-level config lacks it (e.g. Gemma-3).
# ---------------------------------------------------------------------------

class MockVLMBackbone(nn.Module):
    '''Backbone that accepts **mm_kwargs (like Qwen2VLModel) and records what it got.'''
    def __init__(self, hidden_size):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.param = nn.Parameter(torch.randn(1))
        self.received_mm = None

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **mm_kwargs):
        self.received_mm = mm_kwargs
        B, T = input_ids.shape
        return SimpleNamespace(last_hidden_state=torch.randn(B, T, self.config.hidden_size,
                                                             device=self.param.device, dtype=self.param.dtype))

def test_value_network_forward_passes_mm_kwargs():
    hidden_size = 32
    B, T = 2, 6
    backbone = MockVLMBackbone(hidden_size)
    base_model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size), model=backbone)
    vn = ValueNetwork(base_model)

    pixel_values = torch.zeros(12, 1176)
    image_grid_thw = torch.tensor([[1, 16, 16]], dtype=torch.long)
    out = vn(torch.zeros(B, T, dtype=torch.long),
             pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    assert out.logits.shape == (B, T, 1)
    # the vision tensors reached the backbone
    assert set(backbone.received_mm.keys()) == {"pixel_values", "image_grid_thw"}
    assert backbone.received_mm["pixel_values"].shape == (12, 1176)

def test_value_network_forward_llm_empty_mm_kwargs():
    '''Text-only call: backbone receives no vision kwargs (empty), forward still works.'''
    hidden_size = 16
    backbone = MockVLMBackbone(hidden_size)
    base_model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size), model=backbone)
    vn = ValueNetwork(base_model)
    out = vn(torch.zeros(2, 4, dtype=torch.long))
    assert out.logits.shape == (2, 4, 1)
    assert backbone.received_mm == {}

def test_value_network_hidden_size_from_text_config():
    '''VLMs like Gemma-3 nest hidden_size under text_config; the head must still size right.'''
    hidden_size = 48
    backbone = MockVLMBackbone(hidden_size)
    # top-level config has NO hidden_size -> must fall back to text_config.hidden_size
    base_model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden_size)),
        model=backbone,
    )
    vn = ValueNetwork(base_model)
    assert vn.value_head.in_features == hidden_size
