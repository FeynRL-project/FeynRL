import torch
import pytest
import os
from unittest.mock import MagicMock, patch
from algs.RL.common import COMMON

def test_save_checkpoint_logic(tmp_path):

    # Create a dummy object
    dummy_self = SimpleNamespace()
    dummy_self.alg_name = "test"
    dummy_self.policy_engine = MagicMock()
    dummy_self.peft_config = SimpleNamespace(use_peft=False)
    
    # Mock distributed
    import torch.distributed as dist
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.barrier'):
        
        output_dir = str(tmp_path / "policy")
        COMMON.save_checkpoint(dummy_self, output_dir, "tag_v1")
        
        # Verify save_16bit_model was called
        dummy_self.policy_engine.save_16bit_model.assert_called_once_with(output_dir)

def test_save_checkpoint_peft(tmp_path):

    # Create a dummy object
    dummy_self = SimpleNamespace()
    dummy_self.alg_name = "test"
    dummy_self.policy_engine = MagicMock()
    dummy_self.policy_engine.module = MagicMock()
    dummy_self.peft_config = SimpleNamespace(use_peft=True, lora_alpha=32, lora_rank=8)
    
    # Mocking _merge_peft_state_dict
    dummy_self._merge_peft_state_dict = MagicMock(return_value={"weight": torch.tensor([1.0])})
    
    # Mock deepspeed.zero.GatheredParameters
    import deepspeed
    
    import torch.distributed as dist
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.barrier'), \
         patch('algs.RL.common.save_file') as mock_save:
        
        output_dir = str(tmp_path / "peft_policy")
        os.makedirs(output_dir, exist_ok=True)
        COMMON.save_checkpoint(dummy_self, output_dir, "tag_peft")
        
        mock_save.assert_called_once()
        dummy_self._merge_peft_state_dict.assert_called_once()

from types import SimpleNamespace
