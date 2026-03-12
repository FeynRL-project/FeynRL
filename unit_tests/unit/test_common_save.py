import torch
import torch.nn as nn
import pytest
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from algs.RL.common import COMMON

def test_save_checkpoint_logic(tmp_path):

    # Create a dummy object
    dummy_self = SimpleNamespace()
    dummy_self.alg_name = "test"
    dummy_self.policy_engine = MagicMock()
    dummy_self.peft_config = SimpleNamespace(use_peft=False)
    dummy_self.gather_params_for_save = MagicMock(return_value={"weight": torch.tensor([1.0])})
    dummy_self.save_state_dict_sharded = MagicMock()
    dummy_self.barrier_with_error_check = MagicMock()

    # Mock distributed
    import torch.distributed as dist
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.barrier'):

        output_dir = str(tmp_path / "policy")
        COMMON.save_checkpoint(dummy_self, output_dir, "tag_v1")

        # Verify gather + sharded save was called (replaces save_16bit_model)
        dummy_self.gather_params_for_save.assert_called_once()
        dummy_self.save_state_dict_sharded.assert_called_once()

def test_save_checkpoint_peft(tmp_path):

    # Create a dummy object
    dummy_self = SimpleNamespace()
    dummy_self.alg_name = "test"
    dummy_self.policy_engine = MagicMock()
    dummy_self.policy_engine.module = MagicMock()
    dummy_self.peft_config = SimpleNamespace(use_peft=True, lora_alpha=32, lora_rank=8)
    dummy_self.gather_params_for_save = MagicMock(return_value={"weight": torch.tensor([1.0])})

    # Mocking merge_peft_state_dict
    dummy_self.merge_peft_state_dict = MagicMock(return_value={"weight": torch.tensor([1.0])})
    dummy_self.save_state_dict_sharded = MagicMock()
    dummy_self.barrier_with_error_check = MagicMock()

    # Mock deepspeed.zero.GatheredParameters
    import deepspeed

    import torch.distributed as dist
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.barrier'):

        output_dir = str(tmp_path / "peft_policy")
        os.makedirs(output_dir, exist_ok=True)
        COMMON.save_checkpoint(dummy_self, output_dir, "tag_peft")

        dummy_self.save_state_dict_sharded.assert_called_once()
        dummy_self.merge_peft_state_dict.assert_called_once()


def test_barrier_with_error_check_single_process_failure():
    dummy_self = SimpleNamespace(
        alg_name="test",
        policy_engine=SimpleNamespace(device=torch.device("cpu")),
    )

    with patch('torch.distributed.is_initialized', return_value=False):
        with pytest.raises(RuntimeError, match="Checkpoint operation failed"):
            COMMON.barrier_with_error_check(dummy_self, succeeded=False)


def test_barrier_with_error_check_distributed_failure_propagates():
    dummy_self = SimpleNamespace(
        alg_name="test",
        policy_engine=SimpleNamespace(device=torch.device("cpu")),
    )

    def _force_failure(flag, op=None):
        flag.zero_()

    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.all_reduce', side_effect=_force_failure), \
         patch('torch.distributed.get_rank', return_value=3):
        with pytest.raises(RuntimeError, match="rank 3"):
            COMMON.barrier_with_error_check(dummy_self, succeeded=True)


class _TwoParamModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.tensor([1.0]))
        self.second = nn.Parameter(torch.tensor([2.0]))


def test_gather_params_for_save_detects_zero3_if_later_param_has_ds_id():
    module = _TwoParamModule()
    module.second.ds_id = 17
    gather_calls = []

    class _FakeGatheredParameters:
        def __init__(self, params, modifier_rank=0):
            self.params = params
            self.modifier_rank = modifier_rank

        def __enter__(self):
            gather_calls.append(([id(p) for p in self.params], self.modifier_rank))
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    with patch('algs.RL.common.deepspeed.zero.GatheredParameters', _FakeGatheredParameters):
        state_dict = COMMON.gather_params_for_save(SimpleNamespace(), module, rank=0)

    assert list(state_dict.keys()) == ["first", "second"]
    assert len(gather_calls) == 2
    assert all(modifier_rank == 0 for _, modifier_rank in gather_calls)
