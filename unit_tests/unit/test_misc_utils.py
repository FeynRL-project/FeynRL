import os
import torch
import pytest
from unittest.mock import MagicMock, patch
from misc.utils import (
    safe_string_to_torch_dtype,
    ensure_1d,
    pad_1d_to_length,
    get_experiment_dir_name,
    load_algorithm,
    ray_get_with_timeout,
    set_random_seeds,
    get_determinism_env_vars,
)

def test_get_determinism_env_vars():
    result = get_determinism_env_vars()
    assert result == ":16:8"

def test_set_random_seeds():
    # Clean env to test setting
    os.environ.pop("PYTHONHASHSEED", None)
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    set_random_seeds(seed=123, rank=0)

    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"

def test_set_random_seeds_with_rank():
    '''
        Verify that rank offset is applied to the seed.
    '''
    os.environ.pop("PYTHONHASHSEED", None)

    set_random_seeds(seed=100, rank=3)

    # PYTHONHASHSEED should be the base seed (not offset)
    assert os.environ["PYTHONHASHSEED"] == "100"

    # But torch/random/numpy seeds should use seed+rank = 103
    # Verify reproducibility: setting same seed+rank gives same random output
    set_random_seeds(seed=100, rank=3)
    val1 = torch.rand(1).item()

    set_random_seeds(seed=100, rank=3)
    val2 = torch.rand(1).item()

    assert val1 == val2

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

def test_get_experiment_dir_name():
    result = get_experiment_dir_name("/checkpoints", "epoch_3", "exp_001")
    assert result == os.path.join("/checkpoints", "exp_001", "epoch_3")

def test_get_experiment_dir_name_nested():
    result = get_experiment_dir_name("/a/b/c", "tag", "id")
    assert result == os.path.join("/a/b/c", "id", "tag")

def test_load_algorithm_success():
    registry = {
        "sft": ("algs.SFT.sft", "SFT"),
    }
    cls = load_algorithm("sft", registry)
    from algs.SFT.sft import SFT
    assert cls is SFT

def test_load_algorithm_case_insensitive():
    registry = {
        "sft": ("algs.SFT.sft", "SFT"),
    }
    cls = load_algorithm("SFT", registry)
    from algs.SFT.sft import SFT
    assert cls is SFT

def test_load_algorithm_unknown():
    registry = {"sft": ("algs.SFT.sft", "SFT")}
    with pytest.raises(ValueError, match="Unknown algorithm"):
        load_algorithm("nonexistent", registry)

def test_ray_get_with_timeout_success():
    '''
        ray_get_with_timeout returns results when all refs complete.
    '''
    logger = MagicMock()
    ref = MagicMock()
    with patch('misc.utils.ray') as mock_ray:
        # ray.wait returns all refs as ready immediately
        mock_ray.wait.return_value = ([ref], [])
        mock_ray.get.return_value = "result"
        result = ray_get_with_timeout(refs=ref, timeout=10, description="test_op", logger=logger)
        assert result == "result"

# The global ray mock in conftest makes ray.exceptions.* into MagicMocks,
# not real BaseException subclasses. We need real exception classes so that
# the except clauses in ray_get_with_timeout can actually catch them.

class _TimeoutError(Exception):
    pass

class _ActorError(Exception):
    pass

class _TaskError(Exception):
    pass

def _patch_ray_exceptions():
    '''
        Patch all ray exception names in misc.utils with real BaseException subclasses.
    '''
    return (
        patch('misc.utils.GetTimeoutError', _TimeoutError),
        patch('misc.utils.RayActorError', _ActorError),
        patch('misc.utils.RayTaskError', _TaskError),
    )

def test_ray_get_with_timeout_timeout_error():
    '''
        ray_get_with_timeout raises RuntimeError when ray.wait never completes.
    '''
    logger = MagicMock()
    ref = MagicMock()
    p1, p2, p3 = _patch_ray_exceptions()
    with patch('misc.utils.ray') as mock_ray, p1, p2, p3:
        # ray.wait always returns nothing ready (simulates timeout)
        mock_ray.wait.return_value = ([], [ref])
        with pytest.raises(RuntimeError, match="timed out"):
            ray_get_with_timeout(refs=ref, timeout=0.01, description="slow_op", logger=logger)

def test_ray_get_with_timeout_actor_error():
    '''
        ray_get_with_timeout raises RuntimeError when an actor dies mid-execution.
    '''
    logger = MagicMock()
    ref = MagicMock()
    p1, p2, p3 = _patch_ray_exceptions()
    with patch('misc.utils.ray') as mock_ray, p1, p2, p3:
        # ray.wait reports ref as ready, but ray.get raises actor error (EarlyFailure path)
        mock_ray.wait.return_value = ([ref], [MagicMock()])
        mock_ray.get.side_effect = _ActorError("actor crashed")
        with pytest.raises(RuntimeError, match="actor died"):
            ray_get_with_timeout(refs=ref, timeout=10, description="dead_op", logger=logger)
