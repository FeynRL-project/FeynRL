import torch
import os
import random
import importlib
import time
import numpy as np
import ray
from ray.exceptions import GetTimeoutError, RayActorError, RayTaskError

def get_determinism_env_vars():
    '''
        Returns environment variables for deterministic runs.
    '''
    num_results = 16
    bytes_per_result = 8
    # use in os.environ.get('CUBLAS_WORKSPACE_CONFIG', f":{num_results}:{bytes_per_result}") which
    # returns ":16:8"
    return f":{num_results}:{bytes_per_result}"

def set_random_seeds(seed, rank=0):
    '''
        Set random seeds and other flags to make runs more reproducible (still not guaranteed).
        With distributed training, floating-point math, rollout engines, non-deterministic ops (e.g., torch.Tensor.index_add_), etc.,
        can still cause differences, seeding just reduces the variance a bit.
    '''
    # Set PYTHONHASHSEED for deterministic string hashing (e.g., dict keys, sets).
    # Note: it only affects processes that read it at startup.  For the
    # current (already-running) process this has no effect, but it IS inherited by
    # any child processes spawned later (e.g., DataLoader workers).  For Ray actors
    # and torchrun workers, we need to set it via runtime_env / launch command.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # :16:8 forces cuBLAS to use a fixed workspace allocation and the format is :num_results:bytes_per_result.
    # Specifically 16 results with 8 bytes each. This constrains cuBLAS to only use deterministic algorithm paths
    # that always accumulate in the same order.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", get_determinism_env_vars())

    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    # Force deterministic cuDNN algorithm selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def safe_string_to_torch_dtype(dtype_in):
    '''
        dtype_in might be a string in config (e.g., "fp16", "float16"). transformers expects torch.float16 or torch.bfloat16 etc.,
        when passed as torch_dtype. We must convert strings safely.
    '''

    if isinstance(dtype_in, torch.dtype):
        return dtype_in

    if dtype_in is None:
        return None

    if isinstance(dtype_in, str):
        s = dtype_in.lower()
        if s in ("fp16", "float16"):
            return torch.float16

        if s in ("bf16", "bfloat16"):
            return torch.bfloat16

        if s in ("fp32", "float32"):
            return torch.float32

        if s in ("fp64", "float64"):
            return torch.float64

    raise ValueError(f"Unsupported model_dtype: {dtype_in}")

def ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    '''
        Sanity check to make sure the input is a 1D tensor.
    '''
    if x.dim() != 1:
        raise ValueError(f"Expected {name} to be 1D, got {x.dim()}D")

    return x

def pad_1d_to_length(x: torch.Tensor, pad_value: float, target_len: int) -> torch.Tensor:
    '''
        Pad/truncate 1D sequence x[T] to target_len.
        Always returns length == target_len.
    '''
    seq_len = x.numel()

    if seq_len > target_len:
        return x[:target_len]

    if seq_len < target_len:
        pad = torch.full((target_len - seq_len,),
                            pad_value,
                            dtype=x.dtype,
                            device=x.device)
        return torch.cat([x, pad], dim=0)

    return x

def get_experiment_dir_name(output_dir: str, tag: str, experiment_id: str):
    '''
       It creates output_dir/experiment_id/tag
    '''
    experiment_dir = os.path.join(output_dir, experiment_id, tag)
    return experiment_dir


def get_gpus_per_node(ray_obj):
    '''
        Get number of GPUs per node from Ray cluster state.
        Falls back to torch.cuda.device_count() if Ray fails.
        Multi-node cluster: queries Ray, gets actual GPUs per node
        Single-node: queries Ray, falls back to torch if needed
        Orchestrator on CPU head node: queries Ray (gets worker node GPUs per node)
        Ray not initialized properly: falls back to torch/1
    '''
    try:
        # Get all nodes in the cluster
        nodes = ray_obj.nodes()
        # Filter for alive nodes that have GPUs
        gpu_nodes = [n for n in nodes if n['Alive'] and n.get('Resources', {}).get('GPU', 0) > 0]

        if gpu_nodes:
            # We assume homogeneous cluster for now
            return int(gpu_nodes[0]['Resources']['GPU'])

    except Exception as e:
        print(f"Warning: Could not get GPU count from Ray: {e}")

    # Fallback
    if torch.cuda.is_available():
        return torch.cuda.device_count()

    else:
        return 1

def load_algorithm(alg_name: str, registry: dict):
    '''
        Load algorithm class from registry.
    '''
    alg_name = alg_name.lower()
    if alg_name not in registry:
        available = list(registry.keys())
        raise ValueError(f"Unknown algorithm: {alg_name}. Available: {available}")

    module_path, class_name = registry[alg_name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def ray_get_with_timeout(refs, timeout, description, logger):
    '''
        Wrapper around ray.get() that adds a timeout and clear error reporting.
        This provides periodic logging for long-running operations.

        Args:
            refs: ObjectRef or list of ObjectRefs to wait on.
            timeout: seconds before raising (None disables timeout).
            description: human-readable label for log/error messages.
            logger: logger instance.
    '''
    try:
        if timeout is None:
            return ray.get(refs)

        start_time = time.time()
        deadline = start_time + timeout
        poll_interval = 60.0  # Log every 60 seconds

        # Ensure refs is a list for ray.wait
        is_single = not isinstance(refs, list)
        wait_refs = [refs] if is_single else list(refs)

        unready = wait_refs
        while unready:
            now = time.time()
            if now >= deadline:
                raise GetTimeoutError()

            current_timeout = min(poll_interval, deadline - now)
            ready, unready = ray.wait(unready, num_returns=len(unready), timeout=current_timeout)

            # Check if any completed refs have errors — a failed engine stops
            # participating in ZeRO-3 collectives, causing the remaining engines
            # to hang forever at the next all-gather/reduce-scatter.
            for ref in ready:
                try:
                    ray.get(ref, timeout=0)
                except RayTaskError as e:
                    logger.error(f"[EarlyFailure] {description}: one task failed, "
                                 f"remaining {len(unready)} tasks will likely deadlock. Error: {e}")
                    raise RuntimeError(f"{description} failed: {e}") from e
                except RayActorError as e:
                    logger.error(f"[EarlyFailure] {description}: actor died, "
                                 f"remaining {len(unready)} tasks will likely deadlock. Error: {e}")
                    raise RuntimeError(f"{description} failed: actor died: {e}") from e
                except GetTimeoutError:
                    pass  # ref is ready but ray.get(timeout=0) can race, so ignore

            if unready:
                elapsed = int(time.time() - start_time)
                total = len(wait_refs)
                done = total - len(unready)
                logger.info(f"[Heartbeat] {description}: {done}/{total} done, "
                            f"{len(unready)} pending (elapsed {elapsed}s, timeout {timeout}s)")

        # All ready, this will return immediately
        return ray.get(refs)

    except GetTimeoutError as e:
        logger.error(f"[Timeout] {description} did not complete within {timeout}s")
        raise RuntimeError(f"{description} timed out after {timeout}s. Check actor logs for OOM, GPU faults, or NCCL hangs.") from e

    except RayActorError as e:
        logger.error(f"[ActorDied] {description} failed: actor died: {e}")
        raise RuntimeError(f"{description} failed because a Ray actor died: {e}") from e

    # Debug remote execution failures exactly as if they had occurred locally.
    except RayTaskError as e:
        logger.error(f"[TaskError] {description} failed: {e}")
        raise RuntimeError(f"{description} failed with a remote exception: {e}") from e