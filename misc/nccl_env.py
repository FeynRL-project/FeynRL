import os

def install_nccl_watchdog(timeout_seconds=1800):
    '''
        Enable PyTorch's NCCL watchdog so a wedged collective aborts after
        `timeout_seconds` instead of hanging the process forever.

        Without this, a stuck NCCL collective (e.g. vllm stalls
        mid-receive_all_weights_nccl, leaving training rank 0 wedged inside
        PyNcclCommunicator.broadcast) sits on the CUDA stream indefinitely.
        ray_get_with_timeout would catch the Python-side hang, but the GPU
        kernel stays wedged and the next training step's collective deadlocks
        the entire DeepSpeed group.

        With the watchdog enabled:
          - TORCH_NCCL_ASYNC_ERROR_HANDLING=1 spawns a monitor thread that
            calls ncclCommAbort on the wedged communicator after timeout.
          - The next CUDA op on that communicator raises RuntimeError, which
            Ray surfaces to the driver as RayActorError.
          - The job fails fast with a real stack trace instead of hanging.

        Async mode has no reinit_nccl_weight_sync_group fallback, so a clean
        fail is the only correct outcome — and that's exactly what this gives
        us.

        MUST be called BEFORE any NCCL process group is created (i.e. before
        deepspeed.initialize() on training engines, and before vLLM engine
        creation on rollout engines). Both sides of the collective need it.

        Idempotent — uses os.environ.setdefault so caller-overrides win.
    '''
    for k, v in nccl_watchdog_env_vars(timeout_seconds).items():
        os.environ.setdefault(k, v)


def nccl_watchdog_env_vars(timeout_seconds=1800):
    '''
        Return the nccl watchdog env vars as a dict, suitable for injection
        into Ray's runtime_env={"env_vars": ...}. Preferred over
        install_nccl_watchdog() when launching actors via Ray, because Ray
        sets these in the child process BEFORE Python starts — eliminating
        any risk of the call being too late relative to deepspeed/vllm pg
        creation.
    '''
    return {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_BLOCKING_WAIT":        "0",
            "NCCL_TIMEOUT":                    str(int(timeout_seconds))}
