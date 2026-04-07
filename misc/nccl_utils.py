from torch.distributed.distributed_c10d import (
Backend, PrefixStore, _new_process_group_helper, _world
    )
from torch.distributed import TCPStore
from datetime import timedelta


def is_nccl_fatal_error(exc):
    '''
        Pattern-match an exception to detect NCCL communicator destruction.
        Returns True if the comm is no longer usable and any retry will
        immediately re-raise — i.e. the job must fail fast rather than
        looping over a dead pg.
        Used by the async engine's sync_weights_nccl callers to distinguish
        recoverable failures (partial load on one engine, transient network
        blip during gather) from fatal ones (watchdog-induced ncclCommAbort,
        hardware/network errors that destroyed the communicator).
        Async mode has no reinit_nccl_weight_sync_group runtime path, so a
        fatal error means the job must fail. Caller is expected to re-raise
        when this returns True.
    '''
    # Substrings that indicate the NCCL communicator was destroyed by the
    # watchdog (TORCH_NCCL_ASYNC_ERROR_HANDLING) or otherwise made unusable.
    # Kept inside the function so the constant is colocated with its only
    # use site.
    fatal_patterns = ("communicator was aborted",
                      "communicator is aborted",
                      "ncclcommabort",
                      "communicator is destroyed",
                      "nccl error",
                      "internal error",
                      "nccl unhandled cuda error",
                      "watchdog caught collective operation timeout",
                      # ray_get_with_timeout wraps GetTimeoutError as
                      # "<description> timed out after Xs. Check actor logs for OOM,
                      # GPU faults, or NCCL hangs." A Ray-side timeout on a nccl
                      # collective means the collective is wedged on the worker's CUDA
                      # stream, we cannot reuse the comm even if NCCL_TIMEOUT hasn't
                      # fired yet. Treat as fatal so the job exits cleanly instead of
                      # waiting another ~30 minutes for NCCL_TIMEOUT to abort the comm.
                      "timed out after",
                      # ray_get_with_timeout also wraps RayActorError as
                      # "<description> failed because a Ray actor died: ...". A dead
                      # actor mid-sync means the NCCL group has lost a rank, which is
                      # unrecoverable in async mode (no reinit_nccl_weight_sync_group
                      # path).
                      "ray actor died",
                      "actor died",
                      )
    msg = str(exc).lower()
    return any(p in msg for p in fatal_patterns)

def create_nccl_process_group(init_method, rank, world_size, group_name, timeout_seconds, backend="nccl"):
    '''
        Create a process group for weight broadcast between training rank 0
        and vllm rollout workers. We can't reuse ds's or vllm's groups because
        neither spans both training and rollout participants. We use pytorch internals
        (_new_process_group_helper) instead of init_process_group() to avoid overwriting
        the default process group that ds and vllm depend on.
        backend: nccl for gpu-to-gpu broadcast, gloo for cpu-based broadcast.
    '''
    timeout = timedelta(seconds=timeout_seconds)

    # Parse host and port from tcp:// init_method
    # init_method format: "tcp://host:port"
    addr = init_method.replace("tcp://", "")
    host, port = addr.rsplit(":", 1)
    port = int(port)

    # Rank 0 creates the TCP store server; others connect as clients.
    store = TCPStore(host_name=host,
                     port=port,
                     world_size=world_size,
                     is_master=(rank == 0),
                     timeout=timeout,
                     wait_for_workers=True,
                     )

    # Namespace the store so keys don't collide with ds/vllm groups
    store = PrefixStore(prefix=group_name, store=store)

    # nccl backend requires ProcessGroupNCCL.Options for proper GPU communicator init
    # across separate Ray actor processes.
    pg_options = None
    if str(backend) == "nccl":
        from torch.distributed.distributed_c10d import ProcessGroupNCCL
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = False

    # Create the group without overwriting the default process group.
    # PyTorch 2.7+ renamed: world_size→group_size, rank→group_rank, ranks→global_ranks_in_group
    pg, _ = _new_process_group_helper(group_size=world_size,
                                      group_rank=rank,
                                      global_ranks_in_group=list(range(world_size)),
                                      backend=Backend(backend),
                                      store=store,
                                      group_name=group_name,
                                      timeout=timeout,
                                      backend_options=pg_options,
                                      )

    # Register rank mapping so torch.distributed.broadcast(..., group=pg) can
    # resolve rank numbers within this group
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg