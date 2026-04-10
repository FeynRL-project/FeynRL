import torch
import pickle
from misc.nccl_utils import create_nccl_process_group

class WeightSyncExtension:
    '''
        vllm WorkerExtension mixin that enables in-place weight updates on vllm workers.
        Used with worker_extension_cls parameter when creating vllm llm instances.
        For 0.19, it injects this as a base class of the Worker via __bases__ manipulation,
        so self.model_runner is inherited from the Worker instance.
    '''

    def update_weights_from_state(self, serialized_state):
        '''
            Update model weights in-place on this vllm worker.
            vllm's load_weights handles name remapping and tp sharding internally.
            serialized_state: file path to pickled state_dict on /dev/shm,
            or a raw dict for backward compatibility.

            Returns the number of parameters in the state_dict that were loaded.
            This is used by the caller to verify all TP workers loaded the same weights.
        '''
        if isinstance(serialized_state, str):
            with open(serialized_state, 'rb') as f:
                state_dict = pickle.load(f)

        elif isinstance(serialized_state, dict):
            state_dict = serialized_state

        else:
            raise TypeError(f"Unsupported weight payload type: {type(serialized_state)}")

        num_params = len(state_dict)

        # Sanity-check that state_dict keys have some overlap with model params.
        # vllm fuses some layers internally (e.g. q_proj + k_proj + v_proj -> qkv_proj,
        # gate_proj + up_proj -> gate_up_proj), so a strict 1:1 match is not expected.
        # load_weights handles the remapping, but if zero keys match, the naming
        # convention is completely wrong and load_weights would silently no-op.
        model_params = set(name for name, _ in self.model_runner.model.named_parameters())
        matched = sum(1 for k in state_dict if k in model_params)
        if matched == 0 and num_params > 0:
            raise RuntimeError(f"Weight sync failed: none of the {num_params} state_dict keys "
                               f"matched model parameters. This likely means the naming convention "
                               f"changed between vllm versions. "
                               f"Sample state_dict keys: {list(state_dict.keys())[:3]}, "
                               f"sample model params: {list(model_params)[:3]}")

        self.model_runner.model.load_weights(weights=state_dict.items())
        torch.cuda.synchronize()
        return num_params

    def check_weights_hash(self, param_name):
        '''
            Return a hash of a specific parameter for verification.
            Useful for confirming weights were updated correctly.
            param_name: name of the parameter to hash.
        '''
        for name, param in self.model_runner.model.named_parameters():
            if name == param_name or param_name in name:
                return param.data.float().sum().item()
        return None

    def init_weight_nccl_group(self, master_addr, master_port, rank_offset, world_size, group_name, timeout_seconds, backend):
        '''
            Initialize a process group for weight broadcast from training rank 0.
            For gloo backend: we use create_nccl_process_group (torch.distributed internals).
            For nccl backend: we use vllm's StatelessProcessGroup + PyNcclCommunicator.
                This bypasses torch.distributed entirely and creates a raw nccl communicator
                inside the EngineCore subprocess, avoiding conflicts with vllm's own cuda context.
                The trainer side (rank 0) uses a standard torch.distributed ProcessGroupNCCL
                created via init_extra_process_group. Both sides share the same underlying nccl
                communicator bootstrapped via the same TCP store (same pattern as PipelineRL).
        '''
        tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        my_rank = rank_offset + tp_rank
        self.weight_sync_rank = my_rank
        self.weight_sync_backend = backend

        if backend == "nccl":
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            pg = StatelessProcessGroup.create(
                host=master_addr, port=master_port, rank=my_rank, world_size=world_size)
            device = torch.device("cuda", torch.cuda.current_device())
            self.weight_sync_pynccl = PyNcclCommunicator(pg, device=device)
            self.weight_sync_group = None

        else:
            self.weight_sync_group = create_nccl_process_group(
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=my_rank, world_size=world_size,
                group_name=group_name, timeout_seconds=timeout_seconds,
                backend=backend)
            self.weight_sync_pynccl = None

        return True

    def update_weights_nccl(self, param_name, dtype_raw, shape, empty_cache=False):
        '''
            Receive a single weight tensor via broadcast from training rank 0
            and load it into the model. dtype_raw is a torch.dtype or a string
            (e.g. "torch.bfloat16") when arriving through vLLM's collective_rpc.
        '''
        dtype = getattr(torch, dtype_raw.replace("torch.", "")) if isinstance(dtype_raw, str) else dtype_raw
        if self.weight_sync_backend == "nccl":
            buffer = torch.empty(shape, dtype=dtype, device="cuda")
            self.weight_sync_pynccl.broadcast(buffer, src=0, stream=torch.cuda.current_stream())

        else:
            buffer = torch.empty(shape, dtype=dtype, device="cpu")
            torch.distributed.broadcast(buffer, src=0, group=self.weight_sync_group)

        self.model_runner.model.load_weights(weights=[(param_name, buffer)])
        torch.cuda.synchronize()
        del buffer
        if empty_cache:
            torch.cuda.empty_cache()

        return 1

    def receive_all_weights_nccl(self, param_metadata):
        '''
            Receive ALL weight tensors via broadcast in a single collective_rpc call.
            For nccl: uses PyNcclCommunicator.broadcast.
            For gloo: uses torch.distributed.broadcast.
        '''
        use_pynccl = (self.weight_sync_backend == "nccl")
        num_loaded = 0
        corrupt_params = []

        for name, dtype_raw, shape in param_metadata:
            # dtype may arrive as a string (e.g. "torch.bfloat16") when sent
            # through vLLM's msgpack-based collective_rpc serializer.
            dtype = getattr(torch, dtype_raw.replace("torch.", "")) if isinstance(dtype_raw, str) else dtype_raw
            if use_pynccl:
                buffer = torch.empty(tuple(shape), dtype=dtype, device="cuda")
                self.weight_sync_pynccl.broadcast(buffer, src=0, stream=torch.cuda.current_stream())

            else:
                buffer = torch.empty(tuple(shape), dtype=dtype, device="cpu")
                torch.distributed.broadcast(buffer, src=0, group=self.weight_sync_group)

            # Receiver-side verificatioas transport-layer corruption
            # (NCCL bug, network glitch, CUDA error) can still produce a bad
            # tensor on the receiver. We do noty raise error here if any. Instead,
            # we record the corruption, skip the load, and CONTINUE receiving
            # so the collective completes cleanly. finalize_weight_nccl will then
            # see num_loaded != expected_params and return False
            if not torch.isfinite(buffer).all():
                num_bad = (~torch.isfinite(buffer)).sum().item()
                corrupt_params.append((name, dtype, tuple(shape), num_bad))
                del buffer
                continue

            self.model_runner.model.load_weights(weights=[(name, buffer)])
            torch.cuda.synchronize()
            del buffer
            num_loaded += 1

        if corrupt_params:
            print(f"[receive_all_weights_nccl] DETECTED {len(corrupt_params)} "
                  f"corrupted tensors in broadcast (NaN/Inf): "
                  f"{[p[0] for p in corrupt_params[:5]]}{'...' if len(corrupt_params) > 5 else ''}. "
                  f"Skipped load for these — NCCL collective drained cleanly. "
                  f"finalize_weight_nccl will report partial load to driver.",
                  flush=True)

        return num_loaded

    def close_weight_nccl_group(self):
        '''
            Destroy the weight sync group. Called during shutdown.
        '''
        if hasattr(self, 'weight_sync_pynccl') and self.weight_sync_pynccl is not None:
            del self.weight_sync_pynccl
            self.weight_sync_pynccl = None

        if hasattr(self, 'weight_sync_group') and self.weight_sync_group is not None:
            try:
                torch.distributed.destroy_process_group(self.weight_sync_group)
            except Exception:
                pass
            self.weight_sync_group = None