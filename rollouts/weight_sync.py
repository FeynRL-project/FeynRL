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

        model = self.model_runner.model
        # Sanity-check that state_dict keys have some overlap with model params.
        # vllm fuses some layers internally (e.g. q_proj + k_proj + v_proj -> qkv_proj,
        # gate_proj + up_proj -> gate_up_proj), so a strict 1:1 match is not expected.
        # load_weights handles the remapping, but if zero keys match, the naming
        # convention is completely wrong and load_weights would silently no-op.
        model_params = set(name for name, _ in model.named_parameters())
        matched = sum(1 for k in state_dict if k in model_params)
        if matched == 0 and num_params > 0:
            raise RuntimeError(f"Weight sync failed: none of the {num_params} state_dict keys "
                               f"matched model parameters. This likely means the naming convention "
                               f"changed between vllm versions. "
                               f"Sample state_dict keys: {list(state_dict.keys())[:3]}, "
                               f"sample model params: {list(model_params)[:3]}")

        # When the engine was loaded with online quantization (e.g. fp8), live
        # weight params are quantized with frozen scales. vllm's layerwise
        # path restores -> loads -> re-quantizes.
        quant = getattr(self.vllm_config.model_config, "quantization", None)
        if quant is not None:
            self.load_weights_layerwise_reload(model, state_dict)
            self.log_quant_reload_summary(model, num_params, quant)

        else:
            model.load_weights(weights=state_dict.items())

        torch.cuda.synchronize()
        return num_params

    def check_weights_hash(self, param_name):
        '''
            Return a hash of a specific parameter for verification.
            Useful for confirming weights were updated correctly.
            param_name: name of the parameter to hash.

            For FP8-quantized weights the raw FP8 storage is dequantized
            (weight_fp8 * weight_scale, mirroring fp8.py's batch-invariant
            path) so the hash is comparable to a bf16 reference computed by
            the trainer. Quantization noise still introduces small drift, so
            this remains an approximate check.
        '''
        model = self.model_runner.model
        for name, param in model.named_parameters():
            if name == param_name or param_name in name:
                data = param.data
                if data.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    module_path = name.rpartition(".")[0]
                    module = model.get_submodule(module_path) if module_path else model
                    w_bf16 = data.to(torch.bfloat16)
                    scale = getattr(module, "weight_scale", None)
                    scale_inv = getattr(module, "weight_scale_inv", None)
                    if scale is not None:
                        s = scale.data.to(torch.bfloat16)
                        if s.numel() == 1:
                            deq = w_bf16 * s
                        elif s.dim() == 1 and s.shape[0] == w_bf16.shape[0]:
                            deq = w_bf16 * s.unsqueeze(1)
                        else:
                            deq = w_bf16 * s
                    elif scale_inv is not None:
                        # Block quant: per-block scale. Approximate the hash with
                        # the mean scale so it stays the right order of magnitude.
                        deq = w_bf16 * scale_inv.data.to(torch.bfloat16).mean()
                    else:
                        deq = w_bf16
                    return deq.float().sum().item()
                return data.float().sum().item()
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

    def load_weights_layerwise_reload(self, model, state_dict):
        '''
            Load bf16 weights into a quantized engine via vllm's layerwise reload:
            restore -> load -> re-quantize per layer. Logger is silenced because
            finalize warns for every container module with no direct params
            (benign — leaves re-quantize during the load itself).
        '''
        import logging
        from vllm.model_executor.model_loader.reload import (initialize_layerwise_reload,
                                                             finalize_layerwise_reload,)

        reload_logger = logging.getLogger("vllm.model_executor.model_loader.reload.layerwise")
        prev_level = reload_logger.level
        reload_logger.setLevel(logging.ERROR)
        try:
            with torch.device(self.device):
                initialize_layerwise_reload(model)
                model.load_weights(weights=state_dict.items())
                finalize_layerwise_reload(model, self.vllm_config.model_config)

        finally:
            reload_logger.setLevel(prev_level)

    def log_quant_reload_summary(self, model, num_params, quant):
        '''
            TP-rank-0 only: log FP8 param count + sample after reload.
            fp8_params=0 means quantization silently fell back to full precision.
        '''
        tp_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if tp_rank != 0:
            return

        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        fp8_count = 0
        sample_name = sample_dtype = sample_scale_shape = None

        for n, p in model.named_parameters():
            if p.data.dtype in fp8_dtypes:
                fp8_count += 1
                if sample_name is None:
                    sample_name = n
                    sample_dtype = p.data.dtype
                    module_path = n.rpartition(".")[0]
                    m = model.get_submodule(module_path) if module_path else model
                    s = getattr(m, "weight_scale", None) or getattr(m, "weight_scale_inv", None)
                    sample_scale_shape = tuple(s.shape) if s is not None else None

        print(f"[WeightSyncExtension] FP8 layerwise reload OK | quant={quant} | "
              f"fp8_params={fp8_count}/{num_params} | "
              f"sample={sample_name} dtype={sample_dtype} scale_shape={sample_scale_shape}",
              flush=True)

    def get_quantization_info(self):
        '''
            Report what quantization the engine was loaded with and how many of
            the live model parameters are actually FP8 vs full-precision. Called
            once after engine creation so the orchestrator can confirm FP8 is
            wired through end-to-end (not just configured).
        '''
        quant = getattr(self.vllm_config.model_config, "quantization", None)
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        fp8_count = 0
        non_fp8_count = 0
        sample_name = sample_dtype = sample_scale_shape = None
        model = self.model_runner.model
        for n, p in model.named_parameters():
            if p.data.dtype in fp8_dtypes:
                fp8_count += 1
                if sample_name is None:
                    sample_name = n
                    sample_dtype = str(p.data.dtype)
                    module_path = n.rpartition(".")[0]
                    m = model.get_submodule(module_path) if module_path else model
                    s = getattr(m, "weight_scale", None) or getattr(m, "weight_scale_inv", None)
                    sample_scale_shape = tuple(s.shape) if s is not None else None
            else:
                non_fp8_count += 1

        return {"quantization": quant,
                "fp8_params": fp8_count,
                "non_fp8_params": non_fp8_count,
                "sample_name": sample_name,
                "sample_dtype": sample_dtype,
                "sample_scale_shape": sample_scale_shape,}