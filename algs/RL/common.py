import os
import torch
import torch.distributed
from transformers import AutoModelForCausalLM, AutoConfig
import deepspeed

class COMMON:
    '''
        This class provides common functions for policy gradient algorithms.
        Only contains methods that are 100% identical across all PG algorithms.
    '''
    def _load_single_model(self, model_path: str, dtype: torch.dtype):
        '''
            Helper to load a single model from HuggingFace.
        '''
        assert dtype != 'auto', "dtype must not be auto to avoid any precision issues"
        assert self.attn_impl is None or self.attn_impl == '' or self.attn_impl in ['eager', 'flash_attention_2'], \
            "attn_impl must be one of None, '', 'eager', 'flash_attention_2'"

        config = AutoConfig.from_pretrained(model_path)
        model  = AutoModelForCausalLM.from_pretrained(
                                model_path,
                                dtype=dtype,
                                trust_remote_code=self.trust_remote_code,
                                config=config,
                                attn_implementation=None if self.attn_impl == '' else self.attn_impl
                            )
        return model

    def init_training_engine(self):
        '''
            Initialize deepspeed training engines for policy, reference, and optionally value models.
            Since we are using deepspeed with ray, each ray actor MUST create its own deepspeed engine.
            This is because each ray actor process is a separate process (1 actor = 1 gpu = 1 ds rank).
        '''
        # Convert pydantic model to python Dict for DeepSpeed
        ds_config_dict = self.deepspeed_config.model_dump()

        # check to avoid re-initializing distributed backend
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()
        
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Initializing training engine...")

        # Load models
        models = self.load_model()

        # Unpack models flexibly
        policy_model = models["policy_model"]
        ref_model    = models["ref_model"] if "ref_model" in models and models["ref_model"] is not None else None
        value_model  = models["value_model"] if "value_model" in models and models["value_model"] is not None else None

        # Log model paths
        if value_model is not None:
            # PPO has separate value model
            print(f"[Alg:{self.alg_name}][Rank {rank}] Models loaded: policy={self.model_path}, value={self.value_model_path}, ref={self.ref_model_path}")

        else:
            # SGRPO/CISPO has policy only
            print(f"[Alg:{self.alg_name}][Rank {rank}] Model loaded: {self.model_path}")

        # Enable gradient checkpointing on the HF model before DS wrapping
        if self.gradient_checkpointing:
            policy_model.gradient_checkpointing_enable()
            if value_model is not None:
                value_model.gradient_checkpointing_enable()

            print(f"[Alg:{self.alg_name}][Rank {rank}] Gradient checkpointing enabled"
                  f"{' (policy + value)' if value_model is not None else ''}")

        # Initialize policy engine
        self.policy_engine, self.policy_optimizer , _, _ = deepspeed.initialize(
                                                        model=policy_model,
                                                        model_parameters=policy_model.parameters(),
                                                        config=ds_config_dict
                                                        )
        print(f"[Alg:{self.alg_name}][Rank {rank}] DeepSpeed engine initialized on device: {self.policy_engine.device}")

        # Initialize reference model if provided
        self.ref_model_engine = None
        if ref_model is not None:
            ref_model.eval()
            ref_ds_config = self.deepspeed_ref_config.model_dump()
            self.ref_model_engine, _, _, _ = deepspeed.initialize(
                                                    model=ref_model,
                                                    config=ref_ds_config
                                                    )
            print(f"[Alg:{self.alg_name}][Rank {rank}] Reference model initialized with DeepSpeed")

        # Initialize value model
        if value_model is not None:
            # Use separate DS config for value model if available (different lr, weight decay, grad clip)
            value_ds_config = self.deepspeed_value_config
            value_ds_dict = value_ds_config.model_dump() if value_ds_config is not None else ds_config_dict
            self.value_engine, self.value_optimizer, _, _ = deepspeed.initialize(
                                                    model=value_model,
                                                    model_parameters=value_model.parameters(),
                                                    config=value_ds_dict
                                                    )
            print(f"[Alg:{self.alg_name}][Rank {rank}] Value model initialized with DeepSpeed")

    def policy_forward(self, input_ids, att_mask, pos_ids):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            Returns:
                logits is [B, T-1, vocab_size]
                entropies is [B, T-1]
        '''
        # if pos_ids is not provided, HF will add that automatically.
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        # feed data to model
        # use_cache=False: KV cache is never useful during training
        output = self.policy_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=False)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        B, T_minus_1, vocab_size = logits.shape

        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        # cross_entropy return -logprobs but we need logprobs
        # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
        # target_ids is [B, T-1] --> [B * (T-1)]
        # Compute token logprobs in float32 to avoid bf16/fp16 quantization
        neg_logprobs = self.cross_entropy(logits.to(torch.float32).view(-1, vocab_size), target_ids.view(-1))
        logprobs = -neg_logprobs.view(B, T_minus_1)
        # we can also do this, but it is less efficient I guess
        #   logprobs = logits.log_softmax(dim=-1)
        #   logprobs = torch.gather(logprobs, dim=-1, index=target_ids)

        entropies = None
        if self.ent_coeff > 0.0:
            entropies = torch.distributions.Categorical(logits=logits.to(torch.float32)).entropy()

        return logprobs, entropies, target_ids

    def ref_forward(self, input_ids, att_mask, pos_ids):
        '''
            input_ids and att_mask are [B, T]
            pos_ids is [B, T] or None
            Returns:
                ref_logprobs is [B, T-1]
        '''
        # feed data to model
        with torch.no_grad():
            if pos_ids is not None:
                pos_ids = pos_ids.to(input_ids.device)

            # use_cache=False: full-sequence forward
            output = self.ref_model_engine(input_ids=input_ids,
                                           attention_mask=att_mask,
                                           position_ids=pos_ids,
                                           use_cache=False)

            # [B, T, V] -> [B, T-1, V]
            logits = output.logits[:, :-1, :].contiguous()
            B, T_minus_1, vocab_size = logits.shape

            # [B, T] -> [B, T-1]
            target_ids = input_ids[:, 1:].contiguous()

            # cross_entropy return -logprobs but we need logprobs
            # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
            # target_ids is [B, T-1] --> [B * (T-1)]
            # Match policy path: keep logprob computation in float32 for KL stability.
            neg_logprobs = self.cross_entropy(logits.to(torch.float32).view(-1, vocab_size), target_ids.view(-1))
            ref_logprobs = -neg_logprobs.view(B, T_minus_1)

        return ref_logprobs

    def compute_kl_distance(self, logprobs, ref_logprobs):
        '''
            Compute KL divergence between two policies.
            using var_reduced form:
            kl = E[log pi/pi_ref] + pi_ref/pi - 1
        '''
        # [B, T-1]
        # Perform KL math in float32 for numerical stability under bf16/fp16.
        logprobs = logprobs.to(torch.float32)
        ref_logprobs = ref_logprobs.to(torch.float32)

        log_ratio = logprobs - ref_logprobs
        # pi_ref/pi = exp(ref_logprobs - logprobs)
        exponent = ref_logprobs - logprobs
        if exponent.max().item() > 10.0:
            print(f"[WARNING] compute_kl_distance: extreme divergence detected, max exponent={exponent.max().item():.1f}")

        ratio_inv = torch.exp(exponent)
        kl_dist = log_ratio + ratio_inv - 1
        return kl_dist

    def save_checkpoint(self, output_dir: str, tag: str, value_output_dir: str = None):
        '''
            Saves the model in hf compatible format for vllm, etc.
            We rely on save_16bit_model which handles gathering partitioned weights in zero-3.
            Args:
                output_dir: Directory to save policy model
                tag: Checkpoint tag/identifier
                value_output_dir: Optional directory to save value model (whenever we meed value function). 
                                  If None and value model exists, defaults to {output_dir}_value.
            Note we must call this on ALL ranks for zero-3 correctness.
        '''
        rank = torch.distributed.get_rank()
        print(f"[Alg:{self.alg_name}][Rank {rank}] Saving checkpoint with tag {tag}...")

        try:
            # 1. Save policy model weights (gathered fp16/bf16)
            # os.makedirs(output_dir, exist_ok=True)
            self.policy_engine.save_16bit_model(output_dir)

            # Barrier to ensure all ranks finished writing before rank 0 saves config
            # Without this, rank 0 might save config before other ranks write their shards
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # 2. Save policy config (required for vllm) on rank 0 ONLY
            if rank == 0:
                if hasattr(self.policy_engine.module, 'config'):
                    self.policy_engine.module.config.save_pretrained(output_dir)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Policy config saved")

                else:
                    print(f"[Alg:{self.alg_name}][Rank {rank}] WARNING: Could not find model config to save for policy")

            # Make sure rank 0 finished writing policy config
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # 3. Save value model if provided
            if hasattr(self, 'value_engine') and self.value_engine is not None:
                # Auto-derive value output directory if not provided
                if value_output_dir is None:
                    value_output_dir = output_dir.rstrip('/') + "_value"
                
                self.value_engine.save_16bit_model(value_output_dir)

                # Barrier to ensure all ranks finished writing before rank 0 saves config
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                # 4. Save value config on rank 0 ONLY
                if rank == 0:
                    if hasattr(self.value_engine.module, 'config'):
                        self.value_engine.module.config.save_pretrained(value_output_dir)
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Value config saved")

                    else:
                        print(f"[Alg:{self.alg_name}][Rank {rank}] WARNING: Could not find model config to save for value")

                # Make sure rank 0 finished writing value config
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

            print(f"[Alg:{self.alg_name}][Rank {rank}] Checkpoint save completed!")

        except Exception as e:
            # log error but don't crash allows other ranks to continue
            print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving checkpoint: {e}")
            if torch.distributed.is_initialized():
                # still need barrier even on error to prevent deadlock
                torch.distributed.barrier()
            raise

    def gather_state_dict(self):
        '''
            Gather policy weights into a full state_dict on rank 0.
            Works for all ds stages (0/1/2/3):
              - Stage 0/1/2: params are already full on each rank, so rank 0 copies directly.
              - Stage 3: uses GatheredParameters to collect partitioned params before copying.
            Only rank 0 returns the actual state_dict; others return {}.
            This is used for direct sync of weights with vllm rather than saving them on disk.

            Returns:
                dict: {param_name: cpu_tensor} on rank 0, empty dict on other ranks.
        '''
        rank = torch.distributed.get_rank()
        state_dict = {}

        # 1. Get all parameters
        # Must be called on ALL ranks as zero-3 requires collective participation.
        params = []
        names = []
        for name, param in self.policy_engine.module.named_parameters():
            params.append(param)
            names.append(name)

        # 2. Check if we're using stage-3 where parameters are partitioned
        is_zero3 = hasattr(params[0], 'ds_id') if params else False

        if is_zero3:
            # gather partitioned parameters in a single collective call
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if rank == 0:
                    for name, param in zip(names, params):
                        # .data avoids autograd overhead.
                        # .cpu().clone() ensures the tensor lives in CPU memory and is independent of the
                        # ZeRO-3 partition buffer which gets freed after the context.
                        state_dict[name] = param.data.cpu().clone()
        else:
            # Stages 0/1/2: params are already full, just copy on rank 0
            if rank == 0:
                for name, param in zip(names, params):
                    state_dict[name] = param.data.cpu().clone()

        if rank == 0:
            print(f"[Alg:{self.alg_name}][Rank {rank}] Gathered state_dict (zero3={is_zero3})!")

        return state_dict