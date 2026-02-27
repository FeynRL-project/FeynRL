import os
import torch
import torch.distributed
from transformers import AutoModelForCausalLM, AutoConfig
import deepspeed
from peft import get_peft_model, LoraConfig
from safetensors.torch import save_file

class COMMON:
    '''
        This class provides common functions for policy gradient algorithms.
        Only contains methods that are 100% identical across all PG algorithms.
    '''
    def _load_single_model(self, model_path: str, dtype: torch.dtype, model_name: str):
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
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # apply PEFT module if enabled
        if model_name != "ref" and self.peft_config.use_peft:
            model = self._apply_peft_module(model)
            if rank == 0:
                print(f"[Alg:{self.alg_name}][Rank {rank}] PEFT module applied for {model_name}")
                model.print_trainable_parameters()

            # Catch misconfigured lora_target_modules early, if no params are trainable,
            # ds init will fail late or silently train nothing.
            num_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            assert num_trainable > 0, "PEFT produced zero trainable parameters. Check peft.lora_target_modules"

        # Enable gradient checkpointing on the HF model before DS wrapping
        if model_name != "ref" and self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print(f"[Alg:{self.alg_name}][Rank {rank}] Gradient checkpointing enabled for {model_name}")

            # With gradient checkpointing + peft/lora, pytorch may require that at least
            # one input to each checkpointed block has requires_grad=True. When the base model
            # is frozen which is the case in lora (i.e.,requires_grad=False), it causes backward to
            # fail or skip grads. Hence, we need to force the inputs to require grad so lora params
            # inside checkpointed blocks still receive gradients.
            if self.peft_config.use_peft and self.peft_config.peft_type == "lora":
                if hasattr(model, 'enable_input_require_grads'):
                    model.enable_input_require_grads()
                    if rank == 0:
                        print(f"[Alg:{self.alg_name}][Rank {rank}] enable_input_require_grads() for {model_name}")

        return model

    def _apply_peft_module(self, model):
        '''
            Apply PEFT module to the model if it is enabled.
        '''
        if self.peft_config.peft_type == 'lora':
            lora_config = LoraConfig(r=self.peft_config.lora_rank,
                                    lora_alpha=self.peft_config.lora_alpha,
                                    lora_dropout=self.peft_config.lora_dropout,
                                    target_modules=self.peft_config.lora_target_modules,
                                    task_type=self.peft_config.task_type)

            model_peft = get_peft_model(model, lora_config)
            print("LoRA model loaded successfully")
            return model_peft

        else:
            raise ValueError(f"Unsupported PEFT type: {self.peft_config.peft_type}")

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

        # Initialize policy engine
        # only pass trainable params so ds doesn't waste memory on frozen ones (e.g. LoRA)
        trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
        self.policy_engine, self.policy_optimizer , _, _ = deepspeed.initialize(
                                                        model=policy_model,
                                                        model_parameters=trainable_params,
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
            trainable_value_params = [p for p in value_model.parameters() if p.requires_grad]
            self.value_engine, self.value_optimizer, _, _ = deepspeed.initialize(
                                                    model=value_model,
                                                    model_parameters=trainable_value_params,
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
            if self.peft_config.use_peft:
                # Save full merged model (base + LoRA merged) so vllm can load it
                # directly for disk-based refresh. Must gather ALL params (base + adapter)
                # since ZeRO-3 partitions everything, not just trainable params.
                all_params = list(self.policy_engine.module.parameters())
                with deepspeed.zero.GatheredParameters(all_params, modifier_rank=0):
                    if rank == 0:
                        raw_sd = {name: param.data.cpu().clone()
                                  for name, param in self.policy_engine.module.named_parameters()}
                        merged_sd = self._merge_peft_state_dict(raw_sd)
                        save_file(merged_sd, os.path.join(output_dir, "model.safetensors"))
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Saved merged PEFT policy model")

            else:
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
                
                if self.peft_config.use_peft:
                    # Gather ALL params (base + adapter) since ZeRO-3 partitions everything.
                    # Then merge lora into base and save as a standalone model for vllm.
                    all_value_params = list(self.value_engine.module.parameters())
                    with deepspeed.zero.GatheredParameters(all_value_params, modifier_rank=0):
                        if rank == 0:
                            os.makedirs(value_output_dir, exist_ok=True)
                            raw_sd = {name: param.data.cpu().clone()
                                      for name, param in self.value_engine.module.named_parameters()}
                            merged_sd = self._merge_peft_state_dict(raw_sd)
                            save_file(merged_sd, os.path.join(value_output_dir, "model.safetensors"))
                            print(f"[Alg:{self.alg_name}][Rank {rank}] Saved merged PEFT value model")
                else:
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
            # The normal path has multiple barriers and we cannot know which one failed,
            # so adding one here would cause a barrier count mismatch and deadlock.
            # Instead, we let the error propagate and ray.get catches it and kills other actors.
            print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving checkpoint: {e}")
            raise

    def _merge_peft_state_dict(self, raw_state_dict):
        '''
            Merge LoRA adapter weights into base model weights and remap to
            original HuggingFace parameter names so vllm can load them.

            PeftModel names follow the pattern:
              base weight:  base_model.model.{hf_name}
              lora_A:       base_model.model.{module_path}.lora_A.default.weight
              lora_B:       base_model.model.{module_path}.lora_B.default.weight
              scaling:      alpha / r

            Merged weight = base_weight + (alpha / r) * lora_B @ lora_A
        '''
        peft_prefix = "base_model.model."
        merged = {}

        # 1. Collect LoRA A/B pairs keyed by their module path
        lora_a = {}  # module_path -> tensor
        lora_b = {}  # module_path -> tensor
        base_weights = {}  # peft_name -> tensor (everything that isn't lora_A/B)

        for name, tensor in raw_state_dict.items():
            if ".lora_A." in name:
                # e.g. base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
                module_path = name.split(".lora_A.")[0]
                lora_a[module_path] = tensor

            elif ".lora_B." in name:
                module_path = name.split(".lora_B.")[0]
                lora_b[module_path] = tensor

            else:
                base_weights[name] = tensor

        # 2. Merge lora into base weights
        scaling = self.peft_config.lora_alpha / self.peft_config.lora_rank

        for module_path in lora_a:
            if module_path not in lora_b:
                print(f"[WARNING] _merge_peft_state_dict: found lora_A but no lora_B for {module_path}")
                continue

            A = lora_a[module_path]  # [r, D]
            B = lora_b[module_path]  # [H, r]
            delta = (B @ A) * scaling  # [H, D]

            # Find the corresponding base weight
            # Base weight is at: {module_path}.base_layer.weight (ZeRO-3 + PEFT)
            # or {module_path}.weight (some PEFT versions)
            base_key = module_path + ".base_layer.weight"
            if base_key not in base_weights:
                base_key = module_path + ".weight"

            if base_key in base_weights:
                base_weights[base_key] = base_weights[base_key] + delta.to(base_weights[base_key].dtype)

            else:
                print(f"[WARNING] _merge_peft_state_dict: no base weight found for {module_path}")

        # 3. Remap names: strip peft_prefix and .base_layer suffix
        for peft_name, tensor in base_weights.items():
            # Strip "base_model.model." prefix
            if peft_name.startswith(peft_prefix):
                hf_name = peft_name[len(peft_prefix):]
            else:
                hf_name = peft_name

            # Strip ".base_layer" inserted by PEFT for wrapped linear layers
            hf_name = hf_name.replace(".base_layer.", ".")

            merged[hf_name] = tensor

        return merged

    def gather_state_dict(self):
        '''
            Gather policy weights into a full state_dict on rank 0.
            Works for all ds stages (0/1/2/3):
              - Stage 0/1/2: params are already full on each rank, so rank 0 copies directly.
              - Stage 3: uses GatheredParameters to collect partitioned params before copying.
            Only rank 0 returns the actual state_dict; others return {}.
            This is used for direct sync of weights with vllm rather than saving them on disk.

            When peft is active, merges adapter weights into base weights and remaps
            to original HF names so vllm can load them.

            Returns:
                dict: {param_name: cpu_tensor} on rank 0, empty dict on other ranks.
        '''
        rank = torch.distributed.get_rank()
        state_dict = {}

        # 1. Collect parameter references without GPU allocation.
        # With ZeRO-3, each param is still partitioned here, only the local
        # shard lives on GPU. This just builds python lists of references.
        # Must be called on ALL ranks as zero-3 requires collective participation.
        params = []
        names = []
        for name, param in self.policy_engine.module.named_parameters():
            params.append(param)
            names.append(name)

        # 2. Check if we're using stage-3 where parameters are partitioned
        is_zero3 = hasattr(params[0], 'ds_id') if params else False

        if is_zero3:
            # Gather one parameter at a time to avoid GPU oom. Gathering all params
            # at once would need the full model size in free GPU memory, which is
            # rarely available after training. Per-param gathering only needs space
            # for the largest single parameter (e.g., ~500MB for a 14B model). All ranks
            # must enter each context as a collective op.
            for name, param in zip(names, params):
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if rank == 0:
                        state_dict[name] = param.data.cpu().clone()
        else:
            # Stages 0/1/2: params are already full, just copy on rank 0
            if rank == 0:
                for name, param in zip(names, params):
                    state_dict[name] = param.data.cpu().clone()

        # 3. When peft is active, named_parameters() returns peft-prefixed names. For example:
        # base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight (frozen base)
        # base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight (adapter)
        # however, vllm expects original names like model.layers.0.self_attn.q_proj.weight.
        # So load_weights() won't recognize any of these and the sync silently fails. To fix this,
        # we merge LoRA weights into base model weights and remap to original HF names so vllm can load them.

        if rank == 0 and self.peft_config.use_peft:
            state_dict = self._merge_peft_state_dict(state_dict)

        if rank == 0:
            print(f"[Alg:{self.alg_name}][Rank {rank}] Gathered state_dict (zero3={is_zero3}, peft={self.peft_config.use_peft})!")

        return state_dict