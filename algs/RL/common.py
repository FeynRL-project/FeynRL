import os
import json
import torch
import torch.distributed
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
import deepspeed
from peft import get_peft_model, LoraConfig
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards
from misc.utils import set_random_seeds
import copy
import random
import time
# internal and local import
from misc.nccl_utils import create_nccl_process_group

@torch.compile(dynamic=True)
def compiled_log_softmax_and_gather(logits_f32: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    '''
        Fused log_softmax + gather via torch.compile.
        Avoids materializing the full [B, T, V] logprobs tensor in memory.
        See: https://github.com/allenai/open-instruct/pull/584

        logits_f32: [B*T, V] in float32
        target_ids: [B*T]
        Returns: [B*T] per-token logprobs (negative of nll)
    '''
    logprobs = logits_f32.log_softmax(dim=-1)
    return torch.gather(logprobs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

class COMMON:
    '''
        This class provides common functions for policy gradient algorithms.
        Only contains methods that are 100% identical across all PG algorithms.
    '''
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

        # Inefficient way of computing cross_entropy:
        # neg_logprobs = self.cross_entropy(logits.to(torch.float32).view(-1, vocab_size), target_ids.view(-1))
        # logprobs = -neg_logprobs.view(B, T_minus_1)
        # Above has two kernels:
        # kernel1: log_softmax: reads [B*T, V], writes  [B*T, V]
        # kernel2: gather: reads that entire [B*T, V], picks one value per row, writes [B*T] output
        # torch.compile fuses log_softmax + gather into a single kernel,
        # avoiding the full [B*T, V] logprobs intermediate tensor. In particualr,
        # it computes log_softmax and gathers the target index in the same kernel, in registers
        # logits: [B, T-1, V] --> [B*(T-1), V], target_ids: [B, T-1] --> [B*(T-1)]
        # The [B*T, V] intermediate never gets allocated in GPU memory in fused verson.
        logprobs = compiled_log_softmax_and_gather(logits.to(torch.float32).view(-1, vocab_size),
                                                   target_ids.view(-1)).view(B, T_minus_1)

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

            # Compute logprobs in float32 for KL stability.
            # logits: [B, T-1, V] --> [B*(T-1), V], target_ids: [B, T-1] --> [B*(T-1)]
            ref_logprobs = compiled_log_softmax_and_gather(logits.to(torch.float32).view(-1, vocab_size),
                                                           target_ids.view(-1)).view(B, T_minus_1)

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
        exponent = torch.clamp(ref_logprobs - logprobs, min=-10.0, max=10.0)
        #if exponent.max().item() > 10.0:
        #    print(f"[WARNING] compute_kl_distance: extreme divergence detected, max exponent={exponent.max().item():.1f}")

        ratio_inv = torch.exp(exponent)
        kl_dist = log_ratio + ratio_inv - 1
        return kl_dist

    def sanitize_logprobs(self, logprobs, engine_id, step, num_micro):
        '''
            Prevent NaN from entering into loss.
            Returns (logprobs, nan_mask) where nan_mask is True at positions
            that had NaN/Inf.
        '''
        nan_mask = torch.isnan(logprobs) | torch.isinf(logprobs)
        if nan_mask.any():
            print(f"[Alg:{self.alg_name}][Engine {engine_id}] WARNING: NaN/Inf in logprobs at "
                  f"micro-batch {step}/{num_micro}, masking {nan_mask.sum().item()} tokens out of loss",
                  flush=True)
            # sentinel: logprobs are always <= 0
            logprobs = logprobs.masked_fill(nan_mask, 1.0)

        return logprobs, nan_mask

    def compute_global_token_denom(self, micro_batches, ga_steps, device):
        '''
            Compute global token count for global token normalization.
            ga_denom = total valid tokens across ALL micro-batches on ALL ranks.
            dp_scale = ga_steps * world_size cancels DeepSpeed's internal averaging
            so each token contributes equally to the gradient. This is the exact same logic as what
            we have in sft.py
            micro_batches: list of micro-batch dicts with mask key, shape [B, T].
            ga_steps: gradient accumulation steps from DeepSpeed config.
            device: torch device.
            Returns:
                (ga_denom, dp_scale) where ga_denom is the global token count
                and dp_scale is ga_steps * world_size.
        '''
        local_token_count = sum((mb['mask'][:, :-1] > 0.5).sum().item() for mb in micro_batches)
        ga_denom_tensor   = torch.tensor(local_token_count, dtype=torch.float64, device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(ga_denom_tensor, op=torch.distributed.ReduceOp.SUM)

        ga_denom = ga_denom_tensor.item()
        if ga_denom <= 0:
            raise RuntimeError(f"[Alg:{self.alg_name}] Invalid global token denominator: {ga_denom}")

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        dp_scale   = ga_steps * world_size

        return ga_denom, dp_scale

    def compute_per_group_token_denoms(self, micro_batches, ga_steps, device):
        '''
            Compute pergradient-accumulation-group global token counts for use with
            update_after_full_replay=False. Each GA group gets its own
            denominator so each optimizer step is normalized only by its
            own group's tokens.
            micro_batches: list of micro-batch dicts with mask key, shape [B, T].
            ga_steps: gradient accumulation steps from DeepSpeed config.
            Returns:
                (ga_denoms, dp_scale) where:
                  - ga_denoms is a list of per-GA-group global token counts.
                    Length = ceil(num_micro / ga_steps).
                  - dp_scale is ga_steps * world_size.
        '''
        num_micro  = len(micro_batches)
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        dp_scale   = ga_steps * world_size

        # Per-micro-batch local token counts.
        count = [(mb['mask'][:, :-1] > 0.5).sum().item() for mb in micro_batches]
        local_counts = torch.tensor(count, dtype=torch.float64, device=device)

        # All-reduce so every rank has global per-micro-batch counts.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_counts, op=torch.distributed.ReduceOp.SUM)

        # 12 micro-batches (num_micro), ga_steps=3, 5 ranks, batch_size=4
        # 4 GA groups (micro-batches / ga_steps):
        # [MB 0-2], [MB 3-5], [MB 6-8], [MB 9-11]
        # Each group: num_micro=3 x batch_size=4 x 5 ranks = 60 samples
        # ga_denoms[i] where i in [0,1,2,3] = total valid tokens in MB [...] across all 5 ranks
        # Optimizer steps at MB 2, 5, 8, 11, each step divides by its own
        # group's token count, not the full replay buffer's count.
        # Sum into GA groups. Each group covers ga_steps consecutive micro-batches.
        # E.g., ga_steps=3, 12 MBs -> groups [0:3], [3:6], [6:9], [9:12] -> 4 denoms.
        ga_denoms = []
        for group_start in range(0, num_micro, ga_steps):
            group_end   = min(group_start + ga_steps, num_micro)
            group_denom = local_counts[group_start:group_end].sum().item()
            ga_denoms.append(max(group_denom, 1.0))

        return ga_denoms, dp_scale

    def check_logit_health(self, logits):
        '''
            Check if there is nan or inf in logits.
            logits: [B, T-1, V]

            NaN in logits with NaN in weights = weight corruption (fatal).
            NaN/Inf in logits with clean weights = bf16 overflow on specific
            sequences (non-fatal, masked out downstream by sanitize_logprobs).
        '''
        # Reduce along vocab dim in native dtype -> [B, T-1], no extra [B,T,V] alloc.
        with torch.no_grad():
            row_max = logits.max(dim=-1).values   # [B, T-1]
            all_finite = torch.isfinite(row_max).all().item()

        if not all_finite:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            num_bad = (~torch.isfinite(row_max)).sum().item()
            total = row_max.numel()

            # Check if weights themselves are corrupted
            bad_param = None
            for pname, p in self.policy_engine.module.named_parameters():
                if p.numel() > 0 and torch.isnan(p).any():
                    bad_param = pname
                    break

            if bad_param is not None:
                # Weights are corrupted — fatal, no recovery possible
                raise RuntimeError(f"[Rank {rank}] NaN in logits AND weights: "
                                   f"logits shape={list(logits.shape)}, dtype={logits.dtype}, "
                                   f"first NaN param='{bad_param}'. "
                                   f"Model weights are corrupted — check optimizer step or weight sync.")

            # Weights are clean — bf16 overflow on specific tokens.
            # Log warning but don't crash. The affected positions will produce
            # NaN logprobs which sanitize_logprobs masks out of the loss.
            print(f"[Rank {rank}] WARNING: {num_bad}/{total} non-finite logit rows "
                  f"(bf16 overflow), logits shape={list(logits.shape)}. "
                  f"Weights are clean — affected tokens will be masked from loss.",
                  flush=True)

    def check_weights_health(self, engine_id, location):
        '''
            Pre-training-step guard: exit immediately if weights are already nan.
            Catches corruption carried over from a previous epoch or weight sync.
        '''
        nan_params = [name for name, p in self.policy_engine.module.named_parameters()
                      if p.requires_grad and p.numel() > 0 and torch.isnan(p).any()
                      ]
        if nan_params:
            raise RuntimeError(f"[Alg:{self.alg_name}][Engine {engine_id}] {len(nan_params)} params "
                               f"contain NaN {location}: {nan_params[:10]}. "
                               f"Model is corrupted — restart from a known-good checkpoint.")

    def check_weights_per_microbatch_update(self, engine_id, step, num_micro, pi_loss, loss_total_sum, local_denom, ga_denom):
        '''
            Post-optimizer-step guard: detect NaN introduced by DeepSpeed's step().
            Lightweight — iterates params without cloning or accessing DS internals.
        '''
        bad = [name for name, p in self.policy_engine.module.named_parameters()
               if p.requires_grad and p.numel() > 0 and torch.isnan(p).any()
               ]
        if bad:
            raise RuntimeError(f"[Alg:{self.alg_name}][Engine {engine_id}] NaN in weights "
                               f"after optimizer step at micro-batch {step}/{num_micro}. "
                               f"pi_loss={pi_loss.item():.6f}, "
                               f"loss_total_sum={loss_total_sum.item():.6f}, "
                               f"local_denom={local_denom.item():.0f}, "
                               f"ga_denom={ga_denom}. "
                               f"Corrupted params ({len(bad)}): {bad[:10]}")

    def check_all_masked(self, engine_id, step, num_micro, local_denom,
                         consecutive_nan_steps, nan_abort_threshold=20):
        '''
            Guard against sustained all-masked micro-batches which produce zero
            gradients and can cause ZeRO-3 NCCL hangs.
            Returns updated consecutive_nan_steps count.
        '''
        if local_denom.item() == 0:
            consecutive_nan_steps += 1
            if engine_id == 0:
                print(f"[Alg:{self.alg_name}][Engine {engine_id}] WARNING: all tokens "
                      f"masked at micro-batch {step}/{num_micro} "
                      f"({consecutive_nan_steps}/{nan_abort_threshold} consecutive), "
                      f"loss is zero — model is not learning.",
                      flush=True)
            if consecutive_nan_steps >= nan_abort_threshold:
                raise RuntimeError(f"[Alg:{self.alg_name}][Engine {engine_id}] Aborting: "
                                   f"{nan_abort_threshold} consecutive micro-batches had all "
                                   f"tokens masked by NaN/Inf. The model has collapsed — "
                                   f"restart from a known-good checkpoint.")
        else:
            consecutive_nan_steps = 0
        return consecutive_nan_steps

    def load_single_model(self, model_path: str, dtype: torch.dtype, model_name: str):
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

        # apply PEFT module to both policy and value
        if model_name != "ref" and self.peft_config.use_peft:
            model = self.apply_peft_module(model)
            if rank == 0:
                print(f"[Alg:{self.alg_name}][Rank {rank}] PEFT module applied for {model_name}")
                model.print_trainable_parameters()

            # Catch misconfigured lora_target_modules early, if no params are trainable,
            # ds init will fail late or silently train nothing.
            num_trainable = sum(1 for p in model.parameters() if p.requires_grad)
            assert num_trainable > 0, "PEFT produced zero trainable parameters. Check peft.lora_target_modules"

        # Enable gradient checkpointing on the HF model before DS wrapping
        # only for policy as value model is done in ppo.load_model()
        if model_name == "policy" and self.gradient_checkpointing:
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

    def get_model_info(self):
        '''
            Return model parameter counts and PEFT status.
            Uses ds_numel for ZeRO-3 partitioned params, numel() for non-ZeRO.
        '''
        total = sum(getattr(p, 'ds_numel', p.numel()) for p in self.policy_engine.module.parameters())
        trainable = sum(getattr(p, 'ds_numel', p.numel()) for p in self.policy_engine.module.parameters() if p.requires_grad)
        info = {'total_params': total,
                'trainable_params': trainable,
                'frozen_params': total - trainable,
                'peft_enabled': self.peft_config.use_peft,
                'peft_type': self.peft_config.peft_type if self.peft_config.use_peft else None,
                }

        if hasattr(self, 'value_engine') and self.value_engine is not None:
            info['value_total_params'] = sum(getattr(p, 'ds_numel', p.numel()) for p in self.value_engine.module.parameters())
            info['value_trainable_params'] = sum(getattr(p, 'ds_numel', p.numel()) for p in self.value_engine.module.parameters() if p.requires_grad)
        return info

    def get_training_stats(self):
        '''
            Return current LR and GPU peak memory.
        '''
        stats = {}
        if self.policy_optimizer is not None:
            stats['lr'] = self.policy_optimizer.param_groups[0]['lr']

        if torch.cuda.is_available():
            stats['gpu_peak_mem_gb'] = torch.cuda.max_memory_allocated(self.policy_engine.device) / (1024 ** 3)

        return stats

    def apply_peft_module(self, model):
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
        # All ranks MUST use the same seed for model initialization so that stochastic
        #  components (e.g., LoRA adapter init via kaiming_uniform_) produce identical weights.
        # Under zero-3, parameters are partitioned in-place without broadcast, so differing
        # init weights across ranks would cause a problem. Per-rank seeds are applied AFTER deepspeed.initialize() below.
        set_random_seeds(self.seed)

        # Convert pydantic model to python Dict for DeepSpeed
        ds_config_dict = self.deepspeed_config.model_dump()

        # check to avoid re-initializing distributed backend
        if not torch.distributed.is_initialized():
            deepspeed.init_distributed()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Initializing training engine...")

        # Load models (all ranks get identical weights due to shared seed above)
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
            # GRPO/CISPO has policy only
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
            ref_model.requires_grad_(False)
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

        # Now re-seed with rank-specific seeds so each rank would have different stochastic behavior.
        set_random_seeds(self.seed + rank)

    def shutdown(self):
        '''
            Clean up distributed state. Call before the ray actor is torn down
            to release nccl resources and prevent stale connections on restart.
        '''
        # Destroy the custom weight sync group first before tearing down the default process group.
        self.close_weight_nccl_group()

        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass

    def save_checkpoint(self, output_dir: str, tag: str, value_output_dir: str = None):
        '''
            Saves the model in hf compatible format for vllm, etc.
            We rely on save_16bit_model which handles gathering partitioned weights in zero-3.
            Args:
                output_dir: Directory to save policy model
                tag: Checkpoint tag/identifier
                value_output_dir: Optional directory to save value model (whenever we need value function).
                                  If None and value model exists, defaults to {output_dir}_value.
            Note we must call this on ALL ranks for zero-3 correctness.
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:{self.alg_name}][Rank {rank}] Saving checkpoint with tag {tag}...")

        # 1. Policy weights
        if self.peft_config.use_peft:
            # Save full merged model (base + LoRA merged) so vllm can load it
            # directly for disk-based refresh. Must gather ALL params, base + adapter,
            # since ZeRO-3 partitions everything, not just trainable params.
            # Gather one parameter at a time to avoid gpu OOM as gathering all at once
            # would temporarily materialize the entire model on every gpu.
            raw_sd = self.gather_params_for_save(self.policy_engine.module, rank)
            status = True
            try:
                if rank == 0:
                    os.makedirs(output_dir, exist_ok=True)
                    merged_sd = self.merge_peft_state_dict(raw_sd)
                    self.save_state_dict_sharded(merged_sd, output_dir)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Saved merged PEFT policy model")

            except Exception as e:
                status = False
                print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving PEFT policy weights: {e}")

            self.barrier_with_error_check(status)

        else:
            # Gather non-PEFT weights and save as sharded safetensors, hf-compatible.
            # Uses the same gather + save path as PEFT for consistent output format.
            raw_sd = self.gather_params_for_save(self.policy_engine.module, rank)
            status = True
            try:
                if rank == 0:
                    os.makedirs(output_dir, exist_ok=True)
                    self.save_state_dict_sharded(raw_sd, output_dir)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Saved non-PEFT policy model")

            except Exception as e:
                status = False
                print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving non-PEFT policy weights: {e}")

            self.barrier_with_error_check(status)

        # 2. Policy config for rank 0 only. this is required for vllm
        status = True
        try:
            if rank == 0:
                # When PEFT is active, self.policy_engine.module is a PeftModel.
                # We must save the underlying base model's config so vllm can
                # load the merged checkpoint without peft specific fields.
                model_module = self.policy_engine.module
                if self.peft_config.use_peft and hasattr(model_module, 'get_base_model'):
                    base_config = model_module.get_base_model().config

                elif hasattr(model_module, 'config'):
                    base_config = model_module.config

                else:
                    base_config = None

                if base_config is not None:
                    base_config.save_pretrained(output_dir)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Policy config saved")

                else:
                    print(f"[Alg:{self.alg_name}][Rank {rank}] WARNING: Could not find model config to save for policy")

                # Save generation_config.json if available. This is needed by some hf pipelines or vllm.
                gen_cfg = getattr(model_module, 'generation_config', None)
                # For peft, generation_config lives on the base model.
                if gen_cfg is None and hasattr(model_module, 'get_base_model'):
                    gen_cfg = getattr(model_module.get_base_model(), 'generation_config', None)

                if gen_cfg is not None:
                    gen_cfg.save_pretrained(output_dir)
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Generation config saved")

        except Exception as e:
            status = False
            print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving policy config: {e}")

        self.barrier_with_error_check(status)

        # 3. Value model weights
        if hasattr(self, 'value_engine') and self.value_engine is not None:
            if value_output_dir is None:
                value_output_dir = output_dir.rstrip('/') + "_value"

            if self.peft_config.use_peft:
                # Gather one parameter at a time to avoid gpu OOM (same as policy save).
                raw_sd = self.gather_params_for_save(self.value_engine.module, rank)
                status = True
                try:
                    if rank == 0:
                        os.makedirs(value_output_dir, exist_ok=True)
                        # Value model params use backbone.* prefix from valueNetwork,
                        # not base_model.model.* from PeftModel. We need to save raw gathered
                        # weights without PEFT merge since ValueNetwork already unwrapped
                        # the PeftModel via get_base_model() at init time.
                        self.save_state_dict_sharded(raw_sd, value_output_dir)

                        # Save the PEFT config so the value checkpoint is reloadable.
                        # LoRA injected parameter names (e.g. backbone.model.layers.0.self_attn.q_proj.lora_A.default.weight)
                        # cannot be loaded without knowing the exact PEFT wrapping used. So the following is to fix that:
                        peft_cfg = {"peft_type": self.peft_config.peft_type,
                                    "lora_rank": self.peft_config.lora_rank,
                                    "lora_alpha": self.peft_config.lora_alpha,
                                    "lora_dropout": self.peft_config.lora_dropout,
                                    "lora_target_modules": self.peft_config.lora_target_modules,
                                    "task_type": self.peft_config.task_type,
                                    "is_value_model": True,
                                   }
                        with open(os.path.join(value_output_dir, "peft_config.json"), "w") as f:
                            json.dump(peft_cfg, f, indent=2)
                            f.flush()
                            os.fsync(f.fileno())
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Saved value model weights + peft_config.json")

                except Exception as e:
                    status = False
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving value model weights: {e}")

                self.barrier_with_error_check(status)
            else:
                # Gather non-PEFT value weights and save as sharded safetensors.
                raw_sd = self.gather_params_for_save(self.value_engine.module, rank)
                status = True
                try:
                    if rank == 0:
                        os.makedirs(value_output_dir, exist_ok=True)
                        self.save_state_dict_sharded(raw_sd, value_output_dir)
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Saved non-PEFT value model")

                except Exception as e:
                    status = False
                    print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving non-PEFT value weights: {e}")

                self.barrier_with_error_check(status)

            # 4. Value config for rank 0 only
            # The value model is a ValueNetwork (backbone + value_head), NOT a
            # standard CausalLM. So we need to save its modified config that records this so
            # users don't accidentally load it with AutoModelForCausalLM.
            status = True
            try:
                if rank == 0:
                    if hasattr(self.value_engine.module, 'config'):
                        value_config = copy.deepcopy(self.value_engine.module.config)
                        # Override architectures to prevent AutoModelForCausalLM from
                        # loading weights that have backbone.value_head structure.
                        value_config.architectures = ["ValueNetwork"]
                        value_config.auto_map = {}
                        # hidden_size is already present in the base model config.
                        value_config.save_pretrained(value_output_dir)
                        print(f"[Alg:{self.alg_name}][Rank {rank}] Value config saved (architectures=ValueNetwork)")

                    else:
                        print(f"[Alg:{self.alg_name}][Rank {rank}] WARNING: Could not find model config to save for value")

            except Exception as e:
                status = False
                print(f"[Alg:{self.alg_name}][Rank {rank}] Error saving value config: {e}")

            self.barrier_with_error_check(status)

        print(f"[Alg:{self.alg_name}][Rank {rank}] Checkpoint save completed!")

    def gather_params_for_save(self, module, rank):
        '''
            Gather all parameters from a deepspeed wrapped module one at a time.
            Returns {name: cpu_tensor} on rank 0, empty dict on others.
            Must be called on ALL ranks for ZeRO-3 collective correctness.
        '''
        params = []
        names = []
        for name, param in module.named_parameters():
            params.append(param)
            names.append(name)

        is_zero3 = any(hasattr(p, 'ds_id') for p in params) if params else False
        state_dict = {}

        # Before GatheredParameters:
        # param.data -> tensor([], device=cuda:N), only the local shard is present
        # During GatheredParameters:
        # param.data -> full tensor([...shape...]), all ranks all-gathered
        if is_zero3:
            for name, param in zip(names, params):
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if rank == 0:
                        state_dict[name] = param.data.cpu().clone()
        else:
            if rank == 0:
                for name, param in zip(names, params):
                    state_dict[name] = param.data.cpu().clone()

        return state_dict

    def save_state_dict_sharded(self, state_dict, output_dir, max_shard_size="5GB"):
        '''
            Saves a state dict with automatic sharding for large models.
            For small models that fit in a single shard, saves as model.safetensors.
            For large models, we split it into model-00001-of-NNNNN.safetensors shards
            and write model.safetensors.index.json so HF or vllm can load them.
        '''
        state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size)

        for filename, tensor_names in state_dict_split.filename_to_tensors.items():
            shard = {name: state_dict[name] for name in tensor_names}
            save_file(shard, os.path.join(output_dir, filename))

        if state_dict_split.is_sharded:
            index = {"metadata": state_dict_split.metadata,
                     "weight_map": state_dict_split.tensor_to_filename}

            with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

    def merge_peft_state_dict(self, raw_state_dict):
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
                raise RuntimeError(f"merge_peft_state_dict: found lora_A but no lora_B for {module_path}. "
                                   f"State dict is corrupt or incomplete.")

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
                raise RuntimeError(f"merge_peft_state_dict: no base weight found for {module_path}, "
                                   f"LoRA delta would be silently dropped.")

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
            This is used for direct sync of weights with vllm rather than saving them on disk.
            gather_params_for_save handles ZeRO-3 collective gathering and then
            merges PEFT names when active so vllm can load them.

            Returns:
                dict: {param_name: cpu_tensor} on rank 0, empty dict on other ranks.
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # gather_params_for_save handles ZeRO-3 collective gathering and works
        # for all DS stages. Must be called on ALL ranks.
        state_dict = self.gather_params_for_save(self.policy_engine.module, rank)

        # When peft is active, named_parameters() returns peft-prefixed names. For example:
        # base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight (frozen base)
        # base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight (adapter)
        # however, vllm expects original names like model.layers.0.self_attn.q_proj.weight.
        # So load_weights() won't recognize any of these and the sync silently fails. To fix this,
        # we merge LoRA weights into base model weights and remap to original HF names so vllm can load them.
        ok = True
        if rank == 0 and self.peft_config.use_peft:
            try:
                state_dict = self.merge_peft_state_dict(state_dict)
            except Exception as e:
                ok = False
                print(f"[Alg:{self.alg_name}][Rank {rank}] Error merging PEFT state_dict: {e}")

        # Synchronize to ensure no rank continues until Rank 0 finishes cpu processing.
        # Use barrier_with_error_check so any merge failure on rank 0 propagates
        # immediately instead of hanging other ranks at a plain barrier.
        self.barrier_with_error_check(ok)

        if rank == 0:
            print(f"[Alg:{self.alg_name}][Rank {rank}] Gathered state_dict (peft={self.peft_config.use_peft})!")

        return state_dict

    def init_weight_nccl_group(self, master_addr, master_port, rank, world_size, group_name, timeout_seconds, backend):
        '''
            Initialize a process group for weight broadcast.
            Called only on training rank 0, concurrently with all vllm rollout
            workers, as rendezvous requires all participants to call in together.
            Other training ranks are not part of this group as they only participate
            in broadcast_weights_nccl for the zero-3 gather collective.
            For gloo: we use create_nccl_process_group which is torch.distributed internals.
            For nccl: we use vllm's StatelessProcessGroup + PyNcclCommunicator so both
                trainer and vLLM EngineCore workers share the same nccl communicator
                bootstrapped via the same TCP store and nccl ID exchange protocol.
        '''
        torch.cuda.set_device(self.policy_engine.device)
        self.weight_sync_backend = backend

        if backend == "nccl":
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            pg = StatelessProcessGroup.create(host=master_addr,
                                              port=master_port,
                                              rank=rank,
                                              world_size=world_size)
            device = self.policy_engine.device
            self.weight_sync_pynccl = PyNcclCommunicator(pg, device=device)
            self.weight_sync_group = None

        else:
            self.weight_sync_group = create_nccl_process_group(init_method=f"tcp://{master_addr}:{master_port}",
                                                              rank=rank,
                                                              world_size=world_size,
                                                              group_name=group_name,
                                                              timeout_seconds=timeout_seconds,
                                                              backend=backend)
            self.weight_sync_pynccl = None

        mechanism = "PyNcclCommunicator" if backend == "nccl" else "torch.distributed"
        print(f"[Alg:{self.alg_name}] Training weight sync initialized (backend={backend}, mechanism={mechanism}, rank={rank}, world_size={world_size})", flush=True)
        return True

    def gather_weights_for_nccl(self):
        '''
            Phase 1 of NCCL weight sync: gather the ZeRO-3 state dict.
            Must be called on ALL training ranks. Rank 0 stores the gathered state dict
            and returns param metadata [(name, dtype_str, shape), ...]. Other ranks return [].
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        device = self.policy_engine.device
        torch.cuda.set_device(device)

        # Free cached memory before ZeRO-3 gather. GatheredParameters does
        # all-gather ops that need temporary buffers; a fragmented allocator
        # can cause cache flushes that corrupt persistent param buffers.
        torch.cuda.empty_cache()

        state_dict = self.gather_state_dict()

        # Check live model params after ZeRO-3 gather.
        # GatheredParameters temporarily all-gathers into p.data and reverts on exit.
        # If the all-gather corrupts the buffer (e.g. CUDA stream race), the live
        # model params now have NaN even though the CPU copies might be clean.
        nan_live = [name for name, p in self.policy_engine.module.named_parameters()
                    if p.requires_grad and p.numel() > 0 and torch.isnan(p).any()]
        if nan_live:
            raise RuntimeError(f"[Alg:{self.alg_name}][Rank {rank}] NaN in {len(nan_live)} live model "
                               f"params after ZeRO-3 gather: {nan_live[:10]}. "
                               f"GatheredParameters corrupted the model.")

        if rank == 0 and state_dict:
            self._pending_nccl_state_dict = state_dict
            metadata = [(name, str(param.dtype), tuple(param.shape)) for name, param in state_dict.items()]

            # Check gathered state_dict (CPU copies) for NaN before broadcasting.
            nan_in_sync = [name for name, param in state_dict.items()
                           if param.numel() > 0 and torch.isnan(param).any()]
            if nan_in_sync:
                raise RuntimeError(f"[Alg:{self.alg_name}][Rank 0] NaN in {len(nan_in_sync)} params "
                                   f"gathered for weight sync: {nan_in_sync[:10]}. "
                                   f"Aborting broadcast to prevent corrupting rollout engines.")

            print(f"[Alg:{self.alg_name}][Rank 0] Gathered {len(metadata)} params for NCCL broadcast", flush=True)
            return metadata

        return []

    def nccl_broadcast_gathered(self):
        '''
            Phase 2 of weight sync, training rank 0 only: broadcast all
            gathered params. Rollout engine receives must already be
            scheduled by the driver so they enter broadcast in lockstep.
            For gloo: uses torch.distributed.broadcast with CPU tensors.
            For nccl: uses PyNcclCommunicator.broadcast with GPU tensors.
        '''
        state_dict = getattr(self, '_pending_nccl_state_dict', None)
        if not state_dict:
            return 0

        device = self.policy_engine.device
        torch.cuda.set_device(device)

        param_names = list(state_dict.keys())
        num_params = len(param_names)
        start = time.time()

        backend = self.weight_sync_backend
        use_pynccl = (backend == "nccl")

        for i, name in enumerate(param_names):
            param = state_dict[name]
            if use_pynccl:
                send_buf = param.to(device=device, non_blocking=True) if not param.is_cuda else param.contiguous()
                self.weight_sync_pynccl.broadcast(send_buf, src=0, stream=torch.cuda.current_stream())

            else:
                send_buf = param.cpu() if param.is_cuda else param.contiguous()
                torch.distributed.broadcast(send_buf, src=0, group=self.weight_sync_group)
            del send_buf

        # Ensure all NCCL broadcasts complete before releasing state dict
        if use_pynccl:
            torch.cuda.synchronize()

        elapsed = time.time() - start
        print(f"[Alg:{self.alg_name}][Rank 0] Weight broadcast {num_params} params in {elapsed:.2f}s", flush=True)

        del self._pending_nccl_state_dict
        return num_params

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

    def barrier_with_error_check(self, succeeded: bool):
        '''
            Synchronized barrier that propagates failures across ranks.
            Unlike torch.distributed.barrier(), if any rank fails (e.g. disk full,
            permission error during save), all ranks detect the failure and raise
            immediately instead of hanging until NCCL timeout. Each rank publishes a 0/1 flag
            via all_reduce(MIN). If any rank failed, the reduced flag is 0 and every rank raises.
            In single-process mode, raises directly on failure.
        '''
        if not torch.distributed.is_initialized():
            if not succeeded:
                raise RuntimeError(f"[Alg:{self.alg_name}] Checkpoint operation failed on this process")

            return

        flag = torch.tensor([1 if succeeded else 0], dtype=torch.int32, device=self.policy_engine.device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)

        if flag.item() == 0:
            raise RuntimeError(f"[Alg:{self.alg_name}] Checkpoint operation failed on at least one rank "
                               f"(this is rank {torch.distributed.get_rank()}), all ranks aborting "
                                f"to prevent deadlock")

    def save_engine_state(self, engine_state_dir):
        '''
            Save deepspeed engine state such as optimizer, LR scheduler for training resume.
            Uses DS native save_checkpoint which handles ZeRO partition persistence
            so each rank saves its own optimizer shard. Must be called on ALL ranks.
            RNG states are stored in client_state so each rank preserves its own
            Python/NumPy/PyTorch/CUDA random generator state.
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # Create the directory from inside the actor (rank 0) rather than from the driver
        # to avoid NFS metadata propagation races on multi-node clusters.
        if rank == 0:
            os.makedirs(engine_state_dir, exist_ok=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        client_state = {'rng_python': random.getstate(),
                        'rng_numpy': np.random.get_state(),
                        'rng_torch_cpu': torch.random.get_rng_state(),
                        # Preserve _train_step_calls for reproducible micro-batch shuffling on resume
                        '_train_step_calls': getattr(self, '_train_step_calls', 0),}

        if torch.cuda.is_available():
            client_state['rng_torch_cuda'] = torch.cuda.get_rng_state()

        self.policy_engine.save_checkpoint(engine_state_dir, tag="policy", client_state=client_state)
        print(f"[Alg:{self.alg_name}][Rank {rank}] Saved policy engine state")

        if hasattr(self, 'value_engine') and self.value_engine is not None:
            self.value_engine.save_checkpoint(engine_state_dir, tag="value")
            print(f"[Alg:{self.alg_name}][Rank {rank}] Saved value engine state")

    def load_engine_state(self, engine_state_dir):
        '''
            Load DeepSpeed engine state: model weights, optimizer, and lr scheduler
            from a previous checkpoint for training resume. Must be called on ALL ranks after init_training_engine().
            Returns client_state dict (contains RNG states) or empty dict.
        '''
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        load_path, client_state = self.policy_engine.load_checkpoint(engine_state_dir, tag="policy")

        if load_path is None:
            raise FileNotFoundError(f"[Alg:{self.alg_name}][Rank {rank}] DeepSpeed failed to load policy checkpoint "
                                    f"from {engine_state_dir}/policy. load_checkpoint returned (None, None) — "
                                    f"directory may be missing or corrupt.")

        print(f"[Alg:{self.alg_name}][Rank {rank}] Loaded policy engine state from {load_path}")

        if hasattr(self, 'value_engine') and self.value_engine is not None:
            value_load_path, _ = self.value_engine.load_checkpoint(engine_state_dir, tag="value")
            if value_load_path is None:
                raise FileNotFoundError(f"[Alg:{self.alg_name}][Rank {rank}] DeepSpeed failed to load value checkpoint "
                                        f"from {engine_state_dir}/value. load_checkpoint returned (None, None) — "
                                        f"directory may be missing or corrupt.")

            print(f"[Alg:{self.alg_name}][Rank {rank}] Loaded value engine state from {value_load_path}")

        # Restore per-rank rng states and train_step_calls counter
        if client_state:
            if 'rng_python' in client_state:
                random.setstate(client_state['rng_python'])

            if 'rng_numpy' in client_state:
                np.random.set_state(client_state['rng_numpy'])

            if 'rng_torch_cpu' in client_state:
                torch.random.set_rng_state(client_state['rng_torch_cpu'])

            if 'rng_torch_cuda' in client_state and torch.cuda.is_available():
                torch.cuda.set_rng_state(client_state['rng_torch_cuda'])

            if '_train_step_calls' in client_state:
                self._train_step_calls = client_state['_train_step_calls']
            print(f"[Alg:{self.alg_name}][Rank {rank}] Restored RNG states")

        return client_state or {}