import os
import torch
import numpy as np
from tqdm import tqdm
import ray
import deepspeed
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoConfig

@ray.remote
class PG:
    def __init__(self,
                 model_path: str,
                 model_dtype: torch.dtype,
                 trust_remote_code: bool,
                 attn_impl: str,
                 kl_coeff: float,
                 clip_low: float,
                 clip_high: float,
                 entropy_coeff: float,
                 use_cache: bool,
                 micro_batch_size_per_gpu: int,
                 update_after_full_replay: bool,
                 deepspeed_config: deepspeed.DeepSpeedConfig,
                 ref_model_path: str = None,
                 ):

        # model related parameters
        self.model_path = model_path
        self.ref_model_path = ref_model_path
        self.use_cache = use_cache
        self.attn_impl = attn_impl
        self.model_dtype = model_dtype
        self.trust_remote_code = trust_remote_code

        # training related parameters
        self.deepspeed_config = deepspeed_config
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu

        # policy related parameters
        self.kl_coeff = float(kl_coeff)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.ent_coeff = float(entropy_coeff)

        # use cross entropy loss for policy gradient
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        # if true, it means the update is done after seeing all samples in the reply buffer
        # treating the entire buffer as a single batch.
        self.update_only_after_full_replay = update_after_full_replay

        self.ready = False
        self.init_training_engine()
        self.ready = True

    def is_ready(self):
        '''
            Barrier method to ensure all Ray actors are initialized before DeepSpeed collective ops.
        '''
        return self.ready

    def init_training_engine(self):
        '''
            Since, we are using ray, each ray actor MUST create its own deepspeed engine.
            This is because each ray actor process is a separate process as it should be 1 actor = 1 gpu = 1 ds rank.
        '''
        # Convert pydantic model to python Dict for DeepSpeed
        ds_config_dict = self.deepspeed_config.model_dump()

        # check to avoid re-initializing distributed backend
        if not torch.distributed.is_initialized():
            # 1. Initialize distributed training engine
            deepspeed.init_distributed()

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        print(f"[Alg:PG][Rank {rank}] Initializing training engine...")

        # 2. Load model
        model, ref_model = self.load_model()
        print(f"[Alg:PG][Rank {rank}] Model loaded: {self.model_path}")

        # 2. Initialize model engine
        self.policy_engine, self.optimizer, _, _ = deepspeed.initialize(
                                                            model=model,
                                                            model_parameters=model.parameters(),
                                                            config=ds_config_dict
                                                            )
        print(f"[Alg:PG][Rank {rank}] DeepSpeed engine initialized on device: {self.policy_engine.device}")

        self.ref_model_engine = None
        if ref_model is not None:
            # ref_model is supported here in case if we want to add
            # additional metrics, divergence, etc. Note, ref_model will not be optimized.
            try:
                ref_model.to(self.policy_engine.device)
                ref_model.eval()
                self.ref_model_engine = ref_model
                print(f"[Alg:PG][Rank {rank}] Reference model loaded")

            except:
                # fallback: initialize with DeepSpeed
                self.ref_model_engine, _, _, _ = deepspeed.initialize(
                                                        model=ref_model,
                                                        config=ds_config_dict
                                                        )
                print(f"[Alg:PG][Rank {rank}] Reference model initialized with DeepSpeed fallback")

    def load_model(self):
        '''
            Load models and tokenizer from huggingface.
        '''
        assert self.model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"
        assert self.attn_impl=='' or self.attn_impl in ['eager', 'flash_attention_2'], "attn_impl must be one of 'eager', 'flash_attention_2' or empty string"

        # 1. model and its config initialization
        model_config = AutoConfig.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                    dtype=self.model_dtype,
                                                    trust_remote_code=self.trust_remote_code,
                                                    config=model_config,
                                                    attn_implementation=None if self.attn_impl == '' else self.attn_impl)

        # if ref model is provided to use it in kl for example.
        if self.ref_model_path:
            ref_model = AutoModelForCausalLM.from_pretrained(self.ref_model_path,
                                                            dtype=self.model_dtype,
                                                            trust_remote_code=self.trust_remote_code,
                                                            config=model_config,
                                                            attn_implementation=None if self.attn_impl == '' else self.attn_impl)
        else:
            ref_model = None

        return model, ref_model

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
        output = self.policy_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, V] -> [B, T-1, V]
        logits = output.logits[:, :-1, :].contiguous()
        B, T_minus_1, vocab_size = logits.shape

        # [B, T] -> [B, T-1]
        target_ids = input_ids[:, 1:].contiguous()

        # cross_entropy return -logprobs but we need logprobs
        # logits is [B, T-1, vocab_size] --> [B * (T-1), vocab_size]
        # target_ids is [B, T-1] --> [B * (T-1)]
        neg_logprobs = self.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
        logprobs = -neg_logprobs.view(B, T_minus_1)
        # we can also do this, but it is less efficient I guess
        #   logprobs = logits.log_softmax(dim=-1)
        #   logprobs = torch.gather(logprobs, dim=-1, index=target_ids)

        entropies = None
        if self.ent_coeff > 0.0:
            entropies = torch.distributions.Categorical(logits=logits).entropy()

        return logprobs, entropies

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            entropies: torch.Tensor,
                            ):
        '''
            logprobs: [B, T-1]
            old_logprobs, advantages, mask: [B, T - 1]
            entropies: [B, T-1]
            Compute policy loss:
                1. ratio = exp(logprobs - old_logprobs)
                2. loss = -(min(ratio * adv, clip_adv * adv)) * mask
        '''
        device = logprobs.device
        dtype = logprobs.dtype
        loss_ent = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. make sure advantages are detached and
        # convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)
        mask = (mask.to(device=device) > 0.5).to(dtype=dtype)
        denom = mask.sum().clamp(min=1.0)

        # 2. calculate ratio = pi / pi_old = exp(logprobs - old_logprobs)
        logratio = (logprobs - old_logprobs).to(torch.float32)
        ratio   = torch.exp(logratio)

        # 3. compute loss: -(min(ratio * adv, clip_adv)) * mask
        unclipped = ratio * adv
        clip_adv  = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high) * adv
        loss_pi   = -(torch.minimum(unclipped, clip_adv) * mask).sum() / denom

        # 4. compute entropy loss
        if entropies is not None and self.ent_coeff > 0.0:
            loss_ent = (entropies * mask).sum() / denom

        loss_total = loss_pi - self.ent_coeff * loss_ent

        # 5. useful metrics
        with torch.no_grad():
            # first term too large ==> policy changed too much upward
            # second term too small ==> policy changed too much downward
            clipped_mask = (ratio > (1.0 + self.clip_high)) | (ratio < (1.0 - self.clip_low))
            # fraction of masked tokens that ratio out of ranges
            clipfrac = (clipped_mask.to(dtype=dtype) * mask).sum() / denom

            # approx KL: either E[old_logprobs - logprobs] or E[(ratio - 1) - logratio]
            approx_kl_t = (ratio - 1.0) - logratio
            approx_kl = (approx_kl_t.to(dtype=dtype) * mask).sum() / denom

            # save the metrics for debugging
            metrics = {
                'clipfrac': clipfrac.item(),
                'approx_kl': approx_kl.item(),
                'loss_ent': loss_ent.item(),
                'loss_pi': loss_pi.item(),
                'loss_total': loss_total.item(),
            }

        return loss_total, metrics

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for local_batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.
        '''
        assert self.policy_engine is not None, "DeepSpeed engine not initialized"

        device = self.policy_engine.device

        # 1. Models to train mode
        self.policy_engine.train()

        # 2. zero grads
        self.policy_engine.zero_grad()

        # 3. create progress bar
        num_micro = len(micro_batches)
        # torch.distributed.get_rank() would be the same thing as engine_id
        if engine_id == 0:
            progress_bar = tqdm(micro_batches, total=num_micro, desc="[Alg:PG] Training Step in rank {}".format(engine_id))

        else:
            progress_bar = micro_batches # No tqdm for other ranks

        ga_pi_attr = getattr(self.policy_engine, 'gradient_accumulation_steps', 1)
        ga_pi = int(ga_pi_attr() if callable(ga_pi_attr) else ga_pi_attr)

        # track metrics across all micro-batches
        all_metrics = []
        for step, micro_batch in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))
            is_boundary = (((step + 1) % ga_pi) == 0) or is_last

            ########
            # 1. Data from buffer
            ########
            # all are [B, T]
            # zscore is normalized rewards using the number of samples for each proompt (X -mu) / (std + eps)
            # this is a simple baseline for policy gradients (PPO in this code) as it reflects relative quality
            # among that prompt’s samples.
            advs      = micro_batch['zscore'][:, :-1].to(device, non_blocking=True)
            done      = micro_batch['done'][:, :-1].to(device, non_blocking=True)
            mask      = micro_batch['mask'][:, :-1].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'][:, :-1].to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            ########
            # 2. Compute loss
            ########
            # Forward pass through the current policy.
            pi_logprobs, pi_entropies = self.policy_forward(input_ids=input_ids,
                                                            att_mask=att_mask,
                                                            pos_ids=pos_ids)

            # Compute policy loss using the current policy.
            pi_loss, pi_metrics = self.compute_policy_loss(logprobs=pi_logprobs,
                                                           old_logprobs=old_logprobs,
                                                           advantages=advs,
                                                           mask=mask,
                                                           entropies=pi_entropies)

            # store metrics
            all_metrics.append(pi_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({
                    "loss": f"{pi_loss.item():.4f}",
                    "clip": f"{pi_metrics['clipfrac']:.3f}",
                    "kl": f"{pi_metrics['approx_kl']:.4f}"
                })

            if self.update_only_after_full_replay:
                # Accumulate gradients across all micro-batches, only update at the end
                self.policy_engine.set_gradient_accumulation_boundary(is_boundary)
            else:
                # Update after every micro-batch (treat each as a boundary)
                self.policy_engine.set_gradient_accumulation_boundary(True)

            # backward pass
            self.policy_engine.backward(pi_loss)
            self.policy_engine.step()

        # aggregate metrics across all micro-batches
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])

        return aggregated_metrics

    def save_checkpoint(self, output_dir: str, tag: str):
        '''
            Saves the model in hf compatible format for vllm, etc.
            We rely on save_16bit_model which handles gathering partitioned weights in zero-3.

            Note we must call this on ALL ranks for zero-3 correctness.
        '''
        rank = torch.distributed.get_rank()
        print(f"[Alg:PG][Rank {rank}] Saving checkpoint to {output_dir} with tag {tag}...")

        try:
            # 1. Save model weights (gathered fp16/bf16)
            # save_16bit_model internally handles zero-3 gathering from all ranks
            self.policy_engine.save_16bit_model(output_dir)

            # Barrier to ensure all ranks finished writing before rank 0 saves config
            # Without this, rank 0 might save config before other ranks write their shards
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # 2. Save config (required for vllm) on rank 0 ONLY
            if rank == 0:
                if hasattr(self.policy_engine.module, 'config'):
                    self.policy_engine.module.config.save_pretrained(output_dir)
                    print(f"[Alg:PG][Rank {rank}] Config saved")

                else:
                    # fallback by trying to get config from the model itself
                    if hasattr(self.policy_engine.module, 'module'):
                        # wrapped model e.g., deepspeed wrapper
                        if hasattr(self.policy_engine.module.module, 'config'):
                            self.policy_engine.module.module.config.save_pretrained(output_dir)
                            print(f"[Alg:PG][Rank {rank}] Config saved (fallback)")

            # make sure rank 0 finished writing config
            # this ensures vLLM refresh can safely read all files
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            print(f"[Alg:PG][Rank {rank}] Checkpoint save completed!")

        except Exception as e:
            # log error but don't crash allows other ranks to continue
            print(f"[Alg:PG][Rank {rank}] Error saving checkpoint to {output_dir}: {e}")
            if torch.distributed.is_initialized():
                # still need barrier even on error to prevent deadlock
                torch.distributed.barrier()
            raise