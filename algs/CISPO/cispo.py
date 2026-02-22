import torch
import numpy as np
from tqdm import tqdm
from typing import Any
import ray

# load follwoings from common.py:
# _load_single_model, init_training_engine, policy_forward,
# ref_forward, compute_kl_distance, save_checkpoint
from algs.RL.common import COMMON

@ray.remote
class CISPO(COMMON):
    def __init__(self,
                 model_path: str,
                 model_dtype: torch.dtype,
                 trust_remote_code: bool,
                 attn_impl: str,
                 kl_coeff: float,
                 clip_low: float,
                 clip_high: float,
                 entropy_coeff: float,
                 micro_batch_size_per_gpu: int,
                 update_after_full_replay: bool,
                 deepspeed_config: Any,
                 gradient_checkpointing: bool,
                 ref_model_path: str = None,
                 deepspeed_ref_config: Any = None,
                 peft_config: Any = None,
                 ):

        self.alg_name = self.__class__.__name__
        # model related parameters
        self.model_path = model_path
        self.ref_model_path = ref_model_path
        self.attn_impl = attn_impl
        self.model_dtype = model_dtype
        self.trust_remote_code = trust_remote_code
        self.peft_config = peft_config

        # training related parameters
        self.deepspeed_config = deepspeed_config
        self.deepspeed_ref_config = deepspeed_ref_config
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu
        self.gradient_checkpointing = gradient_checkpointing

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

    def load_model(self):
        '''
            Load policy and reference models from huggingface.
        '''
        # Load policy model
        model = self._load_single_model(model_path=self.model_path, dtype=self.model_dtype, model_name="policy")

        # Load reference model if provided
        if self.ref_model_path and self.kl_coeff > 0.0:
            ref_model = self._load_single_model(model_path=self.ref_model_path, dtype=self.model_dtype, model_name="ref")

        else:
            ref_model = None

        return {"policy_model": model, "ref_model": ref_model}

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            entropies: torch.Tensor,
                            ref_logprobs: torch.Tensor,
                            ):
        '''
            logprobs: [B, T-1]
            old_logprobs, advantages, mask: [B, T - 1]
            entropies: [B, T-1]
            ref_logprobs: [B, T-1]
            Compute policy loss:
                1. ratio = exp(logprobs - old_logprobs)
                2. loss = -(min(ratio * adv, clip_adv * adv)) * mask
        '''
        device = logprobs.device
        dtype = logprobs.dtype
        loss_ent = torch.tensor(0.0, device=device, dtype=dtype)
        kl_ref   = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. make sure advantages are detached and
        # convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)
        mask_bool = (mask.to(device=device) > 0.5)
        mask = mask_bool.to(dtype=dtype)
        denom = mask.sum().clamp(min=1.0)

        # 2. calculate ratio = pi / pi_old = exp(logprobs - old_logprobs)
        raw_logratio = (logprobs - old_logprobs).to(torch.float32)
        # Ignore invalid (padded) positions before exp to avoid inf * 0 -> nan.
        logratio = torch.where(mask_bool, raw_logratio, torch.zeros_like(raw_logratio))
        ratio    = torch.exp(logratio)

        # 3. CISPO loss: clipped_ratio.detach() * log(pi) * advantage
        # Unlike PPO, CISPO clips the importance ratio and uses it as a weighting
        # coefficient for the policy's log-probability more like policy gradient.
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
        loss_pi = -(clipped_ratio.detach() * logprobs * adv * mask).sum() / denom

        # 4. compute entropy loss
        if entropies is not None and self.ent_coeff > 0.0:
            loss_ent = (entropies * mask).sum() / denom

        if ref_logprobs is not None and self.kl_coeff > 0.0:
            kl_dist = self.compute_kl_distance(logprobs=logprobs, ref_logprobs=ref_logprobs)
            # avoid calculating kl for padded tokens.
            kl_dist = torch.where(mask_bool, kl_dist, torch.zeros_like(kl_dist))
            kl_ref  = (kl_dist * mask).sum() / denom

        loss_total = loss_pi - self.ent_coeff * loss_ent + self.kl_coeff * kl_ref

        # 5. useful metrics
        with torch.no_grad():
            # first term too large ==> policy changed too much upward
            # second term too small ==> policy changed too much downward
            clipped_mask = (ratio > (1.0 + self.clip_high)) | (ratio < (1.0 - self.clip_low))
            # fraction of masked tokens that ratio out of ranges
            clipfrac = (clipped_mask.to(dtype=dtype) * mask).sum() / denom

            # approx KL (var-reduced): log(pi/pi_old) + pi_old/pi - 1
            # logratio = log(pi/pi_old)
            ratio_inv = torch.exp(-logratio)
            approx_kl_t = logratio + ratio_inv - 1.0
            approx_kl = (approx_kl_t.to(dtype=dtype) * mask).sum() / denom

            # save the metrics for debugging
            metrics = {
                'clipfrac': clipfrac.item(),
                'approx_kl': approx_kl.item(),
                'ent_loss': loss_ent.item(),
                'pi_loss': loss_pi.item(),
                'pi_loss_total': loss_total.item(),
                'kl_ref': kl_ref.item(),
            }

        return loss_total, metrics

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for local_batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.
        '''
        assert self.policy_engine is not None, "DeepSpeed engine not initialized"
        assert isinstance(micro_batches, list) and len(micro_batches) > 0, \
            "micro_batches must be a non-empty list which should be equal across "
            "ranks via prepare_training_batches padding"

        device = self.policy_engine.device

        # 1. Models to train mode
        self.policy_engine.train()

        # 2. zero grads
        self.policy_engine.zero_grad()

        # 3. create progress bar
        num_micro = len(micro_batches)
        # torch.distributed.get_rank() would be the same thing as engine_id
        if engine_id == 0:
            progress_bar = tqdm(micro_batches, total=num_micro, desc="[Alg:{}] Training Step in rank {}".format(self.alg_name, engine_id))

        else:
            progress_bar = micro_batches # No tqdm for other ranks

        ga_pi_attr = getattr(self.policy_engine, 'gradient_accumulation_steps', 1)
        ga_pi = int(ga_pi_attr() if callable(ga_pi_attr) else ga_pi_attr)

        # track metrics across all micro-batches
        all_metrics = []
        for step, micro_batch in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))

            # If update_only_after_full_replay is True, we only update at the very end
            # of the shard. Otherwise, we respect ga_pi.
            if self.update_only_after_full_replay:
                is_boundary = is_last
            else:
                is_boundary = (((step + 1) % ga_pi) == 0) or is_last

            ########
            # 1. Data from buffer
            ########
            # all are [B, T]
            # zscore is normalized rewards using the number of samples for each proompt (X -mu) / (std + eps)
            # this is a simple baseline for policy gradients (PPO in this code) as it reflects relative quality
            # among that prompt’s samples.
            advs      = micro_batch['zscore'][:, :-1].to(device, non_blocking=True)
            mask      = micro_batch['mask'][:, :-1].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'][:, :-1].to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            ########
            # 2. Compute loss
            ########
            # Forward pass through the current policy.
            pi_logprobs, pi_entropies, target_ids = self.policy_forward(input_ids=input_ids,
                                                                        att_mask=att_mask,
                                                                        pos_ids=pos_ids)

            ref_logprobs = None
            if self.kl_coeff > 0.0 and self.ref_model_engine is not None:
                ref_logprobs = self.ref_forward(input_ids=input_ids,
                                                att_mask=att_mask,
                                                pos_ids=pos_ids,
                                                )

            # Compute policy loss using the current policy.
            pi_loss, pi_metrics = self.compute_policy_loss(logprobs=pi_logprobs,
                                                           old_logprobs=old_logprobs,
                                                           advantages=advs,
                                                           mask=mask,
                                                           entropies=pi_entropies,
                                                           ref_logprobs=ref_logprobs)

            # store metrics
            all_metrics.append(pi_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({
                    "pi_loss": f"{pi_loss.item():.4f}",
                    "clipfrac": f"{pi_metrics['clipfrac']:.3f}",
                    "approx_kl": f"{pi_metrics['approx_kl']:.4f}",
                    "kl_ref": f"{pi_metrics['kl_ref']:.4f}"
                })

            # When accumulating over the full replay shard, normalize the loss
            # by the number of micro-batches so the gradient magnitude equals the
            # mean (not the sum) of per-micro-batch gradients. DeepSpeed will
            # still divide by gradient_accumulation_steps, so we multiply by
            # ga_pi to keep the effective scale consistent with standard GA.
            if self.update_only_after_full_replay:
                pi_loss = pi_loss * (ga_pi / num_micro)

            # For DeepSpeed, we must coordinate is_boundary with the backward pass.
            self.policy_engine.set_gradient_accumulation_boundary(is_boundary)

            # backward pass
            self.policy_engine.backward(pi_loss)
            self.policy_engine.step()

        # aggregate metrics across all micro-batches
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])

        return aggregated_metrics
