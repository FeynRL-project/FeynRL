import torch
import numpy as np
from tqdm import tqdm
from typing import Any
import ray
import random

# load follwoings from common.py:
# load_single_model, init_training_engine, policy_forward,
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
                 normalize_loss: bool,
                 deepspeed_config: Any,
                 gradient_checkpointing: bool,
                 seed: int,
                 train_steps_per_epoch: int,
                 ref_model_path: str = None,
                 deepspeed_ref_config: Any = None,
                 peft_config: Any = None,
                 use_decoupled_loss: bool = False,
                 behave_imp_weight_cap: float = None,
                 ):

        self.alg_name = self.__class__.__name__
        # model related parameters
        self.model_path = model_path
        self.ref_model_path = ref_model_path
        self.attn_impl = attn_impl
        self.model_dtype = model_dtype
        self.trust_remote_code = trust_remote_code
        self.peft_config = peft_config
        self.seed = seed

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

        # params for decoupled ppo loss
        self.use_decoupled_loss = use_decoupled_loss
        self.behave_imp_weight_cap = float(behave_imp_weight_cap) if behave_imp_weight_cap is not None else None

        # if true, it means the update is done after seeing all samples in the reply buffer
        # treating the entire buffer as a single batch.
        self.update_only_after_full_replay = update_after_full_replay
        self.normalize_loss = normalize_loss

        # Following are used to snapshot pi_prox once per epoch on the first train_step
        # and reuse the cached snapshot across the remaining iterations.
        self.train_steps_per_epoch  = int(train_steps_per_epoch)
        self.cached_prox_logprobs   = None
        self.cached_prox_nan_masks  = None

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
        model = self.load_single_model(model_path=self.model_path, dtype=self.model_dtype, model_name="policy")

        # Load reference model if provided
        if self.ref_model_path and self.kl_coeff > 0.0:
            ref_model = self.load_single_model(model_path=self.ref_model_path, dtype=self.model_dtype, model_name="ref")

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
                            prox_logprobs: torch.Tensor = None,
                            ):
        '''
            logprobs: [B, T-1]
            old_logprobs, advantages, mask: [B, T - 1]
            entropies: [B, T-1]
            ref_logprobs: [B, T-1]
            prox_logprobs: [B, T-1]
            Compute policy loss:
                1. ratio = exp(logprobs - old_logprobs)
                2. loss = -(min(ratio * adv, clip_adv * adv)) * mask
            Returns:
                loss_total_sum: scalar tensor, raw sum of masked losses (no normalization).
                denom: local token count in this micro-batch (for metrics and fallback normalization).
                metrics: dict, per-token mean metrics using local denom for interpretability.
        '''
        device = logprobs.device
        dtype = logprobs.dtype
        ent_sum = torch.tensor(0.0, device=device, dtype=dtype)
        kl_sum  = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. make sure advantages are detached and
        # convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)
        mask_bool = (mask.to(device=device) > 0.5)
        mask = mask_bool.to(dtype=dtype)
        denom = mask.sum().clamp(min=1.0)

        if self.use_decoupled_loss:
            # 2. Decoupled policy loss (https://arxiv.org/abs/2110.00641)
            # loss = - E[ (pi_prox/pi_behav) * min(r_prox * A, clip(r_prox) * A)]
            # where r_prox = pi / pi_prox
            # pr_prox = pi / pi_prox
            raw_logratio_prox = (logprobs - prox_logprobs).to(torch.float32)
            logratio_prox     = torch.where(mask_bool, raw_logratio_prox, torch.zeros_like(raw_logratio_prox))
            r_prox            = torch.exp(logratio_prox)

            # Behavioral ratio: w = pi_prox / pi_behav
            raw_log_w = (prox_logprobs - old_logprobs).to(torch.float32)
            log_w     = torch.where(mask_bool, raw_log_w, torch.zeros_like(raw_log_w))
            behave_w  = torch.exp(log_w)
            if self.behave_imp_weight_cap is not None:
                behave_w = torch.clamp(behave_w, min=0.0, max=self.behave_imp_weight_cap)

            # CISPO loss: clip(r_prox).detach() * log(pi) * A * w
            clipped_ratio = torch.clamp(r_prox, 1.0 - self.clip_low, 1.0 + self.clip_high)
            pi_sum = -(clipped_ratio.detach() * logprobs * adv * behave_w * mask).sum()

            ratio    = r_prox
            logratio = logratio_prox

        else:
            # 2. calculate ratio = pi / pi_old = exp(logprobs - old_logprobs)
            raw_logratio = (logprobs - old_logprobs).to(torch.float32)
            # Ignore invalid (padded) positions before exp to avoid inf * 0 -> nan.
            logratio = torch.where(mask_bool, raw_logratio, torch.zeros_like(raw_logratio))
            ratio    = torch.exp(logratio)

            # 3. CISPO loss as raw sum: clipped_ratio.detach() * log(pi) * advantage
            # Unlike PPO, CISPO clips the importance ratio and uses it as a weighting
            # coefficient for the policy's log-probability more like policy gradient.
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
            pi_sum = -(clipped_ratio.detach() * logprobs * adv * mask).sum()

        # 4. compute entropy loss (raw sum)
        if entropies is not None and self.ent_coeff > 0.0:
            ent_sum = (entropies * mask).sum()

        if ref_logprobs is not None and self.kl_coeff > 0.0:
            kl_dist = self.compute_kl_distance(logprobs=logprobs, ref_logprobs=ref_logprobs)
            # avoid calculating kl for padded tokens.
            kl_dist = torch.where(mask_bool, kl_dist, torch.zeros_like(kl_dist))
            kl_sum  = (kl_dist * mask).sum()

        loss_total_sum = pi_sum - self.ent_coeff * ent_sum + self.kl_coeff * kl_sum

        # 5. useful metrics. Here per-token means using local denom for interpretability.
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
            metrics = {'clipfrac': clipfrac.item(),
                       'approx_kl': approx_kl.item(),
                       'ent_loss': (ent_sum / denom).item(),
                       'pi_loss': (pi_sum / denom).item(),
                       'loss_total': (loss_total_sum / denom).item(),
                       'kl_ref': (kl_sum / denom).item(),}

            if self.use_decoupled_loss:
                behave_w_masked = behave_w * mask
                metrics['behave_w_mean'] = (behave_w_masked.sum() / denom).item()
                metrics['behave_w_max'] = behave_w[mask_bool].max().item() if mask_bool.any() else 0.0
                metrics['behave_w_min'] = behave_w[mask_bool].min().item() if mask_bool.any() else 0.0
                metrics['behave_approx_kl'] = (log_w[mask_bool].sum() / denom).item() if mask_bool.any() else 0.0
                if self.behave_imp_weight_cap is not None:
                    capped = (behave_w_masked >= self.behave_imp_weight_cap).to(dtype=dtype)
                    metrics['behave_w_capfrac'] = (capped * mask).sum().item() / denom.item()

        return loss_total_sum, denom, metrics

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for local_batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.
        '''
        assert self.policy_engine is not None, "DeepSpeed engine not initialized"
        assert isinstance(micro_batches, list) and len(micro_batches) > 0, \
            "micro_batches must be a non-empty list which should be equal across " \
            "ranks via prepare_training_batches padding"

        device = self.policy_engine.device

        # When using decoupled loss, snapshot pi_prox once per epoch and
        # pair-shuffle with micro_batches. Otherwise just shuffle micro_batches.
        if self.use_decoupled_loss:
            micro_batches, prox_logprobs, prox_nan_masks = self.snapshot_prox_for_epoch(micro_batches=micro_batches,
                                                                                        engine_id=engine_id,
                                                                                        device=device)
        else:
            call_idx               = getattr(self, '_train_step_calls', 0)
            self._train_step_calls = call_idx + 1
            local_rng              = random.Random(f"{self.seed}_{engine_id}_{call_idx}")
            local_rng.shuffle(micro_batches)
            prox_logprobs          = None
            prox_nan_masks         = None

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

        # If num_micro is not divisible by ga_pi, the last GA bucket has fewer
        # micro-batches. DeepSpeed still divides by ga_pi, so we must scale
        # those losses up by ga_pi/remainder to get the correct mean gradient.
        ga_remainder = num_micro % ga_pi

        # Compute global token count for global token normalization.
        ga_denom  = None
        ga_denoms = None
        dp_scale  = None
        if self.normalize_loss:
            if self.update_only_after_full_replay:
                ga_denom, dp_scale  = self.compute_global_token_denom(micro_batches=micro_batches, ga_steps=ga_pi, device=device)

            else:
                ga_denoms, dp_scale = self.compute_per_group_token_denoms(micro_batches=micro_batches, ga_steps=ga_pi, device=device)

        # Weight health check before any update
        self.check_weights_health(engine_id, "BEFORE training step")

        # track metrics across all micro-batches
        all_metrics = []
        consecutive_nan_steps = 0
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

            # Snapshot pre-NaN valid token count for ga_denom correction.
            pre_nan_valid = (mask > 0.5).sum().item() if self.normalize_loss else 0

            # Sanitize logprobs before loss computation to prevent NaN into loss
            pi_logprobs, pi_nan = self.sanitize_logprobs(logprobs=pi_logprobs, engine_id=engine_id, step=step, num_micro=num_micro)
            mask = mask * (~pi_nan).to(mask.dtype)

            ref_logprobs = None
            if self.kl_coeff > 0.0 and self.ref_model_engine is not None:
                ref_logprobs = self.ref_forward(input_ids=input_ids,
                                                att_mask=att_mask,
                                                pos_ids=pos_ids)

                ref_logprobs, ref_nan = self.sanitize_logprobs(logprobs=ref_logprobs, engine_id=engine_id, step=step, num_micro=num_micro)
                mask = mask * (~ref_nan).to(mask.dtype)

            # Compute policy loss using the current policy.
            # prox_logprobs entries are already [B, T-1].
            if prox_logprobs is not None:
                prox_nan_mask = prox_nan_masks[step]
                mask = mask * (~prox_nan_mask).to(mask.dtype)
                prox_lp_step = prox_logprobs[step]

            else:
                prox_lp_step = None

            loss_total_sum, local_denom, pi_metrics = self.compute_policy_loss(logprobs=pi_logprobs,
                                                                               old_logprobs=old_logprobs,
                                                                               advantages=advs,
                                                                               mask=mask,
                                                                               entropies=pi_entropies,
                                                                               ref_logprobs=ref_logprobs,
                                                                               prox_logprobs=prox_lp_step)

            # store metrics
            all_metrics.append(pi_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({"pi_loss": f"{pi_metrics['pi_loss']:.4f}",
                                          "clipfrac": f"{pi_metrics['clipfrac']:.3f}",
                                          "approx_kl": f"{pi_metrics['approx_kl']:.4f}",
                                          "kl_ref": f"{pi_metrics['kl_ref']:.4f}"})

            # Scale loss for backward pass.
            if self.normalize_loss:
                if self.update_only_after_full_replay:
                    # Single global denominator covering all micro-batches.
                    # Adjust for tokens lost to nan sanitization in this micro-batch.
                    nan_removed  = pre_nan_valid - local_denom.item()
                    ga_denom_adj = max(ga_denom - nan_removed, 1.0) if nan_removed > 0 else ga_denom
                    pi_loss      = loss_total_sum * (dp_scale / ga_denom_adj)

                else:
                    # Per-GA-group denominator so each optimizer step is normalized
                    # only by its own group's tokens (not the entire replay buffer).
                    group_idx    = step // ga_pi
                    nan_removed  = pre_nan_valid - local_denom.item()
                    ga_denom_adj = max(ga_denoms[group_idx] - nan_removed, 1.0) if nan_removed > 0 else ga_denoms[group_idx]
                    pi_loss      = loss_total_sum * (dp_scale / ga_denom_adj)

            else:
                # local per-micro-batch mean + manual GA scaling.
                pi_loss = loss_total_sum / local_denom
                if self.update_only_after_full_replay:
                    pi_loss = pi_loss * (ga_pi / num_micro)

                else:
                    if ga_remainder != 0 and step >= (num_micro - ga_remainder):
                        pi_loss = pi_loss * (ga_pi / ga_remainder)


            # For DeepSpeed, we must coordinate is_boundary with the backward pass.
            self.policy_engine.set_gradient_accumulation_boundary(is_boundary)

            # track consecutive all-masked micro-batches and abort early
            # to avoid a ZeRO-3 nccl hang from sustained zero gradients.
            consecutive_nan_steps = self.check_all_masked(engine_id=engine_id,
                                                          step=step,
                                                          num_micro=num_micro,
                                                          local_denom=local_denom,
                                                          consecutive_nan_steps=consecutive_nan_steps)

            # backward pass
            self.policy_engine.backward(pi_loss)

            # Proactive cache flush before optimizer step to reduce fragmentation.
            if is_boundary:
                torch.cuda.empty_cache()

            self.policy_engine.step()

        # aggregate metrics across all micro-batches
        aggregated_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])

        # check weights health after update
        self.check_weights_health(engine_id, "AFTER training step")

        # Free the prox cache if we just finished the last iteration of the epoch.
        self.release_prox_cache_if_epoch_end()

        return aggregated_metrics
