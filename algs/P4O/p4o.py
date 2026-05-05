import torch
import numpy as np
import math
from tqdm import tqdm
from typing import Any
import ray

# load follwoings from common.py:
# load_single_model, init_training_engine, policy_forward,
# ref_forward, compute_kl_distance, save_checkpoint
from algs.RL.common import COMMON

@ray.remote
class P4O(COMMON):
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
        # p4o does not use clip_low and clip_high in loss calculation
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.ent_coeff = float(entropy_coeff)

        # use cross entropy loss for policy gradient
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        # params for decoupled ppo clipped loss which is not used in p4o for now.
        self.use_decoupled_loss = use_decoupled_loss
        self.behave_imp_weight_cap = float(behave_imp_weight_cap) if behave_imp_weight_cap is not None else None

        # if true, it means the update is done after seeing all samples in the reply buffer
        # treating the entire buffer as a single batch.
        self.update_only_after_full_replay = update_after_full_replay
        self.normalize_loss = normalize_loss

        # Following are used to snapshot pi_prox once per epoch on the first train_step
        # and reuse the cached snapshot across the remaining iterations.
        self.train_steps_per_epoch = int(train_steps_per_epoch)
        self.cached_prox_logprobs  = None
        self.cached_prox_nan_masks = None

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

    def calculate_ess(self, ratio, mask_bool):
        '''
            Calculate the effective sample size (ESS) across all ranks.
            ESS = ||w||^2_1 / ||w||^2_2 , ratio = exp(logprobs - old_logprobs)
            ESS = ESS / n where n is number of valid (non-padded) samples to normalize
            ESS would be between 1/n and 1, effectively 0 < ESS <= 1
        '''
        # Only use valid (non-padded) positions. Padded positions have
        # ratio = exp(0) = 1.0 which would bias ESS toward 1.0.
        valid_ratios = ratio[mask_bool]

        # local_stats_per_rank is 1D tensor of shape (3,).
        local_stats_per_rank = torch.stack([valid_ratios.sum().to(torch.float64),
                                            valid_ratios.pow(2).sum().to(torch.float64),
                                            torch.tensor(float(valid_ratios.numel()), device=ratio.device, dtype=torch.float64)])

        # combine across all ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(local_stats_per_rank, op=torch.distributed.ReduceOp.SUM)

        # One sync to extract scalars after the collective.
        sum_w, sum_w_2, total = local_stats_per_rank.tolist()

        # Avoid division by zero or guard against nans.
        if total < 0.5 or not math.isfinite(sum_w) or not math.isfinite(sum_w_2) or not math.isfinite(total):
            return 1.0

        ess = (sum_w**2) / (sum_w_2 + 1e-8) / total
        return float(ess)

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            entropies: torch.Tensor,
                            ref_logprobs: torch.Tensor,
                            prox_logprobs: torch.Tensor,
                            ):
        '''
            logprobs, old_logprobs, advantages, mask: [B, T-1]
            entropies, ref_logprobs, prox_logprobs:   [B, T-1]
            Compute P4O policy loss:
                1. ratio_b    = exp(logprobs - old_logprobs)        # behavioral IS ratio
                2. ratio_prox = exp(logprobs - prox_logprobs)       # prox IS ratio
                3. ess_b      = ||w_b||^2_1 / ||w_b||^2_2 / n       # global token-ESS over ratio_b
                4. ess_prox   = ||w_prox||^2_1 / ||w_prox||^2_2 / n # global token-ESS over ratio_prox
                5. rho        = min(ratio_b, ess_b, ess_prox)       # detached IS weight (three-way min)
                6. loss = -sg(rho) * logprobs * adv * mask
                          + (1 - ess_b)    * KL(pi || pi_b)
                          + (1 - ess_prox) * KL(pi || pi_prox)
                          + kl_coeff * KL(pi || pi_ref)             # optional, gated on ref_model + kl_coeff > 0
                          - ent_coeff * H(pi)                       # optional, gated on ent_coeff > 0
            Returns:
                loss_total_sum: scalar tensor, raw sum of masked losses (no normalization).
                denom: local token count in this micro-batch (for metrics and fallback normalization).
                metrics: dict, per-token mean metrics using local denom for interpretability.
        '''
        device = logprobs.device
        dtype = logprobs.dtype
        ent_sum = torch.tensor(0.0, device=device, dtype=dtype)
        kl_sum_ref  = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. make sure advantages are detached and
        # convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)
        mask_bool = (mask.to(device=device) > 0.5)
        mask = mask_bool.to(dtype=dtype)
        denom = mask.sum().clamp(min=1.0)

        # 2. calculate ratio = pi / pi_old = exp(logprobs - old_logprobs)
        raw_logratio_b = (logprobs - old_logprobs).to(torch.float32)
        # Ignore invalid (padded) positions before exp to avoid inf * 0 -> nan.
        logratio_b     = torch.where(mask_bool, raw_logratio_b, torch.zeros_like(raw_logratio_b))
        ratio_b        = torch.exp(logratio_b)
        ess_b          = self.calculate_ess(ratio=ratio_b, mask_bool=mask_bool)

        # 3. calculate the KL between policy and old model (behavioral policy) for the KL term in the loss
        kl_dist_b = self.compute_kl_distance(logprobs=logprobs, ref_logprobs=old_logprobs)
        kl_dist_b = torch.where(mask_bool, kl_dist_b, torch.zeros_like(kl_dist_b))
        kl_sum_b  = (kl_dist_b * mask).sum()

        # 4. calculate ratio = pi / pi_prox = exp(logprobs - old_logprobs)
        raw_logratio_prox = (logprobs - prox_logprobs).to(torch.float32)
        logratio_prox     = torch.where(mask_bool, raw_logratio_prox, torch.zeros_like(raw_logratio_prox))
        ratio_prox        = torch.exp(logratio_prox)
        ess_prox          = self.calculate_ess(ratio=ratio_prox, mask_bool=mask_bool)

        # 5. calculate the KL between policy and prox for the KL term in the loss
        kl_dist_prox = self.compute_kl_distance(logprobs=logprobs, ref_logprobs=prox_logprobs)
        kl_dist_prox = torch.where(mask_bool, kl_dist_prox, torch.zeros_like(kl_dist_prox))
        kl_sum_prox  = (kl_dist_prox * mask).sum()

        # 6. P4O loss as raw sum:  min(ratio_b, min(ess_b, ess_prox)).detach() * log(pi) * advantage
        ess_cap      = min(ess_b, ess_prox)
        rho_min_b    = torch.clamp(ratio_b, min=0, max=ess_cap)
        pi_sum       = -(rho_min_b.detach() * logprobs * adv * mask).sum()

        # 4. compute entropy loss (raw sum)
        if entropies is not None and self.ent_coeff > 0.0:
            ent_sum = (entropies * mask).sum()

        if ref_logprobs is not None and self.kl_coeff > 0.0:
            kl_dist_ref = self.compute_kl_distance(logprobs=logprobs, ref_logprobs=ref_logprobs)
            # avoid calculating kl for padded tokens.
            kl_dist_ref = torch.where(mask_bool, kl_dist_ref, torch.zeros_like(kl_dist_ref))
            kl_sum_ref  = (kl_dist_ref * mask).sum()

        # When an ESS factor is ~1, the current policy is close to that anchor (behavioral
        # or prox), so the corresponding KL pull-back is down-weighted to ~0. As an ESS
        # factor approaches 0, the policy has diverged from that anchor and the (1 - ess)
        # weight ramps up the KL penalty to enforce a trust region. The two terms are
        # independent: ess_b governs the pull toward pi_old, ess_prox governs the pull
        # toward the pre-update snapshot.
        loss_total_sum = pi_sum - self.ent_coeff * ent_sum + self.kl_coeff * kl_sum_ref + (1 - ess_b) * kl_sum_b + (1 - ess_prox) * kl_sum_prox

        # 5. useful metrics. here per-token means using local denom for interpretability
        with torch.no_grad():
            # first term too large ==> policy changed too much upward
            # second term too small ==> policy changed too much downward
            clipped_mask = (ratio_b > (1.0 + self.clip_high)) | (ratio_b < (1.0 - self.clip_low))
            # fraction of masked tokens that ratio out of ranges
            clipfrac = (clipped_mask.to(dtype=dtype) * mask).sum() / denom

            # approx KL (var-reduced): log(pi/pi_old) + pi_old/pi - 1
            # logratio = log(pi/pi_old)
            ratio_inv_b   = torch.exp(-logratio_b)
            approx_kl_t_b = logratio_b + ratio_inv_b - 1.0
            approx_kl_b   = (approx_kl_t_b.to(dtype=dtype) * mask).sum() / denom

            # mc entropy proxy at the realized token: -log pi(a|s)
            ent_mc = -(logprobs * mask).sum() / denom

            # save the metrics for debugging
            metrics = {'clipfrac': clipfrac.item(),
                       'approx_kl_b': approx_kl_b.item(),
                       'ent_mc': ent_mc.item(),
                       'pi_loss': (pi_sum / denom).item(),
                       'loss_total': (loss_total_sum / denom).item(),
                       'kl_ref': (kl_sum_ref / denom).item(),
                       'kl_b': (kl_sum_b / denom).item(),
                       'kl_prox':  (kl_sum_prox / denom).item(),
                       'ess_b': ess_b,
                       'ess_prox': ess_prox,
                       'ess_cap': ess_cap,
                       }

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

        # Snapshot pi_prox once per epoch and pair-shuffle with micro_batches.
        micro_batches, prox_logprobs, prox_nan_masks = self.snapshot_prox_for_epoch(micro_batches=micro_batches,
                                                                                    engine_id=engine_id,
                                                                                    device=device)

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

        # Weight health check before any update and exit immediately if weights are already nan
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

            # apply masking to the prox logprobs
            prox_nan_mask = prox_nan_masks[step]
            mask          = mask * (~prox_nan_mask).to(mask.dtype)
            prox_lp_step  = prox_logprobs[step]


            # Compute policy loss using the current policy.
            loss_total_sum, local_denom, pi_metrics = self.compute_policy_loss(logprobs=pi_logprobs,
                                                                               old_logprobs=old_logprobs,
                                                                               advantages=advs,
                                                                               mask=mask,
                                                                               entropies=pi_entropies,
                                                                               ref_logprobs=ref_logprobs,
                                                                               prox_logprobs=prox_lp_step,
                                                                               )

            # store metrics
            all_metrics.append(pi_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({"pi_loss": f"{pi_metrics['pi_loss']:.4f}",
                                          "clipfrac": f"{pi_metrics['clipfrac']:.3f}",
                                          "approx_kl_b": f"{pi_metrics['approx_kl_b']:.4f}",
                                          "kl_ref": f"{pi_metrics['kl_ref']:.4f}",
                                          "kl_b": f"{pi_metrics['kl_b']:.4f}",
                                          "kl_prox": f"{pi_metrics['kl_prox']:.4f}",
                                          "ess_b": f"{pi_metrics['ess_b']:.4f}",
                                          "ess_prox": f"{pi_metrics['ess_prox']:.4f}",
                                          "ess_cap": f"{pi_metrics['ess_cap']:.4f}",
                                          })

            # Scale loss for backward pass.
            if self.normalize_loss:
                if self.update_only_after_full_replay:
                    # Single global denominator covering all micro-batches.
                    # Adjust for tokens lost to nan sanitization in this micro-batch.
                    nan_removed  = pre_nan_valid - local_denom.item()
                    ga_denom_adj = max(ga_denom - nan_removed, 1.0) if nan_removed > 0 else ga_denom
                    pi_loss      = loss_total_sum * (dp_scale / ga_denom_adj)

                else:
                    # Per-ga-group denominator so each optimizer step is normalized
                    # only by its own group's tokens (not the entire replay buffer).
                    group_idx    = step // ga_pi
                    nan_removed  = pre_nan_valid - local_denom.item()
                    ga_denom_adj = max(ga_denoms[group_idx] - nan_removed, 1.0) if nan_removed > 0 else ga_denoms[group_idx]
                    pi_loss      = loss_total_sum * (dp_scale / ga_denom_adj)

            else:
                # local per-micro-batch mean and manual GA scaling.
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

        # check weights health after update to avoid nan in the next step
        self.check_weights_health(engine_id, "AFTER training step")

        # Free the prox cache if we just finished the last iteration of the epoch.
        self.release_prox_cache_if_epoch_end()

        return aggregated_metrics