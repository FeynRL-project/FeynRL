import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Any
import ray

# Since the following functions are the same for all algorithms
# we load them from common.py:
# _load_single_model, init_training_engine,policy_forward,
# ref_forward, compute_kl_distance, save_checkpoint
from algs.RL.common import COMMON
from algs.PPO.value_net import ValueNetwork

@ray.remote
class PPO(COMMON):
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
                 # ppo specific
                 value_model_path: str = None,
                 tau: float = None,
                 gamma: float = None,
                 deepspeed_value_config: Any = None,
                 ):
        assert tau is not None and gamma is not None, 'tau and gamma must be provided for PPO'
        assert value_model_path is not None, 'value_model_path must be provided for PPO'
        self.alg_name = self.__class__.__name__
        self.model_path = model_path
        self.model_dtype = model_dtype
        self.ref_model_path = ref_model_path
        self.attn_impl = attn_impl
        self.trust_remote_code = trust_remote_code
        self.peft_config = peft_config

        # training related parameters
        self.deepspeed_config = deepspeed_config
        self.deepspeed_ref_config = deepspeed_ref_config
        self.deepspeed_value_config = deepspeed_value_config
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu
        self.gradient_checkpointing = gradient_checkpointing

        # policy related parameters
        self.kl_coeff = float(kl_coeff)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.ent_coeff = float(entropy_coeff)

        # value model params
        self.value_model_path = value_model_path
        self.tau = float(tau)
        self.gamma = float(gamma)

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
            Load policy, reference, and value models.
            Value model has a scalar value head.
        '''
        # Load policy model
        policy_model = self._load_single_model(self.model_path, self.model_dtype)

        # Load reference model if provided
        if self.ref_model_path and self.kl_coeff > 0.0:
            ref_model = self._load_single_model(self.ref_model_path, self.model_dtype)
        else:
            ref_model = None

        # Load value model
        # we assume the value model has the same dtype as the policy model
        base_value_model = self._load_single_model(self.value_model_path, self.model_dtype)
        value_model = ValueNetwork(base_value_model)

        return {"policy_model": policy_model, "ref_model": ref_model, "value_model": value_model}

    def compute_advantages(self,
                           rewards: torch.Tensor,
                           values: torch.Tensor,
                           done: torch.Tensor,
                           mask: torch.Tensor,
                           last_val: torch.Tensor | None = None,
                          ):
        '''
            rewards, values: [B, T]
            done, mask: [B, T]
            done:    1 if t is terminal (EOS/stop), 0 otherwise.
            mask:    1 if valid token, 0 if padding.
            GAE and returns: [B, T]
            last_val: [B]
            return: rets, advs which would be both [B, T]
        '''
        # 1. float32 for numerical stability under bf16/fp16.
        device = values.device
        B, T   = values.shape
        values = values.to(torch.float32)
        rets   = torch.zeros(B, T, dtype=torch.float32, device=device)
        advs   = torch.zeros(B, T, dtype=torch.float32, device=device)
        last_adv = torch.zeros(B, dtype=torch.float32, device=device)
        rewards  = rewards.to(dtype=torch.float32, device=device)

        # 2. casting
        mask  = mask.to(device=device)
        done  = done.to(device=device)
        mask  = (mask > 0.5)
        done  = (done > 0.5)

        # 3. Check for nan
        if not torch.isfinite(rewards[mask]).all() or not torch.isfinite(values[mask]).all():
            raise ValueError("rewards or values contain NaN on valid positions")

        if (done & (~mask)).any():
            raise ValueError("done flag set on padding positions")

        # 4. reject holes in mask e.g., [1, 1, 0, 1, 1].
        # valid masks have one contiguous block of 1s with optional leading/trailing 0s.
        # A hole exists iff any 0 -> 1 rise occurs after a 1 -> 0 drop.
        drops = (mask[:, :-1] & ~mask[:, 1:])  # 1 -> 0 transitions
        rises = (~mask[:, :-1] & mask[:, 1:])  # 0 -> 1 transitions
        if (rises & (drops.cumsum(dim=1) > 0)).any():
            raise ValueError("mask has non-contiguous valid regions (holes). This is unsupported.")

        # prefill val and reward for invalid tokens (i.e., padding) as they can contain nan in padded slot
        rewards = rewards.masked_fill(~mask, 0.0)
        values  = values.detach().masked_fill(~mask, 0.0)

        # 5. empty sequences
        if T == 0:
            empty = rewards.new_zeros((B, 0))
            return empty, empty

        # 6. next value for bootstrapping
        if last_val is not None:
            next_val = last_val.to(dtype=torch.float32, device=device).detach().reshape(B)
            if not torch.isfinite(next_val).all():
                raise ValueError("last_val contains NaN or Inf")

        else:
            next_val = torch.zeros(B, dtype=torch.float32, device=device)

        # 7. Using (tensor > 0.5) is safer than bool() if inputs are already floats
        # especially in case of bf16/fp16 training.
        mask  = mask.to(dtype=torch.float32, device=device)
        done  = done.to(dtype=torch.float32, device=device)

        # 8. Compute returns and advantages
        for t in reversed(range(T)): # [T-1, 0]
            # Done is 1 if EOS/Terminal, we do NOT bootstrap from t+1.
            not_done = 1.0 - done[:, t]
            is_valid = mask[:, t]

            # GAE: A[t] = delta[t] + gamma * tau * A[t+1] * (1 - done[t])
            delta = rewards[:, t] + (self.gamma * next_val * not_done) - values[:, t]
            last_adv   = is_valid * (delta + (self.gamma * self.tau * last_adv * not_done))
            advs[:, t] = last_adv

            # At valid positions use v(s_t) and at padding keep next_val
            # so the bootstrap survives through padding to the last valid token.
            next_val = torch.where(is_valid > 0.5, values[:, t], next_val)

        rets = advs + values

        return rets, advs

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
        ratio   = torch.exp(logratio)

        # 3. compute loss: -(min(ratio * adv, clip_adv)) * mask
        unclipped = ratio * adv
        clip_adv  = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high) * adv
        loss_pi   = -(torch.minimum(unclipped, clip_adv) * mask).sum() / denom

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

    def value_forward(self, input_ids, att_mask, pos_ids):
        '''
            Input:
                input_ids/att_mask: [B, T]
                pos_ids: [B, T] or None
            Returns:
                values: [B, T-1]
                last_value: [B] value at each row's true last non-pad token
        '''
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        # feed data to model
        output = self.value_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=False)

        # ValueHeadModel outputs [B, T, 1] -> squeeze to [B, T]
        logits = output.logits.squeeze(-1).contiguous()

        # [B, T] -> [B, T-1]
        values = logits[:, :-1].contiguous()
        # last_value is used for bootstrapping. For non-padded rows logits[:, -1] is correct.
        # However, when there is padding, logits[:, -1] would be garbage. this is fixed by picking
        # the last real token's value for each row.
        B = logits.shape[0]
        # attn_mask vs mask:
        # attn_mask:  [1, 1, 1, 1, 1, 0, 0] -> prompt + response which all real tokens would be 1 and pad would be zero
        # masks:      [0, 0, 1, 1, 0, 0, 0] -> only valid prediction positions
        seq_lens = att_mask.sum(dim=1).long() # [B] number of real tokens
        if (seq_lens <= 0).any():
            raise ValueError("att_mask has rows with zero valid tokens; cannot compute bootstrap last_value")
        last_real_idx = (seq_lens - 1).clamp(min=0) # [B]
        last_value = logits[torch.arange(B, device=logits.device), last_real_idx]

        return values, last_value

    def compute_value_loss(self,
                           values: torch.Tensor,
                           returns: torch.Tensor,
                           mask: torch.Tensor,
                           ):
        '''
            values/returns/mask: [B, T-1]
            Compute value loss: 0.5 * (values - returns)^2
        '''
        device = values.device

        # Compute value loss in float32 for numerical stability under bf16/fp16.
        rets   = returns.detach().to(torch.float32)
        v_loss = (values.to(torch.float32) - rets).pow(2)
        mask   = (mask.to(device=device) > 0.5).to(dtype=torch.float32)
        denom  = mask.sum().clamp(min=1.0)

        loss = 0.5 * (v_loss * mask).sum() / denom

        # save the metrics for debugging
        metrics = {
            'loss_v': loss.item(),
        }

        return loss, metrics

    def precompute_gae(self, micro_batches):
        '''
            Precompute values and gae for all batches before any updates.
            Returns list of (returns, advs) tuples on cpu, one per batch.
        '''
        device = self.value_engine.device
        self.value_engine.eval()
        precomputed = []
        with torch.no_grad():
            for mb in micro_batches:
                ids  = mb['input_ids'].to(device, non_blocking=True)
                amsk = mb['attn_mask'].to(device, non_blocking=True)
                pids = mb.get('position_ids', None)
                vals, last_v = self.value_forward(input_ids=ids, att_mask=amsk, pos_ids=pids)
                # all are prediction aligned
                rewards = mb['rewards'][:, :-1].to(device, non_blocking=True)
                done    = mb['done'][:, :-1].to(device, non_blocking=True)
                mask    = mb['mask'][:, :-1].to(device, non_blocking=True)

                rets, advs = self.compute_advantages(rewards=rewards,
                                                     values=vals,
                                                     done=done,
                                                     mask=mask,
                                                     last_val=last_v)
                precomputed.append((rets.cpu(), advs.cpu()))
        return precomputed

    def train_step(self, engine_id, micro_batches):
        '''
           This function implements a training step per rank/gpu for local_batch.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
           micro_batches is a partition of the replay buffer (list of micro-batches) for the current rank/gpu.
        '''
        assert self.policy_engine is not None, "DeepSpeed policy_engine not initialized"
        assert self.value_engine  is not None, "DeepSpeed value_engine not initialized"
        assert isinstance(micro_batches, list) and len(micro_batches) > 0, \
            "micro_batches must be a non-empty list which should be equal across " \
            "ranks via prepare_training_batches padding"

        # 1. Pre-compute values and GAE before any updates.
        precomputed_gae = self.precompute_gae(micro_batches)

        # 1. Pre-compute values and GAE before any updates.
        precomputed_gae = self.precompute_gae(micro_batches)

        device = self.policy_engine.device

        # 2. Models to train mode
        self.policy_engine.train()
        self.value_engine.train()

        # 3. zero grads
        self.policy_engine.zero_grad()
        self.value_engine.zero_grad()

        # 4. create progress bar
        num_micro = len(micro_batches)
        paired = list(zip(micro_batches, precomputed_gae))

        # Shuffle so each steps_per_epoch iteration micro_batches are processed in a
        # different sequence to avoid systematic bias from GA boundary placement.
        random.shuffle(paired)

        # torch.distributed.get_rank() would be the same thing as engine_id
        if engine_id == 0:
            progress_bar = tqdm(paired, total=num_micro, desc="[Alg:{}] Training Step in rank {}".format(self.alg_name, engine_id))

        else:
            progress_bar = paired # No tqdm for other ranks

        # Both engines share the same DS config, so GA steps are identical.
        ga_attr = getattr(self.policy_engine, 'gradient_accumulation_steps', 1)
        ga_steps = int(ga_attr() if callable(ga_attr) else ga_attr)

        # track metrics across all micro-batches
        all_metrics_policy = []
        all_metrics_value = []
        for step, (micro_batch, (returns, advs)) in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))
            # If update_only_after_full_replay is True, we only update at the very end
            # of the shard. Otherwise, we respect ga_pi.
            if self.update_only_after_full_replay:
                is_boundary = is_last

            else:
                is_boundary = (((step + 1) % ga_steps) == 0) or is_last
            ########
            # 1. Data from buffer
            ########
            mask      = micro_batch['mask'][:, :-1].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'][:, :-1].to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            # Pre-computed returns and advantages based on frozen value_net
            returns, advs = precomputed_gae[step]
            returns = returns.to(device, non_blocking=True)
            advs    = advs.to(device, non_blocking=True)

            ########
            # 3. Compute policy loss
            ########
            # Forward pass through the policy.
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
            all_metrics_policy.append(pi_metrics)
            # When accumulating over the full replay shard, normalize the loss
            # by the number of micro-batches so the gradient magnitude equals the
            # mean (not the sum) of per-micro-batch gradients. DeepSpeed will
            # still divide by gradient_accumulation_steps, so we multiply by
            # ga_pi to keep the effective scale consistent with standard GA.
            if self.update_only_after_full_replay:
                pi_loss = pi_loss * (ga_steps / num_micro)

            self.policy_engine.set_gradient_accumulation_boundary(is_boundary)

            # backward pass
            self.policy_engine.backward(pi_loss)
            self.policy_engine.step()

            ########
            # 4. Compute value loss
            ########
            # Forward pass through the value function.
            values, _ = self.value_forward(input_ids=input_ids, att_mask=att_mask, pos_ids=pos_ids)

            # Compute value loss
            v_loss, v_metrics = self.compute_value_loss(values=values, returns=returns, mask=mask)

            # store metrics
            all_metrics_value.append(v_metrics)
            if engine_id == 0:
                progress_bar.set_postfix({
                    "pi_loss": f"{pi_loss.item():.4f}",
                    "clipfrac": f"{pi_metrics['clipfrac']:.3f}",
                    "approx_kl": f"{pi_metrics['approx_kl']:.4f}",
                    "kl_ref": f"{pi_metrics['kl_ref']:.4f}",
                    "v_loss": f"{v_metrics['loss_v']:.4f}",
                })

            # Same loss scaling for value function
            if self.update_only_after_full_replay:
                v_loss = v_loss * (ga_steps / num_micro)

            self.value_engine.set_gradient_accumulation_boundary(is_boundary)

            # backward pass
            self.value_engine.backward(v_loss)
            self.value_engine.step()

        # aggregate metrics across all micro-batches into a single dict
        aggregated_metrics = {}

        # Policy metrics use the same keys as SGRPO for compatibility with run_training_step
        if all_metrics_policy:
            for key in all_metrics_policy[0].keys():
                aggregated_metrics[key] = np.mean([m[key] for m in all_metrics_policy])

        # Value metrics with 'value_' prefix (additional PPO-specific metrics)
        if all_metrics_value:
            for key in all_metrics_value[0].keys():
                aggregated_metrics[f'value_{key}'] = np.mean([m[key] for m in all_metrics_value])

        return aggregated_metrics
