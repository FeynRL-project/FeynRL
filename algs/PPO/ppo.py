import torch
import numpy as np
from tqdm import tqdm
import ray
from typing import Dict

@ray.remote(resources={"training": 1})
class PPO:
    def __init__(self,
                policy_engine,
                value_engine,
                kl_coeff: float,
                clip_low: float,
                clip_high: float,
                vf_clip: float,
                tau: float,
                gamma: float,
                entropy_coeff: float,
                use_cache: bool,
                micro_batch_size_per_gpu: int,
                ref_model=None,
                ):

        # model related parameters
        self.policy_engine = policy_engine
        self.value_engine = value_engine
        self.ref_model = ref_model
        self.use_cache = use_cache
        self.micro_batch_size_per_gpu = micro_batch_size_per_gpu

        # policy related parameters
        self.kl_coeff = float(kl_coeff)
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.ent_coeff = float(entropy_coeff)

        # value related parameters
        self.vf_clip = float(vf_clip)

        # use cross entropy loss for policy gradient
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

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
            done:    1 if t is EOS (terminal), 0 otherwise.
                     MUST be set at every packed sequence boundary so it
                     shows the boundary of each sequence.
            mask:    1 if valid token, 0 if padding.
            GAE and returns: [B, T]
            last_val: [B]
            return: rets, advs which would be both [B, T]
        '''
        # 1. Device and shape setup
        device = values.device
        dtype  = values.dtype
        B, T   = values.shape
        rets   = torch.zeros_like(values)
        advs   = torch.zeros_like(values)
        last_adv = torch.zeros(B, dtype=dtype, device=device)
        rewards  = rewards.to(dtype=dtype, device=device)

        # 2. Delay casting the mask to the same dtype for indexing and checks.
        mask  = mask.to(device=device)
        done  = done.to(device=device)
        mask  = (mask > 0.5)
        done  = (done > 0.5)

        # 3. Check for nan in rewards or values for valid tokens
        if not torch.isfinite(rewards[mask]).all() or not torch.isfinite(values[mask]).all():
            raise ValueError("rewards or values contain NaN on valid positions")

        if (done & (~mask)).any():
            raise ValueError("done flag set on padding positions")

        # 4. reject holes in padding e.g., [x1, x2, x3, pad, x4, x5] --> this is not supported
        #    we only support [x1, x2, x3, pad, pad, pad...] or [x1, x2, x3, eos, pad,..]
        if (mask[:, 1:] & (~mask[:, :-1])).any():
            raise ValueError("mask has 0->1 transitions (padding in the middle). This is unsupported.")

        # prefill val and rerward for invalid tokens (i.e., padding) as they can contain nan in padded slot
        rewards = rewards.masked_fill(~mask, 0.0)
        values  = values.detach().masked_fill(~mask, 0.0)

        # 5. empty sequences
        if T == 0:
            empty = rewards.new_zeros((B, 0))
            return empty, empty

        # 6. next value
        if last_val is not None:
            next_val = last_val.to(dtype=dtype, device=device).detach().reshape(B)

        else:
            # biased estimation espically whenre there is need for bootstrapping, i.e.,
            # no EOS in generation like [x1,x2,x3]
            next_val = torch.zeros(B, dtype=dtype, device=device)

        # 7. Using (tensor > 0.5) is safer than bool() if inputs are already floats
        # espically in case of BF16/FP16 training.
        mask  = mask.to(dtype=dtype, device=device)
        done  = done.to(dtype=dtype, device=device)

        # 8. Compute returns and advantages
        for t in reversed(range(T)): # [T-1, 0]
            # Done is 1 if EOS/Terminal, we do NOT bootstrap from t+1.
            not_done = 1.0 - done[:, t]
            is_valid = mask[:, t]

            # GAE: A[t] = delta[t] + gamma * tau * A[t+1] * (1 - done[t])
            delta = rewards[:, t] + (self.gamma * next_val * not_done) - values[:, t]
            last_adv   = is_valid * (delta + (self.gamma * self.tau * last_adv * not_done))
            advs[:, t] = last_adv

            # to avoid any leaking from padding.
            next_val = values[:, t] * is_valid

        rets = advs + values

        return rets, advs

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
        # logits is [B, T-1, vocab_size]
        # target_ids is [B, T-1]
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
            logprobs, old_logprobs, advantages, mask: [B, T]
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

        # 2. calculate ratio = exp(logprobs - old_logprobs)
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
                'clipfrac': clipfrac,
                'approx_kl': approx_kl,
                'loss_ent': loss_ent.item(),
                'loss_pi': loss_pi.item(),
                'loss_total': loss_total.item(),
            }

        return loss_total, metrics

    def value_forward(self, input_ids, att_mask, pos_ids):
        '''
            Input:
                input_ids/att_mask: [B, T]
                pos_ids: [B, T] or None
            Returns:
                values: [B, T-1] aligned with input state at t, predicting future
                last_value: [B, 1] value of the very last token
        '''
        # if pos_ids is not provided, HF will add that automatically.
        if pos_ids is not None:
            pos_ids = pos_ids.to(input_ids.device)

        # feed data to model
        output = self.value_engine(input_ids=input_ids,
                                   attention_mask=att_mask,
                                   position_ids=pos_ids,
                                   use_cache=self.use_cache)

        # [B, T, 1] -> [B, T]
        every_token_values = output.logits.squeeze(-1)
        # [B, T] -> [B, T-1]
        values = every_token_values[:, :-1].contiguous()
        # Value for terminal state (e.g., t=T-1) for bootstrapping if not EOS
        #[B, T] -> [B]
        last_value = every_token_values[:, -1].contiguous()

        return values, last_value

    def compute_value_loss(self,
                           values: torch.Tensor,
                           v_old: torch.Tensor,
                           returns: torch.Tensor,
                           mask: torch.Tensor,
                           ):
        '''
            Compute value loss:
                1. if v_old:  loss = 0.5 * (max(values, v_clipped) - rets)^2
                2. otherwise: loss = 0.5 * (values - rets)^2
        '''
        # 1. compute unclipped value loss
        rets = returns.detach()
        v_loss = (values - rets).pow(2)
        denom = mask.sum().clamp(min=1.0)

        # 2. compute clipped value loss
        if  self.vf_clip > 0 and v_old is not None:
            v_old = v_old.detach()

            # 3. compute clipped value loss
            v_clipped = v_old + torch.clamp(values - v_old, -self.vf_clip, self.vf_clip)
            v_loss_clipped = (v_clipped - rets).pow(2)
            vmax =  torch.maximum(v_loss, v_loss_clipped)
            loss = 0.5 * (vmax * mask).sum() / denom

            # 4. log how much things are changed
            with torch.no_grad():
                vf_clipfrac = (values - v_old).abs() > self.vf_clip
                vf_clipfrac = (vf_clipfrac * mask).sum() / denom

        else:
            loss = 0.5 * (v_loss * mask).sum() / denom
            vf_clipfrac = 0.0

        # save the metrics for debugging
        metrics = {
            'vf_clipfrac': vf_clipfrac,
        }

        return loss, metrics

    def policy_step_update(self,
                           input_ids: torch.Tensor,
                           att_mask: torch.Tensor,
                           pos_ids,
                           old_logprobs: torch.Tensor,
                           advantages: torch.Tensor,
                           mask: torch.Tensor,
                           is_boundary: bool,
                           ) -> Dict[str, float]:
        '''
            Update policy using the current policy.
            input_ids/att_mask/pos_ids: [B, T]
            old_logprobs/advantages/mask: [B, T]
            is_boundary: bool
            Returns:
                policy_metrics: Dictionary containing policy metrics.
        '''
        # Forward pass through the current policy.
        pi_logprobs, pi_entropies = self.policy_forward(
                                                        input_ids=input_ids,
                                                        att_mask=att_mask,
                                                        pos_ids=pos_ids)

        # Compute policy loss using the current policy.
        pi_loss, pi_metrics = self.compute_policy_loss(
                                                logprobs=pi_logprobs,
                                                old_logprobs=old_logprobs,
                                                advantages=advantages,
                                                mask=mask,
                                                entropies=pi_entropies)

        self.policy_engine.set_gradient_accumulation_boundary(is_boundary)

        # backward pass
        self.policy_engine.backward(pi_loss)
        self.policy_engine.step()

        return pi_metrics

    def value_step_update(self,
                          input_ids: torch.Tensor,
                          att_mask: torch.Tensor,
                          pos_ids: torch.Tensor | None,
                          v_old: torch.Tensor,
                          returns: torch.Tensor,
                          mask: torch.Tensor,
                          is_boundary: bool,
                          device: torch.device,
                          ) -> Dict[str, float]:
        '''
            Update value using the current value.
            input_ids/att_mask/pos_ids: [B, T]
            values/v_old/returns/mask: [B, T]
            is_boundary: bool
            Returns:
                value_metrics: Dictionary containing value metrics.
        '''
        if v_old is not None:
            v_old = v_old.to(device, non_blocking=True)

        values, last_value = self.value_forward(
                                            input_ids=input_ids,
                                            att_mask=att_mask,
                                            pos_ids=pos_ids)

        vl_loss, vl_metrics = self.compute_value_loss(
                                            values=values,
                                            v_old=v_old,
                                            returns=returns,
                                            mask=mask)

        self.value_engine.set_gradient_accumulation_boundary(is_boundary)

        # backward pass
        self.value_engine.backward(vl_loss)
        self.value_engine.step()

        return vl_metrics

    def train_step(self, replay_buffer):
        '''
           This function implements a training step per rank/gpu for full replay buffer.
           The batch size for each gpu/rank should be micro_batch_size_per_gpu.
        '''
        device = self.policy_engine.device

        # 1. Models to train mode
        self.policy_engine.train()
        self.value_engine.train()

        # 2. zero grads
        self.policy_engine.zero_grad()
        self.value_engine.zero_grad()

        # 3. create progress bar
        num_micro = len(replay_buffer) # replay_buffer is already a dataloader of micro-batches
        progress_bar = tqdm(replay_buffer, total=num_micro)

        ga_pi_attr = getattr(self.policy_engine, 'gradient_accumulation_steps', 1)
        ga_pi = int(ga_pi_attr() if callable(ga_pi_attr) else ga_pi_attr)

        for step, micro_batch in enumerate(progress_bar):
            is_last = (step == (num_micro - 1))
            is_boundary = (((step + 1) % ga_pi) == 0) or is_last

            ########
            # 1. Data from buffer
            ########
            rewards   = micro_batch['rewards'].to(device, non_blocking=True)
            done      = micro_batch['done'].to(device, non_blocking=True)
            mask      = micro_batch['mask'].to(device, non_blocking=True)
            old_logprobs = micro_batch['old_logprobs'].to(device, non_blocking=True)

            # PPO-specific: values and last_val from vLLM rollout
            values    = micro_batch.get('v_olds', None)
            if values is not None:
                values = values.to(device, non_blocking=True)
            last_val  = micro_batch.get('last_val', None)
            if last_val is not None:
                last_val = last_val.to(device, non_blocking=True)

            input_ids = micro_batch['input_ids'].to(device, non_blocking=True)
            att_mask  = micro_batch['attn_mask'].to(device, non_blocking=True)
            pos_ids   = micro_batch.get('position_ids', None)

            ########
            # 2. Compute advantages
            ########
            returns, advantages = self.compute_advantages(
                                                    rewards=rewards,
                                                    values=values,
                                                    done=done,
                                                    mask=mask,
                                                    last_val=last_val)

            ########
            # 3. Update policy
            ########
            pi_metrics = self.policy_step_update(
                                    input_ids=input_ids,
                                    att_mask=att_mask,
                                    pos_ids=pos_ids,
                                    old_logprobs=old_logprobs,
                                    advantages=advantages,
                                    mask=mask,
                                    is_boundary=is_boundary)

            ########
            # 4. Update value
            ########
            vl_metrics = self.value_step_update(
                                    input_ids=input_ids,
                                    att_mask=att_mask,
                                    pos_ids=pos_ids,
                                    v_old=values,  # values from rollout (old value estimates)
                                    returns=returns,
                                    mask=mask,
                                    is_boundary=is_boundary,
                                    device=device)