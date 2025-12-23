import torch
import numpy as np

class PPO:
    def __init__(self,
                model_engine:,
                optimizer,
                ref_model=None,
                use_cache: [bool]=False,
                kl_coeff: [float]=0.0,
                clip_low: [float]=0.0,
                clip_high: [float]=1.0,
                vf_clip: [float]=None,
                tau: [float]=0.95,
                gamma: [float]=0.99,
                entropy_coeff: [float]=0.0,
                alg_type: [str]="ppo",
                ):

        self.model_engine = model_engine
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.use_cache = use_cache
        self.kl_coeff = kl_coeff
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.tau = tau
        self.gamma = gamma
        self.alg_type = alg_type.lower()
        self.vf_clip = vf_clip
        self.ent_coeff = entropy_coeff

    @staticmethod
    def compute_advantages(rewards: torch.Tensor,
                           values: torch.Tensor,
                           done: torch.Tensor,
                           mask: torch.Tensor,
                           gamma: float,
                           tau: float,
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
            delta = rewards[:, t] + (gamma * next_val * not_done) - values[:, t]
            last_adv   = is_valid * (delta + (gamma * tau * last_adv * not_done))
            advs[:, t] = last_adv

            # to avoid any leaking from padding.
            next_val = values[:, t] * is_valid

        rets = advs + values

        return rets, advs

    def compute_policy_loss(self,
                            logprobs: torch.Tensor,
                            old_logprobs: torch.Tensor,
                            advantages: torch.Tensor,
                            mask: torch.Tensor,
                            ):
        '''
            logprobs, old_logprobs, advantages, mask: [B, T]
            Compute policy loss:
                1. ratio = exp(logprobs - old_logprobs)
                2. loss = -(min(ratio * adv, clip_adv * adv)) * mask
        '''
        # 1. make sure advantages are detached and convert to float32 for stability under bf16/fp16
        adv = advantages.detach().to(torch.float32)

        # 2. calculate ratio = exp(logprobs - old_logprobs)
        logratio = (logprobs - old_logprobs).to(torch.float32)
        ratio   = torch.exp(logratio)

        # 3. compute loss: -(min(ratio * adv, clip_adv)) * mask
        unclipped = ratio * adv
        clip_adv  = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high) * adv
        loss_pi   = -(torch.minimum(unclipped, clip_adv) * mask).sum() / mask.sum()

        # 4. useful metrics
        with torch.no_grad():
            clipped_mask = (ratio > (1.0 + self.clip_high)) | (ratio < (1.0 - self.clip_low))
            clipfrac = (clipped_mask.to(torch.float32) * mask).sum() / mask.sum()

            # approx KL: either E[old_logprobs - logprobs] or E[(ratio - 1) - logratio]
            approx_kl_t = (ratio - 1.0) - logratio
            approx_kl = (approx_kl_t.to(torch.float32) * mask).sum() / mask.sum()

            # save the metrics for debugging
            metrics = {
                'clipfrac': clipfrac,
                'approx_kl': approx_kl,
            }

        return loss_pi, metrics

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
        # 1. compute unlipped value loss
        rets = returns.detach()
        v_loss = (values - rets).pow(2)

        # 2. compute clipped value loss
        if  self.vf_clip is not None and v_old is not None:
            v_old = v_old.detach()

            # 3. compute clipped value loss
            v_clipped = v_old + torch.clamp(values - v_old, -self.vf_clip, self.vf_clip)
            v_loss_clipped = (v_clipped - rets).pow(2)
            vmax =  torch.maximum(v_loss, v_loss_clipped)
            loss = 0.5 * (vmax * mask).sum() / mask.sum()

            # 4. log how much things are changed
            with torch.no_grad():
                vf_clipfrac = (values - v_old).abs() > self.vf_clip
                vf_clipfrac = (vf_clipfrac * mask).sum() / mask.sum()

        else:
            loss = 0.5 * (v_loss * mask).sum() / mask.sum()
            vf_clipfrac = 0.0

        # save the metrics for debugging
        metrics = {
            'vf_clipfrac': vf_clipfrac,
        }

        return loss, metrics

    def compute_entropy_loss(self,
                            entropies: torch.Tensor,
                            mask: torch.Tensor,
                            ):
        '''
            Compute entropy loss
        '''
        if entropies is None or self.ent_coeff == 0.0:
            return 0.0, {}

        else:
            loss = - (entropies * mask).sum() / mask.sum()
            return loss, {'ent_loss': loss}

    def compute_loss(self,
                     replay_buffer: ReplayBuffer):
        '''
            Compute total ppo loss
        '''
        pass
