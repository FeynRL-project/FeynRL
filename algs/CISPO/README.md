### CISPO (vs GRPO-style clipping)

CISPO differs from typical PPO/GRPO-style updates only in the **policy loss / clipping form**. GRPO-style clipping uses the "min of unclipped vs clipped objective": with ratio $r=\exp(\log p_\theta-\log q)$, it optimizes

$$
\mathcal{L}_{\text{GRPO}}(\theta)=-\mathbb{E}\Big[\min\big(rA,\ \mathrm{clip}(r,1-\epsilon_l,1+\epsilon_h)A\big)\Big].
$$

CISPO instead clips the ratio and uses the **clipped ratio as a stop-gradient coefficient** multiplying the log-prob term:

$$
\mathcal{L}_{\text{CISPO}}(\theta)= -\mathbb{E}\Big[\mathrm{sg}\big(\mathrm{clip}(r,1-\epsilon_l,1+\epsilon_h)\big)\ \log p_\theta(\cdot)\ A\Big].
$$

In this repo, CISPO is therefore "SGRPO with a different `compute_policy_loss(...)`": the replay format, uniform replay sampling, masking, DeepSpeed micro-batching/accumulation, micro-batch shuffling, GA remainder scaling, and optional entropy/KL terms are all unchanged — only the clipping/weighting in the policy objective differs.

#### How CISPO differs in practice

In standard PPO/SGRPO clipping, the gradient is zeroed when the ratio moves outside the clip range (in the direction that the advantage would encourage further movement). CISPO takes a different approach: it always passes a gradient through `log p_θ`, but weights it by the *detached* clipped ratio. This means:

- When the ratio is within `[1 - clip_low, 1 + clip_high]`, the gradient magnitude scales with the importance weight (the ratio itself).
- When the ratio exceeds the clip bounds, the gradient still flows but the weighting coefficient is clamped, preventing the effective step size from growing unboundedly.

The result is that CISPO never fully "turns off" the gradient for a token (as PPO clipping can), but limits how much the importance weight can amplify it.

#### Algorithm box

**Input:** initial policy parameters $\theta_0$, replay shards $\mathcal{B}$ (`micro_batches`)

**Hyperparams:** clip range $(1-\epsilon_\ell,\ 1+\epsilon_h)$, entropy weight $\beta_{\mathrm{ent}}$, KL weight $\beta_{\mathrm{kl}}$

**Replay fields:** `mask`, `old_logprobs`, group-normalized advantages `zscore`

For each training step:

1. For each micro-batch $\mathcal{B}$:

   - Forward policy: $(\log \pi_{\theta},\, H_{\theta}) \leftarrow \pi_{\theta}(\mathcal{B})$
   - PPO ratio: $\rho \leftarrow \exp(\log \pi_{\theta} - \texttt{old\\_logprobs})$

   - CISPO policy loss (masked mean, using `zscore`):

$$
\mathcal{L}_{\pi}
\leftarrow
-\mathrm{Mean}_{\texttt{mask}}\Big(
\mathrm{sg}\big(\mathrm{clip}(\rho, 1-\epsilon_\ell, 1+\epsilon_h)\big)
\cdot \log \pi_{\theta}
\cdot \texttt{zscore}
\Big)
$$

   - (Optional) entropy term:

$$
\mathcal{L}_{\mathrm{ent}} \leftarrow \mathrm{Mean}_{\texttt{mask}}(H_{\theta})
$$

   - (Optional) KL penalty (vs. a reference policy $\pi_{\mathrm{ref}}$):

$$
\mathcal{L}_{\mathrm{kl}} \leftarrow \mathrm{Mean}_{\texttt{mask}}\big(\mathrm{KL}(\pi_{\theta}\,\|\,\pi_{\mathrm{ref}})\big)
$$

   - Total loss:

$$
\mathcal{L}
\leftarrow
\mathcal{L}_{\pi}
-\beta_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}}
+\beta_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}}
$$

   - DeepSpeed backward/step (grad accumulation; optionally one step at shard end)

**Return:** $\theta$
