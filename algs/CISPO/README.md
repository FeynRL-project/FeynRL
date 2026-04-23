### CISPO (Clipped Importance Sampling Policy Optimization)

CISPO is a PPO-style policy optimization algorithm that clips the importance ratio but uses it as a **stop-gradient weighting coefficient** on the log-probability term, rather than as a factor inside a min-clip surrogate. Gradient flows through `log π_θ` on every valid token; the clip only bounds how strongly the importance weight can amplify that gradient. This removes the "gradient zeroed on out-of-range tokens" failure mode of standard PPO/GRPO clipping while keeping a hard ceiling on effective step size.

#### CISPO loss vs. PPO/GRPO-style clipping

With ratio $r = \exp(\log \pi_\theta - \log \pi_{\mathrm{old}})$, the standard PPO/GRPO clipping uses the "min of unclipped vs. clipped objective":

$$
\mathcal{L}_{\mathrm{PPO}}(\theta) = -\mathbb{E}\Big[\min\big(r\,A,\ \mathrm{clip}(r, 1-\epsilon_\ell, 1+\epsilon_h)\,A\big)\Big].
$$

CISPO instead treats the clipped ratio as a detached weighting coefficient:

$$
\mathcal{L}_{\mathrm{CISPO}}(\theta) = -\mathbb{E}\Big[\mathrm{sg}\big(\mathrm{clip}(r, 1-\epsilon_\ell, 1+\epsilon_h)\big)\ \log \pi_\theta(\cdot)\ A\Big].
$$

It is worth noting that this stop-gradient clipped-ratio formulation was originally introduced by P3O [[2]](#references), which used an ESS-derived one-sided cap instead of a fixed two-sided clip range. CISPO adapts the same mechanism with fixed $[1 - \epsilon_\ell, 1 + \epsilon_h]$ bounds.

In our implementation, CISPO reuses the full GRPO training loop (replay format, uniform replay sampling, masking, DeepSpeed micro-batching and gradient accumulation, micro-batch shuffling, GA remainder scaling, optional entropy and KL terms). Only `compute_policy_loss(...)` differs.

#### How CISPO differs in practice

Under standard PPO/GRPO clipping, the gradient is zeroed when the ratio moves outside the clip range in the direction the advantage would push further. CISPO instead always passes a gradient through `log π_θ` and weights it by the detached clipped ratio:

- When the ratio is within $[1 - \epsilon_\ell, 1 + \epsilon_h]$, the gradient magnitude scales with the importance weight (the ratio itself).
- When the ratio exceeds the clip bounds, the gradient still flows, but the weighting coefficient is clamped so the effective step size cannot grow unboundedly.

The net effect: CISPO never fully "turns off" the gradient for a token, but still limits how much the importance weight can amplify it.

#### Notes

- **Global token normalization instead of per-micro-batch means.** Standard CISPO-style implementations typically use `loss_sum / mask.sum()` per micro-batch. With variable-length rollouts this produces "mean of means ≠ global mean", giving disproportionate gradient weight to micro-batches with fewer valid tokens. With `normalize_loss=True`, the global valid-token count across all micro-batches and all ranks is computed before the training loop and used as the loss denominator, so every action token contributes equally regardless of which micro-batch or rank it lands on. See [RL Common README — Global Token Normalization](../RL/README.md#global-token-normalization-for-rl) and [SFT README — Loss Normalization](../SFT/README.md#loss-normalization) for the derivation.

- **Sync vs. async considerations**: CISPO is more tolerant of off-policy data than PPO/GRPO because the gradient is never zeroed out on clipped tokens; updates continue to flow through `log π_θ` even when ratios drift. However, the clip range is still **fixed**, so as data ages more tokens' weighting coefficients saturate at $1 \pm \epsilon$, capping how much importance-weight amplification the update can carry. For heavy off-policy use, consider: tightening the clip range, lowering `train_steps_per_epoch` / `max_lag`, enabling a **decoupled loss** [[1]](#references) (separate importance ratios for the clip vs. the gradient, so the clip doesn't zero out the gradient on stale tokens), or switching to an algorithm with a data-driven clip such as [P3O](../P3O/README.md) [[2]](#references).

- **Tracked metrics** (averaged across micro-batches): `clipfrac` (fraction of masked tokens where the ratio falls outside the clip range), `approx_kl` (variance-reduced approximate KL between current and old policy), `ent_loss`, `pi_loss`, `loss_total`, `kl_ref`.

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

#### References

[1] J. Hilton, K. Cobbe, and J. Schulman. *Batch size-invariance for policy optimization.* arXiv:2110.00641, 2021. [https://arxiv.org/abs/2110.00641](https://arxiv.org/abs/2110.00641)

[2] R. Fakoor, P. Chaudhari, and A. J. Smola. *P3O: Policy-on Policy-off Policy Optimization.* arXiv:1905.01756, 2019. [https://arxiv.org/abs/1905.01756](https://arxiv.org/abs/1905.01756)
