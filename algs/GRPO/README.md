### GRPO (Group Relative Policy Optimization)

GRPO [[1]](#references) is a PPO-style policy optimization algorithm that samples multiple completions for each prompt and derives advantages from their relative rewards within the sampled group. This group-relative formulation removes the need for a separate value model and provides a simple variance-reduction mechanism for policy updates.

#### Differences vs. standard GRPO implementations

**Training batches are not group-structured.** Common GRPO implementations build each training step around prompt-groups: generate $G$ completions per prompt, keep them together in the training batch, and compute both the normalization and the update within that group. Our implementation uniformly samples from the replay buffer instead, so each (micro-)batch is a mixture of tokens from many prompts and many generations. The group normalization still happens at rollout time and is baked into the stored `zscore`; only the training-time batching is decoupled from the group structure. We prefer this for two reasons:

- **Flexibility**: groups don't need to be materialized during training, we can handle variable numbers of samples per prompt-group, and we can drop or down-weight degenerate samples (e.g., duplicate completions within a group) at rollout time without restructuring training batches.
- **Broader per-step mix**: each update sees tokens from many prompts and many generations rather than being confined to the completions from a single prompt-group.

**Global token normalization instead of per-micro-batch means.** Standard GRPO typically uses `loss_sum / mask.sum()` per micro-batch. With variable-length rollouts this produces "mean of means ≠ global mean", giving disproportionate gradient weight to micro-batches with fewer valid tokens. With `normalize_loss=True`, the global valid-token count across all micro-batches and all ranks is computed before the training loop and used as the loss denominator, so every action token contributes equally regardless of which micro-batch or rank it lands on. See [RL Common README: Global Token Normalization](../RL/README.md#global-token-normalization-for-rl) and [SFT README: Loss Normalization](../SFT/README.md#loss-normalization) for the derivation.

#### `update_only_after_full_replay=True`

This flag does **not** change sampling as we still sample uniformly from replay. It only changes the **optimizer step boundary**:

* If `False`, we step according to DeepSpeed gradient-accumulation boundaries (typical micro-batch accumulation).
* If `True`, we accumulate gradients over the entire replay shard and apply **one optimizer step at the end** (often with a scaling to keep gradient magnitude comparable).

#### Implementation details

- **Micro-batch shuffling**: At each training step, the list of micro-batches is randomly shuffled before iteration. This ensures that across multiple `train_steps_per_epoch` calls, the gradient-accumulation boundary falls on different micro-batches, avoiding systematic bias from always having the same micro-batches grouped together in the same accumulation window.

- **Loss scaling for GA remainder**: When the number of micro-batches is not divisible by `gradient_accumulation_steps`, the last GA bucket has fewer micro-batches. DeepSpeed still divides by `gradient_accumulation_steps`, so the code scales the loss in the final bucket by `ga_steps / remainder` to produce the correct mean gradient. When `update_only_after_full_replay=True`, the loss is instead scaled by `ga_steps / num_micro` for all micro-batches.

- **KL divergence form**: The KL penalty uses the variance-reduced estimator: $\text{KL} = \log(\pi/\pi_{\text{ref}}) + \pi_{\text{ref}}/\pi - 1$. Computation is performed in float32 for numerical stability under bf16/fp16.

- **Masking**: Padded and prompt positions are zeroed out in both the loss and all metrics. The denominator for mean computation is `mask.sum()` (clamped to ≥ 1).

- **Tracked metrics** (averaged across micro-batches): such as `clipfrac` (fraction of masked tokens where ratio falls outside the clip range), `approx_kl` (variance-reduced approximate KL between current and old policy), `ent_loss`, `pi_loss`, `loss_total`, `kl_ref`, etc.

- **Sync vs. async considerations**: GRPO uses a **fixed** clip range $(1-\epsilon_\ell, 1+\epsilon_h)$ that does not self-adjust with data staleness. In sync mode each update sees nearly on-policy data and most ratios fall inside the clip range. In async / overlap mode, as the replay buffer mixes older policy versions, a larger fraction of tokens can fall outside the clip range and contribute no gradient (higher `clipfrac`). Overlap mode therefore **auto-enables the decoupled loss** (see [Decoupled loss](#decoupled-loss-overlap-mode) below) [[2]](#references) so the clip doesn't zero out the gradient on tokens whose ratio against the rollout policy has drifted but whose ratio against the in-shard snapshot is still close to 1. For heavy off-policy use you can also tighten the clip range, lower `train_steps_per_epoch` / `overlap.max_lag`, or switch to a data-driven-clip algorithm such as [P3O](../P3O/README.md) [[3]](#references).

#### Decoupled loss (overlap mode)

The decoupled loss is automatically enabled when `overlap.enabled=True` (wired by `use_decoupled_loss = overlap.enabled`). It is **shared across GRPO, PPO, and CISPO**. P3O does not use it; its ESS-based one-sided clip and adaptive trust-region KL already absorb off-policy drift from the same statistic, so the proximal snapshot would be redundant. (P3O's constructor accepts `use_decoupled_loss` and `behave_imp_weight_cap` for backward compatibility but treats them as no-ops.)

**Why it exists.** The standard PPO-style loss uses a single ratio between the current policy $\pi_\theta$ and the behavior policy $\pi_{\mathrm{old}}$ (the policy that generated the rollouts), and applies a symmetric clip to it:

$$
\mathcal{L}_{\mathrm{std}}(\theta) = -\mathbb{E}\Big[ \min\big(r\,A,\ \mathrm{clip}(r, 1-\epsilon_\ell, 1+\epsilon_h)\,A\big) \Big],\qquad r = \frac{\pi_\theta}{\pi_{\mathrm{old}}}.
$$

In overlap mode the replay buffer mixes multiple policy versions, so $\pi_{\mathrm{old}}$ for any given token can be several versions behind $\pi_\theta$ before the first training step even runs. Under the standard loss, tokens whose ratio against $\pi_{\mathrm{old}}$ lands outside the clip range contribute no gradient, even when the step the policy has taken *within the current training shard* is small.

**The decoupled objective.** The decoupled loss [[2]](#references) separates the two roles of the ratio: the **clip** (bounding the step size the policy takes within the current shard), and the **importance-sampling correction** (reweighting the rollout data). Given a proximal policy $\pi_{\mathrm{prox}}$ that is "recent but not $\pi_{\mathrm{old}}$", the objective uses two different ratios:

$$
\mathcal{L}_{\mathrm{dec}}(\theta) = -\mathbb{E}\Big[ w \cdot \min\big(r_{\mathrm{prox}}\,A,\ \mathrm{clip}(r_{\mathrm{prox}}, 1-\epsilon_\ell, 1+\epsilon_h)\,A\big) \Big]
$$

$$
r_{\mathrm{prox}} = \frac{\pi_\theta}{\pi_{\mathrm{prox}}},\qquad w = \mathrm{sg}\!\left(\frac{\pi_{\mathrm{prox}}}{\pi_{\mathrm{old}}}\right).
$$

- The **clip is on $r_{\mathrm{prox}}$**, i.e. how far $\pi_\theta$ has drifted from the proximal policy. Tokens whose in-shard step is small are no longer clipped just because the replay data is stale.
- The **behavioral importance weight $w = \pi_{\mathrm{prox}} / \pi_{\mathrm{old}}$** reweights rollout samples onto the proximal distribution. It is detached (neither $\pi_{\mathrm{prox}}$ nor $\pi_{\mathrm{old}}$ carry gradient for the current $\theta$) and only rescales the per-token loss.

**How $\pi_{\mathrm{prox}}$ is obtained (shard-start snapshot, differs from the paper).** The original paper [[2]](#references) defines $\pi_{\mathrm{prox}}$ as an **exponentially-weighted moving average (EWMA)** of the policy network parameters, updated every gradient step with decay rate $\beta_{\mathrm{prox}}$. This codebase uses a simpler, memory-cheaper approximation: at the start of each training shard (*after* micro-batch shuffling, *before* any optimizer step), each training rank runs `policy_forward` under `torch.no_grad()` over every micro-batch and stores the resulting detached per-token logprobs ([`algs/RL/common.py:223-245`](../RL/common.py#L223-L245)). Those logprobs are $\log \pi_{\mathrm{prox}}$ for the entire inner-update loop; the proximal policy is effectively "the policy at the start of this shard" and does not move during the inner loop. Memory cost is `num_micro × [B, T-1]` detached logprob tensors instead of a full EWMA copy of the model. The decoupled objective above is the same either way; only the definition of $\pi_{\mathrm{prox}}$ differs.

**`overlap.behave_imp_weight_cap` (addition beyond the paper).** Not part of the original decoupled objective. The weight $w$ can grow large on individual tokens when $\pi_{\mathrm{prox}}$ diverges from $\pi_{\mathrm{old}}$ (for example, after many training steps on stale replay data), and a single runaway $w$ can dominate the micro-batch gradient. This codebase therefore adds an optional cap under the `overlap:` config block:

$$
w \leftarrow \min(w,\ \texttt{behave\\_imp\\_weight\\_cap}).
$$

Set to `null` for no cap (the paper's default). The config validator requires `> 1.0` when set for non-P3O algorithms (P3O ignores it). Typical values: `2.0`–`5.0`. The `behave_w_mean`, `behave_w_max`, `behave_w_min`, and `behave_w_capfrac` metrics report the distribution and cap-hit rate so you can tune this against your observed replay staleness.


**Input:** initial policy parameters $\theta_0$, replay shards $\mathcal{B}$ (`micro_batches`)

**Hyperparams:** clip range $(1-\epsilon_\ell,\ 1+\epsilon_h)$, entropy weight $\beta_{\mathrm{ent}}$, KL weight $\beta_{\mathrm{kl}}$

**Replay fields:** `mask`, `old_logprobs`, group-normalized advantages `zscore`

*Training samples are uniformly drawn from replay; no prompt-group batching at training time.*

For each training step:

1. For each micro-batch $\mathcal{B}$:

   - Forward policy: $(\log \pi_{\theta},\, H_{\theta}) \leftarrow \pi_{\theta}(\mathcal{B})$
   - PPO ratio: $\rho \leftarrow \exp(\log \pi_{\theta} - \texttt{old\\_logprobs})$

   - Clipped policy loss (masked mean, using `zscore`):

$$
\mathcal{L}_{\pi}
\leftarrow
-\mathrm{Mean}_{\texttt{mask}}\Big(
\min\big(
\rho\,\texttt{zscore},\
\mathrm{clip}(\rho,1-\epsilon_\ell,1+\epsilon_h)\,\texttt{zscore}
\big)
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

[1] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

[2] J. Hilton, K. Cobbe, and J. Schulman. *Batch size-invariance for policy optimization.* arXiv:2110.00641, 2021. [https://arxiv.org/abs/2110.00641](https://arxiv.org/abs/2110.00641)

[3] R. Fakoor, P. Chaudhari, and A. J. Smola. *P3O: Policy-on Policy-off Policy Optimization.* arXiv:1905.01756, 2019. [https://arxiv.org/abs/1905.01756](https://arxiv.org/abs/1905.01756)