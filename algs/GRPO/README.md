### GRPO (Group Relative Policy Optimization)

GRPO [[1]](#references) is a PPO-style policy optimization algorithm that samples multiple completions for each prompt and derives advantages from their relative rewards within the sampled group. This group-relative formulation removes the need for a separate value model and provides a simple variance-reduction mechanism for policy updates.

#### Differences vs. standard GRPO implementations

**Training batches are not group-structured.** Common GRPO implementations build each training step around prompt-groups: generate $G$ completions per prompt, keep them together in the training batch, and compute both the normalization and the update within that group. Our implementation uniformly samples from the replay buffer instead, so each (micro-)batch is a mixture of tokens from many prompts and many generations. The group normalization still happens at rollout time and is baked into the stored `zscore`; only the training-time batching is decoupled from the group structure. We prefer this for two reasons:

- **Flexibility**: groups don't need to be materialized during training, we can handle variable numbers of samples per prompt-group, and we can drop or down-weight degenerate samples (e.g., duplicate completions within a group) at rollout time without restructuring training batches.
- **Stability**: updates driven by a broader mix of recent experiences tend to be more stable than updates confined to a single prompt's completions.

**Global token normalization instead of per-micro-batch means.** Standard GRPO typically uses `loss_sum / mask.sum()` per micro-batch. With variable-length rollouts this produces "mean of means ≠ global mean", giving disproportionate gradient weight to micro-batches with fewer valid tokens. With `normalize_loss=True`, the global valid-token count across all micro-batches and all ranks is computed before the training loop and used as the loss denominator, so every action token contributes equally regardless of which micro-batch or rank it lands on. See [RL Common README — Global Token Normalization](../RL/README.md#global-token-normalization-for-rl) and [SFT README — Loss Normalization](../SFT/README.md#loss-normalization) for the derivation.

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

- **Sync vs. async considerations**: GRPO uses a **fixed** clip range $(1-\epsilon_\ell, 1+\epsilon_h)$ that does not self-adjust with data staleness. It works well in sync mode, where each update sees nearly on-policy data. In async / overlap mode, as the replay buffer mixes older policy versions, a larger fraction of tokens can fall outside the clip range and contribute no gradient (high `clipfrac`, reduced effective sample size). For heavy off-policy use, consider: tightening the clip range, using a smaller `train_steps_per_epoch` / `max_lag`, enabling a **decoupled loss** [[2]](#references) (separate importance ratios for the clip vs. the gradient, so the clip doesn't zero out the gradient on stale tokens), or switching to an algorithm with a data-driven clip such as [P3O](../P3O/README.md) [[3]](#references).


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