### GRPO (our implementation)

GRPO is a PPO-style update trained from a replay buffer. During rollout, for each prompt we generate multiple completions and compute group-normalized advantages (e.g., z-scored rewards within the set of completions for the same prompt). These advantages, along with the corresponding old policy log-probabilities (`old_logprobs`) and a token mask (to exclude prompt/padding), are stored in the replay buffer.

At training time, we do **not** construct batches that keep all completions of the same prompt together. Instead, we **uniformly sample** from the replay buffer, so each (micro-)batch contains a mixture of tokens from many prompts and many generations. For each micro-batch we run a forward pass under the current policy to obtain token log-probabilities, form the PPO ratio, and apply the standard clipped surrogate objective using the stored advantages. Optionally, we add an entropy bonus and an optional KL-to-reference penalty implemented in a variance-reduced form.

We do this replay-style, uniform sampling for two practical reasons. First, it makes the training loop more flexible in how batches are constructed: groups do not need to be materialized explicitly during training, and we can easily support variable numbers of samples per prompt-group. This is also convenient when rollouts contain duplicates (e.g., a group where multiple generations collapse to the same completion for a prompt), since we can ignore or down-weight such samples directly during generation without restructuring group batches. Second, optimizing over an uniform samples from replay buffer rather than only the current set of grouped completions tends to improve stability, because each update is informed by a broader, more diverse mix of recent experiences.

#### Difference vs official GRPO-style implementations

Common GRPO implementations typically build each training step around prompt-groups: start from a batch of prompts, generate G completions per prompt, and compute normalization (advantages/scaling) within each group (across the G completions for the same prompt). Training batches therefore preserve group structure by construction.

In contrast, our GRPO uses uniform sampling from replay buffer, so training batches are not group-structured (even though the stored advantages are computed using per-prompt group normalization at rollout time). This changes the update statistics: instead of operating on a self-contained set of completions for a prompt, each update is driven by a mixture of replay samples across many prompts.

Additionally, standard GRPO implementations typically use a per-micro-batch masked mean for loss normalization (`loss_sum / mask.sum()`). When micro-batches have different numbers of valid tokens, which is common with variable-length rollouts, this produces "mean of means != global mean", giving disproportionate gradient weight to micro-batches with fewer valid tokens. Our implementation supports global token normalization (`normalize_loss=True`): the global token count (total valid tokens across all micro-batches and all GPUs) is computed before the training loop and used as the loss denominator. This ensures every action token contributes equally to the gradient regardless of which micro-batch or rank it lands on. See [RL Common README — Global Token Normalization](../RL/README.md#global-token-normalization-for-rl) and [SFT README — Loss Normalization](../SFT/README.md#loss-normalization) for the full derivation.

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