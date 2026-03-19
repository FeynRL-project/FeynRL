### P3O (Policy Performance on Previous Policies Optimization)

P3O uses the **Effective Sample Size (ESS)** of importance weights to adaptively clip the importance ratio, enabling effective learning from off-policy (stale) data without fixed clip bounds which can be hard sometimes to tune. The clipped ratio serves as a stop-gradient weighting coefficient for the log-probability. 

Based on [Fakoor et al., 2019 (arXiv:1905.01756)](https://arxiv.org/abs/1905.01756).

#### P3O loss (Eq. 1 in the paper)

$$
\mathcal{L}_{\text{P3O}}(\theta)= -\mathbb{E}\Big[\mathrm{sg}\big(\mathrm{clip}(r,\, 0,\, \text{ESS})\big)\ \log p_\theta(\cdot)\ A\Big].
$$

where the Effective Sample Size is:

$$
\text{ESS} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2} \cdot \frac{1}{n}
$$

with $w_i = \exp(\log \pi_\theta - \log \pi_{\text{old}})$ being the importance ratios over valid (non-padded) tokens.

CISPO uses the same stop-gradient formulation but with fixed clip bounds $[1-\epsilon_l, 1+\epsilon_h]$ instead of the data-driven $[0, \text{ESS}]$.

#### ESS behavior

ESS $\in [1/n,\, 1]$ (effectively $[0,\, 1]$ for typical LLM batch sizes):

- **ESS $\approx$ 1**: Data is fresh (ratios $\approx$ 1), clipping is loose, gradients flow normally.
- **ESS $\to$ 0**: Data is stale (ratios diverge), clipping becomes very tight, effectively zeroing out the gradient for that micro-batch.

This makes P3O naturally suited for settings where the replay buffer contains data from multiple policy versions (off-policy training with staleness).

#### Algorithm box

**Input:** initial policy parameters $\theta_0$, replay shards $\mathcal{B}$ (`micro_batches`)

**Hyperparams:** entropy weight $\beta_{\mathrm{ent}}$, KL weight $\beta_{\mathrm{kl}}$

**Replay fields:** `mask`, `old_logprobs`, group-normalized advantages `zscore`

For each training step:

1. For each micro-batch $\mathcal{B}$:

   - Forward policy: $(\log \pi_{\theta},\, H_{\theta}) \leftarrow \pi_{\theta}(\mathcal{B})$
   - Importance ratio: $r \leftarrow \exp(\log \pi_{\theta} - \texttt{old\\_logprobs})$
   - Effective Sample Size: $\text{ESS} \leftarrow \frac{(\sum r)^2}{\sum r^2} \cdot \frac{1}{n}$ (over valid tokens only)

   - P3O policy loss (masked mean, using `zscore`):

$$
\mathcal{L}_{\pi}
\leftarrow
-\mathrm{Mean}_{\texttt{mask}}\Big(
\mathrm{sg}\big(\mathrm{clip}(r,\, 0,\, \text{ESS})\big)
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

#### Differences from the original P3O paper

The original P3O was originally applied to discrete and continuous games like Atari and MuJoCo. This implementation adapts it for post-training which can be used for autoregressive tasks like text generation, which introduces several differences such as token-level action spaces and group-normalized rewards. However, the core idea of ESS-clipping remains the same.

#### Notes

- `clip_low` and `clip_high` are stored but **not used in the loss calculation**. They are only used for the `clipfrac` monitoring metric (fraction of tokens where the ratio falls outside `[1 - clip_low, 1 + clip_high]`).
