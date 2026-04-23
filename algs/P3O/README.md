### P3O (Policy-on Policy-off Policy Optimization)

P3O [[1]](#references) learns from mixed on/off-policy replay data without manually tuned clip bounds or KL coefficients. It uses a single statistic, the **Effective Sample Size (ESS)** of the token-level importance weights, to **simultaneously** govern two complementary mechanisms:

- **How much to trust the policy gradient** from the current batch (via a one-sided ESS cap on the importance ratio), and
- **How strongly to pull the policy back** toward the data-generating distribution (via an adaptive trust-region KL weighted by `(1 − ESS)`).

When data is fresh, ESS ≈ 1: the cap is loose and the trust region is inactive, so learning behaves like standard policy gradient. As data ages, ESS drops: the cap tightens and the trust region takes over, preventing the policy from drifting beyond what the off-policy data can reliably support. This makes P3O a natural fit for replay buffers that mix multiple policy versions.

This implementation is based on the method introduced by Fakoor et al. [[1]](#references).


#### P3O loss

The total per-token loss combines a policy-gradient term, the adaptive trust-region KL, and optional entropy / reference-KL terms:

$$
\mathcal{L} = \mathcal{L}_{\pi} - \beta_{\mathrm{ent}} \mathcal{L}_{\mathrm{ent}} + (1-\text{ESS})\mathcal{L}_{\mathrm{kl,old}} + \beta_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl,ref}}
$$

**Policy-gradient term (one-sided ESS cap, stop-gradient on the ratio):**

$$
\mathcal{L}_{\pi}(\theta)
= -\mathbb{E}\Big[
\mathrm{sg}\big(\mathrm{clip}(r,0,\text{ESS})\big)\ \log \pi_\theta(\cdot)\ A
\Big],
\qquad
r = \frac{\pi_\theta}{\pi_{\mathrm{old}}}
$$

The clipped ratio acts as a **non-negative weighting coefficient** for the log-probability; it is detached so it does not contribute gradient itself. This is a REINFORCE-style score-function update with the weight bounded by ESS.

**Adaptive trust-region KL (paper's `(1 − ESS) · KL` term):**

$$
\mathcal{L}_{\mathrm{kl,old}}(\theta) = \mathrm{KL}\big(\pi_\theta\ \big\|\ \pi_{\mathrm{old}}\big)
$$

pulls $\pi_\theta$ back toward the behavior policy $\pi_{\mathrm{old}}$, weighted adaptively by $(1 - \text{ESS})$ so the pull is strong exactly when the data is stale.

**Effective Sample Size** (computed over valid / non-padded tokens, aggregated across all DP ranks by all-reducing $\sum w$, $\sum w^2$, $n$):

$$
\text{ESS} = \frac{\left(\sum_i w_i\right)^2}{n \cdot \sum_i w_i^2} \in \left[\tfrac{1}{n}, 1\right],
\qquad
w_i = \exp(\log \pi_\theta - \log \pi_{\mathrm{old}})
$$

#### The dual role of ESS, in one picture

| Data regime        | ESS value      | Ratio cap $[0, \text{ESS}]$ | KL weight $(1 - \text{ESS})$ | Net effect                              |
| ------------------ | -------------- | --------------------------- | ---------------------------- | --------------------------------------- |
| Fresh (on-policy)  | ≈ 1            | Loose, gradient flows       | ≈ 0, KL inactive             | Standard policy gradient                |
| Stale (off-policy) | → 0            | Tight, PG term suppressed   | ≈ 1, KL dominates            | Trust-region pull toward $\pi_{\mathrm{old}}$ |

Both terms share the **same ESS**, so there is no separate knob to tune; the replay data's own statistics decide the schedule.

#### Algorithm box

**Input:** initial policy parameters $\theta_0$, replay shards $\mathcal{B}$ (`micro_batches`)

**Hyperparams:** entropy weight $\beta_{\mathrm{ent}}$, reference-KL weight $\beta_{\mathrm{kl}}$

**Replay fields:** `mask`, `old_logprobs`, group-normalized advantages `zscore`

For each training step, for each micro-batch $\mathcal{B}$:

1. Forward policy: $(\log \pi_{\theta}, H_{\theta}) \leftarrow \pi_{\theta}(\mathcal{B})$
2. Importance ratio: $r \leftarrow \exp(\log \pi_{\theta} - \texttt{old\\_logprobs})$ on valid tokens
3. ESS: $\text{ESS} \leftarrow \frac{(\sum r)^2}{n \cdot \sum r^2}$ (all-reduced across DP ranks)
4. Policy-gradient term (masked sum, using `zscore` as the advantage):

$$
\mathcal{L}_{\pi}
\leftarrow
-\sum_{\texttt{mask}}
\mathrm{sg}\big(\mathrm{clip}(r, 0, \text{ESS})\big)
\cdot \log \pi_{\theta}
\cdot \texttt{zscore}
$$

5. Adaptive trust-region KL:

$$
(1 - \text{ESS})\ \cdot\ \sum_{\texttt{mask}} \mathrm{KL}\big(\pi_{\theta}\ \big\|\ \pi_{\mathrm{old}}\big)
$$

6. (Optional) entropy bonus $\sum_{\texttt{mask}} H_\theta$ and reference-model KL $\sum_{\texttt{mask}} \mathrm{KL}(\pi_\theta \|\ \pi_{\mathrm{ref}})$
7. Combine into the total loss above (all per-token sums, normalized downstream by the global token count)
8. DeepSpeed backward / step (gradient accumulation; optionally one step at shard end)

**Return:** $\theta$


#### Notes

- **Same loss in sync and async**: because the ESS schedule is derived from each batch's own importance weights rather than from a separately-tuned staleness coefficient, the P3O loss runs unchanged in both the sync and async training engines. Other algorithms typically need extra machinery when moved from sync to async, e.g. a decoupled loss (separate ratios for the clip vs. the gradient, as in PPO/GRPO off-policy variants), a Polyak / EWMA proximal-policy snapshot (PPO-EWMA style), tighter clip ranges re-tuned for the staleness budget, or an explicit off-policy importance-weight correction. P3O self-adjusts as data freshness changes using only the current batch's ratios.
- `clip_low` / `clip_high` are stored but **not used in the P3O loss**. They control only the `clipfrac` monitoring metric (fraction of tokens where the ratio falls outside $[1 - \text{clip\\_low}, 1 + \text{clip\\_high}]$), so you can read P3O's actual clip behavior on the same scale as a PPO clip range.
- The constructor accepts `use_decoupled_loss` and `behave_imp_weight_cap` for backward compatibility with older configs, but they are **no-ops** in this paper-faithful formulation.
- Tracked metrics (averaged across micro-batches): `clipfrac`, `approx_kl`, `ent_loss`, `pi_loss`, `loss_total`, `kl_ref` (KL to frozen reference), `kl_behavioral` (KL to behavior, pre-weighting), `ess_factor`.

#### References

[1] R. Fakoor, P. Chaudhari, and A. J. Smola. *P3O: Policy-on Policy-off Policy Optimization.* arXiv:1905.01756, 2019. [https://arxiv.org/abs/1905.01756](https://arxiv.org/abs/1905.01756)