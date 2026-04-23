### PPO (Proximal Policy Optimization)

PPO [[1]](#references) is a policy gradient algorithm that uses a clipped surrogate objective to constrain policy updates, together with a learned value function for variance reduction through Generalized Advantage Estimation (GAE) [[2]](#references).

Our PPO training step (`train_step`) runs on a replay shard (a list of `micro_batches`) and uses DeepSpeed for micro-batching and gradient accumulation. Before any policy/value updates, we call `precompute_gae(micro_batches)`, which runs the value model in `eval()` mode and computes returns and advantages via `compute_advantages(...)`. Then, for each `micro_batch`, we update the policy with a PPO clipped objective using stored `old_logprobs` and `mask`, plus optional entropy regularization (`ent_coeff`) and optional KL-to-reference penalty (`kl_coeff` if a reference model exists). We also update the value model by regressing `values` to the precomputed `returns` with a masked MSE. If `update_only_after_full_replay=True`, we take one optimizer step at the end of the replay shard (and scale losses by `ga_steps/num_micro` to keep gradient magnitude consistent).

#### Value network

The value model (`value_net.py`) wraps a HuggingFace causal LM backbone with a scalar value head. The LM head (`hidden_dim → vocab_size`) is replaced with a linear projection (`hidden_dim → 1`), initialized to zero so initial value predictions don't dominate early training. The backbone is extracted via `.model` (LLaMA, Gemma, Mistral, Qwen) or `.transformer` (GPT-2, GPT-Neo). The value network outputs `[B, T, 1]` which is squeezed to `[B, T]`.

The `value_forward` method returns `values [B, T-1]` (prediction-aligned, dropping the last position) and `last_value [B]` for bootstrapping. The `last_value` is computed by picking the value at each row's last non-pad token, which correctly handles variable-length sequences with padding.

#### Key implementation details

- **Advantage normalization**: Unlike GRPO/CISPO which use pre-computed z-scored rewards from the replay buffer, PPO normalizes advantages **inside `calculate_gae`** to have mean=0 and std=1 across all valid (masked) positions globally across all ranks (see point 3 under GAE below).

- **GAE**: Unlike GRPO/CISPO, which use group-normalized rewards directly as advantages and do not learn a value function, PPO computes advantages via **Generalized Advantage Estimation (GAE)** using a learned value function. For each valid position \(t\), the backward pass computes

$$
\delta_t = r_t + \gamma \, V_\phi(s_{t+1}) \, (1 - d_t) - V_\phi(s_t)
$$

$$
A_t = \delta_t + \gamma \, \lambda \, A_{t+1} \, (1 - d_t)
$$

where \(d_t\) indicates a **true terminal transition** for bootstrapping purposes, and \(\lambda\) is the GAE parameter. Returns are then computed as

$$
R_t = A_t + V_\phi(s_t).
$$

In practice, implementations often append a `last_value` and run the GAE recursion backward from the end of the sequence. The key detail is how this final bootstrap is handled. If the rollout ends because of a **true terminal state**, the correct bootstrap value is zero. If the rollout ends because of **truncation**, such as a max-length cutoff or epoch boundary, the correct bootstrap is the critic prediction at the next state, \(V_\phi(s_T)\), not zero.

This distinction matters. If a truncated rollout is incorrectly treated as terminal, the TD residual at the last valid step becomes

$$
\delta_{T-1} = r_{T-1} - V_\phi(s_{T-1}),
$$

instead of

$$
\delta_{T-1} = r_{T-1} + \gamma \, V_\phi(s_T) - V_\phi(s_{T-1}).
$$

That effectively assumes there is no future value beyond the cutoff, which introduces a downward bias at the end of the sequence. This bias then propagates backward through the GAE recursion and systematically underestimates advantages near the end of truncated rollouts.


- **GAE precomputation**: `precompute_gae` runs the value model in `eval()` mode over all micro-batches before any updates begin, so the value estimates used for GAE are consistent (not affected by value model updates during the training step). The precomputed `(returns, advs)` are stored on CPU and moved back to GPU per micro-batch during the update loop.

- **Paired shuffling**: Micro-batches and their precomputed GAE values are zipped together and shuffled as pairs, so the alignment between replay data and precomputed returns/advantages is maintained.

- **Dual engine updates**: Both the policy and value engines share the same gradient accumulation config and boundary logic. Both engines are updated within the same micro-batch loop, policy loss backward/step first, then value loss backward/step.

- **GAE validation checks**: `compute_advantages` validates that rewards and values contain no NaN on valid positions, that `done` flags are not set on padding positions, and that the mask has no non-contiguous holes (e.g., `[1,1,0,1,1]` is rejected).

- **Tracked metrics**: Policy metrics (`clipfrac`, `approx_kl`, `ent_loss`, `pi_loss`, `loss_total`, `kl_ref`) plus `value_loss_v` (value function MSE loss).

- **Global token normalization instead of per-micro-batch means.** Standard PPO implementations typically use `loss_sum / mask.sum()` per micro-batch. With variable-length rollouts this produces "mean of means ≠ global mean", giving disproportionate gradient weight to micro-batches with fewer valid tokens. With `normalize_loss=True`, the global valid-token count across all micro-batches and all ranks is computed before the training loop and used as the loss denominator, so every action token contributes equally regardless of which micro-batch or rank it lands on. See [RL Common README — Global Token Normalization](../RL/README.md#global-token-normalization-for-rl) and [SFT README — Loss Normalization](../SFT/README.md#loss-normalization) for the derivation.

- **Sync vs. async considerations**: PPO uses a **fixed** clip range $(1-\epsilon_\ell, 1+\epsilon_h)$ and a learned value function whose predictions also become stale as the policy drifts. It is best suited to sync mode, where each update sees near-on-policy data and the critic is fresh. In async / overlap mode, value estimates for older replay samples can become inaccurate and the fixed clip range does not self-adjust to rising off-policyness. For heavy off-policy use, consider: tightening the clip range, lowering `train_steps_per_epoch` / `max_lag`, enabling a **decoupled loss** [[3]](#references) (separate importance ratios for the clip vs. the gradient, so the clip doesn't zero out the gradient on stale tokens), or switching to a critic-free algorithm with a data-driven clip such as [P3O](../P3O/README.md) [[4]](#references).


**Input:** initial policy parameters $\theta_0$, initial value parameters $\phi_0$, replay shards $\mathcal{B}$ (`micro_batches`)

**Hyperparams:** discount $\gamma$, GAE $\tau$, clip range $(1-\epsilon_\ell,\ 1+\epsilon_h)$, entropy weight $\beta_{\mathrm{ent}}$, KL weight $\beta_{\mathrm{kl}}$

For each training step:

1. **Precompute GAE:** for all micro-batches, compute returns $R$ and advantages $A$ using $\gamma,\tau$.

2. For each micro-batch $\mathcal{B}$:

   - Forward policy: $(\log \pi_{\theta},\, H_{\theta}) \leftarrow \pi_{\theta}(\mathcal{B})$
   - PPO ratio: $\rho \leftarrow \exp(\log \pi_{\theta} - \texttt{old\\_logprobs})$
   - Normalize advantages: $A \leftarrow (A - \mu_A) / (\sigma_A + 10^{-8})$ over valid (masked) positions

   - Clipped policy loss (masked mean):

$$
\mathcal{L}_{\pi}
\leftarrow
-\mathrm{Mean}_{\texttt{mask}}\Big(
\min\big(
\rho A,\ \mathrm{clip}(\rho,1-\epsilon_\ell,1+\epsilon_h)\,A
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

   - Policy loss:

$$
\mathcal{L}
\leftarrow
\mathcal{L}_{\pi}
-\beta_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}}
+\beta_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}}
$$

   - DeepSpeed backward/step for policy (grad accumulation; optionally one step at shard end)

   - Forward value: $V_{\phi} \leftarrow V_{\phi}(\mathcal{B})$

   - Value loss (masked MSE):

$$
\mathcal{L}_{V}
\leftarrow
\frac{1}{2}\,\mathrm{Mean}_{\texttt{mask}}\big((V_{\phi}-R)^2\big)
$$

   - DeepSpeed backward/step for value (same boundary)

**Return:** $\theta,\phi$

#### References

[1] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017. [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

[2] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438, 2015. [https://arxiv.org/abs/1506.02438](https://arxiv.org/abs/1506.02438)

[3] J. Hilton, K. Cobbe, and J. Schulman. *Batch size-invariance for policy optimization.* arXiv:2110.00641, 2021. [https://arxiv.org/abs/2110.00641](https://arxiv.org/abs/2110.00641)

[4] R. Fakoor, P. Chaudhari, and A. J. Smola. *P3O: Policy-on Policy-off Policy Optimization.* arXiv:1905.01756, 2019. [https://arxiv.org/abs/1905.01756](https://arxiv.org/abs/1905.01756)