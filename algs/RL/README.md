# RL Common (`common.py`)

The `COMMON` base class is the shared infrastructure for all policy-gradient RL algorithms in this repo (PPO, GRPO, DAPO, CISPO, P3O). It keeps algorithm-specific files focused on **what each algorithm does differently in `compute_policy_loss`**, while everything that should be identical across algorithms: forward passes, KL estimation, numerical safeguards, loss normalization, health checks, model loading, distributed weight sync, and checkpointing; lives here.

If you're implementing a new RL algorithm, you typically inherit from `COMMON` and only provide your own `compute_policy_loss(...)` and `train_step(...)`.

## What's shared

### Forward passes
- **`policy_forward`** runs the current policy under the training engine and returns `(logprobs, entropies, target_ids)` on the action tokens only (shape `[B, T-1]`). `entropies` is only computed when `ent_coeff > 0`, otherwise `None`. Log-softmax + gather is fused via `torch.compile` to avoid materializing a full `[B, T, V]` logprobs tensor.
- **`ref_forward`** runs the frozen reference model (if configured) under `torch.no_grad()` for the optional KL-to-reference penalty; returns `ref_logprobs` only.

Both methods handle causal-LM shift alignment (dropping the last position, gathering target log-probs in float32) consistently, so every algorithm sees the same `[B, T-1]` layout.

### Proximal-policy snapshot (overlap mode only)
**`snapshot_prox_logprobs`** runs `policy_forward` under `no_grad` over every micro-batch in the shard *before any optimizer step* and returns one detached `[B, T-1]` tensor per micro-batch. This is the "proximal policy" reference $\log \pi_{\mathrm{prox}}$ consumed by the **decoupled loss** that PPO, GRPO, and CISPO all auto-enable when `overlap.enabled=True` (see [`core/rl_engines.py:56`](../../core/rl_engines.py#L56): `use_decoupled_loss = overlap.enabled`). P3O does not use it because it derives its trust region directly from the stored `old_logprobs`.

For the full decoupled-loss formulation, the role of the snapshot, and the `overlap.behave_imp_weight_cap` knob, see [**GRPO README: Decoupled loss (overlap mode)**](../GRPO/README.md#decoupled-loss-overlap-mode).

### KL divergence estimator
`compute_kl_distance(logprobs, ref_logprobs)` uses the variance-reduced, always-non-negative sample-based KL estimator introduced by Schulman [[1]](#references):

$$
\mathrm{KL}(\pi \,\|\, \pi_{\mathrm{ref}}) \approx \log(\pi/\pi_{\mathrm{ref}}) + \pi_{\mathrm{ref}}/\pi - 1.
$$

Compared to the naive estimator $\log(\pi/\pi_{\mathrm{ref}})$, this form has the same expectation but is non-negative on every sample (whereas $\log(\pi/\pi_{\mathrm{ref}})$ itself can be negative per-sample even though the true KL is non-negative). Per-token KL values logged as metrics therefore have the correct sign for every token.

The estimator is computed in float32 for numerical stability under bf16/fp16, with the exponent clamped to $\pm 10$ to prevent overflow. All algorithms use the same estimator for both the KL-to-reference penalty and, where applicable, the KL-to-behavior trust region.

### Numerical safeguards
- **`sanitize_logprobs`** replaces any NaN/Inf positions in the logprobs with a sentinel value (`1.0`, since real logprobs are always ≤ 0) and returns a `nan_mask` so the caller can zero those positions out of the loss via `mask = mask * (~nan_mask)`. This keeps the loss value finite. It does not keep the gradient finite when a policy logits row contains NaN or `+inf` (see [Tokens dropped by NaN sanitization](#tokens-dropped-by-nan-sanitization)).
- **`check_weights_health`** iterates the policy engine's parameters and raises if any contain NaN. Called before and after every `train_step` in every algorithm, so weight corruption carried over from a previous epoch or a failed weight sync is caught immediately.
- **`check_all_masked`** counts consecutive all-masked micro-batches (where `local_denom == 0` after sanitization) and aborts after a threshold (default 20). A sustained all-masked regime produces zero gradients that can wedge ZeRO-3 NCCL collectives.
- **`check_logit_health`** and **`check_weights_per_microbatch_update`** are additional diagnostics available on `COMMON` but not wired into the active training loop today; they are exercised by `unit_tests/unit/test_common_health_checks.py`.

### Loss normalization
- **`compute_global_token_denom`** computes a single denominator for the whole replay shard (used with `update_only_after_full_replay=True`). A single scalar float64 `all_reduce` (8 bytes) yields the global valid-token count.
- **`compute_per_group_token_denoms`** computes one denominator per gradient-accumulation group (used with `update_only_after_full_replay=False`). A single `all_reduce` over a `[num_micro]` float64 tensor yields the global per-micro-batch counts, which are then summed into GA-group buckets locally.

See [Global Token Normalization for RL](#global-token-normalization-for-rl) below for the derivation.

### Model loading, PEFT, and engine init
- **`load_single_model`** loads a HuggingFace causal LM backbone with the configured dtype and attention implementation, optionally wraps it with the PEFT adapter (policy/value only, not the frozen reference), and enables gradient checkpointing on the policy when requested.
- **`apply_peft_module`** wraps the model with a LoRA adapter (the only PEFT type currently supported) when `peft_config.use_peft` is true.
- **`init_training_engine`** seeds all ranks identically for reproducible model init, calls `load_model` (the alg-specific override), then wires up the DeepSpeed engines (policy, reference if used, and value for PPO), passing only the trainable params to DeepSpeed so frozen weights (e.g., LoRA base) don't consume optimizer memory.

### Weight sync and checkpointing
- **`gather_state_dict`** / **`save_checkpoint`** perform a ZeRO-3-aware full-model gather and a sharded save.
- **`gather_weights_for_nccl`** / **`nccl_broadcast_gathered`** are the training-side half of the NCCL weight broadcast to rollout engines.
- **`init_weight_nccl_group`** / **`close_weight_nccl_group`** manage the lifecycle of the dedicated NCCL process group used for weight sync.
- **`save_engine_state`** / **`load_engine_state`** persist and restore the full DeepSpeed engine state for resume.

These are the same across every RL algorithm because the training↔rollout contract is uniform.

---

## Global Token Normalization for RL

In RL training, the replay buffer produces variable-length rollouts. After collation and sharding across GPUs, each micro-batch can have a different number of valid action tokens. The same "mean of means $\neq$ global mean" problem from SFT applies here, but with an additional source of variance: **rollout lengths change every epoch** because the policy generates different-length responses as it improves.

For the full derivation of why naive per-micro-batch normalization is biased and how `dp_scale / ga_denom` corrects it, see [`algs/SFT/README.md`](../SFT/README.md). For `normalize_loss=True` the formula and DeepSpeed interaction are identical. The `normalize_loss=False` fallbacks differ: SFT scales the raw sum by `dp_scale` (a global *sum*), while RL keeps its original per-micro-batch mean with gradient-accumulation rescaling (a mean of local means).

### Why it matters more in RL than SFT

In SFT, sequence lengths are fixed by the dataset and only vary due to packing/padding. In RL:

- **Rollout lengths vary across epochs.** Early in training the policy may produce short responses; later it may produce longer ones. Without global normalization, epochs with longer responses would have smaller per-token gradient magnitude (more tokens $\Rightarrow$ smaller mean), creating an implicit length-dependent learning rate.
- **Replay buffer sharding is uneven.** `prepare_training_batches` pads the number of micro-batches across ranks, but the token counts within those micro-batches can differ significantly when prompts have different lengths or the policy generates variable-length completions.
- **Group-based advantages (GRPO/DAPO/CISPO/P3O) amplify the imbalance.** When `n_samples > 1`, some prompts produce many short responses and others produce fewer long ones. Without global normalization, short-response micro-batches get disproportionate gradient weight.

### What `compute_global_token_denom` does

```
ga_denom = total valid action tokens across ALL micro-batches on ALL ranks
dp_scale = ga_steps * world_size
loss_for_backward = loss_sum * (dp_scale / ga_denom)
```

This single `all_reduce` (one scalar, 8 bytes) runs once before the training loop in each `train_step` call. After DeepSpeed's internal gradient averaging ($\div$ `ga_steps` $\div$ `world_size`), the net effect is:

$$\nabla_{\text{final}} = \frac{1}{N_{\text{global}}} \sum_{\text{all tokens}} m_t \cdot \nabla \ell_t$$

Every action token contributes equally to the gradient, regardless of which micro-batch or rank it landed on. When `normalize_loss=False`, the original per-micro-batch normalization is used (backward compatible).

### One denominator per optimizer step

`ga_denom` is a single number shared by every contribution to one optimizer step: every micro-batch, on every rank. That is what makes the weighting uniform. Two micro-batches with 10 and 100 tokens both divide by the same global total, so a token in the short one carries exactly the same weight as a token in the long one.

Which micro-batches are counted depends on where the optimizer step lands:

- `update_only_after_full_replay=True` (config `train.update_after_full_replay: True`) gives one optimizer step per `train_step` call over the replay shard, so `compute_global_token_denom` counts every micro-batch on every rank. Gradient accumulation does not subdivide anything here, because the boundary is set only on the last micro-batch.
- `update_only_after_full_replay=False` (also when `train.update_after_full_replay` is unset) gives one optimizer step per gradient-accumulation group, so `compute_per_group_token_denoms` gives each group its own total, counted over that group's micro-batches on every rank.

### Tokens dropped by NaN sanitization

`sanitize_logprobs` masks out any position whose log-probability came back NaN or Inf, so those tokens contribute nothing to `loss_sum`. They are still counted in `ga_denom`, which is fixed by an `all_reduce` before the training loop's first forward, so it cannot know which positions the policy or reference forward will return as bad. Proximal NaN positions, found by the no-grad snapshot that the decoupled loss takes (always in P4O), are already known at that point, but `ga_denom` still counts them like every other dropped token.

Write `Z` for this shared denominator and `D` for the number of tokens dropped across the whole optimizer step, summed over every micro-batch on every rank. The accumulated gradient is then `(Z - D) / Z` times the per-token mean over the surviving tokens. Every surviving token keeps the same weight `1 / Z` as every other token in the step, so a micro-batch that loses tokens contributes proportionally less, and only the magnitude of the gradient moves, never its direction. In a healthy step nothing is dropped and the factor is exactly 1.

Keeping one shared denominator is what preserves that property. A per-micro-batch adjustment would need `D` for the *whole* step, which is not knowable when the first micro-batch is scaled and backpropagated, long before the later ones have run their forwards. Exactness would need either a no-grad pre-pass that locates the policy and reference NaNs before the first backward (an extra forward per micro-batch), or a rescaling of the accumulated gradient by `Z / (Z - D)` after one more all-reduce right before the optimizer step, which means reaching into DeepSpeed's gradient buffers. Neither is done, since the cost buys back a one-step scalar of order `D / Z` on a fault path.

`nan_tokens` reports, per micro-batch, the number of tokens that were in the loss mask before sanitization and left it (policy, reference, or proximal NaN). It is logged as `train/nan_tokens`, averaged over the micro-batches of one `train_step` call and then over ranks. A shrunken step is therefore visible in the metrics rather than silent.

This accounting assumes that a dropped position carries no gradient. That holds for a NaN in the reference or proximal log-probabilities and for a `-inf` target logit in an otherwise finite row. A policy logits row that contains NaN or `+inf` (e.g., a bf16 overflow), or is entirely `-inf`, is different, whether or not its position is in the loss mask. Its entropy is NaN as well (`torch.compile` turns `torch.distributions` validation off for the whole process, so `Categorical` returns NaN instead of raising), and since NaN times a zero mask is still NaN, `compute_policy_loss` zeroes the entropies at masked positions with `torch.where`, as it does for the KL term. The loss value therefore stays finite, but the `log_softmax` backward of that row is still NaN, so the gradient becomes NaN instead of shrinking, and `nan_tokens` counts the position only if it was live.

## References

[1] J. Schulman. *Approximating KL Divergence.* Blog post, 2020. [http://joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)