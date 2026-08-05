### DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)

DAPO [[1]](#references) is a GRPO-style policy optimization algorithm: it samples multiple completions per prompt, derives advantages from group-relative z-scores, and needs no value model (see [GRPO](../GRPO/README.md)). On top of that base recipe it applies the paper's **four key techniques** — clip-higher, dynamic sampling, a token-level policy-gradient loss, and soft overlong punishment — plus the removal of the KL penalty, which the paper states as part of its setup. All target long chain-of-thought RL.

The objective:

$$
\mathcal{L}(\theta) = -\frac{1}{\sum_i |o_i|}\sum_i \sum_t \min\big(r_{i,t}\,\hat{A}_{i,t},\ \mathrm{clip}(r_{i,t},\,1-\epsilon_\ell,\,1+\epsilon_h)\,\hat{A}_{i,t}\big),\qquad
\hat{A}_{i,t} = \frac{R_i - \mathrm{mean}(R)}{\mathrm{std}(R)}
$$

subject to the dynamic-sampling constraint that groups where every completion received the same reward are excluded from training.

#### DAPO vs. GRPO at a glance

The trainer ([`dapo.py`](dapo.py)) shares GRPO's clipped-surrogate code; every difference below is expressed through config and the rollout path. This table is the fastest way to understand what DAPO adds:

| # | Component | GRPO | DAPO | Where in this codebase |
|---|-----------|------|------|------------------------|
| 1 | **Clip range** | Symmetric: $\epsilon_\ell = \epsilon_h$ (e.g. 0.2 / 0.2) | **Clip-higher**: $\epsilon_h > \epsilon_\ell$ (paper: 0.2 / **0.28**) raises the ceiling for low-probability "exploration" tokens without loosening the downside clip | `train.clip_low`, `train.clip_high` (already decoupled for all algorithms) |
| 2 | **Loss normalization** | Sample-level in the original paper: mean over tokens per sequence, then mean over sequences (long and short answers weigh equally) | **Token-level**: one global sum divided by the total token count, so every token weighs equally and long-CoT tokens are not down-weighted | `train.normalize_loss: True` — this codebase's global token normalization is already the DAPO form; it is **required** for `alg_name: dapo` (enforced in [`configs/load.py`](../../configs/load.py)) |
| 3 | **Degenerate groups** | Kept: a group where all completions score the same has zero advantage everywhere; its tokens still inflate the loss denominator and dilute the gradient | **Dynamic sampling**: such groups are removed from training | `rollout.dynamic_sampling` (auto-on for dapo on the sync engine; stays off in async — see [why](#why-async-mode-does-not-need-dynamic-sampling)); drop happens in the replay buffer, [`rollouts/replay_buffer.py`](../../rollouts/replay_buffer.py) |
| 4 | **Truncated / overlong responses** | Full (usually 0) reward from the verifier; truncation noise leaks into the advantage | **Soft overlong punishment**: linear length penalty over the last $L_{\mathrm{buffer}}$ tokens of the generation budget, reaching $-c$ at `max_tokens` | `rollout.overlong_buffer_tokens`, `rollout.overlong_penalty_factor`, applied in [`rollouts/base.py`](../../rollouts/base.py) `normalize_rewards` |
| 5 | **KL penalty** | Optional KL vs. a frozen reference policy ($\beta_{\mathrm{kl}}$) | **Removed** — long-CoT training intentionally drifts far from the init policy, so anchoring to it is counterproductive; also saves the ref-model forward pass | `train.kl_coeff: 0.0` (the default; no reference model is loaded) |

Everything else — group z-score advantages baked into `zscore` at rollout time, uniform (non-group-structured) training batches, GA/DeepSpeed handling, micro-batch shuffling — is identical to GRPO; see the [GRPO README](../GRPO/README.md) for those mechanics. One hard requirement beyond GRPO, enforced at config load: **`reward.broadcast: True` is mandatory** — the token-level loss only means something if every response token carries the group advantage; with `broadcast: False` only the terminal prediction position would receive gradient. DAPO runs on both engines; in overlap (async) mode it runs **without dynamic sampling** ([why](#why-async-mode-does-not-need-dynamic-sampling)) and, like GRPO, trains with the decoupled loss.

#### Component details

1. **Clip-higher** — in a symmetric clip, a token with probability 0.01 can at most grow to 0.012 per update while a 0.9 token can grow to 1.08: the upper clip suppresses exactly the low-probability tokens that drive exploration, leading to entropy collapse. Raising only $\epsilon_h$ gives those tokens more upward room while keeping the downside clip tight.

2. **Token-level policy-gradient loss** — the loss is a raw sum normalized by the **global** valid-token count across all micro-batches and ranks, which is exactly the paper's $1/\sum_i|o_i|$ normalization extended correctly to data parallelism. See [RL Common README: Global Token Normalization](../RL/README.md#global-token-normalization-for-rl).

3. **Dynamic sampling** — as training progresses, more prompts become "all correct" (or stay "all wrong"), so a growing fraction of each batch carries zero gradient and the effective batch shrinks noisily. During reward normalization, a group whose **shaped** rewards (verifier reward + overlong penalty) are all identical — i.e. zero group std, hence zero advantage on every token — is flagged degenerate, and the replay buffer drops its samples before they are stored, so the token-normalized loss only sees informative samples. Sync engine only — with `overlap.enabled: True` it stays off (`dynamic_sampling: null` auto-resolves to off, and an explicit `true` is rejected; [why async does not need it](#why-async-mode-does-not-need-dynamic-sampling)); requires `rollout.n_samples > 1` (validated at config load).

4. **Soft overlong punishment** — with $L_{\max}$ = `rollout.max_tokens`, $L_{\mathrm{buffer}}$ = `overlong_buffer_tokens`, $c$ = `overlong_penalty_factor`:

$$
R_{\mathrm{length}}(y) =
\begin{cases}
0, & |y| \le L_{\max} - L_{\mathrm{buffer}} \\
-\dfrac{|y| - (L_{\max} - L_{\mathrm{buffer}})}{L_{\mathrm{buffer}}}\cdot c, & \text{otherwise (capped at } -c\text{)}
\end{cases}
$$

   Responses truncated at `max_tokens` receive the full $-c$. The paper uses $L_{\mathrm{buffer}} = L_{\max}/5$ and $c = 1$. The penalty is added **before** group normalization so it shapes the z-scores; pass@k metrics keep reflecting raw verifier correctness (see [Tracked metrics](#tracked-metrics) for exactly which metrics are shaped vs. raw).

**No entropy bonus** either — leave `train.entropy_coeff: 0.0` (the default); clip-higher is DAPO's entropy-collapse remedy.

#### Example config

```yaml
train:
  alg_name: "dapo"
  kl_coeff: 0.0        # DAPO removes the KL term (default)
  clip_low: 0.2
  clip_high: 0.28      # clip-higher
  normalize_loss: True # token-level loss (required for dapo)

reward:
  broadcast: True      # required for dapo: every response token gets the group advantage

rollout:
  n_samples: 8
  dynamic_sampling: null      # null = auto-enabled for dapo (sync engine; stays off in async)
  overlong_buffer_tokens: 102 # ~ max_tokens / 5
  overlong_penalty_factor: 1.0
  max_tokens: 512
```

Switching an existing GRPO run to DAPO is: `alg_name: "dapo"`, `reward.broadcast: True`, raise `clip_high` to 0.28, and optionally set `overlong_buffer_tokens` (if the GRPO run used a KL term, also set `kl_coeff: 0.0` and drop `model.ref_model`).

#### Deviations from the paper

- **Filtering instead of resample-to-fill.** The paper over-samples and keeps generating until the batch is full of non-degenerate groups, so the trained batch size stays constant. Here, degenerate groups are dropped at the replay buffer and the epoch simply trains on fewer samples; the gradient is identical (the loss is normalized by the surviving token count), only the per-epoch sample count varies. The drop rate is logged as `rollout/dynamic_sampling_drop_rate`.
- **Zero-advantage test instead of accuracy test.** The paper filters groups whose accuracy is 0 or 1. This codebase flags a group when all of its **shaped** rewards (verifier reward + overlong penalty) are identical, i.e. the group std is exactly zero. This is equivalent for binary rewards, and deliberately keeps all-correct groups whose overlong penalties differ — those still carry the length-punishment gradient.
- **Z-score details.** The group std uses Bessel's correction ($n-1$, appropriate for small `n_samples`) and a `1e-8` epsilon in the denominator; the paper's formula has neither.
- **All-degenerate fallback.** If *every* group in an epoch is degenerate (e.g. the policy has saturated the dataset), the dropped samples are restored so the training step doesn't crash on an empty replay buffer; the resulting gradient is ~0 and a warning is logged.
- **No dynamic sampling in overlap (async) mode.** DAPO runs on the async engine as "DAPO minus dynamic sampling": clip-higher, the token-level loss, the overlong penalty, and KL removal all apply, and training uses the decoupled loss with behavioral importance weighting, as for every algorithm under `overlap.enabled: True`. Zero-advantage groups are kept — see the next section for why that is a sound trade rather than a gap. Training batches are not group-structured, exactly as for GRPO — see the [GRPO README](../GRPO/README.md).

#### Why async mode does not need dynamic sampling

Dynamic sampling is implemented as the replay buffer's `drop_zero_advantage_groups` flag ([`rollouts/replay_buffer.py`](../../rollouts/replay_buffer.py)). In overlap (async) mode the flag stays off, deliberately:

1. **Dropping only rescales the gradient; it never changes its direction.** A degenerate group has advantage exactly 0 on every token, so its tokens contribute exactly 0 to the loss *numerator* whether or not they are stored; keeping them only enlarges the token *denominator* $\sum_i |o_i|$. The resulting gradient points in the same direction, uniformly scaled down by the informative-token fraction — and AdamW's per-parameter normalization absorbs much of a uniform rescale. Dynamic sampling is therefore a signal-to-noise / effective-batch-size improvement, not a correctness requirement; nothing in the DAPO objective breaks when the groups are kept, which is exactly how GRPO and CISPO treat them in every mode.

2. **The dilution the paper targets is a per-batch effect that the async buffer already smooths.** In sync mode each epoch trains on exactly one fresh, self-contained rollout batch, so a round with many degenerate groups directly shrinks that epoch's effective batch — the paper's motivation for resample-to-fill. The async replay buffer is persistent and version-evicted: it holds up to `overlap.max_lag` policy versions of data (several rounds' worth), so the informative-token fraction of any training pass is averaged over a much larger pool, and per-round swings in the degenerate rate move the denominator far less.

3. **The all-degenerate safety fallback is inherently sync-shaped.** Dropping needs an escape hatch for the epoch where *every* group is degenerate (a saturated dataset): in sync mode, `collect_rollouts` restores the spilled samples into the then-empty buffer before the empty-buffer check, and the spill is discarded as soon as real samples land. Both triggers key off "buffer is empty" — which is essentially never true for the async buffer, since it persists across rounds. The async driver's protection is instead a hard buffer-underfill check at round start (`train_batch_size × num_engines`); enabling the drop there would turn a saturated dataset from a graceful ~0-gradient epoch into a mid-training crash, unless the restore logic were redesigned around FIFO eviction, version eviction, and carryover-shard accounting. Note the async pipeline itself is drop-tolerant (round completion counts *shards*, not items), so this is purely about the fallback: real complexity for a benefit that points 1–2 already discount.

Config behavior matches: `rollout.dynamic_sampling: null` auto-resolves to **on** for dapo on the sync engine and **off** under `overlap.enabled: True`; an explicit `dynamic_sampling: true` with overlap is rejected at config load.

#### Tracked metrics

In addition to the GRPO trainer metrics (`clipfrac`, `approx_kl`, `pi_loss`, `loss_total`, ...), dynamic sampling reports `rollout/dynamic_sampling_kept`, `rollout/dynamic_sampling_dropped`, `rollout/dynamic_sampling_restored` (all-degenerate fallback, normally 0), and `rollout/dynamic_sampling_drop_rate`.

Metric semantics with overlong punishment enabled: `rollout/avg_reward` (and the other reward aggregates computed from stored sample rewards) reflect the **shaped training reward** including the penalty, while pass@k / `pass_rate` / `reward_std_per_prompt` always reflect **raw verifier correctness** (the applied penalty is stamped on each sample as `overlong_penalty` so consumers can undo it).

#### References

[1] Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, T. Fan, G. Liu, L. Liu, X. Liu, H. Lin, Z. Lin, B. Ma, G. Sheng, Y. Tong, C. Zhang, M. Zhang, W. Zhang, H. Zhu, J. Zhu, J. Chen, J. Chen, C. Wang, H. Yu, W. Dai, Y. Song, X. Wei, H. Zhou, J. Liu, W.-Y. Ma, Y.-Q. Zhang, L. Yan, M. Qiao, Y. Wu, and M. Wang. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* arXiv:2503.14476, 2025. [https://arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)

[2] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300, 2024. [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
