# Direct Preference Optimization (DPO)

DPO [[1]](#references) optimizes a language model against a preference dataset without training a reward model or running policy-gradient rollouts. It recasts the RLHF reward objective as a classification loss on the log-ratio between the policy and a frozen reference model, evaluated on pairs of chosen / rejected completions.

This implementation follows the standard DPO formulation but applies **per-completion length normalization** to the log-ratio: instead of summing token log-ratios per completion (which biases learning toward longer sequences), it averages the masked token log-ratios over the number of supervised (unmasked) tokens in each completion. The loss is the standard DPO logistic objective on the difference between the chosen and rejected length-normalized rewards. The length-bias motivation follows SimPO [[2]](#references); note that SimPO itself uses a **reference-free** reward (average policy log-probability with an added margin $\gamma$), whereas this implementation keeps DPO's reference-model log-ratio structure and only adds length normalization on top. In practice, the pseudocode "minibatch" below corresponds to an effective batch assembled from micro-batches with gradient accumulation.

#### Implementation details

- **Data layout**: Chosen and rejected completions are interleaved in a single batch as `[chosen_0, rejected_0, chosen_1, rejected_1, ...]` via `torch.stack([chosen, rejected], dim=0)`. The input tensors are `[B, 2, T]` and reshaped to `[2B, T]` for the forward pass. Even rows (0::2) are chosen, odd rows (1::2) are rejected.

- **Reference model**: The reference model is initialized in `eval()` mode at construction and never updated. Its forward pass runs inside `torch.no_grad()` and executes **before** the policy forward so that the full `[2B, T-1, vocab_size]` ref logits tensor can be reduced to `[2B, T-1]` logprobs and freed before the policy forward allocates its own logits. During reduction, a transient fp32 copy of the ref logits is created for the `CrossEntropyLoss` computation and freed immediately after.

- **Log-probability computation**: Both policy and reference logprobs are reduced from logits to per-token logprobs `[2B, T-1]` inside `forward()` via `CrossEntropyLoss(reduction="none")` in float32 (negated to get logprobs), avoiding bf16/fp16 quantization in the softmax. By reducing inside `forward()`, the full vocab-dimension tensors are freed before `compute_loss()` runs, so only `[2B, T-1]` logprob tensors flow through the loss and backward pass.

- **Length normalization**: Token log-ratios are summed per sequence and divided by `max(L, 1)` where `L` is the number of valid (unmasked) tokens for that completion. This prevents longer completions from having disproportionately larger reward magnitudes. See SimPO [[2]](#references) for related discussion of length normalization in preference optimization.

- **Symmetric NaN/Inf guard**: `train_step` all-reduces a non-finiteness flag with `ReduceOp.MAX` across ranks and raises on every rank if any rank produced a non-finite loss; this avoids a partial-rank crash that would deadlock the next collective.

- **Tracked metrics**: `loss` (DPO loss), `chosen_rewards` (mean length-normalized chosen reward), `rejected_rewards` (mean length-normalized rejected reward), `reward_accuracies` (fraction of examples where chosen reward > rejected reward).


**Input:** initial policy parameters $\theta_0$, fixed reference parameters $\theta_{\mathrm{ref}}$, preference dataset $\mathcal{D}$, batch size $B$, $\beta>0$, steps $T$.

For $t = 1, \dots, T$:

1. Sample a minibatch $\{(x_i, y_i^{+}, y_i^{-}, m_i^{+}, m_i^{-})\}_{i=1}^{B} \sim \mathcal{D}$
   - $y^+$ preferred, $y^-$ rejected
   - masks $m^{+},m^{-}\in\{0,1\}^{|y|}$ (1 for valid tokens, 0 for invalid tokens like prompt/pad)

2. Compute masked token log-ratios for chosen/rejected:

$$
\ell^{+}_{i,j} = m^{+}_{i,j}\Big(\log p_{\theta_t}(y_{i}^{+,j}\mid x_i,y_i^{+,<j}) - \log p_{\theta_{\mathrm{ref}}}(y_{i}^{+,j}\mid x_i,y_i^{+,<j})\Big)
$$

$$
\ell^{-}_{i,j} = m^{-}_{i,j}\Big(\log p_{\theta_t}(y_{i}^{-,j}\mid x_i,y_i^{-,<j}) - \log p_{\theta_{\mathrm{ref}}}(y_{i}^{-,j}\mid x_i,y_i^{-,<j})\Big)
$$

3. Length-normalized rewards (average log-ratio per unmasked token):

$$
L_i^{+} = \sum_{j} m^{+}_{i,j}, \quad L_i^{-} = \sum_{j} m^{-}_{i,j}
$$

$$
r_i^{+} = \frac{\sum_{j}\ell^{+}_{i,j}}{\max(L_i^{+},1)},\quad
r_i^{-} = \frac{\sum_{j}\ell^{-}_{i,j}}{\max(L_i^{-},1)}
$$

4. DPO loss:

$$
\mathcal{L}_{\mathrm{DPO}}(\theta_t)
= \frac{1}{B}\sum_{i=1}^{B} -\log \sigma\Big(\beta\,(r_i^{+}-r_i^{-})\Big)
$$

5. One step parameter update (e.g., Adam/AdamW):

$$
\theta_{t+1} \leftarrow \mathrm{Update}\left(\theta_t,\nabla_{\theta_t}\mathcal{L}_{\mathrm{DPO}}(\theta_t)\right)
$$

**Return:** $\theta_T$

---

## Loss Normalization and Gradient Accumulation

Unlike SFT (see [SFT README: Loss Normalization](../SFT/README.md#loss-normalization)), DPO does **not** need the global token-count fix for gradient accumulation. The reason is structural:

- **SFT** computes a per-token cross-entropy summed over variable-length sequences. When micro-batches have different numbers of valid tokens, a naive per-micro-batch mean produces "mean of means ≠ global mean", requiring a global token-count denominator computed across the entire gradient-accumulation window and all GPUs.

- **DPO** computes a per-example loss: each example's reward is already length-normalized (sum of token log-ratios divided by the number of valid tokens), and the final loss is `.mean()` over the batch dimension (number of preference pairs). Since every micro-batch contributes equally many *examples* (not tokens), DeepSpeed's default gradient-accumulation averaging (divide accumulated gradients by `ga_steps`) produces the correct global mean over examples. The `normalize_loss` constructor argument is accepted for interface consistency with SFT but is unused in DPO.

  Concretely, the effective gradient after DeepSpeed's internal averaging is:

$$
\nabla_{\mathrm{eff}} = \frac{1}{\texttt{ga-steps} \times \texttt{world-size}} \sum_{\text{rank}} \sum_{s=1}^{\texttt{ga-steps}} \nabla \mathcal{L}_s^{\text{rank}}
$$

  Since each $\mathcal{L}_s$ is already a mean over $B$ preference pairs, this gives the true mean over the full effective batch of $B \times \texttt{ga-steps} \times \texttt{world-size}$ pairs.

#### References

[1] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290, 2023. [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

[2] Y. Meng, M. Xia, and D. Chen. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* arXiv:2405.14734, 2024. [https://arxiv.org/abs/2405.14734](https://arxiv.org/abs/2405.14734)