# Direct Preference Optimization (DPO)

This DPO variant computes the usual policy-vs-reference log-ratio, but instead of summing over tokens which can bias learning toward longer completions, it forms a length-normalized reward for each completion by averaging the masked token log-ratios over the number of supervised (unmasked) tokens. The loss is then the standard DPO logistic objective on the difference between the chosen and rejected rewards. In practice, the pseudocode "minibatch" corresponds to an effective batch assembled from micro-batches with gradient accumulation.

#### Implementation details

- **Data layout**: Chosen and rejected completions are interleaved in a single batch as `[chosen_0, rejected_0, chosen_1, rejected_1, ...]` via `torch.stack([chosen, rejected], dim=0)`. The input tensors are `[B, 2, T]` and reshaped to `[2B, T]` for the forward pass. Even rows (0::2) are chosen, odd rows (1::2) are rejected.

- **Reference model**: The reference model is initialized in `eval()` mode at construction and never updated. Its forward pass runs inside `torch.no_grad()`, and the ref logits are reduced to per-token logprobs `[2B, T-1]` immediately to free the full `[2B, T-1, vocab_size]` tensor from memory.

- **Log-probability computation**: Both policy and reference logprobs are computed via `CrossEntropyLoss(reduction="none")` in float32 (negated to get logprobs), avoiding bf16/fp16 quantization in the softmax.

- **Length normalization**: Token log-ratios are summed per sequence and divided by the number of valid (unmasked) tokens `max(L, 1)` for each completion independently. This prevents longer completions from having disproportionately larger reward magnitudes. See [SimPO (Meng et al., 2024)](https://arxiv.org/abs/2405.14734) and [DPOP (Pal et al., 2024)](https://arxiv.org/abs/2305.18290) for discussion of length normalization in preference optimization.

- **Tracked metrics**: Such as `loss` (DPO loss), `chosen_rewards` (mean length-normalized chosen reward), `rejected_rewards` (mean length-normalized rejected reward), `reward_accuracies` (fraction of examples where chosen reward > rejected reward), etc.


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

Unlike SFT (see [SFT README — Loss Normalization](../SFT/README.md#loss-normalization)), DPO does **not** need the global token-count fix for gradient accumulation. The reason is structural:

- **SFT** computes a per-token cross-entropy summed over variable-length sequences. When micro-batches have different numbers of valid tokens, a naive per-micro-batch mean produces "mean of means != global mean" — requiring a global token-count denominator computed across the entire gradient accumulation window and all GPUs.

- **DPO** computes a per-example loss: each example's reward is already length-normalized (sum of token log-ratios / number of valid tokens), and the final loss is `.mean()` over the batch dimension (number of preference pairs). Since every micro-batch contributes equally many *examples* (not tokens), DeepSpeed's default gradient accumulation averaging (divide accumulated gradients by `ga_steps`) produces the correct global mean over examples. The `normalize_loss` parameter is accepted for interface consistency but is unused.