# Supervised Fine-Tuning (SFT)

The algorithm box summarizes what happens inside `train_step` during Supervised Fine-Tuning (SFT) with teacher forcing: for each training iteration we sample a minibatch, compute the (masked) token-level negative log-likelihood over the supervised tokens (typically the response, excluding prompt/padding), and apply one optimizer update (e.g., AdamW). In the actual training code, we run SFT with DeepSpeed, so a minibatch in the pseudocode corresponds to an effective batch formed by splitting data into micro-batches and using gradient accumulation (and, depending on the configuration, ZeRO partitioning). We keep the algorithm box intentionally simplified for readability as the underlying implementation performs the same objective and update, just executed across micro-batches and distributed workers before producing a single logical parameter update.

#### Implementation details

- **Loss normalization** (`normalize_loss` config flag):
  - When `normalize_loss=True`, the loss is `sum(masked_per_token_loss) / total_possible_tokens` where `total_possible_tokens = B * (T-1)` — the total sequence length including padding. This is a constant across GPUs, which avoids the [gradient accumulation bug](https://unsloth.ai/blog/gradient) that arises when sequence lengths vary across micro-batches (normalizing by `loss_mask.sum()` would give different effective learning rates to different micro-batches).
  - When `normalize_loss=False`, the loss is the raw sum of masked per-token losses.

**Input:** initial parameters $\theta_0$, dataset $\mathcal{D}$, batch size $B$, steps $T$

1. For $t = 1, \dots, T$:
   1. Sample a minibatch $\{(x_i, y_i, m_i)\}_{i=1}^B \sim \mathcal{D}$

      $m_i=(m_{i,1},\dots,m_{i,|y_i|})$, where $m_{i,j}\in\{0,1\}$ masks prompt/pad tokens.

   2. Compute masked token-level negative log-likelihood (NLL):

      $\mathcal{L}(\theta_t) = \frac{1}{\sum_{i=1}^{B}\sum_{j=1}^{|y_i|} m_{i,j}}
      \sum_{i=1}^{B}\sum_{j=1}^{|y_i|}
      m_{i,j}\Big(-\log p_{\theta_t}(y_i^{j}\mid x_i, y_i^{<j})\Big)$

   3. One step parameter update (e.g., Adam/AdamW):

      $\theta_{t+1} \leftarrow \mathrm{Update}\left(\theta_t,\nabla_{\theta_t}\mathcal{L}(\theta_t)\right)$

**Return:** $\theta_T$