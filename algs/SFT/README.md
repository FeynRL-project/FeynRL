# Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning trains a language model on a labeled dataset of input / output pairs via standard cross-entropy. For each example, the model predicts each output token given the prompt and all preceding output tokens (teacher forcing), and the loss is the masked token-level negative log-likelihood over the supervised tokens, typically the response, with the prompt and padding masked out.

This implementation runs SFT with DeepSpeed and gradient accumulation, and handles variable-length sequences correctly via a global-token-count loss normalization. See [Loss Normalization](#loss-normalization) below for the derivation.

**Input:** initial parameters $\theta_0$, dataset $\mathcal{D}$, batch size $B$, steps $T$

1. For $t = 1, \dots, T$:
   1. Sample a minibatch $\{(x_i, y_i, m_i)\}_{i=1}^B \sim \mathcal{D}$

      $m_i=(m_{i,1},\dots,m_{i,|y_i|})$, where $m_{i,j}\in\{0,1\}$ masks prompt / pad tokens.

   2. Compute masked token-level negative log-likelihood (NLL):

      $\mathcal{L}(\theta_t) = \frac{1}{\sum_{i=1}^{B}\sum_{j=1}^{|y_i|} m_{i,j}}
      \sum_{i=1}^{B}\sum_{j=1}^{|y_i|}
      m_{i,j}\Big(-\log p_{\theta_t}(y_i^{j}\mid x_i, y_i^{<j})\Big)$

   3. One step parameter update (e.g., Adam/AdamW):

      $\theta_{t+1} \leftarrow \mathrm{Update}\left(\theta_t,\nabla_{\theta_t}\mathcal{L}(\theta_t)\right)$

**Return:** $\theta_T$

In the actual training code a "minibatch" in the pseudocode above corresponds to an effective batch assembled from micro-batches with gradient accumulation (and, depending on configuration, ZeRO partitioning). The pseudocode is intentionally simplified; the underlying implementation performs the same objective and update, just executed across micro-batches and distributed workers before producing a single logical parameter update.

---

## Loss Normalization

### The Problem

When using gradient accumulation (GA) with variable-length sequences, a naive per-micro-batch mean produces incorrect gradients. This is because **mean of means $\neq$ global mean** when micro-batches have different numbers of valid (non-padding) tokens.

For example, with `ga_steps=2` and two micro-batches having 10 and 100 valid tokens respectively:
- **Naive mean of means**: micro-batch 1 computes `sum_1 / 10`, micro-batch 2 computes `sum_2 / 100`, then the accumulated gradient is `(sum_1/10 + sum_2/100) / 2`. Each token in the 10-token batch has per-token weight `1/20` while each token in the 100-token batch has per-token weight `1/200`, a 10× imbalance.
- **True global mean**: the gradient is `(sum_1 + sum_2) / 110`. Every token contributes equally with per-token weight `1/110`.

This issue is also discussed by Unsloth [[1]](#references) and in Tulu 3 [[2]](#references).

### The Fix

We pre-compute the total number of valid tokens across the **entire GA window** and **all GPUs** via all-reduce, then use that as the denominator. DeepSpeed internally divides gradients by `ga_steps` (gradient-accumulation averaging) and by `W` (data-parallel all-reduce averaging), so we multiply by `dp_scale = ga_steps * W` to cancel those divisions.

#### Formula

For micro-batch $j$ on GPU $g$, with $W$ total GPUs and $K$ GA steps:

$$\text{loss}_{g,j} = \text{loss\\_sum}_{g,j} \times \frac{K \cdot W}{N_{\text{global}}}$$

where:
- $\text{loss\\_sum}_{g,j} = \sum_t m_t \cdot \ell_t$ is the raw sum of masked per-token losses for this micro-batch
- $N_{\text{global}} = \sum_{g=1}^{W} \sum_{j=1}^{K} n_{g,j}$ is the all-reduced total valid tokens across all GPUs and all micro-batches in the GA window
- $n_{g,j}$ is the number of valid tokens in micro-batch $j$ on GPU $g$

#### What DeepSpeed does internally

1. **Backward**: computes $\nabla(\text{loss}_{g,j})$ for each micro-batch
2. **GA averaging**: divides accumulated gradient by $K$
3. **DDP averaging**: all-reduces gradients and divides by $W$

#### Net effect after DeepSpeed's internal scaling

$$\nabla_{\text{final}} = \frac{1}{W} \sum_{g=1}^{W} \left[ \frac{1}{K} \sum_{j=1}^{K} \nabla\left(\text{loss\\_sum}_{g,j} \times \frac{K \cdot W}{N_{\text{global}}}\right) \right]$$

$$= \frac{1}{N_{\text{global}}} \sum_{g,j} \nabla(\text{loss\\_sum}_{g,j})$$

$$= \nabla\left(\frac{\sum_{\text{all tokens}} m_t \cdot \ell_t}{N_{\text{global}}}\right)$$

This is the **true global per-token mean** over the entire effective batch across all GPUs.

#### `normalize_loss=False`

When `normalize_loss=False`, the loss is scaled by `dp_scale` only (no division by $N_{\text{global}}$):

$$\text{loss}_{g,j} = \text{loss\\_sum}_{g,j} \times K \cdot W$$

After DeepSpeed's internal divisions, the effective gradient is the **true global sum** of per-token losses. This mode makes the effective learning rate scale with the number of valid tokens.

#### References

[1] Unsloth AI. *Bugs in LLM training — gradient accumulation fix.* Blog post, 2024. [https://unsloth.ai/blog/gradient](https://unsloth.ai/blog/gradient)

[2] N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi. *Tulu 3: Pushing Frontiers in Open Language Model Post-Training.* arXiv:2411.15124, 2024. [https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)
