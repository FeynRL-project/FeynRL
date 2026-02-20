# Direct Preference Optimization (DPO)

This DPO variant computes the usual policy-vs-reference log-ratio, but instead of summing over tokens which can bias learning toward longer completions, it forms a length-normalized reward for each completion by averaging the masked token log-ratios over the number of supervised (unmasked) tokens. The loss is then the standard DPO logistic objective on the difference between the chosen and rejected rewards. In practice with DeepSpeed, the pseudocode “minibatch” corresponds to an effective batch assembled from micro-batches with gradient accumulation.


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