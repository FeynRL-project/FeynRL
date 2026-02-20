# Direct Preference Optimization (DPO)

This DPO variant computes the usual policy-vs-reference log-ratio, but instead of summing over tokens which can bias learning toward longer completions, it forms a length-normalized reward for each completion by averaging the masked token log-ratios over the number of supervised (unmasked) tokens. The loss is then the standard DPO logistic objective on the difference between the chosen and rejected rewards. In practice with DeepSpeed, the pseudocode “minibatch” corresponds to an effective batch assembled from micro-batches with gradient accumulation.

![DPO algorithm](./alg_dpo.png)
