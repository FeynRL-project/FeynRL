# Supervised Fine-Tuning (SFT)

The algorithm box summarizes what happens inside 'train\_step' during Supervised Fine-Tuning (SFT) with teacher forcing: for each training iteration we sample a minibatch, compute the (masked) token-level negative log-likelihood over the supervised tokens (typically the response, excluding prompt/padding), and apply one optimizer update (e.g., AdamW). In the actual training code, we run SFT with  DeepSpeed, so a minibatch in the pseudocode corresponds to an effective batch formed by splitting data into micro-batches and using gradient accumulation (and, depending on the configuration, ZeRO partitioning). We keep the algorithm box intentionally simplified for readability as the underlying implementation performs the same objective and update, just executed across micro-batches and distributed workers before producing a single logical parameter update.


## Algorithm

The SFT algorithm minimizes the negative log-likelihood of the target sequences given the prompts.

```latex
\begin{algorithm}[H]
\caption{Supervised Fine-Tuning (Teacher Forcing)}
\label{alg:sft}
\begin{algorithmic}[1]
\State \textbf{Input:} initial parameters $\theta_0$, dataset $\mathcal{D}$, batch size $B$, steps $T$
\For{$t = 1, \dots, T$}
    \State Sample a minibatch $\{(x_i, y_i, m_i)\}_{i=1}^{B} \sim \mathcal{D}$
    \Comment{$m_i=(m_{i,1},\dots,m_{i,|y_i|}),\; m_{i,j}\in\{0,1\}$ masks prompt/pad tokens}
    \State Compute masked token-level negative log-likelihood (NLL):
    \[
        \mathcal{L}(\theta_t)
        =
        \frac{1}{\sum_{i=1}^{B}\sum_{j=1}^{|y_i|} m_{i,j}}
        \sum_{i=1}^{B}\sum_{j=1}^{|y_i|}
        m_{i,j}\Big(-\log p_{\theta_t}(y_i^{j}\mid x_i, y_i^{<j})\Big)
    \]
    \State One step parameter update (e.g., Adam/AdamW):
    \[
        \theta_{t+1} \leftarrow \mathrm{Update}\!\left(\theta_t,\nabla_{\theta_t}\mathcal{L}(\theta_t;\mathcal{B}_t)\right)
    \]
\EndFor
\State \textbf{Return:} $\theta_T$
\end{algorithmic}
\end{algorithm} 
```

