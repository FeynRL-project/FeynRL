### CISPO (vs GRPO-style clipping)

CISPO differs from typical PPO/GRPO-style updates only in the **policy loss / clipping form**. GRPO-style clipping uses the “min of unclipped vs clipped objective”: with ratio ($r=\exp(\log p_\theta-\log q)$), it optimizes
$$
\mathcal{L}_{\text{GRPO}}(\theta)=-\mathbb{E}\Big[\min\big(rA,\ \mathrm{clip}(r,1-\epsilon_l,1+\epsilon_h)A\big)\Big].
$$
CISPO instead clips the ratio and uses the **clipped ratio as a stop-gradient coefficient** multiplying the log-prob term:
$$
\mathcal{L}_{\text{CISPO}}(\theta)= -\mathbb{E}\Big[\mathrm{sg}\big(\mathrm{clip}(r,1-\epsilon_l,1+\epsilon_h)\big)\ \log p_\theta(\cdot)\ A\Big].
$$
In this repo, CISPO is therefore “SGRPO with a different `compute_policy_loss(...)`”: the replay format, uniform replay sampling, masking, DeepSpeed micro-batching/accumulation, and optional entropy/KL terms are unchanged—only the clipping/weighting in the policy objective differs.  
