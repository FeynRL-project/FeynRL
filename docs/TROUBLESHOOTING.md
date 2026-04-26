# Troubleshooting and Debugging Guide

This guide covers common issues encountered while running FeynRL, including multi-node scaling, memory management, and training stability.

## Recommended defaults (auto-detect)

These knobs can stay at their `null` defaults whenever Ray and NCCL auto-detection works: always for single-node, and typically for multi-node too on well-configured fabrics:
```yaml
run:
  ray_address: null
  nccl_socket_ifname: null
  nccl_ib_hca: null
  nccl_sync_port: null
```
Only set them explicitly if fabric init fails (see Multi-Node & Scaling Issues below).

## NCCL error or hang at `init_weight_nccl_group`

Typical symptom is an `[EarlyFailure]` within seconds of starting weight sync, with a `RuntimeError: NCCL error: internal error` raised from `PyNcclCommunicator.__init__` → `ncclCommInitRank`. Less commonly the job hangs silently or loops on `[Heartbeat] ... 0/N done`.

Root cause: an upstream vLLM issue triggered by combining `nccl_sync_backend: "nccl"` with `rollout.batch_invariant: True`. Either of these works around it:
```yaml
# Option 1 — keep NCCL weight sync, disable batch invariance
rollout:
  batch_invariant: False
run:
  nccl_sync_backend: "nccl"
# Option 2 — keep batch invariance, switch weight sync to gloo
run:
  nccl_sync_backend: "gloo"
rollout:
  batch_invariant: True
```

---

## Multi-Node & Scaling Issues

### NCCL InfiniBand connection timeout

**Symptom:** `ibv_modify_qp failed with 110 Connection timed out` or similar `ncclSystemError` during initialization.

**Cause:** Nodes with multiple InfiniBand HCAs may have some for GPU data traffic and others for management/storage. NCCL can auto-select a management HCA that cannot reach other ranks.

**Diagnosis and fix:**

1. Run `ibdev2netdev` and `ip addr show` to map each HCA to its network interface and subnet. Management links typically have small subnets (`/30`), while data-fabric HCAs have larger subnets (`/24`, `/25`).

2. **Auto-detect the data-fabric HCAs via GPU topology.** The HCAs you want are the ones with a `PIX` (PCIe-switch-local) link to at least one GPU in `nvidia-smi topo -m`. This one-liner prints them in the comma-separated form NCCL expects (the `-f1-9` assumes an 8-GPU box; widen the range if you have more GPUs):
   ```bash
   nvidia-smi topo -m | grep ^NIC | cut -f1-9 | grep PIX | cut -f1 | sed 's/NIC/mlx5_/;s/$/:1/' | paste -sd,
   # example output: mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
   ```
   Any HCA *not* in this list (e.g. `mlx5_bond_0`, or an HCA that only shows `SYS`/`NODE` connections) is a management/storage link. Use the output directly as the include list, or invert it to build an exclude list.

3. Match the device in the error message (e.g. `on dev mlx5_3:1`) to the mapping. Configure `nccl_ib_hca` in your run config — either include-only (drop the `^`) or exclude-only (prefix with `^`):
   ```yaml
   run:
     nccl_socket_ifname: "<interface_used_by_ray>"
     # include-only (preferred when the auto-detect list is reliable):
     nccl_ib_hca: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
     # or exclude-only (when you'd rather blacklist management HCAs):
     # nccl_ib_hca: "^<mgmt_hca_0>,<mgmt_hca_1>"
   ```

4. To quickly confirm the issue, disable IB and fall back to TCP: `export NCCL_IB_DISABLE=1`

---

### RL run hangs during rollout or training step
**Possible causes:**
- **GPU over-allocation**: `training_gpus + rollout_gpus` exceeds available cluster GPUs.
- **Ray actor crash**: A Ray actor crashed silently (OOM, CUDA error) and the remaining actors are stuck waiting.
- **NCCL timeout**: Network-level issue between nodes on the training side.

**How to diagnose:**
1. **Verify GPU budget**:
   ```bash
   python -c "import ray; ray.init(); print(ray.cluster_resources())"
   ```
   Confirm that the `GPU` count ≥ `training_gpus + rollout_gpus` in your config.
2. **Check Ray actor status**: Surface dead or failed actors using:
   ```bash
   ray status
   ```
3. **Debug NCCL**: If hangs occur during training (not rollout), add `NCCL_DEBUG=INFO` to the environment:
   ```bash
   NCCL_DEBUG=INFO python main_rl.py --config_file ./configs/rl_args.yaml --experiment_id debug_run
   ```
4. **Network Connectivity**: Ensure all nodes can communicate over the specified ports.

5. **Shared Filesystem**: For multi-node runs, ensure your `checkpoint_dir` is on a shared filesystem so all nodes can access saved models.

---

## Memory & OOM Issues

### vLLM Rollout OOM (Out of Memory)
vLLM is memory-intensive. If you encounter OOM:
1. **Reduce `gpu_memory_utilization`**: Lower this value in the `rollout` config (e.g., from 0.9 to 0.7) to leave more headroom for the KV cache.
2. **Increase `tensor_parallel_size`**: Distribute the model across more GPUs to reduce per-GPU memory usage.
3. **Decrease `rollout_batch_size_per_gpu`**: Smaller batches use less memory during generation.
4. **Check `max_seq_len`**: Ensure it's not unnecessarily large for your specific task.

---

## Weight Synchronization & Loading

### vLLM reload/update failures
**Possible causes:**
- **Missing Files**: Checkpoint directory is missing `config.json` or tokenizer files; vLLM cannot load a model without them.
- **Local vs Shared Paths**: On multi-node setups, the checkpoint path is on a local disk that rollout workers on other nodes cannot see.

**How to diagnose:**
1. **Verify Files**:
   ```bash
   ls <checkpoint_dir>/<experiment_id>/
   # expect: config.json, tokenizer.json, tokenizer_config.json, model*.safetensors
   ```
2. **Use Shared Storage**: For multi-node, use a **shared filesystem** for `checkpoint_dir`.
3. **Sync Check**: With `weight_sync_method: nccl` or `direct`, disk checkpoints are only written at the periodic save schedule (or final epoch); verify the sync logs show success between saves.

### Weight Synchronization Methods
The `weight_sync_method` config knob selects how training rank 0 transfers updated weights to the rollout workers. The config validator at [`configs/load.py:680-686`](../configs/load.py#L680-L686) couples this knob tightly to `overlap.enabled`: **overlap (async) mode requires `nccl`, and sync mode forbids `nccl`** (because the non-async vLLM engine has no NCCL weight-sync path).

- **`nccl`** (overlap-only): Training rank 0 broadcasts weights directly to all rollout engine TP workers over a dedicated NCCL process group. Zero serialization, lowest latency. Async mode pairs this with the NCCL watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `NCCL_TIMEOUT=1800s`, set by [`misc/nccl_env.py`](../misc/nccl_env.py)) so wedged collectives abort cleanly instead of hanging the GPU stream.
- **`direct`** (sync-only at runtime): Gathers weights to CPU via DeepSpeed, transfers them through Ray's shared-memory object store to rollout workers, and loads them into vLLM in-place. Async mode uses `direct` exactly once during `load_checkpoint_for_resume`, before the NCCL group is initialized.
- **`disk`** (sync-only): Saves weights to a checkpoint on disk, and rollout workers reload from the saved path. Slowest, but the most robust.

In sync mode, if `direct` fails mid-run FeynRL falls back to `disk` (save + rollout reload) so the rollout engines always receive updated weights. In async mode there is no runtime fallback; instead, fatal NCCL errors (communicator destruction, Ray timeout, dead actor) cause the job to exit immediately so you can fix the root cause.

---

## Training & Algorithmic Issues

### No reward signal or reward is always 0
**Possible causes:**
- **Default/Failure Reward**: The reward function returns 0 for all samples, e.g., the model never produces the expected format (like `#### <number>` for GSM8K), so `extract_solution` returns `None` and the reward is a zero tensor.
- **Truncation at generation**: Responses are truncated at `rollout.max_tokens` before the model can produce a complete answer, cutting off the terminal reward.
- **Truncation at replay buffer**: The replay buffer silently drops sequences where `prompt_len + response_len > data.max_seq_len`. If most sequences exceed this limit, the buffer may be nearly empty or contain only short (possibly degenerate) responses. Look for `[ReplayBuffer] ... sequences truncated` messages in the logs.
- **Zero-length responses**: Sequences with `response_len == 0` are silently dropped by the replay buffer.

**How to diagnose:**
1. **Inspect Samples**: Look at raw rollout samples to see what the model generates and check if they are reasonable.
2. **Check Length Limits**:
   - `max_tokens` = max generation length (response only)
   - `max_seq_len` = max total length (prompt + response)
   - `max_tokens` must be strictly less than `max_seq_len`
3. **Increase Room**: Try increasing `max_tokens` to give the model more room to produce a complete answer.
4. **Reward Logic**: Verify your reward function handles edge cases (empty responses, truncated responses, unexpected format) correctly.

---

### Loss is NaN or Inf
**Possible causes:**
- **Extreme importance ratios**: When the policy diverges significantly from the old policy (e.g., after too many gradient steps on the same replay data), `exp(logprobs - old_logprobs)` can overflow to `inf`. The clipping mechanism bounds the ratio in the loss, but the raw ratio may still cause issues in metrics or when padding is not properly masked.
- **KL divergence overflow**: If the policy and reference model diverge significantly, `exp(ref_logprobs - logprobs)` in the KL computation can overflow. FeynRL clamps the exponent to $\pm 10$ in [`algs/RL/common.py:128`](../algs/RL/common.py#L128) before `torch.exp`, so a silent saturation there (rather than an `Inf` in the loss) is the tell-tale sign of this regime.
- **Learning rate too high**: A high learning rate combined with large batch sizes can cause gradient explosions.
- **Mixed-precision issues**: bf16/fp16 training can cause underflow/overflow in log-probability computations. The code promotes certain operations to float32 for stability, but custom reward functions or model architectures may introduce their own precision issues.

**How to diagnose:**
1. **Monitor `approx_kl`**: A rapidly increasing `approx_kl` metric indicates the policy is changing too fast; sustained large values are the main signal that the policy and reference model are diverging.
2. **Reduce learning rate** or increase `clip_low`/`clip_high` to constrain updates.
3. **Check `gradient_clipping`**: Ensure `clip_grad_norm` is set (e.g., 1.0) to prevent gradient explosions.

---

### High `clipfrac` or policy collapse
**Possible causes:**
- **Stale replay data**: The replay buffer contains old-policy rollouts that are too far from the current policy, causing most importance ratios to be clipped (PPO/GRPO/CISPO) or pushing ESS toward 0 and zeroing out the gradient (P3O).
- **Learning rate too high**: Large policy updates cause the ratio `π/π_old` to frequently fall outside the monitoring range `[1 - clip_low, 1 + clip_high]`.
- **Too many `train_steps_per_epoch`**: Multiple passes over the same replay buffer compound policy shift.

**How to diagnose:**
1. **Watch `clipfrac`**: Note that for PPO/GRPO/CISPO this is the fraction of tokens where the symmetric clip is actually active, while for P3O it is a *monitoring-only* metric (P3O's actual clip is one-sided, `[0, ESS]`, derived from the data rather than from `clip_low/clip_high`). Persistently high values still indicate the policy is updating too aggressively relative to the replay data.
2. **Watch `approx_kl`**: This measures the KL divergence between the current policy and the old policy used to collect the replay data. Large values suggest drift.
3. **For P3O, watch `ess_factor` and `kl_behavioral`**: As ESS drops, the policy-gradient term is suppressed and the adaptive KL term `(1 − ESS) · KL(π ‖ π_old)` takes over to pull the policy back toward the behavior distribution. A persistently low ESS combined with a large `kl_behavioral` (the pre-weighting behavioral KL, logged under that name in [`algs/P3O/p3o.py:238`](../algs/P3O/p3o.py#L238)) signals that data is too stale (reduce `overlap.max_lag`) or that updates are too aggressive (reduce `lr`).
4. **Reduce `train_steps_per_epoch`** or **reduce `lr`** to slow down policy updates.
5. **Enable KL-to-reference penalty** (`kl_coeff > 0` with a reference model) to regularize the policy against long-term drift from the base distribution.

---

### Replay buffer dropping too many samples
The replay buffer logs `[ReplayBuffer] X/Y sequences truncated` when sequences exceed `max_seq_len`. It also silently drops sequences with `response_len == 0`.

**How to diagnose:**
1. **Check logs** for truncation messages. If a large fraction of samples is being dropped, either increase `data.max_seq_len` or decrease `rollout.max_tokens`.
2. **Length constraint**: `max_tokens` (response only) must be strictly less than `max_seq_len` (prompt + response). The config loader enforces this.
3. **Prompt length**: If your prompts are long, you may need a larger `max_seq_len` to accommodate both the prompt and a meaningful response.

---

### PPO: `rewards or values contain NaN on valid positions`
This error is raised by `compute_advantages` (GAE computation) in PPO.

**Possible causes:**
- **Value model divergence**: The value model produced NaN predictions, often due to learning rate being too high or numerical instability.
- **Reward function returning NaN**: The reward function produced non-finite values.

**How to diagnose:**
1. **Check `v_loss`**: If the value loss is growing rapidly, reduce `lr` (value model inherits `lr` if `value_lr` is null).
2. **Verify reward function**: Ensure your reward function always returns finite values.

### PPO: `mask has non-contiguous valid regions (holes)`
This error is raised by `compute_advantages` when the mask has gaps like `[1,1,0,1,1]`.

**Possible causes:**
- **Data corruption**: The rollout produced inconsistent mask/done/reward tensors.
- **Custom reward function**: A per-token reward function that inadvertently sets some mid-sequence mask values to 0.

**How to diagnose:**
1. **Check rollout data**: Inspect the raw rollout output to verify mask, done, and reward tensors are consistent.

---

### DPO: `reward_accuracies` stuck near 0.5
**Possible causes:**
- **Beta too low**: With a small `cl_beta`, the DPO loss signal is weak and the model cannot easily separate chosen from rejected.
- **Chosen/rejected too similar**: If the chosen and rejected completions are nearly identical, the policy-vs-reference log-ratio difference is negligible.
- **Reference model already strong**: If `ref_model` is the same as the initial policy and the preference pairs are subtle, the length-normalized rewards for chosen and rejected will be close.

**How to diagnose:**
1. **Monitor `chosen_rewards` and `rejected_rewards`**: If both are close to zero and barely separating, increase `cl_beta`.
2. **Check data quality**: Verify that chosen responses are meaningfully better than rejected ones in your training data.
