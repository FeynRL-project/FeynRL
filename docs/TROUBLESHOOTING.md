# Troubleshooting Guide

This guide covers common issues encountered while running FeynRL, including multi-node scaling, memory management, and training stability.

## Multi-Node & Scaling Issues

### RL run hangs during rollout or training step
**Possible causes:**
- **GPU over-allocation**: `training_gpus + rollout_gpus` exceeds available cluster GPUs.
- **Ray actor crash**: A Ray actor crashed silently (OOM, CUDA error) and the remaining actors are stuck waiting.
- **NCCL timeout**: Network-level issue between nodes on the training side.

**How to fix:**
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
   NCCL_DEBUG=INFO python main_rl.py --config-file ./configs/rl_args.yaml --experiment_id debug_run
   ```
4. **Network Connectivity**: Ensure all nodes can communicate over the specified ports.

### Troubleshooting multi-node scaling
Multi-node training with Ray and DeepSpeed can be complex. If your run hangs:
1. **GPU Allocation**: Ensure `training_gpus + rollout_gpus` does not exceed your cluster's total GPUs.
2. **Network Connectivity**: Ensure all nodes can communicate over the specified ports. Use `NCCL_DEBUG=INFO` to surface network-level errors.
3. **Ray Status**: Check `ray status` to see if any actors have failed or are pending resources.
4. **Shared Filesystem**: For multi-node runs, ensure your `checkpoint_dir` is on a shared filesystem so all nodes can access saved models.

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

### Strict on-policy error (`policy_version != loaded_version`)
**Possible causes:**
- **Sync Failure**: Weight sync (`direct` or `disk`) failed silently in a previous epoch, so rollout engines still hold stale weights.
- **Strict Mode**: `force_strict_on_policy: True` in the config makes the engine reject any version mismatch.

**How to fix:**
1. Search the logs for earlier `[WeightSync]` warnings; these indicate a failed sync attempt.
2. If the problem persists, switch to `weight_sync_method: disk` in `rl_args.yaml` as a fallback (slower but more robust).

### vLLM reload/update failures
**Possible causes:**
- **Missing Files**: Checkpoint directory is missing `config.json` or tokenizer files; vLLM cannot load a model without them.
- **Local vs Shared Paths**: On multi-node setups, the checkpoint path is on a local disk that rollout workers on other nodes cannot see.

**How to fix:**
1. **Verify Files**:
   ```bash
   ls <checkpoint_dir>/<experiment_id>/
   # expect: config.json, tokenizer.json, tokenizer_config.json, model*.safetensors
   ```
2. **Use Shared Storage**: For multi-node, use a **shared filesystem** for `checkpoint_dir`.
3. **Sync Check**: If using `weight_sync_method: direct`, disk checkpoints are only written at save intervals; verify the sync logs show success.

### Direct vs. Disk Weight Synchronization
- **`direct` (Default)**: Pushes weights directly via GPU memory. It's much faster as it avoids disk I/O, but it's more sensitive to network stability and Ray object store limits.
- **`disk`**: Saves weights to disk and has vLLM reload them. It's slower but more robust, especially on clusters with high network latency or when troubleshooting sync issues.

---

## Training & Algorithmic Issues

### Unexpected zero rewards in RL
**Possible causes:**
- **Default/Failure Reward**: The reward function returns 0 for all samples (e.g., the model never produces EOS, so the default reward assigns 0).
- **Truncation**: Responses are truncated at `rollout.max_tokens` before the model can produce a correct answer, and the terminal reward is lost.
- **Clipping**: `data.max_seq_len` is too small, so prompt + response gets clipped during training.

**How to fix:**
1. **Inspect Samples**: Look at raw rollout samples to see what the model generates — check `response_len` and whether EOS is present.
2. **Check Length Limits**:
   - `max_tokens` = max generation length (response only)
   - `max_seq_len` = max total length (prompt + response)
   - `max_tokens` must be ≤ `max_seq_len`
3. **Increase Room**: Try increasing `max_tokens` to give the model more room to produce a complete answer.
4. **Reward Logic**: Verify your reward function handles edge cases (empty responses, truncated responses) correctly.
