import os
import sys
import glob
import torch
import gc
import ray
from vllm import LLM, SamplingParams
from typing import Optional, List, Callable, Any, Dict
import numpy as np
import pickle
from misc.utils import set_random_seeds
from misc.metrics import compute_pass_metrics
from rollouts.base import Base

@ray.remote
class VLLMRolloutEngine(Base):
    def __init__(self,
                 seed: int,
                 model_path: str,
                 trust_remote_code: bool,
                 temperature: float,
                 max_tokens: int,
                 n_samples: int,
                 top_p: float,
                 top_k: int,
                 ignore_eos: bool,
                 stop: Optional[List[str]],
                 stop_token_ids: Optional[List[int]],
                 prompt_logprobs: bool,
                 force_strict_on_policy: bool,
                 reward_func: Callable,
                 tensor_parallel_size: int,
                 eos_id: int,
                 reward_broadcast: bool,
                 gpu_memory_utilization: float,
                 model_dtype: str,
                 max_seq_len: int,
                 max_model_len: int | None = None,
                 engine_id: int = 0,
                 batch_invariant: bool = False,
                 ):
        # This can reduce throughput depending on model size and batch composition
        # because it forces batch-invariant kernels.
        # https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/reproducibility.py
        if batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"

        # Seed the rollout actor's Python/NumPy/PyTorch RNGs so any
        # non-vLLM operations (reward computation, normalization) are deterministic.
        set_random_seeds(seed + engine_id)

        # Ensure current working directory is in sys.path for this actor
        # and spawned vllm workers. This is required the model so
        # worker_extension_cls resolves to local source.
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())

        # reward function
        self.reward_func = reward_func
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.eos_id = eos_id

        # sampling config
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.n_samples = int(n_samples)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.seed = seed
        self.ignore_eos = bool(ignore_eos)
        self.stop = stop if stop else None
        self.stop_token_ids = stop_token_ids if stop_token_ids else None
        self.prompt_logprobs = prompt_logprobs
        self.force_strict_on_policy = bool(force_strict_on_policy)
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.engine_id = int(engine_id)
        self.batch_invariant = bool(batch_invariant)
        # prompt + response max length also known as context window size
        self.max_seq_len = int(max_seq_len)
        self.max_model_len = int(max_model_len) if max_model_len is not None else None

        # vllm engine config
        self.model_path = model_path
        self.model_dtype = model_dtype
        self.loaded_version = -1
        self.trust_remote_code = trust_remote_code
        self.vllm_engine = None
        self.refresh_model(model_path, 0)
        self.sampling_params = self.make_sampling_params()

        # If True, broadcast a single scalar reward across all tokens in the sequence.
        self.reward_broadcast = bool(reward_broadcast)

    def log(self, msg: str) -> None:
        '''
            Log message only if this is the first engine to avoid clutter.
        '''
        if self.engine_id == 0:
            print(f"[VLLMEngine][Rank {self.engine_id}] {msg}")


    def refresh_model(self, model_path: str, version: int) -> bool:
        '''
           Refresh model only if version changed.
        '''
        if self.vllm_engine is not None and \
           self.loaded_version == version and \
           model_path == self.model_path:
            self.log(f"Model already at version {version}, skipping refresh")
            return False

        self.log(f"Refreshing model to version {version} from {model_path}")

        # only for local paths not HF model identifier (e.g., google/gemma-3-1b-it)
        if os.path.exists(model_path):
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"config.json not found in {model_path}")

        self.model_path = model_path
        self.load_model()
        self.loaded_version = version
        self.log(f"Model refreshed to version {version}")
        return True

    def load_model(self) -> None:
        '''
           Load vLLM engine with cleanup and error handling steps.
        '''
        if self.vllm_engine is not None:
            # Synchronize first so all pending cuda ops complete before teardown.
            # pytorch and vllm execute cuda kernels asynchronously. Without this synchronization
            # step, calling del might destroy the object reference while kernels are still
            # actively running on the gpu, causing memory leaks.
            try:
                if torch.cuda.is_available():
                    # Blocks the cpu until all pending gpu tasks finish.
                    torch.cuda.synchronize()
            except Exception:
                pass

            # Shut down the vllm engine properly so engine-core subprocesses
            # is terminated before we create a new engine.
            try:
                self.vllm_engine.shutdown()

            except Exception as e:
                self.log(f"Engine shutdown warning (non-fatal): {e}")

            try:
                del self.vllm_engine
            except Exception as e:
                print(f"Error deleting vllm_engine: {e}")

            self.vllm_engine = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Load new model
        llm_kwargs = dict(model=self.model_path,
                          trust_remote_code=self.trust_remote_code,
                          tensor_parallel_size=self.tensor_parallel_size,
                          gpu_memory_utilization=self.gpu_memory_utilization,
                          dtype=self.model_dtype,
                          # This enables collective_rpc("update_weights_from_state", ...) on all TP workers within one
                          # rollout engine for direct weight updates. If update_weights_direct is never called,
                          # this sits idle and there is no real overhead since it is just a class
                          # attached to the vllm worker. The orchestrator in main_rl.py calls update_weights_direct on each
                          # rollout engine separately via ray remote so the weight sync happens at both levels.
                          worker_extension_cls="rollouts.weight_sync.WeightSyncExtension",
                          )
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.batch_invariant:
            llm_kwargs["attention_backend"] = "FLASH_ATTN"

        self.vllm_engine = LLM(**llm_kwargs)
        self.log(f"Successfully loaded vllm model from {self.model_path}")

    def update_weights_direct(self, state_dict: dict, version: int) -> bool:
        '''
            Update vllm model weights directly in gpu memory without disk I/O.
            Uses collective_rpc with WeightSyncExtension (works for any TP size).
            Note: state_dict is created in RL/common.py
            state_dict: {param_name: cpu_tensor} from training engine rank 0.
        '''
        if self.vllm_engine is None:
            self.log("Cannot update weights: vLLM engine not loaded")
            return False

        if self.loaded_version == version:
            self.log(f"Model already at version {version}, skipping weight update")
            return True

        # vllm collective_rpc serializes args with msgspec, which cannot encode
        # strings or bytes larger than 4GB. For large models the pickled
        # state_dict easily exceeds that. Instead of sending weight data through
        # msgspec, we pickle to /dev/shm, a ram-backed tmpfs which has no real disk I/O,
        # just a memcpy into kernel page cache, and pass only the ~50-byte file
        # path through collective_rpc. Each TP worker then reads the file
        # independently. This works across multi-node setups because:
        #   1. Ray delivers state_dict to this actor on its node via object store
        #   2. This actor writes to /dev/shm on its node
        #   3. collective_rpc fans out to TP workers on the same node
        #   4. Workers read from the same local /dev/shm
        # Each rollout engine on a different node writes its own file (PID in name).
        # collective_rpc is synchronous, so the file is guaranteed to exist until
        # all workers finish, and the finally block cleans it up afterward.
        shm_path = f"/dev/shm/feynrl_weights_{os.getpid()}_v{version}.pkl"
        with open(shm_path, 'wb') as f:
            pickle.dump(state_dict, f)

        # Free the CPU state_dict now that it's persisted to /dev/shm.
        # The TP workers will read from the file, so we don't need this copy.
        del state_dict
        self.log(f"Updating weights directly to version {version}")
        try:
            results = self.vllm_engine.collective_rpc("update_weights_from_state", args=(shm_path,))

        finally:
            os.remove(shm_path)

        # Verify that weights were actually updated on all tp workers.
        # collective_rpc may silently swallow errors on non-rank-0 tp workers,
        # which would leave some shards with stale weights. update_weights
        # returns the number of parameters loaded where all workers should agree.
        # collective_rpc broadcasts to all TP workers within one rollout engine
        if results is not None:
            valid = [r for r in results if r is not None and r != 0]
            # Complete failure when all workers returned None or 0
            if not valid:
                raise RuntimeError(f"Weight sync verification failed: all {len(results)} TP workers "
                                   f"returned {results} after update_weights. No parameters were loaded. "
                                   f"This likely means load_weights silently failed on all shards.")

            if len(valid) < len(results):
                # Some workers returned a count and some didn't — partial failure
                failed = [i for i, r in enumerate(results) if r is None or r == 0]
                raise RuntimeError(f"Weight sync verification failed: TP workers {failed} "
                                   f"returned {[results[i] for i in failed]} after update_weights. "
                                   f"Weights may be out of sync across TP shards.")

            # check if all tp workers loaded the same number of parameters.
            # it should be one, otherwise there is a problem
            if len(set(valid)) > 1:
                raise RuntimeError(f"Weight sync verification failed: TP workers loaded different "
                                   f"param counts: {results}. Weights may be out of sync.")

        self.loaded_version = version
        self.log(f"Weights updated to version {version}")
        return True

    def generate(self,
                prompts: List[Dict[str, List[int]]],
                current_iter: int,
                policy_version: int,
                log_batch_metrics: bool = False) -> List[Dict[str, Any]]:
                ''' 
                    prompts: Data provided by the dataloader. For example:
                        [{'prompt_token_ids': [2,..], 'solution': '1'}, {'prompt_token_ids': [...], 'solution': '2'}, ...]
                    Returns a list of rollout samples. length ~ B * n_samples.

                    token-aligned and prediction-aligned logprobs/mask/done are returned.
                    Prediction-aligned here means: logit position t predicts token at t+1 (SFT-style shift).
                '''
                if not isinstance(prompts, list) or len(prompts) == 0:
                    raise TypeError(f"prompts must be a non-empty list, got {type(prompts)}")

                if self.force_strict_on_policy and int(policy_version) != int(self.loaded_version):
                    raise ValueError(
                                     f"Off-policy rollout: policy_version={int(policy_version)} "
                                     f"but loaded_version={int(self.loaded_version)}. ")

                assert self.vllm_engine is not None, f"{self.model_path} not loaded."
                # Rotate seed each epoch so sampling RNG varies across iterations.
                # For batch invariance mode, exclude engine_id so the same prompt
                # yields the same output regardless of how many engines are used
                # (topology-invariant: 1-engine and N-engine runs match).
                epoch_offset = (current_iter + 1) * 1000000000
                if self.batch_invariant:
                    self.sampling_params.seed = self.seed + epoch_offset

                else:
                    self.sampling_params.seed = self.seed + self.engine_id * 1000 + epoch_offset

                self.log(f"Generating completions for {len(prompts)} prompts with {self.n_samples} samples each")
                generated_outputs = self.vllm_engine.generate(prompts,
                                                             sampling_params=self.sampling_params,
                                                             use_tqdm=False)
                self.log(f"Generation complete for {len(prompts)} prompts with policy version {policy_version}")

                # generated_outputs has prompt_ids and other outputs
                # this works even if n_samples >= 1
                rollout_samples = []
                batch_num_prompts = 0
                batch_num_passes_at_ks = {i: 0.0 for i in range(1, self.n_samples + 1)}
                batch_num_passes_caret_k = 0.0
                batch_pass_rate_sum = 0.0
                batch_prompt_reward_sum = 0.0
                batch_best_of_k_reward_sum = 0.0
                batch_reward_std_sum = 0.0

                # If the reward function exposes a batch interface, score all
                # (prompt, response) pairs in one call so the reward function
                # can submit all work to its process pool before blocking.
                # This lets e.g. math_verify run 8 verifications concurrently
                # instead of sequentially submitting and waiting one at a time.
                all_pairs = [(pd, resp) for pd, data_item in zip(prompts, generated_outputs)
                             for resp in data_item.outputs]
                if hasattr(self.reward_func, 'batch'):
                    all_rewards = self.score_responses_batch(all_pairs)

                else:
                    all_rewards = [self.score_response(p, r) for p, r in all_pairs]
                reward_idx = 0

                for prompt_data, data in zip(prompts, generated_outputs):
                    group_samples = []
                    group_stats   = {'rewards': [], 'lengths': [], 'correct_threshold': []}
                    prompt_ids = list(data.prompt_token_ids or [])
                    prompt_len = len(prompt_ids)
                    if prompt_len == 0:
                        raise ValueError(f"No prompt token ids found in generated output: {data}")

                    # process generated responses
                    for response in data.outputs:
                        response_ids = list(response.token_ids)
                        response_len = len(response_ids)
                        finish_reason = getattr(response, "finish_reason", None)
                        stop_reason   = getattr(response, "stop_reason", None)

                        # all have length [T] and token_aligned as described above
                        seq_len = prompt_len + response_len
                        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int64, device='cpu')

                        token_masks      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        token_dones      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        token_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        # prediction-level
                        pred_masks      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        pred_dones      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        pred_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        rewards       = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')
                        pred_rewards  = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        rewards_resp, is_per_token, correct_threshold = all_rewards[reward_idx]
                        reward_idx += 1
                        rewards[prompt_len:] = rewards_resp
                        # correct_threshold must be collected from all responses, including empty
                        # correct_threshold is required in pass@k calculation
                        group_stats['correct_threshold'].append(correct_threshold)

                        if response_len > 0:
                            # is_per_token is False, then rewards_resp will only have value for the last element
                            group_stats['rewards'].append(rewards_resp.sum().item())
                            group_stats['lengths'].append(len(response_ids))
                            if response.logprobs is None:
                                raise ValueError("response.logprobs is None. Check if SamplingParams(logprobs=1) is set.")

                            #####
                            # token-aligned
                            #####
                            token_masks[prompt_len:] = 1 # 1 if valid token which we want to update.
                            response_logprobs, nan_mask = self.extract_logprobs(response_ids, response.logprobs)
                            token_old_logprobs[prompt_len:] = response_logprobs
                            token_masks[prompt_len:] = token_masks[prompt_len:] * (~nan_mask).to(token_masks.dtype)
                            #####
                            # pred-aligned
                            #####
                            # To recall how autoregressive models work:
                            # - response token j is at token index prompt_len + j in input_ids
                            # - and this is predicted by logits index prompt_len + j - 1
                            # pred_aligned which would be one we will use in policy update
                            # and to avoid any weired indexing later in the training loop.
                            pred_start = prompt_len - 1
                            pred_end   = seq_len - 1
                            pred_masks[pred_start:pred_end] = 1
                            pred_masks[pred_start:pred_end] = pred_masks[pred_start:pred_end] * (~nan_mask).to(pred_masks.dtype)
                            pred_old_logprobs[pred_start:pred_end] = response_logprobs
                            pred_rewards[pred_start:pred_end] = rewards[prompt_len:]

                            # Terminal handling:
                            #  1. stop: ended due to EOS or a stop condition so done should be 1.
                            #  2. length: truncated which should not be done=1 and we need to bootstrap
                            if finish_reason == "stop":
                                token_dones[seq_len - 1] = 1

                                # pred-aligned terminal is at the logit index that predicts last token
                                # seq_len >= 2 is guaranteed since prompt_len >= 1 and response_len >= 1
                                pred_dones[seq_len - 2] = 1

                            # if stop_reason is None, it means it ended on eos
                            # see here https://docs.vllm.ai/en/stable/api/vllm/outputs/#vllm.outputs.CompletionOutput
                            eos_in_tokens = (response_ids[-1] == self.eos_id)
                            ended_on_eos  = (finish_reason == "stop" and stop_reason is None and eos_in_tokens)

                            group_samples.append({ "iter": int(current_iter),
                                                "policy_version": int(policy_version),
                                                "loaded_version": int(self.loaded_version),

                                                # token-aligned
                                                "input_ids": input_ids, #[T]
                                                "token_rewards": rewards, #[T]
                                                "token_zscores": rewards.clone(), #[T] if len(group_samples) > 1 it will be replaced in normalize_rewards
                                                "token_masks": token_masks, #[T] 1 on response/valid tokens
                                                "token_dones": token_dones, #[T] 1 on last token if terminal
                                                "token_old_logprobs": token_old_logprobs, #[T] 0 on prompt since we don't backprop on it.

                                                # pred-aligned
                                                "pred_rewards": pred_rewards, #[T]
                                                "pred_masks": pred_masks, #[T]
                                                "pred_dones": pred_dones, #[T]
                                                "pred_old_logprobs": pred_old_logprobs, #[T]
                                                "pred_zscores": pred_rewards.clone(), #[T] if len(group_samples) > 1 it will be replaced in normalize_rewards

                                                "finish_reason": finish_reason,
                                                "stop_reason": stop_reason,
                                                "ended_on_eos": ended_on_eos,

                                                "response_ids": response_ids, # list[int]
                                                "prompt_ids": prompt_ids, # list[int]
                                                "response_text": getattr(response, "text", ""),
                                                "response_len": response_len,
                                                "truncated": 1 if finish_reason == "length" else 0,
                                                "seq_truncated": 1 if (prompt_len + response_len) > self.max_seq_len else 0,
                                                    })
                    self.normalize_rewards(samples=group_samples,
                                           stats=group_stats,
                                           prompt_len=prompt_len,
                                           is_per_token=is_per_token)

                    # compute pass@k metrics and update related variables
                    assert len(set(group_stats['correct_threshold'])) == 1, 'all correct_thresholds should be the same from a reward function'
                    correct_threshold = group_stats['correct_threshold'][0]
                    pass_at_k_metrics = compute_pass_metrics(group_stats['rewards'], n_total=self.n_samples,
                                                             correct_threshold=correct_threshold)

                    batch_num_prompts += 1
                    for i in range(1, self.n_samples + 1):
                        batch_num_passes_at_ks[i] += pass_at_k_metrics['pass_at_ks'].get(i, 0.0)

                    batch_num_passes_caret_k += pass_at_k_metrics['pass_caret_k']
                    batch_pass_rate_sum += pass_at_k_metrics['pass_rate']
                    batch_prompt_reward_sum += pass_at_k_metrics['group_mean_reward']
                    batch_best_of_k_reward_sum += pass_at_k_metrics['best_of_k_reward']
                    batch_reward_std_sum += pass_at_k_metrics['reward_std_per_prompt']

                    for s in group_samples:
                        s["pass_at_ks"] = pass_at_k_metrics['pass_at_ks']
                        s["pass_caret_k"] = pass_at_k_metrics['pass_caret_k']
                        s["pass_rate"] = pass_at_k_metrics['pass_rate']
                        s["k"] = pass_at_k_metrics['k']
                        s["group_mean_reward"] = pass_at_k_metrics['group_mean_reward']
                        s["best_of_k_reward"] = pass_at_k_metrics['best_of_k_reward']
                        s["reward_std_per_prompt"] = pass_at_k_metrics['reward_std_per_prompt']

                    rollout_samples.extend(group_samples)

                if log_batch_metrics and batch_num_prompts > 0:
                    pass_at_k_items = ", ".join(
                        f"avg_pass@{i}={batch_num_passes_at_ks[i] / batch_num_prompts:.4f}"
                        for i in range(1, self.n_samples + 1)
                    )
                    self.log(f"Batch metrics: prompts={batch_num_prompts}, "
                             f"{pass_at_k_items}, "
                             f"avg_pass^k={batch_num_passes_caret_k / batch_num_prompts:.4f}, "
                             f"avg_pass_rate={batch_pass_rate_sum / batch_num_prompts:.4f}, "
                             f"avg_reward_per_prompt={batch_prompt_reward_sum / batch_num_prompts:.4f}, "
                             f"avg_best_of_k_reward={batch_best_of_k_reward_sum / batch_num_prompts:.4f}, "
                             f"avg_reward_std_per_prompt={batch_reward_std_sum / batch_num_prompts:.4f}")

                return rollout_samples

if __name__ == "__main__":
    # to test cd ~/FeynRL && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python rollouts/vllm_engine.py
    from transformers import AutoTokenizer
    import ray
    ray.init(local_mode=True)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

    def default_reward_func(prompt, response):
        is_per_token = False
        correct_threshold = 0.0
        response_ids = response.token_ids
        finish_reason = getattr(response, "finish_reason", None)
        r = torch.zeros((len(response_ids),), dtype=torch.float32)

        if len(response_ids) == 0:
            return r, is_per_token, correct_threshold

        r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

        return r, is_per_token, correct_threshold

    vllm = VLLMRolloutEngine.remote(model_path='google/gemma-3-1b-it',
                                    trust_remote_code=True,
                                    temperature=1,
                                    max_tokens=1024,
                                    n_samples=5,
                                    top_p=1,
                                    top_k=-1,
                                    seed=50,
                                    ignore_eos=False,
                                    stop=None,
                                    stop_token_ids=None,
                                    prompt_logprobs=None,
                                    force_strict_on_policy=True,
                                    reward_func=default_reward_func,
                                    tensor_parallel_size=1,
                                    eos_id=tokenizer.eos_token_id,
                                    reward_broadcast=True,
                                    gpu_memory_utilization=0.5,
                                    engine_id=0,
                                    max_seq_len=2048,
                                    max_model_len=32768,
                                    model_dtype='bfloat16',
                                    batch_invariant=True,
                                    )

    dummy_data = ["Hello, how are you?",
                  "Summer is the best season!",
                  "I love playing chess.",
                  ]
    samples_ids = []
    for i in dummy_data:
        prompt_ids = tokenizer.apply_chat_template(
                                        conversation= [{"role": "user", "content": i}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors=None,
                                        )
        samples_ids.append({"prompt_token_ids": prompt_ids})
    output_ref = vllm.generate.remote(prompts=samples_ids, current_iter=1, policy_version=0, log_batch_metrics=True)
    output = ray.get(output_ref)
    print(output)
    print('Done')
    ray.shutdown()