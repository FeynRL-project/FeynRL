import os
import sys
import torch
import gc
import ray
from vllm import LLM, SamplingParams
from typing import Optional, List, Callable, Any, Dict
import numpy as np
import pickle
from misc.utils import set_random_seeds

@ray.remote
class VLLMRolloutEngine:
    def __init__(self,
                 seed:int,
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
                 eps_reward_norm: float,
                 gpu_memory_utilization: float,
                 model_dtype: str,
                 max_seq_len: int,
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

        # vllm engine config
        self.model_path = model_path
        self.model_dtype = model_dtype
        self.loaded_version = -1
        self.trust_remote_code = trust_remote_code
        self.vllm_engine = None
        self.refresh_model(model_path, 0)
        self.sampling_params = self.make_sampling_params()

        # reward normalization
        self.eps_reward_norm = float(eps_reward_norm)
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
            # delete the old engine and free up memory
            try:
                del self.vllm_engine
            except Exception as e:
                print(f"Error deleting vllm_engine: {e}")
                pass

            self.vllm_engine = None
            # memory cleanup
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            # more cleanup
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

        llm_extra_kwargs = {}
        if self.batch_invariant:
            # vLLM batch invariance requires FlashAttention backend.
            # We still keep a compatibility fallback below for older vLLM versions.
            llm_extra_kwargs["attention_backend"] = "FLASH_ATTN"

        # Load new model
        llm_kwargs = dict(model=self.model_path,
                          trust_remote_code=self.trust_remote_code,
                          tensor_parallel_size=self.tensor_parallel_size,
                          gpu_memory_utilization=self.gpu_memory_utilization,
                          dtype=self.model_dtype,
                          # This enables collective_rpc("update_weights", ...) on all TP workers
                          # for direct weight updates. If update_weights_direct is never called,
                          # this sits idle and there is no real overhead since it is just a class
                          # attached to the vllm worker.
                          worker_extension_cls="rollouts.weight_sync.WeightSyncExtension",
                          **llm_extra_kwargs)
        try:
            self.vllm_engine = LLM(**llm_kwargs)
            self.log(f"Successfully loaded vllm model from {self.model_path}")

        except Exception as e:
            # Compatibility fallback: some vLLM versions do not expose
            # the attention_backend argument on LLM(...).
            if self.batch_invariant and "attention_backend" in llm_kwargs:
                msg = str(e)
                backend_kw_error = ("attention_backend" in msg) or ("unexpected keyword" in msg)
                if backend_kw_error:
                    try:
                        self.log("LLM(..., attention_backend=...) unsupported in this vLLM version; retrying without keyword.")
                        llm_kwargs.pop("attention_backend", None)
                        self.vllm_engine = LLM(**llm_kwargs)
                        self.log(f"Successfully loaded vllm model from {self.model_path} (fallback mode)")
                        return
                    except Exception:
                        pass

            print(f"Failed to load vllm model from {self.model_path}: {e}")
            self.vllm_engine = None
            raise

    def update_weights_direct(self, state_dict: dict, version: int) -> bool:
        '''
            Update vllm model weights directly in gpu memory without disk I/O.
            Uses collective_rpc with WeightSyncExtension (works for any TP size).
            Requires vllm >= 0.7 with worker_extension_cls support.
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
            self.vllm_engine.collective_rpc("update_weights", args=(shm_path,))

        finally:
            os.remove(shm_path)

        self.loaded_version = version
        self.log(f"Weights updated to version {version}")
        return True

    def make_sampling_params(self) -> SamplingParams:
        '''
           This function makes sure that sampling policy stays in on-policy regime
           (i.e., same policy as training)
        '''
        if self.force_strict_on_policy:
            if self.temperature != 1.0:
                raise ValueError("Strict on-policy requires temperature = 1.0 (no scaling).")

            if self.top_p != 1.0:
                raise ValueError("Strict on-policy requires top_p = 1.0 (no nucleus truncation).")

            if self.top_k != -1:
                raise ValueError("Strict on-policy requires top_k = -1 (no top-k truncation).")

            if self.n_samples < 1:
                raise ValueError("Strict on-policy requires n_samples >= 1.")

            # vllm can return empty responses for max_tokens <= 0 which will break the rest of the code.
            if self.max_tokens <= 0:
                raise ValueError("max_tokens must be > 0.")

            if self.stop is not None or self.stop_token_ids is not None or self.ignore_eos:
                raise ValueError(
                    "Strict on-policy requires stop=None, stop_token_ids=None, ignore_eos=False "
                    "(these change the trajectory distribution)."
                )

        # When batch_invariant=True, all engines use the same seed so the same
        # prompt always produces the same output regardless of which engine or
        # batch it lands in (topology-invariant).  Sharding in shard_batch_for_engines
        # ensures each prompt goes to exactly one engine, so duplicates are impossible.
        seed_base = self.seed if self.batch_invariant else (self.seed + self.engine_id * 1000)
        return SamplingParams(
            seed=seed_base,
            n=self.n_samples,

            temperature=self.temperature,
            top_p=self.top_p, 
            top_k=self.top_k,
            min_p=0.0,

            max_tokens=self.max_tokens,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,

            # Neutral penalties and no shaping
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
            logit_bias=None,
            allowed_token_ids=None,
            bad_words=None,
            logits_processors=None,

            # setup to returns required info
            logprobs=1, # it returns logprobs for each token
            prompt_logprobs=(1 if self.prompt_logprobs else None), # it returns logprobs for each token in the prompt which is memory intensive
        )

    def extract_logprobs(self, response_ids: List[int], logprobs_by_pos: Any) -> torch.Tensor:
        '''
           Extract logprobs for each token in response_ids from logprobs.
           logprobs_by_pos: list of dict {token_id -> logprob_info}
        '''
        if logprobs_by_pos is None:
            raise ValueError("logprobs_by_pos must not be None.")

        if not isinstance(logprobs_by_pos, list):
            raise TypeError(f"logprobs_by_pos must be a list, got {type(logprobs_by_pos)}")

        if len(response_ids) != len(logprobs_by_pos):
            raise ValueError(f"logprobs_by_pos must have the same len as response_ids. Got {len(logprobs_by_pos)} vs {len(response_ids)}.")

        token_logprobs = []
        for t_id, lgp_dict in zip(response_ids, logprobs_by_pos):
            if lgp_dict is None:
                raise ValueError(f"No logprobs for token {t_id} in {response_ids}.")

            key = t_id
            if key not in lgp_dict and str(key) in lgp_dict:
                key = str(key)

            if key not in lgp_dict:
                raise ValueError(f"No logprobs for token {t_id} in {response_ids}.")

            # account for different formats of logprobs
            v = lgp_dict[key]
            if hasattr(v, 'logprob'):
                token_logprobs.append(float(v.logprob))

            elif isinstance(v, (int, float)):
                token_logprobs.append(float(v))

            elif isinstance(v, dict) and 'logprob' in v:
                token_logprobs.append(float(v['logprob']))

            else:
                raise TypeError(f"Unexpected logprob type: {type(v)}")

        return torch.tensor(token_logprobs, dtype=torch.float32, device='cpu')

    def generate(self,
                prompts: List[Dict[str, List[int]]],
                current_iter: int,
                policy_version: int) -> List[Dict[str, Any]]:
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
                for prompt_data, data in zip(prompts, generated_outputs):
                    group_samples = []
                    group_stats   = {'rewards': [], 'lengths': []}
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

                        # Score every response (including empty) so reward_func can see them,
                        # but only responses > 0 contribute to group stats and normalization.
                        rewards_resp, is_per_token = self.score_response(prompt_data, response)
                        rewards[prompt_len:] = rewards_resp

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
                            response_logprobs = self.extract_logprobs(response_ids, response.logprobs)
                            token_old_logprobs[prompt_len:] = response_logprobs

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
                                                "truncated": 1 if (prompt_len + response_len) > self.max_seq_len else 0,
                                                    })
                    self.normalize_rewards(samples=group_samples,
                                           stats=group_stats,
                                           prompt_len=prompt_len,
                                           is_per_token=is_per_token)
                    rollout_samples.extend(group_samples)

                return rollout_samples

    def score_response(self, prompt: Dict[str, Any], response: Any) -> torch.Tensor:
        '''
            Calculate the reward for each response token.
            it returns a float tensor of len(response_ids).
        '''
        with torch.no_grad():
            # per token rewards or scalar reward
            rewards, is_per_token = self.reward_func(prompt, response)

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(dtype=torch.float32, device='cpu')

        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device='cpu')

        if rewards.numel() != len(response.token_ids):
            raise ValueError(f"score_response must return len={len(response.token_ids)} rewards, got {rewards.numel()}")

        return rewards, is_per_token

    def normalize_rewards(self,
                          samples: List[Dict[str, Any]],
                          stats: Dict[str, List[int]],
                          prompt_len: int,
                          is_per_token: bool) -> None:
        '''
            Normalize rewards for each group of samples for a given prompt.
            samples: list of different responses for a given prompt e.g., [{"prompt_ids": [...], "response_ids": [...],...}, ...]
            stats: {"reward": [...], "length": [...]} or {"reward": [...], "length": [...], "reward": [...], "length": [...]} if reward_broadcast is True
         '''
        denom = len(samples) # number of samples in the group
        if len(samples) > 1:
            rewards_array = np.array(stats['rewards'])
            mean_scores = rewards_array.sum() / denom
            # Bessel's correction (n-1) for unbiased sample std with small n_samples
            std_scores  = np.sqrt(((rewards_array - mean_scores)**2).sum() / max(denom - 1, 1))

        else:
            # For a single sample, we don't normalize (i.e. advantage is 0 if we subtract mean)
            # but usually for n=1 we keep the raw reward.
            mean_scores = 0.0
            std_scores  = 1.0

        if is_per_token:
            raise ValueError("per token rewards are not supported yet as normalization is done assuming per response rewards")

        # now update the rewards in the samples
        for i, sample in enumerate(samples):
            # sample['reward']: [T] where prompt tokens would get 0
            # sample['reward'][-1]: means the last token reward
            zscore = torch.zeros_like(sample['token_rewards'], dtype=torch.float)
            zscore[-1] = (sample['token_rewards'][-1] - mean_scores) / (std_scores + self.eps_reward_norm)
            sample["token_zscores"] = zscore
            if self.reward_broadcast:
                sample["token_zscores"][prompt_len:] = zscore[-1]

            # prediction-aligned zscores
            # zscore[prompt_len:] corresponds to response tokens 0..N-1
            pred_zscores = torch.zeros_like(sample['token_zscores'], dtype=torch.float)
            pred_start = prompt_len - 1
            pred_end   = len(sample['token_zscores']) - 1
            pred_zscores[pred_start:pred_end] = sample["token_zscores"][prompt_len:]
            sample["pred_zscores"] = pred_zscores

if __name__ == "__main__":
    # to test cd ~/FeynRL && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python rollouts/vllm_engine.py
    from transformers import AutoTokenizer
    import ray
    ray.init(local_mode=True)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')

    def default_reward_func(prompt, response):
        is_per_token = False
        response_ids = response.token_ids
        finish_reason = getattr(response, "finish_reason", None)
        r = torch.zeros((len(response_ids),), dtype=torch.float32)

        if len(response_ids) == 0:
            return r, is_per_token

        r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

        return r, is_per_token

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
                                    eps_reward_norm=1e-8,
                                    gpu_memory_utilization=0.5,
                                    engine_id=0,
                                    max_seq_len=2048,
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
    output_ref = vllm.generate.remote(samples_ids, 1, 0)
    output = ray.get(output_ref)
    print(output)
    print('Done')
    ray.shutdown()