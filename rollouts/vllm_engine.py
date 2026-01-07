import torch
import gc
import ray
from vllm import LLM, SamplingParams
from typing import Optional, List, Callable, Any, Dict
import numpy as np

@ray.remote
class VLLMRolloutEngine:
    def __init__(self,
                 model_path: str,
                 trust_remote_code: bool,
                 temperature: float,
                 max_tokens: int,
                 n_samples: int,
                 top_p: float,
                 top_k: int,
                 seed: Optional[int],
                 ignore_eos: bool,
                 stop: Optional[List[str]],
                 stop_token_ids: Optional[List[int]],
                 prompt_logprobs: Optional[int],
                 force_strict_on_policy: bool,
                 reward_func: Callable,
                 tensor_parallel_size: int,
                 eos_id: int,
                 reward_broadcast: bool,
                 eps_reward_norm: float,
                 ):

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
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.force_strict_on_policy = bool(force_strict_on_policy)

        # vllm engine config
        self.model_path = model_path
        self.loaded_version = -1
        self.trust_remote_code = trust_remote_code
        self.vllm_engine = None
        self.refresh_model(model_path, 0)
        self.sampling_params = self.make_sampling_params()

        # reward normalization for group normalization
        self.eps_reward_norm = float(eps_reward_norm)
        # If True, broadcast a single scalar reward across all tokens in the sequence.
        # This is used when the reward function provides a single score for the entire completion.
        self.reward_broadcast = bool(reward_broadcast)

    def refresh_model(self, model_path: str, version: int) -> bool:
        '''
           refresh only if the model is changed.
        '''
        if self.vllm_engine is not None and \
           self.loaded_version == version and \
           model_path == self.model_path:
            return False

        self.model_path = model_path
        self.load_model()
        self.loaded_version = version
        return True

    def load_model(self) -> None:
        '''
           load the model.
        '''
        if self.vllm_engine is not None:
            # delete the old engine and free up memory
            try:
                del self.vllm_engine
            except Exception as e:
                print(f"Error deleting vllm_engine: {e}")
                pass

            self.vllm_engine = None
            gc.collect()

            # empty cache to prevent fragmentation/oom when re-loading
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self.vllm_engine = LLM(model=self.model_path,
                               trust_remote_code=self.trust_remote_code,
                               tensor_parallel_size=self.tensor_parallel_size
                              )
    
    def make_sampling_params(self) -> SamplingParams:
        '''
           This function makes sure that sampling policy stays in on-policy regime
           (i.e., same policy as training)
        '''
        if not self.force_strict_on_policy:
            raise ValueError("force_strict_on_policy must be True during training (strict mode).")

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

        return SamplingParams(
            seed=self.seed,
            n=self.n_samples,
            best_of=None, # best_of != n implies reranking/selection

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
            guided_decoding=None,
            logits_processors=None,

            # setup to returns required info
            logprobs=1, # it returns logprobs for each token
            prompt_logprobs=self.prompt_logprobs if self.prompt_logprobs is not None else 0, # it returns logprobs for each token in the prompt which is memory intensive
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
                    prompts: [{'prompt_token_ids': [2,..]}, {'prompt_token_ids': [...]}, ...]
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
                generated_outputs = self.vllm_engine.generate(prompts,
                                                             sampling_params=self.sampling_params,
                                                             use_tqdm=False)

                # generated_outputs has prompt_ids and other outputs
                # this works even if n_samples >= 1
                rollout_samples = []
                for data in generated_outputs:
                    group_samples = []
                    group_stats   = {'reward': [], 'length': []}
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

                        rewards   = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        # it is important to score the response regardless of its length if it is empty
                        rewards_resp, is_per_token = self.score_response(prompt_ids, response_ids, finish_reason)
                        rewards[prompt_len:] = rewards_resp

                        # is_per_token is False, then rewards_resp will only have value for the last element
                        group_stats['reward'].append(rewards_resp.sum().item())
                        group_stats['length'].append(len(response_ids))

                        if response_len > 0:
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

                        else:
                            ended_on_eos = False

                        # rollout sample in group if n_samples >= 1
                        # I didn't drop response_len == 0 here as it can be useful for logging, or even reward normalization as
                        # reward function should be designed in such way that it assigns negative rewards for example to empty responses.
                        group_samples.append({
                                                "iter": int(current_iter),
                                                "policy_version": int(policy_version),
                                                "loaded_version": int(self.loaded_version),

                                                # token-aligned
                                                "input_ids": input_ids, #[T]
                                                "rewards": rewards, #[T]
                                                "zscores": rewards.clone(), #[T] if len(group_samples) > 1 it will be replaced in normalize_rewards
                                                "token_masks": token_masks, #[T] 1 on response/valid tokens
                                                "token_dones": token_dones, #[T] 1 on last token if terminal
                                                "token_old_logprobs": token_old_logprobs, #[T] 0 on prompt since we don't backprop on it.

                                                # pred-aligned
                                                "pred_masks": pred_masks, #[T]
                                                "pred_dones": pred_dones, #[T]
                                                "pred_old_logprobs": pred_old_logprobs, #[T]

                                                "finish_reason": finish_reason,
                                                "stop_reason": stop_reason,
                                                "ended_on_eos": ended_on_eos,

                                                "response_ids": response_ids, # list[int]
                                                "prompt_ids": prompt_ids, # list[int]
                                                "response_text": getattr(response, "text", ""),
                                                "response_len": response_len,
                                                })
                    if len(group_samples) > 1:
                        # add zscore to the above samples
                        self.normalize_rewards(
                                                samples=group_samples,
                                                stats=group_stats,
                                                prompt_len=prompt_len,
                                                is_per_token=is_per_token)

                    rollout_samples.extend(group_samples)

                return rollout_samples

    def score_response(self, prompt_ids, response_ids, finish_reason) -> torch.Tensor:
        '''
            Calculate the reward for each response token.
            it returns a float tensor of len(response_ids).
        '''
        with torch.no_grad():
            # per token rewards or scalar reward
            rewards, is_per_token = self.reward_func(prompt_ids, response_ids, finish_reason)

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(dtype=torch.float32, device='cpu')

        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device='cpu')

        if rewards.numel() != len(response_ids):
            raise ValueError(f"score_response must return len={len(response_ids)} rewards, got {rewards.numel()}")

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
        rewards_array = np.array(stats['reward'])
        mean_scores = rewards_array.sum() / denom
        std_scores  = np.sqrt(((rewards_array - mean_scores)**2).sum() / denom)

        if is_per_token:
            raise ValueError("per token rewards are not supported yet as normalization is done assuming per response rewards")

        # now update the rewards in the samples
        for i, sample in enumerate(samples):
            # sample['reward']: [T] where prompt tokens would get 0
            # sample['reward'][-1]: means the last token reward
            zscore = torch.zeros_like(sample['reward'], dtype=torch.float)
            zscore[-1] = (sample['reward'][-1] - mean_scores) / (std_scores + self.eps_reward_norm)
            sample["zscores"] = zscore
            if self.reward_broadcast:
                sample["zscores"][prompt_len:] = zscore[-1]