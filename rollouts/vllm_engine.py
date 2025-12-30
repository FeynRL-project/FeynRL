import torch
import gc
import ray
from vllm import LLM, SamplingParams
import numpy as np
from typing import Optional, List, Callable, Any, Dict

@ray.remote(resources={"vllm": 1})
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


    def refresh_model(self, model_path: str, version: int) -> bool:
        '''
           refresh only if the model is changed.
        '''
        if self.vllm_engine is not None and \
           self.loaded_version == version and \
           model_path == self.model_path:
            return False

        self.model_path     = model_path
        self.loaded_version = version
        self.load_model()
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
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self.vllm_engine = LLM(
                                model=self.model_path,
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

        if self.stop is not None or self.stop_token_ids is not None or self.ignore_eos == True:
            raise ValueError(
                "Strict on-policy requires stop=None, stop_token_ids=None, ignore_eos=False "
                "(these change the trajectory distribution)."
            )

        return SamplingParams(
            seed=self.seed,
            n=self.n_samples,
            best_of=None, # best_of != n implies reranking/selectio

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
        if logprobs_by_pos is None or len(response_ids) != len(logprobs_by_pos):
            raise ValueError("logprobs_by_pos must be a list of dict with the same len as response_ids.")

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
                    Each sample includes:
                      - input_ids (prompt+response)
                      - mask (0 for prompt, 1 for response)
                      - old_logprobs (per response token)
                      - finish_reason
                '''
                assert self.vllm_engine is not None, f"{self.model_path} not loaded."
                generated_outputs = self.vllm_engine.generate(prompts,
                                                             sampling_params=self.sampling_params,
                                                             use_tqdm=False)
                # generated_outputs has prompt_ids and other outputs
                # this works even if n_samples >= 1
                rollout_samples = []
                for data in generated_outputs:
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

                        # all of the following are on cpu and have length (prompt_len + response_len)
                        # build input_ids (prompt + response) on cpu
                        seq_len = prompt_len + response_len
                        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int32, device='cpu')
                        mask      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        done      = torch.zeros((seq_len,), dtype=torch.int32, device='cpu')
                        old_logps = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')
                        rewards   = torch.zeros((seq_len,), dtype=torch.float32, device='cpu')

                        if response_len > 0:
                            # 1 if valid token which we want to update.
                            mask[prompt_len:] = 1

                            if response.logprobs is None:
                                raise ValueError("response.logprobs is None. Check if SamplingParams(logprobs=1) is set.")

                            # now get data for policy update like old_log_probs, rewards, etc.
                            response_logprobs = self.extract_logprobs(response_ids, response.logprobs)
                            old_logps[prompt_len:] = response_logprobs
                            rewards   = self.score_response(prompt_ids, response_ids, finish_reason)
                            rewards[prompt_len:] = rewards

                            # Terminal handling:
                            #  1. stop: ended due to EOS or a stop condition so done should be 1.
                            #  2. length: truncated which should not be done=1 and we need to bootstrap
                            if finish_reason == "stop":
                                done[seq_len - 1] = 1

                            # if stop_reason is None, it means it ended on eos
                            # see here https://docs.vllm.ai/en/stable/api/vllm/outputs/#vllm.outputs.CompletionOutput
                            eos_in_tokens = response_ids[-1] == self.eos_id
                            ended_on_eos = (finish_reason == "stop" and stop_reason is None and eos_in_tokens)

                        else:
                            ended_on_eos = False

                        # rollout sample
                        rollout_samples.append({
                                                "iter": int(current_iter),
                                                "policy_version": int(policy_version),
                                                "loaded_version": int(self.loaded_version),

                                                "input_ids": input_ids, #[T]
                                                "mask": mask, #[T] 1 on response/valid tokens
                                                "done": done, #[T] 1 on last token if terminal
                                                "rewards": rewards, #[T]
                                                "old_logps": old_logps, #[T] 0 on prompt since we don't backprop on it

                                                "finish_reason": finish_reason,
                                                "stop_reason": stop_reason,
                                                "ended_on_eos": ended_on_eos,

                                                "response_ids": response_ids,
                                                "prompt_ids": prompt_ids,
                                                "response_text": getattr(response, "text", ""),
                                                })

                return rollout_samples

    def score_response(self, prompt_ids, response_ids, finish_reason) -> List[float]:
        '''
            Calculate the reward for each response token.
            it returns a float tensor of len(response_ids).
        '''
        rewards = self.reward_func(prompt_ids, response_ids, finish_reason)
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(dtype=torch.float32, device='cpu')

        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device='cpu')

        if rewards.numel() != len(response_ids):
            raise ValueError(f"score_response must return len={len(response_ids)} rewards, got {rewards.numel()}")

        return rewards