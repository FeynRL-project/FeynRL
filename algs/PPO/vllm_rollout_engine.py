import torch
import gc
import ray
from vllm import LLM, SamplingParams
import numpy as np
from typing import Optional, List, Callable

@ray.remote(num_gpus=1, resources={"vllm": 1})
class VLLMRolloutEngine:
    def __init__(self,
                 model_name: str,
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
                 ):

        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        # reward function
        self.reward_func = reward_func

        # sampling config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_samples = n_samples
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.ignore_eos = ignore_eos
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.force_strict_on_policy = force_strict_on_policy

        self.vllm_engine = LLM(model=self.model_name, trust_remote_code=self.trust_remote_code)
        self.sampling_params = self.make_sampling_params()

    
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
            logprobs=1, # it will return logprobs for each token 
            prompt_logprobs=self.prompt_logprobs,
        )

    def collect(self, 
                prompt_ids: List[Dict[str, List[int]]], 
                current_iter: int, 
                policy_version: int) -> List[Dict[str, Any]]:
                ''' 
                    prompt_ids: [{'prompt_token_ids': [2,..]}, {'prompt_token_ids': [...]}, ...]
                    Returns a list of rollout samples. length ~ B * n_samples.
                    Each sample includes:
                      - input_ids (prompt+response)
                      - mask (0 for prompt, 1 for response)
                      - old_logprobs (per response token)
                      - finish_reason
                '''

                generated_outputs = self.vllm_engine.generate(prompt_ids, 
                                                             self.sampling_params)
                # generated_outputs has prompt_ids and other outputs
                # this works of even if n_samples >= 1
                rollout_samples = []
                for data in generated_outputs:
                    prompt_ids = data.prompt_token_ids if data.prompt_token_ids is not None else []
                    prompt_len = len(prompt_ids)
                    
                    # process generated responses
                    for response in data.outputs:
                        response_ids = list(response.token_ids)
                        response_len = len(response_ids)

                        # 1. now get data for policy update like old_log_probs, rewards, etc.
                        logprobs = response.logprobs
                        
                        # 2. build input_ids (prompt + response) on cpu
                        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int32, device='cpu')
                        mask      = torch.zeros((prompt_len + response_len,), dtype=torch.int32, device='cpu')
                        if response_len > 0:
                            mask[prompt_len:] = 1

                        # 1 if t is EOS (terminal), 0 otherwise.
                        done  = torch.zeros((prompt_len + response_len,), dtype=torch.int32, device='cpu')
                        # possible values for finish_reason in vLLM:
                        #   stop: generation stopped because a stop condition was met like EOS
                        #   length: generation stopped because max_tokens was reached
                        #   abort: generation stopped because of an error
                        #   None: for unknown reason
                        if "finish_reason" in response and response.finish_reason == "stop": 
                            done[-1] = 1
                        
                        # get rewards per token same as length as input_ids
                        rewards = self.calculate_rewards(prompt_ids, response_ids, response.finish_reason)
                        
                        # 3. build old_logprobs
                        old_logprobs = logprobs[prompt_len:]
                        
                        # 5. build rollout sample
                        rollout_samples.append({
                            "iter": int(current_iter),
                            "policy_version": int(policy_version),
                            "input_ids": input_ids,
                            "mask": mask,
                            "done": done,
                            "old_logprobs": old_logprobs,
                            "response_text": response.text,  
                        })

                return rollout_samples

    def calculate_rewards(prompt_ids, response_ids, finish_reason) -> List[float]:
        '''
            Calculate the reward for each response token.
            reward function assumes getting the prompt_ids, response_ids and finish_reason
            as input.
        '''
        return self.reward_func(prompt_ids, response_ids, finish_reason)