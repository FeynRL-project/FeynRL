import torch
import numpy as np
from typing import List, Dict, Any
from vllm import SamplingParams

class Base:

    def make_sampling_params(self) -> SamplingParams:
        '''
            Create sampling params for generation.
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
        return SamplingParams(seed=seed_base,
                              n=self.n_samples,

                              temperature=self.temperature,
                              top_p=self.top_p,
                              top_k=self.top_k,
                              min_p=0.0,

                              max_tokens=self.max_tokens,
                              stop=self.stop,
                              stop_token_ids=self.stop_token_ids,
                              ignore_eos=self.ignore_eos,

                              # disable penalties
                              presence_penalty=0.0,
                              frequency_penalty=0.0,
                              repetition_penalty=1.0,
                              logit_bias=None,
                              allowed_token_ids=None,
                              bad_words=None,

                              # setup to returns required info
                              logprobs=1, # it returns logprobs for each token
                              prompt_logprobs=(1 if self.prompt_logprobs else None), # it returns logprobs for each token in the prompt which is memory intensive
                              )

    def sanitize_logprobs(self, token_logprobs):
        '''
            Prevent NaN from entering into calculations.
        '''
        logprobs_t = torch.tensor(token_logprobs, dtype=torch.float32, device='cpu')
        nan_mask   = torch.isnan(logprobs_t) | torch.isinf(logprobs_t)
        if nan_mask.any():
            print(f"[vLLM] WARNING: {nan_mask.sum().item()} NaN/Inf in logprobs from vLLM, "
                  f"replacing with sentinel 1.0", flush=True)
            logprobs_t = logprobs_t.masked_fill(nan_mask, 1.0)

        return logprobs_t, nan_mask

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

        return self.sanitize_logprobs(token_logprobs=token_logprobs)

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
            zscore[-1] = (sample['token_rewards'][-1] - mean_scores) / (std_scores + 1e-8)
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

    def score_response(self, prompt: Dict[str, Any], response: Any) -> torch.Tensor:
        '''
            Calculate the reward for each response token.
            it returns a float tensor of len(response_ids).

            The reward function may return either:
              (rewards, is_per_token, correct_threshold)
              (rewards, is_per_token, correct_threshold, named_rewards)
              (rewards, is_per_token, correct_threshold, named_rewards, correctness_reward)
            where named_rewards is a Dict[str, float] of per-component scalars
            for logging purposes. Missing named_rewards defaults to {}.
            correctness_reward is an optional scalar used for pass@k metrics.
        '''
        with torch.no_grad():
            result = self.reward_func(prompt, response)

        rewards, is_per_token, correct_threshold = result[:3]
        named_rewards = result[3] if len(result) > 3 else {}
        correctness_reward = result[4] if len(result) > 4 else None

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(dtype=torch.float32, device='cpu')

        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device='cpu')

        if rewards.numel() != len(response.token_ids):
            raise ValueError(f"score_response must return len={len(response.token_ids)} rewards, got {rewards.numel()}")

        return rewards, is_per_token, correct_threshold, named_rewards, correctness_reward

    def score_responses_batch(self, pairs: List[tuple]) -> List[tuple]:
        '''
            Score all (prompt, response) pairs in one batch call.
            The reward function's .batch method handles concurrency internally
            (e.g. ProcessPoolExecutor with spawn context for math_verify).

            Each result is normalized to a 4-tuple:
              (rewards, is_per_token, correct_threshold, named_rewards, correctness_reward)
            where named_rewards is a Dict[str, float] of per-component scalars
            for logging purposes. Missing named_rewards defaults to {}.
            correctness_reward is an optional scalar used for pass@k metrics.
        '''
        with torch.no_grad():
            raw_results = self.reward_func.batch(pairs)

        validated = []
        for result, (_, response) in zip(raw_results, pairs):
            rewards, is_per_token, correct_threshold = result[:3]
            named_rewards = result[3] if len(result) > 3 else {}
            correctness_reward = result[4] if len(result) > 4 else None

            if isinstance(rewards, torch.Tensor):
                rewards = rewards.to(dtype=torch.float32, device='cpu')

            else:
                rewards = torch.tensor(rewards, dtype=torch.float32, device='cpu')

            if rewards.numel() != len(response.token_ids):
                raise ValueError(f"score_responses_batch must return len={len(response.token_ids)} rewards, got {rewards.numel()}")

            validated.append((rewards, is_per_token, correct_threshold, named_rewards, correctness_reward))

        return validated
