"""
ExternalPolicyEngine: generates completions from an OpenAI-compatible API endpoint
and returns samples in the same format as VLLMRolloutEngine.generate().

Design
------
* All per-prompt API calls within one generate() are issued concurrently via a
  thread pool so latency equals ~one prompt's generation time, not N×that.
* Prompts are sent as token-ID lists to avoid re-applying the chat template.
* Logprobs come from the completions endpoint (not chat completions).
* A trailing EOS logprob that the API appends but tokenizer.encode omits is
  silently trimmed (common off-by-one with models that stop on EOS).
* Samples beyond that tolerance are dropped with a warning (tokeniser family
  mismatch).
* policy_version is tagged -1 in returned samples to identify external origin.
* token_zscores / pred_zscores are placeholders; the caller (collect_rollouts)
  calls renormalize_groups() after merging with main-policy samples.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from openai import OpenAI


@dataclass
class _MockVLLMResponse:
    """Minimal shim that satisfies reward-function expectations."""
    token_ids: List[int]
    text: str
    finish_reason: str
    stop_reason: Optional[str] = None


class ExternalPolicyEngine:
    def __init__(
        self,
        base_url: str,
        model: str,
        n_samples: int,
        tokenizer,
        reward_func: Callable,
        reward_broadcast: bool,
        eos_id: int,
        max_tokens: int,
        temperature: float,
        max_seq_len: int,
        api_key: str = "dummy",
        request_timeout: float = 120.0,
        max_concurrent: int = 32,
    ) -> None:
        self.model = model
        self.n_samples = int(n_samples)
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.reward_broadcast = bool(reward_broadcast)
        self.eos_id = int(eos_id)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.max_seq_len = int(max_seq_len)
        self.request_timeout = float(request_timeout)
        self.max_concurrent = int(max_concurrent)

        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
        print(
            f"[ExternalPolicy] Ready: base_url={base_url}, model={model}, "
            f"n_samples={n_samples}, max_concurrent={max_concurrent}, "
            f"request_timeout={request_timeout}s",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(self, prompt_data: Dict[str, Any], mock_resp: _MockVLLMResponse):
        with torch.no_grad():
            rewards, is_per_token, correct_threshold = self.reward_func(prompt_data, mock_resp)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, dtype=torch.float32, device="cpu")
        else:
            rewards = rewards.to(dtype=torch.float32, device="cpu")
        if rewards.numel() != len(mock_resp.token_ids):
            raise ValueError(
                f"Reward numel mismatch: got {rewards.numel()}, "
                f"expected {len(mock_resp.token_ids)}"
            )
        return rewards, is_per_token, correct_threshold

    @staticmethod
    def _sanitize_logprobs(lp_list: List[float]):
        arr = np.array(lp_list, dtype=np.float32)
        nan_mask_np = ~np.isfinite(arr)
        if nan_mask_np.any():
            print(
                f"[ExternalPolicy] {nan_mask_np.sum()} NaN/Inf logprobs, "
                "replacing with sentinel 1.0",
                flush=True,
            )
            arr[nan_mask_np] = 1.0
        return torch.from_numpy(arr), torch.from_numpy(nan_mask_np)

    def _process_one_prompt(
        self, prompt_data: Dict[str, Any], current_iter: int
    ) -> List[Dict[str, Any]]:
        """
        Issue one API call for a single prompt and return all valid samples.
        Called concurrently from generate() via a thread pool.
        """
        prompt_ids = list(prompt_data["prompt_token_ids"])
        prompt_len = len(prompt_ids)

        try:
            api_response = self.client.completions.create(
                model=self.model,
                prompt=prompt_ids,      # token IDs — skips double chat-template
                n=self.n_samples,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                logprobs=1,             # log-prob of the chosen token at each step
                timeout=self.request_timeout,
            )
        except Exception as exc:
            print(f"[ExternalPolicy] API call failed: {exc}", flush=True)
            return []

        samples: List[Dict[str, Any]] = []

        for choice in api_response.choices:
            response_text = choice.text or ""
            finish_reason = choice.finish_reason or "stop"

            if not response_text:
                continue

            response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
            response_len = len(response_ids)

            if response_len == 0:
                continue

            raw_lps: List[Optional[float]] = (
                choice.logprobs.token_logprobs
                if (choice.logprobs and choice.logprobs.token_logprobs)
                else []
            )
            if not raw_lps:
                print("[ExternalPolicy] No logprobs returned, skipping choice.", flush=True)
                continue

            lp_floats = [float(lp) if lp is not None else 0.0 for lp in raw_lps]

            # Trim a single trailing EOS logprob that the API appends when the
            # model stopped on EOS but tokenizer.encode omits that token.
            # This is the common off-by-one seen as "(N vs N-1)".
            if len(lp_floats) == response_len + 1:
                lp_floats = lp_floats[:response_len]

            if len(lp_floats) != response_len:
                print(
                    f"[ExternalPolicy] Logprob/token length mismatch "
                    f"({len(lp_floats)} vs {response_len}) — "
                    "tokeniser families may differ; skipping choice.",
                    flush=True,
                )
                continue

            seq_len = prompt_len + response_len
            input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int64, device="cpu")

            token_masks        = torch.zeros(seq_len, dtype=torch.int32,   device="cpu")
            token_dones        = torch.zeros(seq_len, dtype=torch.int32,   device="cpu")
            token_old_logprobs = torch.zeros(seq_len, dtype=torch.float32, device="cpu")
            pred_masks         = torch.zeros(seq_len, dtype=torch.int32,   device="cpu")
            pred_dones         = torch.zeros(seq_len, dtype=torch.int32,   device="cpu")
            pred_old_logprobs  = torch.zeros(seq_len, dtype=torch.float32, device="cpu")
            rewards_t          = torch.zeros(seq_len, dtype=torch.float32, device="cpu")
            pred_rewards_t     = torch.zeros(seq_len, dtype=torch.float32, device="cpu")

            response_logprobs, nan_mask = self._sanitize_logprobs(lp_floats)

            # Token-aligned
            token_masks[prompt_len:] = 1
            token_old_logprobs[prompt_len:] = response_logprobs
            token_masks[prompt_len:] *= (~nan_mask).to(token_masks.dtype)

            # Pred-aligned
            pred_start = prompt_len - 1
            pred_end   = seq_len - 1
            pred_masks[pred_start:pred_end] = 1
            pred_masks[pred_start:pred_end] *= (~nan_mask).to(pred_masks.dtype)
            pred_old_logprobs[pred_start:pred_end] = response_logprobs

            mock_resp = _MockVLLMResponse(
                token_ids=response_ids,
                text=response_text,
                finish_reason=finish_reason,
            )
            try:
                rewards_resp, _is_per_token, _correct_threshold = self._score(
                    prompt_data, mock_resp
                )
            except Exception as exc:
                print(f"[ExternalPolicy] Reward scoring error: {exc}", flush=True)
                continue

            rewards_t[prompt_len:] = rewards_resp
            pred_rewards_t[pred_start:pred_end] = rewards_resp

            eos_in_tokens = bool(response_ids and response_ids[-1] == self.eos_id)
            ended_on_eos  = finish_reason == "stop" and eos_in_tokens
            if finish_reason == "stop":
                token_dones[seq_len - 1] = 1
                if seq_len >= 2:
                    pred_dones[seq_len - 2] = 1

            samples.append(
                {
                    "iter": int(current_iter),
                    "policy_version": -1,       # marks as external
                    "loaded_version": -1,
                    # token-aligned
                    "input_ids":           input_ids,
                    "token_rewards":       rewards_t,
                    "token_zscores":       rewards_t.clone(),  # renormalize_groups will fix
                    "token_masks":         token_masks,
                    "token_dones":         token_dones,
                    "token_old_logprobs":  token_old_logprobs,
                    # pred-aligned
                    "pred_rewards":        pred_rewards_t,
                    "pred_masks":          pred_masks,
                    "pred_dones":          pred_dones,
                    "pred_old_logprobs":   pred_old_logprobs,
                    "pred_zscores":        pred_rewards_t.clone(),  # renormalize_groups will fix
                    # metadata
                    "finish_reason":  finish_reason,
                    "stop_reason":    None,
                    "ended_on_eos":   ended_on_eos,
                    "response_ids":   response_ids,
                    "prompt_ids":     prompt_ids,
                    "response_text":  response_text,
                    "response_len":   response_len,
                    "truncated":      1 if finish_reason == "length" else 0,
                    "seq_truncated":  1 if seq_len > self.max_seq_len else 0,
                }
            )

        return samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: List[Dict[str, Any]],
        current_iter: int,
        policy_version: int,  # unused; kept for call-site symmetry with vllm engine
    ) -> List[Dict[str, Any]]:
        """
        Generate self.n_samples completions for every prompt in parallel.

        All per-prompt API calls are issued concurrently (up to max_concurrent
        threads) so total latency ≈ one prompt's generation time, not N×that.
        This runs while the main vLLM engines generate their shards, so it
        overlaps with GPU generation at the batch level too.
        """
        workers = min(len(prompts), self.max_concurrent)
        rollout_samples: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._process_one_prompt, pd, current_iter): pd
                for pd in prompts
            }
            for future in as_completed(futures):
                try:
                    rollout_samples.extend(future.result())
                except Exception as exc:
                    print(f"[ExternalPolicy] Prompt processing error: {exc}", flush=True)

        return rollout_samples
