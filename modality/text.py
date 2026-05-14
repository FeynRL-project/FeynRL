from __future__ import annotations

import torch
from typing import Any

from modality.base import ModalityAdapter


class TextOnlyAdapter(ModalityAdapter):
    """Identity adapter for the existing text-only RL path.

    Wraps the prompt/response format already used by ``PromptsFeed`` and
    ``VLLMRolloutEngine`` with no behaviour changes.  Metrics and loss values
    are identical to the pre-adapter code path.  The training-signal dict
    additionally carries ``encoder_inputs: None`` to establish a uniform schema
    across all adapters.

    Parameters
    ----------
    tokenizer:
        The HuggingFace tokenizer.  Required only by ``render_output`` when
        ``metadata["response_text"]`` is absent (e.g. standalone/test use).
        Pass ``None`` in the engine context — vLLM already decodes the text
        and it is provided via ``metadata["response_text"]``.
    """

    def __init__(self, tokenizer: Any = None) -> None:
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Data feed layer
    # ------------------------------------------------------------------

    def load_sample(self, raw: dict) -> tuple[Any, dict]:
        """Pass-through for a ``PromptsFeed`` output dict.

        Returns
        -------
        inp:
            ``list[int]`` of prompt token ids.
        metadata:
            ``{"text": str, "solution": str | None}``
        """
        inp = raw["prompt_token_ids"]
        meta: dict = {
            "text": raw.get("text", ""),
            "solution": raw.get("solution"),
        }
        return inp, meta

    # ------------------------------------------------------------------
    # Rollout layer
    # ------------------------------------------------------------------

    def build_rollout_request(self, inp: Any, metadata: dict) -> dict:
        """Reconstruct the dict that ``VLLMRolloutEngine.generate`` receives."""
        req: dict = {
            "prompt_token_ids": inp,
            "text": metadata.get("text", ""),
        }
        if metadata.get("solution") is not None:
            req["solution"] = metadata["solution"]
        return req

    def parse_rollout_output(self, vllm_out: dict, request: dict) -> torch.Tensor:
        """Slice the response token ids from the full ``input_ids`` sequence.

        ``vllm_out["input_ids"]`` is shape ``[T]`` (prompt + response).
        The first ``len(request["prompt_token_ids"])`` tokens are the prompt.
        """
        prompt_len = len(request["prompt_token_ids"])
        full_ids: torch.Tensor = vllm_out["input_ids"]
        return full_ids[prompt_len:]

    def render_output(self, inp: Any, action: torch.Tensor, metadata: dict) -> str:
        """Return the reward-function-visible text form of the action.

        In the rollout-engine context ``metadata["response_text"]`` is
        populated with the text already decoded by vLLM, so no tokenizer is
        needed.  Falls back to ``self.tokenizer.decode`` for standalone use.
        """
        if "response_text" in metadata:
            return metadata["response_text"]
        return self.tokenizer.decode(action.tolist(), skip_special_tokens=False)

    # ------------------------------------------------------------------
    # Training layer
    # ------------------------------------------------------------------

    def build_training_signal(
        self,
        inp: Any,
        action: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        metadata: dict,
        policy_version: int = 0,
    ) -> dict:
        """Build the full sample dict consumed by ``normalize_rewards`` and
        ``ReplayBuffer.add_batch_seqs``.

        Reproduces the token-aligned and prediction-aligned tensor construction
        that ``VLLMRolloutEngine.generate`` previously performed inline.

        Parameters
        ----------
        inp:
            Prompt token ids, ``list[int]``.
        action:
            Response token ids, shape ``[R]``.
        logprobs:
            Per-response-token log-probabilities ``[R]``, already sanitized
            by ``Base.extract_logprobs`` (NaN/Inf replaced with sentinel 1.0).
        rewards:
            Per-response-token rewards from the reward function, shape ``[R]``.
        metadata:
            Engine-supplied context.  Expected keys:
                ``nan_mask``      bool tensor ``[R]`` — positions with sentinel logprobs
                ``finish_reason`` str | None
                ``stop_reason``   str | None
                ``response_text`` str — text decoded by vLLM
                ``eos_id``        int | None
                ``max_seq_len``   int
                ``iter``          int
                ``loaded_version`` int
            All keys are optional; sensible defaults are used when absent.
        policy_version:
            Policy version that generated this response.
        """
        prompt_ids: list[int] = inp
        response_ids: list[int] = action.tolist()
        response_logprobs: torch.Tensor = logprobs     # [R]
        rewards_resp: torch.Tensor = rewards           # [R]

        nan_mask: torch.Tensor = metadata.get(
            "nan_mask", torch.zeros(len(response_ids), dtype=torch.bool)
        )
        finish_reason = metadata.get("finish_reason")
        stop_reason   = metadata.get("stop_reason")
        response_text = metadata.get("response_text", "")
        eos_id        = metadata.get("eos_id")
        max_seq_len   = metadata.get("max_seq_len", 2 ** 31)
        current_iter  = metadata.get("iter", 0)
        loaded_version = metadata.get("loaded_version", 0)

        prompt_len   = len(prompt_ids)
        response_len = len(response_ids)
        seq_len      = prompt_len + response_len

        input_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.int64, device="cpu")

        # Zero-initialise all token-aligned and pred-aligned arrays.
        token_masks       = torch.zeros((seq_len,), dtype=torch.int32,   device="cpu")
        token_dones       = torch.zeros((seq_len,), dtype=torch.int32,   device="cpu")
        token_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device="cpu")
        pred_masks        = torch.zeros((seq_len,), dtype=torch.int32,   device="cpu")
        pred_dones        = torch.zeros((seq_len,), dtype=torch.int32,   device="cpu")
        pred_old_logprobs = torch.zeros((seq_len,), dtype=torch.float32, device="cpu")
        token_rewards     = torch.zeros((seq_len,), dtype=torch.float32, device="cpu")
        pred_rewards      = torch.zeros((seq_len,), dtype=torch.float32, device="cpu")

        token_rewards[prompt_len:] = rewards_resp

        # --- token-aligned ---
        token_masks[prompt_len:] = 1
        token_old_logprobs[prompt_len:] = response_logprobs
        # zero out positions that had NaN/Inf logprobs
        token_masks[prompt_len:] = token_masks[prompt_len:] * (~nan_mask).to(token_masks.dtype)

        # --- prediction-aligned (SFT shift) ---
        # Logit at position t predicts token t+1, so the logit for response
        # token 0 is at index prompt_len-1.
        pred_start = prompt_len - 1
        pred_end   = seq_len - 1
        pred_masks[pred_start:pred_end] = 1
        pred_masks[pred_start:pred_end] = (
            pred_masks[pred_start:pred_end] * (~nan_mask).to(pred_masks.dtype)
        )
        pred_old_logprobs[pred_start:pred_end] = response_logprobs
        pred_rewards[pred_start:pred_end] = token_rewards[prompt_len:]

        # --- terminal handling ---
        # "stop"   → episode ended (EOS or stop string) → done = 1
        # "length" → truncated                          → done = 0, needs bootstrap
        # Guard: seq_len >= 2 is required (prompt_len >= 1 and response_len >= 1).
        if finish_reason == "stop" and response_len > 0:
            token_dones[seq_len - 1] = 1
            # pred-aligned terminal: logit that predicts the last token
            pred_dones[seq_len - 2] = 1

        eos_in_tokens = (response_ids[-1] == eos_id) if (eos_id is not None and response_len > 0) else False
        ended_on_eos  = finish_reason == "stop" and stop_reason is None and eos_in_tokens

        return {
            "iter":              int(current_iter),
            "policy_version":    int(policy_version),
            "loaded_version":    int(loaded_version),

            # token-aligned
            "input_ids":            input_ids,             # [T]
            "token_rewards":        token_rewards,          # [T]
            "token_zscores":        token_rewards.clone(),  # [T] placeholder; updated by normalize_rewards
            "token_masks":          token_masks,            # [T]
            "token_dones":          token_dones,            # [T]
            "token_old_logprobs":   token_old_logprobs,     # [T]

            # pred-aligned
            "pred_rewards":     pred_rewards,               # [T]
            "pred_masks":       pred_masks,                 # [T]
            "pred_dones":       pred_dones,                 # [T]
            "pred_old_logprobs": pred_old_logprobs,         # [T]
            "pred_zscores":     pred_rewards.clone(),       # [T] placeholder; updated by normalize_rewards

            "finish_reason":  finish_reason,
            "stop_reason":    stop_reason,
            "ended_on_eos":   ended_on_eos,
            "response_ids":   response_ids,                 # list[int]
            "prompt_ids":     prompt_ids,                   # list[int]
            "response_text":  response_text,
            "response_len":   response_len,
            "truncated":      1 if finish_reason == "length" else 0,
            "seq_truncated":  1 if seq_len > max_seq_len else 0,
            "encoder_inputs": None,
        }
