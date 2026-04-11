import logging
from typing import Any, Dict

import torch
from concurrent.futures import TimeoutError as FuturesTimeoutError

# Reuse the process pool and verification logic from math_verify_reward_func.
# _get_reward_pool / _run_verification are pure-CPU subprocess helpers; sharing
# the pool avoids spawning a second set of workers for a combined reward.
from rewards.math_verify_reward_func import (
    _get_reward_pool,
    _run_verification,
)

logger = logging.getLogger(__name__)

EOS_BONUS = 0.1  # added to the total reward when the model ends on EoS


def compute_scores_batch(
    pairs: list[tuple[Dict[str, Any], Any]],
    timeout_score: float = 0.0,
    per_call_timeout: float = 60.0,
) -> list[tuple[torch.Tensor, bool, float, Dict[str, float]]]:
    '''
    Score all (prompt_data, response_data) pairs with a combined reward:

        total = math_score + eos_bonus

    where math_score comes from math_verify (0.0–1.0) and eos_bonus is
    EOS_BONUS (0.1) when the model ended generation on EoS, else 0.0.

    Named reward components are returned for separate logging:
        {"math": math_score, "eos": eos_bonus}

    Returns a list with the same length and order as `pairs`, where each
    element is (rewards_tensor, is_per_token, correct_threshold, named_rewards).
    '''
    pool = _get_reward_pool()
    is_per_token = False
    correct_threshold = 0.0

    # Phase 1: submit all math verification jobs without blocking
    submissions = []  # (future | None, n_tokens, eos_bonus)
    for prompt_data, response_data in pairs:
        n_tokens = len(response_data.token_ids)

        finish_reason = getattr(response_data, 'finish_reason', None)
        eos_bonus = EOS_BONUS if str(finish_reason) == 'stop' else 0.0

        if n_tokens == 0:
            submissions.append((None, n_tokens, eos_bonus))
            continue

        ground_truth = prompt_data["solution"]
        ground_truth_boxed = ground_truth
        if "\\boxed" not in ground_truth:
            ground_truth_boxed = "\\boxed{" + ground_truth + "}"

        future = pool.submit(_run_verification, ground_truth_boxed,
                             response_data.text, timeout_score)
        submissions.append((future, n_tokens, eos_bonus))

    # Phase 2: collect results (math workers run concurrently in between)
    results = []
    for future, n_tokens, eos_bonus in submissions:
        r = torch.zeros((n_tokens,), dtype=torch.float32, device='cpu')

        math_score = 0.0
        if future is not None:
            try:
                math_score = float(future.result(timeout=per_call_timeout))
            except FuturesTimeoutError:
                future.cancel()
                logger.warning(
                    f"Reward computation exceeded {per_call_timeout}s wall-clock cap "
                    f"(internal signal.alarm should have fired at 30s). "
                    f"Returning timeout_score={timeout_score}."
                )
                math_score = float(timeout_score)
            except Exception as e:
                logger.error(f"Error in math_eos compute_scores_batch: {e}")
                math_score = 0.0

            r[-1] = math_score + eos_bonus

        named_rewards = {"math": math_score, "eos": eos_bonus}
        results.append((r, is_per_token, correct_threshold, named_rewards))

    return results


def compute_score(
    prompt_data: Dict[str, Any],
    response_data: Any,
    timeout_score: float = 0.0,
    per_call_timeout: float = 60.0,
) -> tuple[torch.Tensor, bool, float, Dict[str, float]]:
    '''
    Single-pair scoring. Delegates to compute_scores_batch for consistency.

    input args:
      prompt_data: Dict[str, Any]
      response_data: response object with .token_ids, .text, and .finish_reason
      timeout_score: score to return for math verification on timeout
      per_call_timeout: hard wall-clock cap (seconds) for math verification

    output args:
      r: torch.Tensor of length len(response_data.token_ids)
         r[-1] = math_score + eos_bonus
      is_per_token: False
      correct_threshold: 0.0
      named_rewards: {"math": float, "eos": float} for separate logging
    '''
    return compute_scores_batch(
        [(prompt_data, response_data)],
        timeout_score=timeout_score,
        per_call_timeout=per_call_timeout,
    )[0]


# Attach batch function so the engine can discover it via
# hasattr(reward_func, 'batch') without changing the constructor interface.
compute_score.batch = compute_scores_batch
