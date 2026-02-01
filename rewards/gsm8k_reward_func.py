import torch
from typing import List, Any

def compute_score(
    prompt_ids: List[int],
    response_ids: List[int],
    finish_reason: Any,
    answer_ids: List[int] = None,
) -> float:
    """
    Simple GSM8K-style terminal reward based on token suffix.
    Only strips last token (<eos>) and compares the end of the response to answer_ids.
    """

    rewards = torch.zeros(len(response_ids), dtype=torch.float32)

    is_per_token = False

    if not response_ids or finish_reason != "stop" or not answer_ids:
        return rewards, is_per_token

    # remove last token (<eos>)
    response_tail = response_ids[:-1]

    # reward if the last tokens match answer_ids
    if len(response_tail) >= len(answer_ids) and response_tail[-len(answer_ids):] == answer_ids:
        rewards[-1] = 1.0
    return rewards, is_per_token