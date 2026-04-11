import torch
from typing import Any, Dict


def compute_score(prompt_data: Dict[str, Any], response_data: Any):
    '''
      Returns a scalar terminal reward of 1.0 when generation ended on EOS.
      Auxiliary rewards like this should typically be weighted in config.
    '''
    del prompt_data

    response_ids = list(response_data.token_ids)
    correct_threshold = 0.0
    is_per_token = False
    rewards = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0:
        return rewards, is_per_token, correct_threshold

    finish_reason = getattr(response_data, "finish_reason", None)
    stop_reason = getattr(response_data, "stop_reason", None)
    ended_on_eos = (str(finish_reason) == "stop" and stop_reason is None)
    rewards[-1] = 1.0 if ended_on_eos else 0.0

    return rewards, is_per_token, correct_threshold
