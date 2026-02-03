import torch
from typing import List, Any, Dict

def compute_score(prompt_data: Dict[str, Any], response_data: Dict[str, Any]):
    '''
      input args:
        reward_data: Dict[str, Any] - dictionary containing reward data
      output args:
        r: torch.Tensor - reward tensor
        is_per_token: bool - whether the reward is per token
    '''

    response_ids = list(response_data.token_ids)
    finish_reason = response_data.finish_reason

    is_per_token = False
    r = torch.zeros((len(response_ids),), dtype=torch.float32)
    
    if len(response_ids) == 0:
        return r, is_per_token

    r[-1] = 1.0 if str(finish_reason) == "stop" else 0.0

    return r, is_per_token