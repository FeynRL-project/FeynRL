import re
from typing import Dict, Any
import torch

def extract_solution(solution_str, clip_chars=300):
    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    
    if len(solution_str) > clip_chars:
        solution_str = solution_str[-clip_chars:]

    # this also tests the formatting of the model
    solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
    if len(solutions) == 0:
        final_answer = None
    else:
        # take the last solution
        final_answer = solutions[-1].replace(",", "").replace("$", "")

    return final_answer

def compute_score(prompt_data: Dict[str, Any], response_data: Dict[str, Any], method="flexible", format_score=0.0, score=1.0):
    '''
      input args:
        reward_data: Dict[str, Any] - dictionary containing reward data
      output args:
        r: torch.Tensor - reward tensor
        is_per_token: bool - whether the reward is per token
    '''
    solution_str = response_data.text
    ground_truth = prompt_data["solution"]

    r = torch.zeros((len(response_data.token_ids),), dtype=torch.float32)
    answer = extract_solution(solution_str=solution_str, method=method)

    is_per_token = False

    if answer is None:
        return r, is_per_token
    else:
        if answer == ground_truth:
            r[-1] = score
            return r, is_per_token
        else:
            r[-1] = format_score
            return r, is_per_token