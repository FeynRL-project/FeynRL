import torch
import pytest
from types import SimpleNamespace
from rewards.gsm8k_reward_func import extract_solution, compute_score

def test_extract_solution():
    assert extract_solution("The answer is #### 42") == "42"
    assert extract_solution("The answer is #### -12.5") == "-12.5"
    assert extract_solution("#### 1,000") == "1000"
    assert extract_solution("No answer here") is None

def test_compute_score_correct():
    prompt_data = {"solution": "42"}
    response_data = SimpleNamespace(
        text="Blah blah #### 42",
        token_ids=[1, 2, 3]
    )
    r, is_per_token = compute_score(prompt_data, response_data)
    assert not is_per_token
    assert r[-1] == 1.0
    assert r[:-1].sum() == 0.0

def test_compute_score_incorrect():
    prompt_data = {"solution": "42"}
    response_data = SimpleNamespace(
        text="Blah blah #### 43",
        token_ids=[1, 2, 3]
    )
    r, is_per_token = compute_score(prompt_data, response_data)
    assert r[-1] == 0.0

def test_compute_score_no_format():
    prompt_data = {"solution": "42"}
    response_data = SimpleNamespace(
        text="Blah blah no answer",
        token_ids=[1, 2, 3]
    )
    r, is_per_token = compute_score(prompt_data, response_data)
    assert r.sum() == 0.0
