import torch
import pytest
from types import SimpleNamespace
import rewards.math_verify_reward_func as reward_mod


class _InlineFuture:
    def __init__(self, fn, *args):
        self._fn = fn
        self._args = args

    def result(self, timeout=None):
        return self._fn(*self._args)


class _InlinePool:
    def submit(self, fn, *args):
        return _InlineFuture(fn, *args)


@pytest.fixture(autouse=True)
def inline_reward_pool(monkeypatch):
    monkeypatch.setattr(reward_mod, "_get_reward_pool", lambda: _InlinePool())

def test_compute_score_correct_simple():
    prompt_data = {"solution": "42"}
    response_data = SimpleNamespace(
        text="The final answer is 42",
        token_ids=[1, 2, 3, 4, 5]
    )
    r, is_per_token, _ = reward_mod.compute_score(prompt_data, response_data)
    assert not is_per_token
    assert r[-1] == 1.0
    assert r[:-1].sum() == 0.0

def test_compute_score_correct_latex():
    prompt_data = {"solution": "x^2 + y^2"}
    response_data = SimpleNamespace(
        text="It simplifies to $x^2 + y^2$.",
        token_ids=[1, 2, 3, 4, 5, 6]
    )
    r, is_per_token, _ = reward_mod.compute_score(prompt_data, response_data)
    assert r[-1] == 1.0

def test_compute_score_correct_greek():
    prompt_data = {"solution": "\\pi"}
    response_data = SimpleNamespace(
        text="The ratio is 3.14159, which is $\\pi$.",
        token_ids=[1, 2, 3, 4, 5, 6, 7]
    )
    r, is_per_token, _ = reward_mod.compute_score(prompt_data, response_data)
    assert r[-1] == 1.0

def test_compute_score_incorrect():
    prompt_data = {"solution": "42"}
    response_data = SimpleNamespace(
        text="The final answer is 43",
        token_ids=[1, 2, 3, 4, 5]
    )
    r, is_per_token, _ = reward_mod.compute_score(prompt_data, response_data)
    assert r[-1] == 0.0

def test_compute_score_boxed_gt():
    prompt_data = {"solution": "\\boxed{42}"}
    response_data = SimpleNamespace(
        text="The answer is 42",
        token_ids=[1, 2, 3, 4, 5]
    )
    r, is_per_token, _ = reward_mod.compute_score(prompt_data, response_data)
    assert r[-1] == 1.0

def test_compute_score_empty_response():
    prompt_data = {"solution": "42"}
    response_data = SimpleNamespace(
        text="",
        token_ids=[]
    )
    r, is_per_token, _ = reward_mod.compute_score(prompt_data, response_data)
    assert r.numel() == 0
    assert not is_per_token
