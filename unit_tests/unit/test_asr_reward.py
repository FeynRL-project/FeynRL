from __future__ import annotations


class _Resp:
    def __init__(self, text: str):
        self.text = text
        self.token_ids = [1, 2, 3]


def test_asr_reward_exact_match_gives_one():
    from rewards.asr_reward_func import compute_score

    r, is_per_token, _ = compute_score({"solution": "hello world"}, _Resp("Hello, world!"))
    assert is_per_token is False
    assert float(r[-1].item()) == 1.0


def test_asr_reward_mismatch_less_than_one():
    from rewards.asr_reward_func import compute_score

    r, _, _ = compute_score({"solution": "hello world"}, _Resp("hello there"))
    assert 0.0 <= float(r[-1].item()) < 1.0

