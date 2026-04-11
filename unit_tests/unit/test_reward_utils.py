import pytest
from types import SimpleNamespace

import torch

from misc.reward_utils import RewardComponent, compose_reward_functions


def _response(token_ids, finish_reason="stop", stop_reason=None):
    return SimpleNamespace(
        token_ids=token_ids,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        text="resp",
    )


def test_compose_reward_functions_weights_training_reward_and_keeps_primary_correctness():
    def math_reward(prompt, response):
        rewards = torch.zeros((len(response.token_ids),), dtype=torch.float32)
        rewards[-1] = 1.0 if prompt["correct"] else 0.0
        return rewards, False, 0.0

    def eos_reward(prompt, response):
        rewards = torch.zeros((len(response.token_ids),), dtype=torch.float32)
        rewards[-1] = 1.0 if response.stop_reason is None else 0.0
        return rewards, False, 0.0

    reward_fn = compose_reward_functions([
        RewardComponent(name="math_verify", weight=1.0, fn=math_reward),
        RewardComponent(name="eos", weight=0.1, fn=eos_reward),
    ])

    rewards, is_per_token, correct_threshold, named_rewards, correctness_reward = reward_fn(
        {"correct": False},
        _response([1, 2, 3], finish_reason="stop", stop_reason=None),
    )

    assert not is_per_token
    assert correct_threshold == 0.0
    assert rewards[-1].item() == pytest.approx(0.1)
    assert correctness_reward == 0.0
    assert named_rewards["math_verify"] == 0.0
    assert named_rewards["eos"] == 1.0


def test_compose_reward_functions_batch_preserves_primary_correctness_scores():
    def math_reward(prompt, response):
        rewards = torch.zeros((len(response.token_ids),), dtype=torch.float32)
        rewards[-1] = float(prompt["math"])
        return rewards, False, 0.0

    def math_reward_batch(pairs):
        return [math_reward(prompt, response) for prompt, response in pairs]

    math_reward.batch = math_reward_batch

    def eos_reward(prompt, response):
        rewards = torch.zeros((len(response.token_ids),), dtype=torch.float32)
        rewards[-1] = 1.0 if response.stop_reason is None else 0.0
        return rewards, False, 0.0

    reward_fn = compose_reward_functions([
        RewardComponent(name="math_verify", weight=1.0, fn=math_reward),
        RewardComponent(name="eos", weight=0.1, fn=eos_reward),
    ])

    results = reward_fn.batch([
        ({"math": 1.0}, _response([1, 2], stop_reason=None)),
        ({"math": 0.0}, _response([3, 4], stop_reason=None)),
    ])

    assert results[0][0][-1].item() == pytest.approx(1.1)
    assert results[0][4] == 1.0
    assert results[1][0][-1].item() == pytest.approx(0.1)
    assert results[1][4] == 0.0
