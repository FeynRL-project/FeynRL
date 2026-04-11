import importlib
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class RewardComponent:
    name: str
    weight: float
    fn: Callable[..., Any]


def _normalize_reward_name(module_name: str) -> str:
    suffix = "_reward_func"
    return module_name[:-len(suffix)] if module_name.endswith(suffix) else module_name


def _load_reward_function(module_name: str) -> Callable[..., Any]:
    reward_module = importlib.import_module("rewards." + module_name)
    return getattr(reward_module, "compute_score")


def load_reward_function(config_reward) -> Callable[..., Any]:
    if config_reward.reward_funcs:
        weights = config_reward.reward_weights
        components = [
            RewardComponent(name=_normalize_reward_name(module_name),
                            weight=float(weight),
                            fn=_load_reward_function(module_name))
            for module_name, weight in zip(config_reward.reward_funcs, weights)
        ]
        return compose_reward_functions(components)

    if config_reward.reward_func:
        return _load_reward_function(config_reward.reward_func)

    raise ValueError("Reward function not specified")


def compose_reward_functions(components: list[RewardComponent]) -> Callable[..., Any]:
    if not components:
        raise ValueError("At least one reward component is required")

    primary_component = components[0]

    def _normalize_result(result, response):
        rewards, is_per_token, correct_threshold = result[:3]
        named_rewards = result[3] if len(result) > 3 else {}

        if isinstance(rewards, torch.Tensor):
            rewards = rewards.to(dtype=torch.float32, device="cpu")
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32, device="cpu")

        if rewards.numel() != len(response.token_ids):
            raise ValueError(
                f"Reward function must return len={len(response.token_ids)} rewards, got {rewards.numel()}"
            )

        return rewards, bool(is_per_token), float(correct_threshold), dict(named_rewards)

    def _merge_named_rewards(base_name: str, component_total: float, named_rewards: dict[str, float]) -> dict[str, float]:
        merged = {base_name: component_total}
        for sub_name, value in named_rewards.items():
            key = f"{base_name}_{sub_name}"
            merged[key] = float(value)
        return merged

    def _combine_one(prompt, response):
        total_rewards = None
        total_named_rewards = {}
        primary_correct_threshold = None
        primary_reward_total = None
        expected_is_per_token = None

        for component in components:
            rewards, is_per_token, correct_threshold, named_rewards = _normalize_result(component.fn(prompt, response), response)

            if expected_is_per_token is None:
                expected_is_per_token = is_per_token
            elif expected_is_per_token != is_per_token:
                raise ValueError("All reward components must agree on is_per_token")

            if component is primary_component:
                primary_correct_threshold = correct_threshold
                primary_reward_total = float(rewards.sum().item())

            weighted_rewards = rewards * component.weight
            total_rewards = weighted_rewards if total_rewards is None else (total_rewards + weighted_rewards)

            component_total = float(rewards.sum().item())
            total_named_rewards.update(_merge_named_rewards(component.name, component_total, named_rewards))

        return total_rewards, expected_is_per_token, primary_correct_threshold, total_named_rewards, primary_reward_total

    def compute_score(prompt_data, response_data):
        return _combine_one(prompt_data, response_data)

    def compute_scores_batch(pairs):
        component_results = []
        for component in components:
            if hasattr(component.fn, "batch"):
                raw_results = component.fn.batch(pairs)
            else:
                raw_results = [component.fn(prompt, response) for prompt, response in pairs]

            normalized = [
                _normalize_result(result, response)
                for result, (_, response) in zip(raw_results, pairs)
            ]
            component_results.append((component, normalized))

        combined_results = []
        for idx, (_, response) in enumerate(pairs):
            total_rewards = None
            total_named_rewards = {}
            primary_correct_threshold = None
            primary_reward_total = None
            expected_is_per_token = None

            for component, normalized in component_results:
                rewards, is_per_token, correct_threshold, named_rewards = normalized[idx]

                if expected_is_per_token is None:
                    expected_is_per_token = is_per_token
                elif expected_is_per_token != is_per_token:
                    raise ValueError("All reward components must agree on is_per_token")

                if component is primary_component:
                    primary_correct_threshold = correct_threshold
                    primary_reward_total = float(rewards.sum().item())

                weighted_rewards = rewards * component.weight
                total_rewards = weighted_rewards if total_rewards is None else (total_rewards + weighted_rewards)

                component_total = float(rewards.sum().item())
                total_named_rewards.update(_merge_named_rewards(component.name, component_total, named_rewards))

            combined_results.append((total_rewards, expected_is_per_token, primary_correct_threshold, total_named_rewards, primary_reward_total))

        return combined_results

    compute_score.batch = compute_scores_batch
    return compute_score
