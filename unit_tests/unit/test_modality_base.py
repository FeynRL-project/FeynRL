"""Unit tests for modality/base.py, modality/text.py, and modality/__init__.py.

These tests cover:
  - ModalityAdapter cannot be instantiated directly (it is abstract)
  - TextOnlyAdapter implements every abstract method correctly
  - Round-trip: load_sample → build_rollout_request preserves all fields
  - parse_rollout_output extracts exactly the response slice
  - render_output uses vLLM-decoded text when available, falls back to tokenizer.decode
  - build_training_signal produces a ReplayBuffer-compatible dict
  - build_adapter registry: valid names, unknown names, pass-through
"""

import pytest
import torch
from unittest.mock import MagicMock

from modality.base import ModalityAdapter
from modality.text import TextOnlyAdapter
from modality import build_adapter, _ADAPTER_REGISTRY


# ---------------------------------------------------------------------------
# ModalityAdapter — abstract
# ---------------------------------------------------------------------------

def test_modality_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ModalityAdapter()  # type: ignore[abstract]


def test_modality_adapter_subclass_must_implement_all_methods():
    class Incomplete(ModalityAdapter):
        pass  # no abstract methods implemented

    with pytest.raises(TypeError):
        Incomplete()


# ---------------------------------------------------------------------------
# TextOnlyAdapter — construction
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.decode = MagicMock(return_value="decoded text")
    return tok


@pytest.fixture
def adapter(tokenizer):
    return TextOnlyAdapter(tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# load_sample
# ---------------------------------------------------------------------------

def test_load_sample_with_solution(adapter):
    raw = {"prompt_token_ids": [1, 2, 3], "text": "Hello?", "solution": "42"}
    inp, meta = adapter.load_sample(raw)
    assert inp == [1, 2, 3]
    assert meta["text"] == "Hello?"
    assert meta["solution"] == "42"


def test_load_sample_without_solution(adapter):
    raw = {"prompt_token_ids": [10, 20], "text": "Hi"}
    inp, meta = adapter.load_sample(raw)
    assert inp == [10, 20]
    assert meta["solution"] is None


def test_load_sample_missing_text_defaults_to_empty(adapter):
    raw = {"prompt_token_ids": [1]}
    _, meta = adapter.load_sample(raw)
    assert meta["text"] == ""


# ---------------------------------------------------------------------------
# build_rollout_request
# ---------------------------------------------------------------------------

def test_build_rollout_request_with_solution(adapter):
    inp = [1, 2, 3]
    meta = {"text": "Hello?", "solution": "42"}
    req = adapter.build_rollout_request(inp, meta)
    assert req["prompt_token_ids"] == [1, 2, 3]
    assert req["text"] == "Hello?"
    assert req["solution"] == "42"


def test_build_rollout_request_without_solution(adapter):
    inp = [5, 6]
    meta = {"text": "Hi", "solution": None}
    req = adapter.build_rollout_request(inp, meta)
    assert "solution" not in req


# ---------------------------------------------------------------------------
# Round-trip: load_sample → build_rollout_request
# ---------------------------------------------------------------------------

def test_round_trip_preserves_prompt_ids(adapter):
    raw = {"prompt_token_ids": [7, 8, 9], "text": "Q?", "solution": "A"}
    inp, meta = adapter.load_sample(raw)
    req = adapter.build_rollout_request(inp, meta)
    assert req["prompt_token_ids"] == raw["prompt_token_ids"]
    assert req["text"] == raw["text"]
    assert req["solution"] == raw["solution"]


# ---------------------------------------------------------------------------
# parse_rollout_output
# ---------------------------------------------------------------------------

def test_parse_rollout_output_extracts_response(adapter):
    prompt_ids = [1, 2, 3]
    response_ids = [4, 5, 6]
    full_ids = torch.tensor(prompt_ids + response_ids)
    request = {"prompt_token_ids": prompt_ids}
    vllm_out = {"input_ids": full_ids}

    action = adapter.parse_rollout_output(vllm_out, request)
    assert action.tolist() == response_ids


def test_parse_rollout_output_empty_response(adapter):
    prompt_ids = [1, 2]
    full_ids = torch.tensor(prompt_ids)
    request = {"prompt_token_ids": prompt_ids}
    vllm_out = {"input_ids": full_ids}

    action = adapter.parse_rollout_output(vllm_out, request)
    assert action.numel() == 0


# ---------------------------------------------------------------------------
# render_output
# ---------------------------------------------------------------------------

def test_render_output_uses_response_text_from_metadata(adapter, tokenizer):
    action = torch.tensor([10, 20, 30])
    result = adapter.render_output(inp=[1, 2], action=action, metadata={"response_text": "vllm decoded"})
    tokenizer.decode.assert_not_called()
    assert result == "vllm decoded"


def test_render_output_calls_tokenizer_decode(adapter, tokenizer):
    action = torch.tensor([10, 20, 30])
    result = adapter.render_output(inp=[1, 2], action=action, metadata={})
    tokenizer.decode.assert_called_once_with([10, 20, 30], skip_special_tokens=False)
    assert result == "decoded text"


# ---------------------------------------------------------------------------
# build_training_signal
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "input_ids", "pred_rewards", "pred_zscores", "pred_masks",
    "pred_dones", "pred_old_logprobs", "policy_version", "response_len",
    "encoder_inputs",
}


def _make_signal(adapter, prompt_ids, response_len, policy_version=0, finish_reason="stop"):
    action = torch.arange(response_len, dtype=torch.long)
    logprobs = torch.zeros(response_len)
    rewards = torch.ones(response_len)
    return adapter.build_training_signal(
        inp=prompt_ids,
        action=action,
        logprobs=logprobs,
        rewards=rewards,
        metadata={
            "nan_mask":       torch.zeros(response_len, dtype=torch.bool),
            "finish_reason":  finish_reason,
            "stop_reason":    None,
            "response_text":  "test",
            "eos_id":         None,
            "max_seq_len":    2 ** 31,
            "iter":           0,
            "loaded_version": 0,
        },
        policy_version=policy_version,
    )


def test_build_training_signal_has_required_keys(adapter):
    sig = _make_signal(adapter, prompt_ids=[1, 2, 3], response_len=4)
    assert REQUIRED_KEYS.issubset(sig.keys())


def test_build_training_signal_input_ids_shape(adapter):
    prompt_ids = [1, 2, 3]
    R = 5
    sig = _make_signal(adapter, prompt_ids=prompt_ids, response_len=R)
    T = len(prompt_ids) + R
    assert sig["input_ids"].shape == (T,)


def test_build_training_signal_prompt_portion_of_input_ids(adapter):
    prompt_ids = [7, 8, 9]
    sig = _make_signal(adapter, prompt_ids=prompt_ids, response_len=3)
    assert sig["input_ids"][:3].tolist() == prompt_ids


def test_build_training_signal_mask_covers_only_action(adapter):
    prompt_ids = [1, 2]
    R = 4
    sig = _make_signal(adapter, prompt_ids=prompt_ids, response_len=R)
    T = len(prompt_ids) + R
    masks = sig["pred_masks"]
    assert masks.shape == (T,)
    # Pred-aligned: logit at prompt_len-1 predicts the first response token,
    # so the active range is [prompt_len-1, prompt_len+R-1) — exactly R positions.
    pred_start = len(prompt_ids) - 1
    pred_end   = len(prompt_ids) + R - 1
    assert masks[:pred_start].sum().item() == 0.0          # before pred window
    assert masks[pred_start:pred_end].sum().item() == float(R)  # pred window
    assert masks[pred_end:].sum().item() == 0.0            # last token excluded


def test_build_training_signal_done_at_last_action_token(adapter):
    prompt_ids = [1]
    R = 3
    sig = _make_signal(adapter, prompt_ids=prompt_ids, response_len=R, finish_reason="stop")
    dones = sig["pred_dones"]
    T = len(prompt_ids) + R
    # Pred-aligned terminal: logit at seq_len-2 predicts the last response token.
    assert dones[T - 2].item() == 1.0
    assert dones[T - 1].item() == 0.0   # last position is NOT the terminal
    assert dones[: T - 2].sum().item() == 0.0


def test_build_training_signal_response_len(adapter):
    R = 6
    sig = _make_signal(adapter, prompt_ids=[0], response_len=R)
    assert sig["response_len"] == R


def test_build_training_signal_policy_version(adapter):
    sig = _make_signal(adapter, prompt_ids=[0], response_len=2, policy_version=7)
    assert sig["policy_version"] == 7


def test_build_training_signal_encoder_inputs_is_none_for_text(adapter):
    sig = _make_signal(adapter, prompt_ids=[1, 2], response_len=2)
    assert sig["encoder_inputs"] is None


def test_build_training_signal_rewards_padded_to_T(adapter):
    prompt_ids = [1, 2, 3]
    R = 4
    rewards = torch.tensor([0.1, 0.2, 0.3, 0.4])
    action = torch.arange(R, dtype=torch.long)
    sig = adapter.build_training_signal(
        inp=prompt_ids, action=action,
        logprobs=torch.zeros(R), rewards=rewards,
        metadata={
            "nan_mask":      torch.zeros(R, dtype=torch.bool),
            "finish_reason": "stop",
            "eos_id":        None,
        },
    )
    T = len(prompt_ids) + R
    assert sig["pred_rewards"].shape == (T,)
    # Pred-aligned: rewards sit at [prompt_len-1, prompt_len+R-1).
    pred_start = len(prompt_ids) - 1
    pred_end   = len(prompt_ids) + R - 1
    assert sig["pred_rewards"][:pred_start].sum().item() == 0.0
    assert torch.allclose(sig["pred_rewards"][pred_start:pred_end], rewards)
    assert sig["pred_rewards"][pred_end:].sum().item() == 0.0


def test_build_training_signal_empty_response(adapter):
    sig = _make_signal(adapter, prompt_ids=[1, 2], response_len=0)
    assert sig["response_len"] == 0
    assert sig["pred_dones"].sum().item() == 0.0


# ---------------------------------------------------------------------------
# build_adapter / _ADAPTER_REGISTRY
# ---------------------------------------------------------------------------

def test_registry_contains_text():
    assert "text" in _ADAPTER_REGISTRY
    assert _ADAPTER_REGISTRY["text"] is TextOnlyAdapter


def test_registry_values_are_modality_adapter_subclasses():
    for name, cls in _ADAPTER_REGISTRY.items():
        assert issubclass(cls, ModalityAdapter), (
            f"Registry entry '{name}' ({cls}) is not a ModalityAdapter subclass"
        )


def test_build_adapter_valid_name_returns_correct_type():
    adapter = build_adapter("text")
    assert isinstance(adapter, TextOnlyAdapter)


def test_build_adapter_returns_fresh_instance_each_call():
    a1 = build_adapter("text")
    a2 = build_adapter("text")
    assert a1 is not a2


def test_build_adapter_instance_is_usable():
    adapter = build_adapter("text")
    raw = {"prompt_token_ids": [1, 2, 3], "text": "hi", "solution": "42"}
    inp, meta = adapter.load_sample(raw)
    assert inp == [1, 2, 3]
    assert meta["solution"] == "42"


def test_build_adapter_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown modality adapter"):
        build_adapter("nonexistent_modality")


def test_build_adapter_unknown_name_error_lists_supported():
    with pytest.raises(ValueError, match="text"):
        build_adapter("audio")


def test_build_adapter_all_registry_names_constructible():
    """Every name in the registry must produce a usable adapter instance."""
    for name in _ADAPTER_REGISTRY:
        adapter = build_adapter(name)
        assert isinstance(adapter, ModalityAdapter)


def test_build_adapter_text_tokenizer_is_none():
    adapter = build_adapter("text")
    assert adapter.tokenizer is None
