import torch
import pytest
import sys
import types
from unittest.mock import MagicMock, call, patch
from algs.PPO.ppo import PPO
from rollouts.replay_buffer import ReplayBuffer
from types import SimpleNamespace

# Stub optional imports pulled in by core.rl_engines.
_df_prompts = types.ModuleType("data_feeds.prompts")
_df_prompts.PromptsFeed = MagicMock()
sys.modules.setdefault("data_feeds.prompts", _df_prompts)

_df_mixed = types.ModuleType("data_feeds.mixed_sampler")
_df_mixed.create_prompt_dataset_and_sampler = MagicMock()
sys.modules.setdefault("data_feeds.mixed_sampler", _df_mixed)

_vllm = types.ModuleType("rollouts.vllm_engine")
_vllm.VLLMRolloutEngine = MagicMock()
sys.modules.setdefault("rollouts.vllm_engine", _vllm)

_vllm_async = types.ModuleType("rollouts.vllm_engine_async")
_vllm_async.VLLMRolloutEngineAsync = MagicMock()
sys.modules.setdefault("rollouts.vllm_engine_async", _vllm_async)

from core.rl_engines import create_rollout_engines, create_training_engines


def _make_params(tp=1, rollout_gpus=1, overlap=None):
    return SimpleNamespace(
        model=SimpleNamespace(
            name="mock/model",
            ref_model=None,
            dtype="float32",
            trust_remote_code=True,
            attn_implementation="",
            gradient_checkpointing=False,
            value_model=None,
        ),
        run=SimpleNamespace(
            seed=42,
            rollout_gpus=rollout_gpus,
            init_timeout=123,
            nccl_socket_ifname=None,
            nccl_ib_hca=None,
        ),
        train=SimpleNamespace(
            alg_name="grpo",
            kl_coeff=0.01,
            clip_low=0.2,
            clip_high=0.2,
            entropy_coeff=0.01,
            train_batch_size_per_gpu=1,
            update_after_full_replay=True,
            normalize_loss=False,
            train_steps_per_epoch=1,
            tau=0.95,
            gamma=0.99,
        ),
        rollout=SimpleNamespace(
            tensor_parallel_size=tp,
            temperature=1.0,
            max_tokens=16,
            n_samples=1,
            top_p=1.0,
            top_k=-1,
            ignore_eos=False,
            stop=None,
            stop_token_ids=None,
            prompt_logprobs=False,
            gpu_memory_utilization=0.9,
            force_strict_on_policy=False,
            batch_invariant=False,
            max_model_len=None,
            quantization=None,
        ),
        reward=SimpleNamespace(broadcast=False),
        data=SimpleNamespace(max_seq_len=128),
        deepspeed=SimpleNamespace(),
        deepspeed_ref=SimpleNamespace(),
        deepspeed_value=SimpleNamespace(),
        peft=None,
        overlap=overlap,
    )


def _mock_actor_options(count, prefix):
    actor = MagicMock()
    option_results = []
    for i in range(count):
        option_result = MagicMock()
        option_result.remote.return_value = f"{prefix}-{i}"
        option_results.append(option_result)
    actor.options.side_effect = option_results
    return actor, option_results


def test_shape_mismatch_error():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 2, 4
    # Mismatched rewards shape
    rewards = torch.randn(B, T + 1)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.ones(B, T)
    
    # In ppo.py, it checks if len(all_len) != 1.
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_nan_reward_error():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 2, 4
    rewards = torch.tensor([[1.0, 2.0, float('nan'), 4.0], [1.0, 2.0, 3.0, 4.0]])
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    mask = torch.ones(B, T)
    
    with pytest.raises(ValueError, match="rewards or values contain NaN"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)

def test_empty_batch_error():
    # ReplayBuffer is a normal class
    rb = ReplayBuffer(pad_token_id=0, max_seq_len=10)
    with pytest.raises(ValueError, match="collate_fn received an empty batch"):
        rb.collate_fn([])

def test_invalid_mask_holes():
    ppo_logic = PPO
    dummy_self = SimpleNamespace(gamma=0.99, tau=0.95)
    B, T = 1, 5
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    done = torch.zeros(B, T)
    
    # ppo.py checks for holes (rises & (drops.cumsum(dim=1) > 0)).any()
    with pytest.raises(ValueError, match="mask has non-contiguous valid regions"):
        ppo_logic.compute_advantages(dummy_self, rewards, values, done, mask)


def test_training_engines_use_pack_placement_group():
    params = _make_params()
    world_size = 3
    alg, _option_results = _mock_actor_options(world_size, "runner")
    pg = object()
    scheduling_strategies = [object() for _ in range(world_size)]
    rank0_node_ip = "10.0.0.17"

    with patch("core.rl_engines.placement_group", return_value=pg) as mock_pg:
        with patch(
            "core.rl_engines._get_placement_group_bundle_node_ip",
            return_value=rank0_node_ip,
        ) as mock_rank0_ip:
            with patch(
                "core.rl_engines.PlacementGroupSchedulingStrategy",
                side_effect=scheduling_strategies,
            ) as mock_strategy:
                runners, training_rank0_addr = create_training_engines(
                    params=params,
                    alg=alg,
                    world_size=world_size,
                    master_addr="127.0.0.1",
                    master_port=1234,
                )

    assert runners == ["runner-0", "runner-1", "runner-2"]
    assert training_rank0_addr == rank0_node_ip
    mock_pg.assert_called_once_with(
        [{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}],
        strategy="PACK",
    )
    mock_rank0_ip.assert_called_once_with(pg, 0, timeout=params.run.init_timeout)
    assert mock_strategy.call_args_list == [
        call(placement_group=pg, placement_group_bundle_index=0),
        call(placement_group=pg, placement_group_bundle_index=1),
        call(placement_group=pg, placement_group_bundle_index=2),
    ]

    for rank, options_call in enumerate(alg.options.call_args_list):
        options_kwargs = options_call.kwargs
        env_vars = options_kwargs["runtime_env"]["env_vars"]
        assert options_kwargs["num_gpus"] == 1
        assert options_kwargs["scheduling_strategy"] is scheduling_strategies[rank]
        assert env_vars["MASTER_ADDR"] == rank0_node_ip
        assert env_vars["MASTER_PORT"] == "1234"
        assert env_vars["RANK"] == str(rank)
        assert env_vars["WORLD_SIZE"] == str(world_size)


def test_rollout_engines_tp1_use_pack_placement_group():
    params = _make_params(tp=1, rollout_gpus=2)
    actor, _option_results = _mock_actor_options(2, "engine")
    pg = object()
    scheduling_strategies = [object(), object()]

    with patch("core.rl_engines.VLLMRolloutEngine", actor):
        with patch("core.rl_engines.placement_group", return_value=pg) as mock_pg:
            with patch(
                "core.rl_engines.PlacementGroupSchedulingStrategy",
                side_effect=scheduling_strategies,
            ) as mock_strategy:
                engines = create_rollout_engines(params, reward_fnc=MagicMock(), eos_id=0)

    assert engines == ["engine-0", "engine-1"]
    mock_pg.assert_called_once_with(
        [{"GPU": 1, "CPU": 1}, {"GPU": 1, "CPU": 1}],
        strategy="PACK",
    )
    assert mock_strategy.call_args_list == [
        call(placement_group=pg, placement_group_bundle_index=0),
        call(placement_group=pg, placement_group_bundle_index=1),
    ]
    for i, options_call in enumerate(actor.options.call_args_list):
        assert options_call.kwargs["num_gpus"] == 1
        assert options_call.kwargs["scheduling_strategy"] is scheduling_strategies[i]


def test_rollout_engines_tp_gt_1_use_pack_placement_group():
    params = _make_params(tp=2, rollout_gpus=4)
    actor, _option_results = _mock_actor_options(2, "engine")
    pg = object()
    scheduling_strategies = [object(), object()]

    with patch("core.rl_engines.VLLMRolloutEngine", actor):
        with patch("core.rl_engines.placement_group", return_value=pg) as mock_pg:
            with patch(
                "core.rl_engines.PlacementGroupSchedulingStrategy",
                side_effect=scheduling_strategies,
            ) as mock_strategy:
                engines = create_rollout_engines(params, reward_fnc=MagicMock(), eos_id=0)

    assert engines == ["engine-0", "engine-1"]
    mock_pg.assert_called_once_with(
        [{"GPU": 2, "CPU": 1}, {"GPU": 2, "CPU": 1}],
        strategy="PACK",
    )
    assert mock_strategy.call_args_list == [
        call(placement_group=pg, placement_group_bundle_index=0),
        call(placement_group=pg, placement_group_bundle_index=1),
    ]
    for i, options_call in enumerate(actor.options.call_args_list):
        assert options_call.kwargs["num_gpus"] == 2
        assert options_call.kwargs["scheduling_strategy"] is scheduling_strategies[i]


def test_rollout_engines_single_gpu_tp1_uses_single_pack_bundle():
    params = _make_params(tp=1, rollout_gpus=1)
    actor, _option_results = _mock_actor_options(1, "engine")
    pg = object()
    scheduling_strategy = object()

    with patch("core.rl_engines.VLLMRolloutEngine", actor):
        with patch("core.rl_engines.placement_group", return_value=pg) as mock_pg:
            with patch(
                "core.rl_engines.PlacementGroupSchedulingStrategy",
                return_value=scheduling_strategy,
            ) as mock_strategy:
                engines = create_rollout_engines(params, reward_fnc=MagicMock(), eos_id=0)

    assert engines == ["engine-0"]
    mock_pg.assert_called_once_with([{"GPU": 1, "CPU": 1}], strategy="PACK")
    mock_strategy.assert_called_once_with(
        placement_group=pg,
        placement_group_bundle_index=0,
    )
    actor.options.assert_called_once()
    assert actor.options.call_args.kwargs["num_gpus"] == 1
    assert actor.options.call_args.kwargs["scheduling_strategy"] is scheduling_strategy
