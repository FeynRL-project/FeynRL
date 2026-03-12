import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch


def _import_main_rl_with_stubs():
    prompts_mod = types.ModuleType("data_feeds.prompts")
    prompts_mod.PromptsFeed = object

    mixed_sampler_mod = types.ModuleType("data_feeds.mixed_sampler")
    mixed_sampler_mod.create_prompt_dataset_and_sampler = MagicMock()

    vllm_mod = types.ModuleType("rollouts.vllm_engine")
    vllm_mod.VLLMRolloutEngine = MagicMock()

    replay_mod = types.ModuleType("rollouts.replay_buffer")
    replay_mod.ReplayBuffer = MagicMock()

    logging_mod = types.ModuleType("misc.logging")
    logging_mod.setup_logging = MagicMock()
    logging_mod.setup_tracker = MagicMock()

    config_mod = types.ModuleType("configs.load")

    with patch.dict(
        sys.modules,
        {
            "configs.load": config_mod,
            "data_feeds.prompts": prompts_mod,
            "data_feeds.mixed_sampler": mixed_sampler_mod,
            "rollouts.vllm_engine": vllm_mod,
            "rollouts.replay_buffer": replay_mod,
            "misc.logging": logging_mod,
        },
    ):
        sys.modules.pop("main_rl", None)
        return importlib.import_module("main_rl")


def _make_training_engine(call_log, label):
    remote = MagicMock(side_effect=lambda **kwargs: call_log.append(label) or f"{label}_ref")
    state_remote = MagicMock(side_effect=lambda *a, **kw: f"{label}_state_ref")
    return SimpleNamespace(
        save_checkpoint=SimpleNamespace(remote=remote),
        save_engine_state=SimpleNamespace(remote=state_remote),
    )


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_save_checkpoint_waits_for_engine_saves_before_tokenizer_write(tmp_path):
    main_rl = _import_main_rl_with_stubs()
    call_log = []
    logger = MagicMock()
    tokenizer = MagicMock()
    tokenizer.save_pretrained.side_effect = lambda path: call_log.append("tokenizer")
    engines = [
        _make_training_engine(call_log, "engine0"),
        _make_training_engine(call_log, "engine1"),
    ]

    def _wait_for_saves(refs, timeout, description, logger):
        call_log.append("wait")
        return [True for _ in refs]

    with patch.object(main_rl, "ray_get_with_timeout", side_effect=_wait_for_saves), \
         patch.object(main_rl.ray, "nodes", return_value=[{"Alive": True, "Resources": {"GPU": 2}}]):
        model_path = main_rl.save_checkpoint(
            epoch=0,
            version=7,
            global_step=42,
            tokenizer=tokenizer,
            training_engines=engines,
            checkpoint_dir=str(tmp_path),
            experiment_id="exp4",
            rank=0,
            logger=logger,
            save_timeout=30,
        )

    assert call_log == ["engine0", "engine1", "wait", "wait", "tokenizer"]
    assert model_path == os.path.join(str(tmp_path), "exp4", "iter000001_v000007")
    tokenizer.save_pretrained.assert_called_once_with(model_path)
    logger.warning.assert_not_called()


def test_save_checkpoint_warns_on_multi_gpu_node_cluster(tmp_path):
    main_rl = _import_main_rl_with_stubs()
    logger = MagicMock()
    tokenizer = MagicMock()
    engines = [_make_training_engine([], "engine0")]
    nodes = [
        {"Alive": True, "Resources": {"CPU": 8, "GPU": 4}},
        {"Alive": True, "Resources": {"CPU": 8, "GPU": 4}},
    ]

    with patch.object(main_rl, "ray_get_with_timeout", return_value=[True]), \
         patch.object(main_rl.ray, "nodes", return_value=nodes):
        main_rl.save_checkpoint(
            epoch=0,
            version=1,
            global_step=10,
            tokenizer=tokenizer,
            training_engines=engines,
            checkpoint_dir=str(tmp_path),
            experiment_id="exp4",
            rank=0,
            logger=logger,
            save_timeout=30,
        )

    logger.warning.assert_called_once()
    assert "shared filesystem" in logger.warning.call_args[0][0]
