import os
import yaml
import pytest
from pydantic import ValidationError
from configs.load import load_and_verify, Config, Run, Train, Model, DeepSpeed, Rollout

def test_config_load_sl_success(tmp_path):
    config_dict = {
        "run": {
            "experiment_id": "test",
            "seed": 42,
            "project_name": "test_proj",
            "tracking_uri": "http://localhost:8181",
            "checkpoint_dir": str(tmp_path / "checkpoints")
        },
        "train": {
            "optimizer_name": "adamw",
            "alg_name": "sl",
            "lr": 1e-5,
            "adam_epsilon": 1e-8,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "warmup_steps_ratio": 0.1,
            "clip_grad_norm": 1.0,
            "lr_scheduler": "WarmupCosineLR",
            "total_number_of_epochs": 1,
            "micro_batches_per_epoch": 10,
            "train_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "val_batch_size_per_gpu": 2,
            "dynamic_ratio_every_step": False,
            "normalize_loss": True
        },
        "model": {
            "name": "test-model",
            "dtype": "fp16",
            "trust_remote_code": True
        },
        "data": {
            "train_files_path": ["data.jsonl"],
            "val_files_path": ["val.jsonl"],
            "num_workers": 2,
            "max_seq_len": 512,
            "prompt_key": "prompt",
            "answer_key": "answer"
        },
        "deepspeed": {
            "zero_optimization": {"stage": 2}
        }
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
        
    config = load_and_verify(method="sl", input_yaml=str(config_file), experiment_id="exp1", rank=0, world_size=1)
    assert config.run.experiment_id == "exp1"
    assert config.deepspeed.train_micro_batch_size_per_gpu == 2

def test_config_load_validation_error(tmp_path):
    config_dict = {
        "run": {"experiment_id": "test", "seed": 42, "project_name": "test", "tracking_uri": "test", "checkpoint_dir": "test"},
        "train": {"lr": -1.0} # Invalid LR
    }
    config_file = tmp_path / "bad_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
        
    # load_and_verify calls sys.exit(1) on ValidationError, so we test Config initialization directly
    with pytest.raises(ValidationError):
        Config(**config_dict)

def test_sync_deepspeed_config_logic():
    run = Run(experiment_id="id", seed=1, project_name="p", tracking_uri="u", method="rl", 
              checkpoint_dir="/tmp", ray_master_port=1, init_timeout=1, rollout_timeout=1, 
              train_step_timeout=1, save_timeout=1, sync_timeout=1)
    train = Train(optimizer_name="adamw", alg_name="ppo", lr=1e-4, adam_epsilon=1e-8, 
                  betas=[0.9, 0.999], weight_decay=0.01, warmup_steps_ratio=0.1, 
                  clip_grad_norm=1.0, lr_scheduler="WarmupCosineLR", total_number_of_epochs=5,
                  train_steps_per_epoch=100, train_batch_size_per_gpu=4, 
                  gradient_accumulation_steps=2, val_batch_size_per_gpu=4,
                  dynamic_ratio_every_step=False, normalize_loss=True, update_after_full_replay=True,
                  tau=0.95, gamma=0.99)
    model = Model(name="m", dtype="bf16", trust_remote_code=True, value_model="v")
    ds = DeepSpeed(zero_optimization={"stage": 3})
    rollout = Rollout(rollout_samples_per_epoch=1000, n_samples=1)
    
    config = Config(run=run, train=train, model=model, deepspeed=ds, rollout=rollout)
    config.sync_deepspeed_config(world_size=4)
    
    assert config.deepspeed.train_batch_size == 4 * 2 * 4 # per_gpu * ga * world
    assert config.deepspeed.bf16["enabled"] is True
    assert config.deepspeed.optimizer["type"] == "AdamW"
    # total steps = epochs(5) * steps_per_epoch(100) = 500
    assert config.deepspeed.scheduler["params"]["total_num_steps"] == 500

def test_sync_deepspeed_config_ref():
    run = Run(experiment_id="id", seed=1, project_name="p", tracking_uri="u", method="rl", checkpoint_dir="/tmp", ray_master_port=1, init_timeout=1, rollout_timeout=1, train_step_timeout=1, save_timeout=1, sync_timeout=1)
    train = Train(optimizer_name="adam", alg_name="ppo", lr=1e-4, adam_epsilon=1e-8, betas=[0.9, 0.99], weight_decay=0.0, warmup_steps_ratio=0.1, clip_grad_norm=1.0, lr_scheduler="WarmupCosineLR", total_number_of_epochs=1, train_steps_per_epoch=10, train_batch_size_per_gpu=4, gradient_accumulation_steps=1, val_batch_size_per_gpu=4, dynamic_ratio_every_step=False, normalize_loss=True, update_after_full_replay=True, tau=0.9, gamma=0.9)
    model = Model(name="m", dtype="fp16", trust_remote_code=True, ref_model="path/to/ref")
    ds = DeepSpeed(zero_optimization={"stage": 2, "offload_optimizer": {"device": "cpu"}})
    rollout = Rollout(rollout_samples_per_epoch=100, n_samples=1)
    
    config = Config(run=run, train=train, model=model, deepspeed=ds, rollout=rollout)
    config.sync_deepspeed_config(world_size=1)
    
    # Check if deepspeed_ref was auto-generated
    assert config.deepspeed_ref is not None
    assert config.deepspeed_ref.zero_optimization["stage"] == 0 # mapped from 2
    assert "offload_optimizer" not in config.deepspeed_ref.zero_optimization

def test_load_and_verify_invalid_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        load_and_verify(method="invalid", input_yaml="dummy", experiment_id="e", rank=0)

def test_ppo_direct_sync_requires_checkpoint_save_interval_1(tmp_path):
    """PPO value model only persists via disk checkpoints.
    With direct weight sync and checkpoint_save_interval > 1,
    a crash between saves would lose the value model."""
    config_dict = {
        "run": {
            "experiment_id": "test", "seed": 42, "project_name": "p",
            "tracking_uri": "", "checkpoint_dir": str(tmp_path),
            "training_gpus": 1, "rollout_gpus": 1,
            "ray_master_port": 29500,
            "weight_sync_method": "direct",
            "checkpoint_save_interval": 5,
            "init_timeout": 60, "rollout_timeout": 60,
            "train_step_timeout": 60, "save_timeout": 60, "sync_timeout": 60,
        },
        "train": {
            "optimizer_name": "adamw", "alg_name": "ppo", "lr": 1e-4,
            "adam_epsilon": 1e-8, "betas": [0.9, 0.99], "weight_decay": 0.01,
            "warmup_steps_ratio": 0.1, "clip_grad_norm": 1.0,
            "lr_scheduler": "WarmupCosineLR", "total_number_of_epochs": 10,
            "train_steps_per_epoch": 10, "train_batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1, "val_batch_size_per_gpu": 4,
            "dynamic_ratio_every_step": False, "normalize_loss": True,
            "update_after_full_replay": True,
            "kl_coeff": 0.0, "clip_low": 0.2, "clip_high": 0.2,
            "entropy_coeff": 0.0, "tau": 0.95, "gamma": 0.99,
        },
        "reward": {"broadcast": False, "reward_func": "dummy"},
        "rollout": {
            "temperature": 1.0, "max_tokens": 128, "n_samples": 4,
            "top_p": 1.0, "top_k": -1, "ignore_eos": False,
            "gpu_memory_utilization": 0.5, "force_strict_on_policy": True,
            "tensor_parallel_size": 1, "rollout_batch_size_per_gpu": 4,
            "rollout_samples_per_epoch": 100,
        },
        "model": {"name": "m", "dtype": "bf16", "trust_remote_code": True, "value_model": "v"},
        "data": {
            "train_files_path": ["data.parquet"], "num_workers": 0,
            "max_seq_len": 256, "prompt_key": "prompt", "answer_key": "answer",
        },
        "deepspeed": {"zero_optimization": {"stage": 2}},
    }

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    # checkpoint_save_interval=5 with direct sync should warn but NOT raise for PPO
    config = load_and_verify(method="rl", input_yaml=str(config_file), experiment_id="e", rank=0)
    assert config.run.checkpoint_save_interval == 5

    # checkpoint_save_interval=1 with direct sync should pass without warning
    config_dict["run"]["checkpoint_save_interval"] = 1
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    config = load_and_verify(method="rl", input_yaml=str(config_file), experiment_id="e", rank=0)
    assert config.run.checkpoint_save_interval == 1

    # disk sync with any interval should pass (disk saves include value model)
    config_dict["run"]["weight_sync_method"] = "disk"
    config_dict["run"]["checkpoint_save_interval"] = 5
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    config = load_and_verify(method="rl", input_yaml=str(config_file), experiment_id="e", rank=0)
    assert config.run.checkpoint_save_interval == 5


# ---------------------------------------------------------------------------
# Eval validator (C1)
# ---------------------------------------------------------------------------

def _base_eval_config(tmp_path):
    """Minimal valid eval config."""
    return {
        "run": {
            "experiment_id": "eval-test", "seed": 42, "project_name": "p",
            "tracking_uri": "", "checkpoint_dir": str(tmp_path),
            "rollout_gpus": 2,
            "rollout_timeout": 3600,
        },
        "reward": {"broadcast": False, "reward_func": "dummy"},
        "rollout": {
            "temperature": 1.0, "max_tokens": 512, "n_samples": 4,
            "top_p": 1.0, "top_k": -1, "ignore_eos": False,
            "gpu_memory_utilization": 0.5, "force_strict_on_policy": False,
            "tensor_parallel_size": 1, "rollout_batch_size_per_gpu": 2,
        },
        "model": {"name": "m", "dtype": "bf16", "trust_remote_code": True},
        "data": {
            "test_files_path": "test.parquet", "num_workers": 0,
            "max_seq_len": 512, "prompt_key": "prompt", "answer_key": "answer",
        },
    }


def test_eval_validator_valid_config(tmp_path):
    """A fully specified eval config must pass load_and_verify."""
    cfg = _base_eval_config(tmp_path)
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)
    config = load_and_verify(method="eval", input_yaml=str(config_file), experiment_id="e", rank=0)
    assert config.run.rollout_gpus == 2
    assert config.rollout.tensor_parallel_size == 1


@pytest.mark.parametrize("field,override,match", [
    # rollout_gpus missing → TypeError replaced by clear ValueError
    ("run.rollout_gpus", None, "rollout_gpus"),
    # tensor_parallel_size missing → TypeError replaced by clear ValueError
    ("rollout.tensor_parallel_size", None, "tensor_parallel_size"),
    # tp > rollout_gpus (rg=2, tp=4)
    ("rollout.tensor_parallel_size", 4, "cannot exceed"),
    # n_samples missing
    ("rollout.n_samples", None, "n_samples"),
    # max_tokens missing
    ("rollout.max_tokens", None, "max_tokens"),
    # rollout_batch_size_per_gpu missing
    ("rollout.rollout_batch_size_per_gpu", None, "rollout_batch_size_per_gpu"),
    # test_files_path empty string (None is rejected by Pydantic; empty string is the realistic bad value)
    ("data.test_files_path", "", "test_files_path"),
    # max_seq_len=0 (None is rejected by Pydantic; 0 tests the < 1 validator)
    ("data.max_seq_len", 0, "max_seq_len"),
    # reward_func missing
    ("reward.reward_func", None, "reward_func"),
    # rollout_timeout missing
    ("run.rollout_timeout", None, "rollout_timeout"),
])
def test_eval_validator_rejects_bad_config(tmp_path, field, override, match):
    """Each required eval field must produce a clear ValueError when missing or invalid."""
    cfg = _base_eval_config(tmp_path)
    section, key = field.split(".", 1)
    cfg[section][key] = override
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError, match=match):
        load_and_verify(method="eval", input_yaml=str(config_file), experiment_id="e", rank=0)


def test_eval_validator_rollout_gpus_not_divisible_by_tp(tmp_path):
    """rollout_gpus not divisible by tensor_parallel_size must be rejected."""
    cfg = _base_eval_config(tmp_path)
    cfg["run"]["rollout_gpus"] = 4
    cfg["rollout"]["tensor_parallel_size"] = 3  # 4 % 3 = 1, not divisible
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)
    with pytest.raises(ValueError, match="divisible"):
        load_and_verify(method="eval", input_yaml=str(config_file), experiment_id="e", rank=0)
