import math
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import yaml
import sys

class Run(BaseModel):
    '''
      This contains general experiment settings.
    '''
    model_config = ConfigDict(extra='forbid')
    experiment_id: str
    distributed_training_strategy: str = None
    seed: int
    project_name: str
    tracking_uri: str
    method: str = None

    # RL-specific fields
    training_gpus: int | None = None
    rollout_gpus: int | None = None
    ray_address: str | None = None
    ray_master_port: int | None = None
    checkpoint_dir: str | None = None

    # Overlap training and rollout generation
    overlap_enabled: bool | None = None # Enable overlap mode
    overlap_max_lag: int | None = None # Max training steps ahead of rollout policy version
    overlap_weight_update_interval: int | None = None # Update rollout weights every N training steps

    # Weight sync: "direct" pushes weights via gpu memory (no disk I/O),
    # "disk" uses save-to-disk + vllm reload.
    weight_sync_method: str | None = "direct"
    # Save a disk checkpoint every N epochs for persistence/resumability.
    # 0 = only at end.
    checkpoint_save_interval: int | None = 1

    # NCCL configuration for multi-node clusters with multiple NICs
    # nccl_socket_ifname: str | None = None  # e.g., "eth0", "ens3", "bond0"
    # nccl_ib_hca: str | None = None         # e.g., "mlx5_0" for InfiniBand HCA selection

class Train(BaseModel):
    '''
        Everything related to training goes here like optimizer, scheduler, etc.
    '''
    model_config = ConfigDict(extra='forbid')
    ###############
    # optimizer related arguments
    ###############
    optimizer_name: str
    alg_name: str
    lr: float = Field(..., gt=0)
    adam_epsilon: float
    betas: list[float]
    weight_decay: float
    warmup_steps_ratio: float
    clip_grad_norm: float
    lr_scheduler: str

    # RL-specific policy arguments
    kl_coeff: float | None = None
    clip_low: float | None = None
    clip_high: float | None = None
    entropy_coeff: float | None = None
    update_after_full_replay: bool | None = None

    ###############
    # general training  loop arguments
    ###############
    # Here, an "epoch" is defined by a fixed number of training steps, not a full sweep of the dataset.
    # Each epoch processes: train_steps_per_epoch * global_batch_size samples.
    # We define epochs this way to control how different datasets are mixed during training and
    # to have more control over the training process.
    total_number_of_epochs: int

    # RL: train_steps_per_epoch = number of optimizer steps per epoch
    train_steps_per_epoch: int | None = None

    # SL: micro_batches_per_epoch = number of micro-batch iterations per epoch
    # Optimizer steps = micro_batches_per_epoch // gradient_accumulation_steps
    micro_batches_per_epoch: int | None = None

    dynamic_ratio_every_step: bool

    ###############
    # Arguments which are common to both deepspeed and standalone training.
    ###############
    # Some of the below arguments also can be set in deepspeed config. However to avoid any confusion and increase code readability,
    # we are setting them here and update deepspeed config accordingly.
    # Note: train_batch_size_per_gpu is same as train_micro_batch_size_per_gpu
    # global batch_size would be train_batch_size_per_gpu * gradient_accumulation_steps * number_of_gpus.
    train_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    val_batch_size_per_gpu: int

    normalize_loss: bool

class Data(BaseModel):
    '''
        Everything related to data goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    train_ratios: dict[str, float] = None
    train_files_path: list[str] = None
    val_files_path: list[str] = None
    test_files_path: str = None
    num_workers: int
    max_seq_len: int
    prompt_key: str
    answer_key: str
    solution_key: str | None = None

class Model(BaseModel):
    '''
        Information like model_name, ref_model_path, dtype, etc.
    '''
    model_config = ConfigDict(extra='forbid')
    name: str
    dtype: str
    ref_model: str = None
    ref_model_offload_to_cpu: bool = False
    trust_remote_code: bool
    model_class: str = None
    attn_implementation: str = None
    gradient_checkpointing: bool = None

class DeepSpeed(BaseModel):
    '''
        Everything related to DeepSpeed goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    train_batch_size: int | None = None  # Calculated automatically
    train_micro_batch_size_per_gpu: int | None = None
    gradient_accumulation_steps: int | None = None
    gradient_clipping: float | None = None

    # Optimizer/Scheduler are usually dicts in DS config
    optimizer: Dict[str, Any] | None = None
    scheduler: Dict[str, Any] | None = None

    fp16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    bf16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})

    # ZeRO Optimization
    zero_optimization: Dict[str, Any] = Field(default_factory=dict)

    # Activation Checkpointing
    activation_checkpointing: Dict[str, Any] = Field(default_factory=dict)

    # Logging
    steps_per_print: int | None = None
    wall_clock_breakdown: bool | None = None

    # Flops profiler
    flops_profiler: Dict[str, Any] | None = None

    # Monitor config
    monitor_config: Dict[str, Any] | None = None

class DeepSpeedRef(BaseModel):
    '''
        Inference-only deepspeed for ref model in rl(no optimizer, no updates).
    '''
    model_config = ConfigDict(extra='forbid')
    fp16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    bf16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    zero_optimization: Dict[str, Any] = Field(default_factory=dict)
    train_micro_batch_size_per_gpu: int | None = None
    # Activation checkpointing (can help with memory)
    activation_checkpointing: Dict[str, Any] | None = None

class InferenceEngine(BaseModel):
    '''
        Everything related to inference goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    name: str

class Reward(BaseModel):
    '''
        Everything related to rewards (RL-specific).
    '''
    model_config = ConfigDict(extra='forbid')
    broadcast: bool | None = None
    eps_reward_norm: float | None = None
    reward_func: str | None = None

class Rollout(BaseModel):
    '''
        Everything related to rollout generation (RL-specific).
    '''
    model_config = ConfigDict(extra='forbid')
    temperature: float | None = None
    max_tokens: int | None = None
    n_samples: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    ignore_eos: bool | None = None
    stop: str | None = None
    gpu_memory_utilization: float | None = None
    stop_token_ids: list[int] | None = None
    prompt_logprobs: bool | None = None
    force_strict_on_policy: bool | None = None
    tensor_parallel_size: int | None = None
    rollout_batch_size_per_gpu: int | None = None
    rollout_samples_per_epoch: int | None = None

class Config(BaseModel):
    '''
        This is the main configuration class for the experiment where it puts all the sub-configurations
        together to form a complete configuration for the experiment.
    '''
    model_config = ConfigDict(extra='forbid')
    run: Run | None = None
    train: Train | None = None
    model: Model | None = None
    data: Data | None = None
    deepspeed: DeepSpeed | None = None
    inference_engine: InferenceEngine | None = None
    # RL-specific sections
    reward: Reward | None = None
    rollout: Rollout | None = None
    # Reference model DeepSpeed config
    deepspeed_ref: DeepSpeedRef | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sync_deepspeed_config(self, world_size: int):
            """
            Sync DeepSpeed config from train/model without duplicating YAML fields.
            """
            # 1 — Batch Sizes (required for both SL and RL)
            self.deepspeed.train_micro_batch_size_per_gpu = self.train.train_batch_size_per_gpu
            self.deepspeed.gradient_accumulation_steps = self.train.gradient_accumulation_steps

            # Explicitly calculate and set train_batch_size for DeepSpeed logging/sanity check.
            # Without this, model_dump() emits train_batch_size: None which some DS
            # versions misinterpret (key present with None vs key absent).
            if world_size is not None:
                self.deepspeed.train_batch_size = self.train.train_batch_size_per_gpu * self.train.gradient_accumulation_steps * world_size

            # 2 — Gradient Clipping
            self.deepspeed.gradient_clipping = float(self.train.clip_grad_norm)

            # 3 — FP16 / BF16
            dtype = self.model.dtype.lower()
            if dtype in ("float16", "fp16"):
                self.deepspeed.fp16["enabled"] = True
                self.deepspeed.bf16["enabled"] = False

            elif dtype in ("bfloat16", "bf16"):
                self.deepspeed.fp16["enabled"] = False
                self.deepspeed.bf16["enabled"] = True

            else:
                self.deepspeed.fp16["enabled"] = False
                self.deepspeed.bf16["enabled"] = False

            # 4 — Optimizer (Auto-Sync)
            # We map generic "optimizer_name" to DeepSpeed's expected structure
            # To use DeepSpeedCPUAdam (for offload), we simply specify "Adam" or "AdamW"
            if "adamw" in self.train.optimizer_name.lower():
                ds_opt_type = "AdamW"
            elif "adam" in self.train.optimizer_name.lower():
                ds_opt_type = "Adam"
            else:
                raise ValueError(f"Unsupported optimizer: {self.train.optimizer_name}")

            self.deepspeed.optimizer = {
                "type": ds_opt_type,
                "params": {
                    "lr": self.train.lr,
                    "betas": self.train.betas,
                    "weight_decay": self.train.weight_decay,
                    "eps": self.train.adam_epsilon
                }
            }

            # 5 — Scheduler (Auto-Sync)
            if self.train.lr_scheduler == "WarmupCosineLR":
                # SL uses micro_batches_per_epoch (convert to optimizer steps)
                # RL uses train_steps_per_epoch (already optimizer steps)
                if self.run.method == "sl":
                    if self.train.micro_batches_per_epoch is None:
                        raise ValueError("micro_batches_per_epoch must be set for SL training")
                    optimizer_steps_per_epoch = self.train.micro_batches_per_epoch // self.train.gradient_accumulation_steps

                else:
                    if self.train.train_steps_per_epoch is None:
                        raise ValueError("train_steps_per_epoch must be set for RL training")

                    # When update_after_full_replay=True, each train_step call produces
                    # exactly 1 optimizer step (boundary only on last micro-batch).
                    # When False, each train_step produces ceil(micro_batches / ga_steps)
                    # optimizer steps, so we must account for this multiplier.
                    if self.train.update_after_full_replay:
                        optimizer_steps_per_epoch = self.train.train_steps_per_epoch
                    else:
                        estimated_replay_size = self.rollout.rollout_samples_per_epoch * self.rollout.n_samples
                        total_micro_batches = math.ceil(estimated_replay_size / self.train.train_batch_size_per_gpu)
                        micro_batches_per_engine = math.ceil(total_micro_batches / world_size)
                        opt_steps_per_train_step = math.ceil(micro_batches_per_engine / self.train.gradient_accumulation_steps)
                        optimizer_steps_per_epoch = self.train.train_steps_per_epoch * opt_steps_per_train_step

                total_optimizer_steps = self.train.total_number_of_epochs * optimizer_steps_per_epoch
                warmup_steps = int(total_optimizer_steps * self.train.warmup_steps_ratio)

                self.deepspeed.scheduler = {
                    "type": self.train.lr_scheduler,
                    "params": {
                        "total_num_steps": total_optimizer_steps,
                        "warmup_min_ratio": 0.0,
                        "cos_min_ratio": 0.1, # standard default, decays to 10% of max LR
                        "warmup_num_steps": warmup_steps
                    }
                }
            else:
                raise ValueError(f"Unsupported scheduler: {self.train.lr_scheduler}")

            # 6 — ZeRO Defaults (Ensure robust ZeRO-3 settings)
            if self.deepspeed.zero_optimization is None:
                self.deepspeed.zero_optimization = {}

            # Remove keys that are None or explicitly disabled via device="none"
            keys_to_remove = []
            for k, v in self.deepspeed.zero_optimization.items():
                if v is None:
                    keys_to_remove.append(k)
                elif isinstance(v, dict) and v.get("device") == "none":
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                del self.deepspeed.zero_optimization[k]

            # Force crucial ZeRO-3 setting if Stage 3 is active
            if self.deepspeed.zero_optimization.get("stage") == 3:
                # This ensures we don't get 500 small files when saving
                if "stage3_gather_16bit_weights_on_model_save" not in self.deepspeed.zero_optimization:
                    self.deepspeed.zero_optimization["stage3_gather_16bit_weights_on_model_save"] = True

            # 7 — Generate ref model config (inference-only, no optimizer/updates)
            if self.deepspeed_ref is None and self.model.ref_model:
                # Start from the main deepspeed config
                ds_dict = self.deepspeed.model_dump()

                # Remove optimizer/scheduler - ref model is frozen, no updates
                ds_dict.pop("optimizer", None)
                ds_dict.pop("scheduler", None)

                # Configure zero_optimization for ref model
                if ds_dict.get("zero_optimization"):
                    # Remove offload_optimizer - no optimizer for ref model
                    ds_dict["zero_optimization"].pop("offload_optimizer", None)

                    # Configure CPU offloading based on ref_model_offload_to_cpu flag
                    if self.model.ref_model_offload_to_cpu:
                        ds_dict["zero_optimization"]["offload_param"] = {
                            "device": "cpu",
                            "pin_memory": True
                        }
                    else:
                        # Remove offload_param if not offloading
                        ds_dict["zero_optimization"].pop("offload_param", None)

                self.deepspeed_ref = DeepSpeedRef(
                    fp16=ds_dict.get("fp16", {"enabled": False}),
                    bf16=ds_dict.get("bf16", {"enabled": False}),
                    zero_optimization=ds_dict.get("zero_optimization", {}),
                    train_micro_batch_size_per_gpu=ds_dict.get("train_micro_batch_size_per_gpu"),
                    activation_checkpointing=ds_dict.get("activation_checkpointing"),
                )

def load_and_verify(method: str, input_yaml: str, experiment_id: str, rank: int, world_size: int | None = None):
    '''
        method: "sl" or "rl"
        input_yaml: path to the yaml file
        experiment_id: experiment identifier
        world_size: number of GPUs for SL training (optional for RL)
    '''
    try:
        with open(input_yaml, "r") as f:
            raw_config = yaml.safe_load(f)

        # now verify the config
        config = Config(**raw_config)
        config.run.method = method
        # Update Run details
        config.run.experiment_id = experiment_id

        # Determine world_size based on method
        if method == "sl":
            if world_size is None:
                raise ValueError("world_size must be specified for SL training")

        elif method == "rl":
            world_size = config.run.training_gpus

        if method != "eval":
            # Sync AFTER updating world_size
            config.sync_deepspeed_config(world_size)

        # Validate rollout engine resources
        if method == "rl" and config.rollout:
            tp = config.rollout.tensor_parallel_size
            rg = config.run.rollout_gpus
            if tp and rg and tp > rg:
                raise ValueError(f"tensor_parallel_size ({tp}) cannot be greater than rollout_gpus ({rg}). "
                                f"Please increase rollout_gpus or decrease tensor_parallel_size.")

        if rank == 0:
            print( "\n" + 20*"=" + "Config" + 20*"=")
            print(f"Contents of {input_yaml}")
            print(config.model_dump_json(indent=4))
            print(46*"=")

    except ValidationError as e:
        print("Configuration Error:")
        print(e)
        sys.exit(1)

    except FileNotFoundError:
        print("Error: Config file not found.")
        sys.exit(1)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    return config

if __name__ == "__main__":
    # load config
    config = load_and_verify(method="sl", input_yaml="./configs/sl_args.yaml", experiment_id="run_1", rank=0, world_size=4)
    config = load_and_verify(method="rl", input_yaml="./configs/rl_args.yaml", experiment_id="run_2", rank=0)
    config = load_and_verify(method="eval", input_yaml="./configs/eval_args.yaml", experiment_id="run_3", rank=0)