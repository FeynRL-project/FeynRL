import math
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict, ValidationError
import yaml
import sys
import copy
import os
import torch

class Run(BaseModel):
    '''
      This contains general experiment settings.
    '''
    model_config = ConfigDict(extra='forbid')
    experiment_id: str
    seed: int
    project_name: str
    tracking_uri: str
    method: str = None
    logger_type : str = "mlflow" # mlflow or wandb

    # RL-specific fields
    training_gpus: int | None = None
    rollout_gpus: int | None = None
    ray_address: str | None = None
    ray_master_port: int | None = None
    checkpoint_dir: str | None = None

    # Weight sync: "direct" pushes weights via gpu memory (no disk I/O),
    # "disk" uses save-to-disk + vllm reload.
    weight_sync_method: str | None = "direct"
    # Save a disk checkpoint every N epochs for persistence/resumability.
    # 0 = only at end.
    checkpoint_save_interval: int | None = 1

    # NCCL configuration for network transport.
    # Leave nccl_socket_ifname and nccl_ib_hca null for single-node or when NCCL auto-detection works.
    # Set nccl_socket_ifname to the network interface carrying inter-node traffic (e.g., "eth0", "bond0"). run ip addr | grep addr
    nccl_socket_ifname: str | None = None
    # Set nccl_ib_hca to the InfiniBand HCA device (e.g., "mlx5_0", "mlx5_0:1"). run ibstat | grep "CA '" to find it
    nccl_ib_hca: str | None = None
    # Port for NCCL weight sync rendezvous (default: ray_master_port + 100)
    nccl_sync_port: int | None = None
    # Backend for the NCCL weight sync process group.
    # nccl for GPU-to-GPU broadcast (fast), gloo for CPU-based broadcast (robust).
    nccl_sync_backend: str | None = None

    # Timeout (seconds) for ray.get() calls to prevent indefinite hangs.
    # None means use the default for each category.
    # training engine init (default: 1800s)
    init_timeout: int | None = None
    # rollout generation per batch (default: 3600s)
    rollout_timeout: int | None = None
    # single training step (default: 1800s)
    train_step_timeout: int | None = None
    # checkpoint save (default: 1800s)
    save_timeout: int | None = None
    # weight sync operations (default: 900s)
    sync_timeout: int | None = None

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

    # PPO-specific arguments
    # GAE lambda
    tau: float | None = None
    # discount factor
    gamma: float | None = None
    # defaults to train.lr if None
    value_lr: float | None = None
    # defaults to train.weight_decay if None
    value_weight_decay: float | None = None
    # defaults to train.clip_grad_norm if None
    value_clip_grad_norm: float | None = None
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
    # Some of the below arguments also can be set in deepspeed. To avoid any
    # confusion, we are setting them here and update deepspeed config accordingly.
    train_batch_size_per_gpu: int
    gradient_accumulation_steps: int
    val_batch_size_per_gpu: int

    normalize_loss: bool

    # DPO specific arguments
    cl_beta: float | None = None

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
    value_model: str = None  # PPO value model path if alg_name is ppo
    ref_model_offload_to_cpu: bool = False
    trust_remote_code: bool
    model_class: str = None
    attn_implementation: str = None
    gradient_checkpointing: bool = None

class Peft(BaseModel):
    model_config = ConfigDict(extra='forbid')
    use_peft: bool
    peft_type: str | None = None
    task_type: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    # if None, apply lora to all linear layers (peft default).
    # otherwise, set explicitly to target specific modules, e.g. ["q_proj", "v_proj"].
    lora_target_modules: list[str] | None = None

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

    # Accumulate gradients in fp32 for numerical stability under bf16/fp16 training.
    data_types: Dict[str, Any] = Field(default_factory=lambda: {"grad_accum_dtype": "fp32"})

    # Disable gradient pre-scaling (scale loss after backward, not before).
    prescale_gradients: bool = False

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

    def model_dump(self, **kwargs):
        # Exclude None values by default for ds compatibility.
        # DS crashes when config contains explicit None values because
        # dict.get(key, default) returns None instead of the default when the key exists.
        kwargs.setdefault('exclude_none', True)
        return super().model_dump(**kwargs)

class DeepSpeedRef(BaseModel):
    '''
        Inference-only deepspeed for ref model in rl(no optimizer, no updates).
    '''
    model_config = ConfigDict(extra='forbid')
    fp16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    bf16: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
    zero_optimization: Dict[str, Any] = Field(default_factory=dict)
    train_micro_batch_size_per_gpu: int | None = None

    def model_dump(self, **kwargs):
        kwargs.setdefault('exclude_none', True)
        return super().model_dump(**kwargs)

class Reward(BaseModel):
    '''
        Everything related to rewards (RL-specific).
    '''
    model_config = ConfigDict(extra='forbid')
    broadcast: bool | None = None
    reward_func: str | None = None

class Overlap(BaseModel):
    '''
        Overlap training and rollout generation (RL-specific).
    '''
    model_config = ConfigDict(extra='forbid')
    enabled: bool  | None = None
    # Max training steps ahead of rollout policy version
    max_lag: int | None = None
    # Dataloader batches per generation chunk.
    # One chunk in-flight at a time; smaller = more frequent NCCL sync windows.
    # =1: max overlap, fastest ESS response. >1: fewer round-trips, coarser sync.
    chunk_size: int | None = None
    # ESS below this triggers sync between training steps (P3O only)
    ess_sync_threshold: float | None = None
    # Static sync interval in training steps for non-P3O algorithms.
    # None = disabled (only ESS-driven sync).
    fixed_sync_interval: int | None = None

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
    batch_invariant: bool = False
    max_model_len: int | None = None

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
    # RL-specific sections
    reward: Reward | None = None
    rollout: Rollout | None = None
    overlap: Overlap | None = None
    # Reference model DeepSpeed config
    deepspeed_ref: DeepSpeedRef | None = None
    # Value model DeepSpeed config
    deepspeed_value: DeepSpeed | None = None

    # peft specific config
    peft: Peft = Field(default_factory=lambda: Peft(use_peft=False))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sync_deepspeed_config(self, world_size: int):
        '''
                Sync DeepSpeed config from train/model without duplicating YAML fields.
        '''
        if self.run is None or self.train is None or self.deepspeed is None or self.model is None:
            raise ValueError("run, train, deepspeed, and model config sections are required for training")

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

        # 5 — Scheduler
        if self.train.lr_scheduler == "WarmupCosineLR":
            # SL uses micro_batches_per_epoch (convert to optimizer steps)
            # RL uses train_steps_per_epoch (already optimizer steps)
            if self.run.method == "sl" or self.run.method == "cl":
                if self.train.micro_batches_per_epoch is None:
                    raise ValueError("micro_batches_per_epoch must be set for SL training")
                optimizer_steps_per_epoch = self.train.micro_batches_per_epoch // self.train.gradient_accumulation_steps

            elif self.run.method == "rl":
                if self.train.train_steps_per_epoch is None:
                    raise ValueError("train_steps_per_epoch must be set for RL training")

                if self.rollout is None:
                    raise ValueError("rollout config is required for RL training")

                # When update_after_full_replay=True, each train_step call produces
                # exactly 1 optimizer step (boundary only on last micro-batch).
                # When False, each train_step produces ceil(micro_batches / ga_steps)
                # optimizer steps, so we must account for this multiplier.
                if self.train.update_after_full_replay:
                    optimizer_steps_per_epoch = self.train.train_steps_per_epoch

                else:
                    # rollout_samples_per_epoch is the number of prompts.
                    # Multiply by n_samples to get total number of samples in the replay buffer.
                    # Note: estimated_replay_size may be slightly larger than actual if some
                    # responses have response_len==0 (filtered in replay_buffer.add_batch_seqs).
                    # This can cause the lr schedule to not fully reach cos_min_ratio.
                    estimated_replay_size = self.rollout.rollout_samples_per_epoch * self.rollout.n_samples
                    total_micro_batches = math.ceil(estimated_replay_size / self.train.train_batch_size_per_gpu)
                    micro_batches_per_engine = math.ceil(total_micro_batches / world_size)
                    opt_steps_per_train_step = math.ceil(micro_batches_per_engine / self.train.gradient_accumulation_steps)
                    optimizer_steps_per_epoch = self.train.train_steps_per_epoch * opt_steps_per_train_step

            else:
                raise ValueError(f"Unsupported method '{self.run.method}' for scheduler calculation. Expected 'sl' or 'rl'.")


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

        # 6 — ZeRO Defaults
        if self.deepspeed.zero_optimization is None:
            self.deepspeed.zero_optimization = {}

        zero_stage = self.deepspeed.zero_optimization.get("stage", 0)

        # Remove keys that are None or explicitly disabled via device="none"
        keys_to_remove = []
        for k, v in self.deepspeed.zero_optimization.items():
            if v is None:
                keys_to_remove.append(k)
            elif isinstance(v, dict) and v.get("device") == "none":
                keys_to_remove.append(k)

        # Strip offload keys that are invalid for the current stage:
        # - offload_param only works at stage 3
        # - offload_optimizer only works at stages 2 and 3
        if zero_stage < 3 and "offload_param" not in keys_to_remove:
            if "offload_param" in self.deepspeed.zero_optimization:
                keys_to_remove.append("offload_param")
        if zero_stage < 2 and "offload_optimizer" not in keys_to_remove:
            if "offload_optimizer" in self.deepspeed.zero_optimization:
                keys_to_remove.append("offload_optimizer")

        # Strip stage3-only keys when stage != 3 (DS ignores them but they clutter config dumps)
        if zero_stage != 3:
            for k in list(self.deepspeed.zero_optimization.keys()):
                if k.startswith("stage3_") and k not in keys_to_remove:
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del self.deepspeed.zero_optimization[k]

        # Force crucial ZeRO-3 setting if stage 3 is active
        if zero_stage == 3:
            # This ensures we don't get 500 small files when saving
            if "stage3_gather_16bit_weights_on_model_save" not in self.deepspeed.zero_optimization:
                self.deepspeed.zero_optimization["stage3_gather_16bit_weights_on_model_save"] = True

        # 7 — Generate ref model config which is inference-only, no optimizer/updates
        if self.deepspeed_ref is None and self.model.ref_model:
            # Start from the main deepspeed config
            ds_dict = self.deepspeed.model_dump()

            # Remove optimizer/scheduler - ref model is frozen, no updates
            ds_dict.pop("optimizer", None)
            ds_dict.pop("scheduler", None)

            # Configure zero_optimization for ref model
            if ds_dict.get("zero_optimization"):
                # Zero stages 1/2 partition optimizer states and gradients respectively when
                # which is not needed for inference-only models.
                # Keep stage 3 as-is since it partitions parameters and DS handles inference at stage 3.
                if zero_stage in (1, 2):
                    ds_dict["zero_optimization"]["stage"] = 0

                # Remove offload_optimizer - no optimizer for ref model
                ds_dict["zero_optimization"].pop("offload_optimizer", None)

                # Configure CPU offloading based on ref_model_offload_to_cpu flag
                # offload_param is only valid at ZeRO stage 3
                if self.model.ref_model_offload_to_cpu:
                    if zero_stage == 3:
                        ds_dict["zero_optimization"]["offload_param"] = {
                            "device": "cpu",
                            "pin_memory": True
                        }
                    else:
                        raise ValueError(
                            f"ref_model_offload_to_cpu=True requires ZeRO stage 3, "
                            f"but current stage is {zero_stage}. "
                            f"Set zero_optimization.stage to 3 or disable ref_model_offload_to_cpu."
                        )
                else:
                    # Remove offload_param if not offloading
                    ds_dict["zero_optimization"].pop("offload_param", None)

            self.deepspeed_ref = DeepSpeedRef(
                fp16=ds_dict.get("fp16", {"enabled": False}),
                bf16=ds_dict.get("bf16", {"enabled": False}),
                zero_optimization=ds_dict.get("zero_optimization", {}),
                train_micro_batch_size_per_gpu=ds_dict.get("train_micro_batch_size_per_gpu"),
            )

        # 8 — Always sync ref model batch size and dtype from main config
        # (handles both auto-generated and user-provided deepspeed_ref)
        if self.deepspeed_ref is not None:
            self.deepspeed_ref.train_micro_batch_size_per_gpu = self.deepspeed.train_micro_batch_size_per_gpu
            self.deepspeed_ref.fp16 = dict(self.deepspeed.fp16)
            self.deepspeed_ref.bf16 = dict(self.deepspeed.bf16)

        # 9 — Generate value model DS config.
        # Clones the policy DS config and overrides optimizer params where specified.
        if self.train.alg_name and self.train.alg_name.lower() == 'ppo':
            value_ds = copy.deepcopy(self.deepspeed)

            # Override optimizer lr if value_lr is set
            v_lr = self.train.value_lr if self.train.value_lr is not None else self.train.lr
            v_wd = self.train.value_weight_decay if self.train.value_weight_decay is not None else self.train.weight_decay
            if value_ds.optimizer is not None:
                value_ds.optimizer['params']['lr'] = v_lr
                value_ds.optimizer['params']['weight_decay'] = v_wd

            # Override gradient clipping if value_clip_grad_norm is set
            v_clip = self.train.value_clip_grad_norm if self.train.value_clip_grad_norm is not None else self.train.clip_grad_norm
            value_ds.gradient_clipping = float(v_clip)

            # in case value_lr, value_weight_decay, value_clip_grad_norm were empty
            # set them to the policy's lr, weight_decay, clip_grad_norm
            print(f"[Config] value_lr: {v_lr}, value_weight_decay: {v_wd}, value_clip_grad_norm: {v_clip}")
            print(f"[Config] policy_lr: {self.train.lr}, policy_weight_decay: {self.train.weight_decay}, policy_clip_grad_norm: {self.train.clip_grad_norm}")
            self.train.value_lr = v_lr
            self.train.value_weight_decay = v_wd
            self.train.value_clip_grad_norm = v_clip

            self.deepspeed_value = value_ds

def load_and_verify(method: str, input_yaml: str, experiment_id: str, rank: int, world_size: int | None = None):
    '''
        method: "sl", "rl", or "eval"
        input_yaml: path to the yaml file
        experiment_id: experiment identifier
        world_size: number of GPUs for SL training (optional for RL)
    '''
    if method not in ("sl", "rl", "eval", "cl"):
        raise ValueError(f"Unsupported method: '{method}'. Must be 'sl', 'rl', 'cl', or 'eval'.")

    try:
        with open(input_yaml, "r") as f:
            raw_config = yaml.safe_load(f)

        assert experiment_id, "experiment_id is required"
        raw_config["run"]["experiment_id"] = experiment_id
        # now verify the config
        config = Config(**raw_config)

        # Normalize empty strings to None to prevent truthy-but-invalid paths
        # (e.g., ref_model=" " is truthy but not a valid model path)
        if config.model and config.model.ref_model is not None and config.model.ref_model.strip() == "":
            config.model.ref_model = None
        if config.model and config.model.value_model is not None and config.model.value_model.strip() == "":
            config.model.value_model = None

        config.run.method = method

        # Common pre-training checks
        if method != "eval":
            # Model checks
            allowed_dtypes = {"float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}
            if config.model.dtype.lower() not in allowed_dtypes:
                raise ValueError(
                    f"model.dtype must be one of {sorted(allowed_dtypes)}, got '{config.model.dtype}'. "
                    f"'auto' is not allowed to avoid precision ambiguity."
                )

            if config.model.attn_implementation is not None and config.model.attn_implementation not in ("", "eager", "flash_attention_2"):
                raise ValueError(f"model.attn_implementation must be '', 'eager', or 'flash_attention_2', got '{config.model.attn_implementation}'")

            # Training loop checks
            if config.train.total_number_of_epochs < 1:
                raise ValueError(f"total_number_of_epochs must be >= 1, got {config.train.total_number_of_epochs}")

            if config.train.train_batch_size_per_gpu < 1:
                raise ValueError(f"train_batch_size_per_gpu must be >= 1, got {config.train.train_batch_size_per_gpu}")

            if config.train.gradient_accumulation_steps < 1:
                raise ValueError(f"gradient_accumulation_steps must be >= 1, got {config.train.gradient_accumulation_steps}")

            if config.train.val_batch_size_per_gpu < 1:
                raise ValueError(f"val_batch_size_per_gpu must be >= 1, got {config.train.val_batch_size_per_gpu}")

            if not (0.0 <= config.train.warmup_steps_ratio <= 1.0):
                raise ValueError(f"warmup_steps_ratio must be in [0.0, 1.0], got {config.train.warmup_steps_ratio}")

            # Data checks
            if not config.data.train_files_path:
                raise ValueError("data.train_files_path must be a non-empty list")

            if config.data.max_seq_len < 1:
                raise ValueError(f"data.max_seq_len must be >= 1, got {config.data.max_seq_len}")

            if config.data.num_workers < 0:
                raise ValueError(f"data.num_workers must be >= 0, got {config.data.num_workers}")

            # Checkpoint directory
            if not config.run.checkpoint_dir or config.run.checkpoint_dir.strip() == "":
                raise ValueError("run.checkpoint_dir must be specified for training")

            # PEFT
            if config.peft.use_peft:
                if config.peft.lora_rank is None or config.peft.lora_rank < 1:
                    raise ValueError(f"lora_rank must be >= 1 when use_peft=True, got {config.peft.lora_rank}")
                if config.peft.lora_alpha is None or config.peft.lora_alpha < 1:
                    raise ValueError(f"lora_alpha must be >= 1 when use_peft=True, got {config.peft.lora_alpha}")

        # Method-specific checks
        if method == "sl":
            if world_size is None:
                raise ValueError("world_size must be specified for SL training")

            if config.train.micro_batches_per_epoch is None or config.train.micro_batches_per_epoch < 1:
                raise ValueError(f"micro_batches_per_epoch must be >= 1 for SL, got {config.train.micro_batches_per_epoch}")

            if not config.data.val_files_path:
                raise ValueError("data.val_files_path must be a non-empty list for SL training")

            if config.train.micro_batches_per_epoch % config.train.gradient_accumulation_steps != 0:
                raise ValueError(f"micro_batches_per_epoch ({config.train.micro_batches_per_epoch}) must be divisible by gradient_accumulation_steps ({config.train.gradient_accumulation_steps})")

        elif method == "cl":
            if world_size is None:
                raise ValueError("world_size must be specified for CL training")

            if not config.model.ref_model:
                raise ValueError("model.ref_model must be specified for CL/DPO training")

            if config.train.micro_batches_per_epoch is None or config.train.micro_batches_per_epoch < 1:
                raise ValueError(f"micro_batches_per_epoch must be >= 1 for CL, got {config.train.micro_batches_per_epoch}")

            if not config.data.val_files_path:
                raise ValueError("data.val_files_path must be a non-empty list for CL training")

            if config.train.micro_batches_per_epoch % config.train.gradient_accumulation_steps != 0:
                raise ValueError(f"micro_batches_per_epoch ({config.train.micro_batches_per_epoch}) must be divisible by gradient_accumulation_steps ({config.train.gradient_accumulation_steps})")

            if config.train.alg_name == "dpo":
                if config.train.cl_beta is None:
                    raise ValueError("cl_beta must be specified for dpo")

                if config.train.cl_beta <= 0:
                    raise ValueError("cl_beta must be > 0 for dpo")

        elif method == "rl":
            if config.run.training_gpus is None or config.run.training_gpus < 1:
                raise ValueError(f"training_gpus must be >= 1 for RL training, got {config.run.training_gpus}")

            if config.run.rollout_gpus is None or config.run.rollout_gpus < 1:
                raise ValueError(f"rollout_gpus must be >= 1 for RL training, got {config.run.rollout_gpus}")

            world_size = config.run.training_gpus

            # [1-clip_low, 1+clip_high] requires non-negative values
            if config.train.clip_low < 0 or config.train.clip_high < 0:
                raise ValueError(f"clip_low and clip_high must be >= 0, got {config.train.clip_low} and {config.train.clip_high}.")

            if config.train.alg_name == "ppo":
                if config.train.tau is None or config.train.gamma is None or not config.model.value_model:
                    raise ValueError("tau and gamma and value_model must be specified for ppo")

                # value model in ppo lives only in ds training engine memory.
                # When using direct weight sync, only the policy is pushed to vllm and the value
                # net is only persisted via disk checkpoints.  If checkpoint_save_interval > 1,
                # a crash between saves loses the value model. However, we only warn users for this case.
                # Otherwise, if we do valueerror, it would be overly restrictive for large models.
                save_interval = config.run.checkpoint_save_interval
                if config.run.weight_sync_method == "direct" and save_interval is not None and save_interval > 1:
                    if rank ==0:
                        print(f"[Config] WARNING: PPO requires checkpoint_save_interval=1 when weight_sync_method='direct' "
                                     f"(got {save_interval}). The value model is not included in direct weight sync and "
                                     f"can only be recovered from disk checkpoints. Set checkpoint_save_interval=1 "
                                     f"or use weight_sync_method='disk'.")

            weight_sync_method = config.run.weight_sync_method
            if weight_sync_method is None:
                raise ValueError("weight_sync_method must be specified for rl training")

            if weight_sync_method not in ["direct", "disk", "nccl"]:
                raise ValueError("weight_sync_method must be 'direct', 'disk', or 'nccl'")

            overlap_enabled = config.overlap and config.overlap.enabled
            if weight_sync_method == "nccl" and not overlap_enabled:
                raise ValueError("weight_sync_method 'nccl' requires overlap.enabled=True "
                                 "(sync rollout engine does not support NCCL)")

            if weight_sync_method == "nccl":
                nccl_backend = config.run.nccl_sync_backend
                if nccl_backend not in ("nccl", "gloo"):
                    raise ValueError(f"nccl_sync_backend must be 'nccl' or 'gloo', got {nccl_backend!r}")

            max_tokens = config.rollout.max_tokens
            max_seq_len = config.data.max_seq_len
            if max_tokens is None or max_seq_len is None:
                raise ValueError("max_tokens and max_seq_len must be specified for rl training")

            if max_tokens > max_seq_len:
                raise ValueError("max_tokens must be < max_seq_len as max_seq_len equals to len(prompt + generation) and max_tokens equals to len(generation)")

            # Training step count
            if config.train.train_steps_per_epoch is None or config.train.train_steps_per_epoch < 1:
                raise ValueError(f"train_steps_per_epoch must be >= 1 for RL, got {config.train.train_steps_per_epoch}")

            # Rollout parameters
            if config.rollout.n_samples is None or config.rollout.n_samples < 1:
                raise ValueError(f"rollout.n_samples must be >= 1, got {config.rollout.n_samples}")

            if config.rollout.rollout_samples_per_epoch is None or config.rollout.rollout_samples_per_epoch < 1:
                raise ValueError(f"rollout.rollout_samples_per_epoch must be >= 1, got {config.rollout.rollout_samples_per_epoch}")

            if config.rollout.rollout_batch_size_per_gpu is None or config.rollout.rollout_batch_size_per_gpu < 1:
                raise ValueError(f"rollout.rollout_batch_size_per_gpu must be >= 1, got {config.rollout.rollout_batch_size_per_gpu}")

            if config.rollout.gpu_memory_utilization is not None and not (0.0 < config.rollout.gpu_memory_utilization <= 1.0):
                raise ValueError(f"rollout.gpu_memory_utilization must be in (0.0, 1.0], got {config.rollout.gpu_memory_utilization}")

            if config.rollout.temperature is not None and config.rollout.temperature < 0:
                raise ValueError(f"rollout.temperature must be >= 0, got {config.rollout.temperature}")

            if config.model.ref_model is not None and (config.train.kl_coeff == 0 or config.train.kl_coeff is None):
                raise ValueError(f"kl_coeff must be > 0 if model.ref is not None")

            if config.model.ref_model is None and (config.train.kl_coeff > 0):
                raise ValueError(f"model.ref must be not None if kl_coeff > 0")

            # Reward function
            if not config.reward or not config.reward.reward_func:
                raise ValueError("reward.reward_func must be specified for RL training")

            # TP must evenly divide rollout GPUs
            tp = config.rollout.tensor_parallel_size
            rg = config.run.rollout_gpus
            if tp and rg and rg % tp != 0:
                raise ValueError(f"rollout_gpus ({rg}) must be divisible by tensor_parallel_size ({tp}). "
                                 f"Currently {rg} % {tp} = {rg % tp}.")

            # Ray port
            if config.run.ray_master_port is None:
                raise ValueError("run.ray_master_port must be specified for RL training")


            # Timeout settings
            if config.run.init_timeout is None:
                raise ValueError("run.init_timeout must be specified for RL training")

            if config.run.rollout_timeout is None:
                raise ValueError("run.rollout_timeout must be specified for RL training")

            if config.run.train_step_timeout is None:
                raise ValueError("run.train_step_timeout must be specified for RL training")

            if config.run.save_timeout is None:
                raise ValueError("run.save_timeout must be specified for RL training")

            if config.run.sync_timeout is None:
                raise ValueError("run.sync_timeout must be specified for RL training")

            # Overlap mode validation
            if config.overlap and config.overlap.enabled:
                if config.overlap.max_lag is None or config.overlap.max_lag < 1:
                    raise ValueError(f"overlap.max_lag must be >= 1, got {config.overlap.max_lag}")

                if config.overlap.chunk_size is None or config.overlap.chunk_size < 1:
                    raise ValueError(f"overlap.chunk_size must be >= 1, got {config.overlap.chunk_size}")

                if config.run.weight_sync_method != "nccl":
                    raise ValueError(f"when overlap is enabled, weight_sync_method must be nccl")

                if config.overlap.ess_sync_threshold is None or not (0.0 < config.overlap.ess_sync_threshold <= 1.0):
                    raise ValueError(f"overlap.ess_sync_threshold must be in (0.0, 1.0], got {config.overlap.ess_sync_threshold}")

                # ESS-driven sync only works with P3O for now.
                # Other algorithms must use fixed_sync_interval for mid-epoch sync.
                alg_name = config.train.alg_name.lower() if config.train.alg_name else ""
                fixed_interval = config.overlap.fixed_sync_interval
                if alg_name != "p3o" and (fixed_interval is None or fixed_interval <= 0):
                    raise ValueError(f"overlap.fixed_sync_interval must be > 0 when overlap is enabled "
                                     f"with algorithm '{config.train.alg_name}' (only P3O supports ESS-driven sync). "
                                     f"Set it to train_steps_per_epoch ({config.train.train_steps_per_epoch}) "
                                     f"for one sync per epoch, or a smaller value for more frequent syncs.")

                steps_per_epoch = config.train.train_steps_per_epoch
                if fixed_interval is not None and fixed_interval > 0 and fixed_interval < steps_per_epoch:
                    if rank == 0:
                        print(f"[Config] WARNING: overlap.fixed_sync_interval={fixed_interval} < "
                              f"train.train_steps_per_epoch={steps_per_epoch}. Training will be truncated to "
                              f"{fixed_interval} step(s) per epoch before sync triggers. "
                              f"Set fixed_sync_interval >= train_steps_per_epoch to use all training steps.")

        # Validate batch_invariant GPU requirements (applies to RL and eval)
        if config.rollout and config.rollout.batch_invariant:
            try:
                if torch.cuda.is_available():
                    supported = False
                    for i in range(torch.cuda.device_count()):
                        cap = torch.cuda.get_device_capability(i)
                        # compute capability 9.0+ which is Hopper/Blackwell
                        # https://docs.vllm.ai/en/latest/features/batch_invariance/
                        if cap[0] >= 9:
                            supported = True
                            break

                    if not supported:
                        cap = torch.cuda.get_device_capability(0)
                        if rank == 0:
                            print(f"[Config] WARNING: batch_invariant=True requires NVIDIA GPUs with "
                                  f"compute capability >= 9.0 (H100, H200, B100, B200). "
                                  f"Detected compute capability: {cap[0]}.{cap[1]}. "
                                  f"Batch invariance will be enabled but reproducibility is not guaranteed.")
            except Exception as e:
                if rank == 0:
                    print(f"[Config] WARNING: Could not check GPU compute capability: {e}")
                    print(f"[Config] Batch invariance will be enabled but reproducibility is not guaranteed.")

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

            # save locally
            os.makedirs(f"{config.run.checkpoint_dir}/{experiment_id}", exist_ok=True)
            with open(f"{config.run.checkpoint_dir}/{experiment_id}/{config.run.method}_config.yaml", "w") as f:
                yaml.dump(config.model_dump(), f)
                f.flush()
                os.fsync(f.fileno())

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
    config = load_and_verify(method="cl", input_yaml="./configs/cl_args.yaml", experiment_id="run_4", rank=0, world_size=4)
