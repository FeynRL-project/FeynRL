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
    world_size: int
    distributed_training_strategy: str
    seed: int

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

    ###############
    # general training  loop arguments
    ###############
    # Here, an "epoch" is defined by a fixed number of training steps, not a full sweep of the dataset.
    # Each epoch processes: train_steps_per_epoch * global_batch_size samples.
    # We define epochs this way to control how different datasets are mixed during training and
    # to have more control over the training process.
    total_number_of_epochs: int
    train_steps_per_epoch: int
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
    train_dnames: list[str]
    train_ratios: dict[str, float]
    train_files_path: str
    val_files_path: str
    num_workers: int
    max_seq_len: int
    prompt_key: str
    answer_key: str

class Model(BaseModel):
    '''
        Information like model_name, ref_model_path, dtype, etc.
    '''
    model_config = ConfigDict(extra='forbid')
    name: str
    dtype: str
    ref_model: str
    ref_model_device: str
    trust_remote_code: bool
    use_cache: bool
    model_class: str
    attn_implementation: str
    gradient_checkpointing: bool

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

class InferenceEngine(BaseModel):
    '''
        Everything related to inference goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    name: str

class Config(BaseModel):
    '''
        This is the main configuration class for the experiment where it puts all the sub-configurations
        together to form a complete configuration for the experiment.
    '''
    model_config = ConfigDict(extra='forbid')
    run: Run
    train: Train
    model: Model
    data: Data
    deepspeed: DeepSpeed
    inference_engine: InferenceEngine

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sync_deepspeed_config(self):
            """
            Sync DeepSpeed config from train/model without duplicating YAML fields.
            """
            # 1 — Batch Sizes
            self.deepspeed.train_micro_batch_size_per_gpu = self.train.train_batch_size_per_gpu
            self.deepspeed.gradient_accumulation_steps = self.train.gradient_accumulation_steps

            # Explicitly calculate and set train_batch_size for DeepSpeed logging/sanity check
            self.deepspeed.train_batch_size = self.train.train_batch_size_per_gpu * self.train.gradient_accumulation_steps * self.run.world_size

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
            # If using ZeRO-3 with Offload, we typically want "DeepSpeedCPUAdam"
            # If no offload, we want "FusedAdam" or "AdamW"
            ds_opt_type = "AdamW"
            if "adam" in self.train.optimizer_name.lower():
                # Check if offload is enabled in YAML
                zero_stage = self.deepspeed.zero_optimization.get("stage", 0)
                offload_opt = self.deepspeed.zero_optimization.get("offload_optimizer", {})

                if zero_stage == 3 and offload_opt.get("device") == "cpu":
                    ds_opt_type = "DeepSpeedCPUAdam"

                else:
                    ds_opt_type = "FusedAdam" # Generally faster on GPU

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
                self.deepspeed.scheduler = {
                    "type": self.train.lr_scheduler,
                    "params": {
                        "total_num_steps": self.train.total_number_of_epochs * self.train.train_steps_per_epoch,
                        "warmup_min_lr": 0,
                        "warmup_max_lr": self.train.lr,
                        "warmup_num_steps": int(self.train.total_number_of_epochs * self.train.train_steps_per_epoch * self.train.warmup_steps_ratio)
                    }
                }
            else:
                raise ValueError(f"Unsupported scheduler: {self.train.lr_scheduler}")

            # 6 — ZeRO Defaults (Ensure robust ZeRO-3 settings)
            if self.deepspeed.zero_optimization is None:
                self.deepspeed.zero_optimization = {}

            # Force crucial ZeRO-3 setting if Stage 3 is active
            if self.deepspeed.zero_optimization.get("stage") == 3:
                # This ensures you don't get 500 small files when you save
                if "stage3_gather_16bit_weights_on_model_save" not in self.deepspeed.zero_optimization:
                    self.deepspeed.zero_optimization["stage3_gather_16bit_weights_on_model_save"] = True

def load_and_verify(input_yaml: str, experiment_id: str, world_size: int):
    '''
        input_yaml: path to the yaml file
    '''
    try:
        with open(input_yaml, "r") as f:
            raw_config = yaml.safe_load(f)

        # now verify the config
        config = Config(**raw_config)
        # Update Run details
        config.run.experiment_id = experiment_id
        config.run.world_size = world_size

        # Sync AFTER updating world_size
        config.sync_deepspeed_config()

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
    config = load_and_verify("./configs/sl_args.yaml", experiment_id="run_1", world_size=1)