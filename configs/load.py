from IPython.core.interactiveshell import PickleShareDB
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
    lr: float = Field(..., gt=0)
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

class DeepSpeed(BaseModel):
    '''
        Everything related to DeepSpeed goes here.
    '''
    model_config = ConfigDict(extra='forbid')
    config_path: str

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

def load_and_verify(input_yaml: str):
    '''
        input_yaml: path to the yaml file
    '''
    try:
        with open(input_yaml, "r") as f:
            raw_config = yaml.safe_load(f)

        # now verify the config
        config = Config(**raw_config)
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
    config = load_and_verify("./config/dummy.yaml")