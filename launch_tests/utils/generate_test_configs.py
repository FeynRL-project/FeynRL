import yaml
import os
import copy

BASE_RL = "launch_tests/configs/base/base_rl.yaml"
BASE_CL = "launch_tests/configs/base/base_cl.yaml"
BASE_SL = "launch_tests/configs/base/base_sl.yaml"

def merge_dicts(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base

def generate_configs():
    with open(BASE_RL, 'r') as f:
        base_rl = yaml.safe_load(f)
    
    with open(BASE_CL, 'r') as f:
        base_cl = yaml.safe_load(f)

    with open(BASE_SL, 'r') as f:
        base_sl = yaml.safe_load(f)

    # Specific subconfigs
    subconfigs = {
        "sgrpo": {"train": {"alg_name": "sgrpo"}},
        "cispo": {"train": {"alg_name": "cispo"}},
        "ppo": {
            "train": {
                "alg_name": "ppo",
                "tau": 0.95,
                "gamma": 0.99
            },
            "model": {
                "value_model": "google/gemma-3-1b-it"
            }
        },
        "dpo": {"train": {"alg_name": "dpo"}},
        "sft": {"train": {"alg_name": "sft"}}
    }

    stages = [0, 1, 2, 3]
    loggers = ["mlflow", "wandb"]

    for i, (alg, override) in enumerate(subconfigs.items()):
        if alg == "dpo":
            base = copy.deepcopy(base_cl)
        elif alg == "sft":
            base = copy.deepcopy(base_sl)
        else:
            base = copy.deepcopy(base_rl)
            
        final_config = merge_dicts(base, override)
        
        # Vary stage and logger based on index for variety
        # This ensures we test different combinations in a single "all" run
        stage = stages[i % len(stages)]
        logger = loggers[i % len(loggers)]
        
        final_config['deepspeed']['zero_optimization']['stage'] = stage
        final_config['run']['logger_type'] = logger
        final_config['run']['experiment_id'] = f"test_{alg}_stage{stage}_{logger}"

        # Logic for ref_model_offload_to_cpu:
        # load.py requires Stage 3 for offloading parameters.
        if 'model' in final_config:
            if stage < 3:
                final_config['model']['ref_model_offload_to_cpu'] = False
            else:
                final_config['model']['ref_model_offload_to_cpu'] = True
        
        output_path = f"launch_tests/configs/{alg}.yaml"
        with open(output_path, 'w') as f:
            yaml.safe_dump(final_config, f, default_flow_style=False)
        print(f"Generated {output_path} (Stage: {stage}, Logger: {logger})")

if __name__ == "__main__":
    generate_configs()
