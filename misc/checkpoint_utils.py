import os
import json
import random
import shutil
import numpy as np
import torch
import torch.distributed
import deepspeed
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards

def gather_params_for_save(module, rank):
    '''
        Gather all parameters from a deepspeed wrapped module.
        Returns {name: cpu_tensor} on rank 0, empty dict on others.
        Must be called on ALL ranks for ZeRO-3 collective correctness.
    '''
    params = []
    names = []
    for name, param in module.named_parameters():
        params.append(param)
        names.append(name)

    is_zero3 = any(hasattr(p, 'ds_id') for p in params) if params else False
    state_dict = {}

    if is_zero3:
        # Gather one param at a time to avoid oon. Each param is materialized
        # on rank 0, copied to cpu, then released before the next param is gathered.
        for name, param in zip(names, params):
            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                if rank == 0:
                    state_dict[name] = param.data.cpu().clone()
    else:
        if rank == 0:
            for name, param in zip(names, params):
                state_dict[name] = param.data.cpu().clone()

    return state_dict

def save_state_dict_sharded(state_dict, output_dir, max_shard_size="5GB"):
    '''
        Saves a state dict with automatic sharding for large models.
        For small models that fit in a single shard, saves as model.safetensors.
        For large models, we split it into model-00001-of-NNNNN.safetensors shards
        and write model.safetensors.index.json so HF or vllm can load them.
    '''
    state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size)

    for filename, tensor_names in state_dict_split.filename_to_tensors.items():
        shard = {name: state_dict[name] for name in tensor_names}
        save_file(shard, os.path.join(output_dir, filename))

    if state_dict_split.is_sharded:
        index = {"metadata": state_dict_split.metadata,
                 "weight_map": state_dict_split.tensor_to_filename}

        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
            f.flush()
            os.fsync(f.fileno())


def merge_peft_state_dict(raw_state_dict, lora_alpha, lora_rank):
    '''
        Merge LoRA adapter weights into base model weights and remap to
        original HuggingFace parameter names so vllm can load them.

        PeftModel names follow the pattern:
          base weight:  base_model.model.{hf_name}
          lora_A:       base_model.model.{module_path}.lora_A.default.weight
          lora_B:       base_model.model.{module_path}.lora_B.default.weight
          scaling:      alpha / r

        Merged weight = base_weight + (alpha / r) * lora_B @ lora_A
    '''
    peft_prefix = "base_model.model."
    merged = {}

    # 1. Collect LoRA A/B pairs keyed by their module path
    lora_a = {}
    lora_b = {}
    base_weights = {}

    for name, tensor in raw_state_dict.items():
        if ".lora_A." in name:
            module_path = name.split(".lora_A.")[0]
            lora_a[module_path] = tensor

        elif ".lora_B." in name:
            module_path = name.split(".lora_B.")[0]
            lora_b[module_path] = tensor

        else:
            base_weights[name] = tensor

    # 2. Merge lora into base weights
    scaling = lora_alpha / lora_rank

    for module_path in lora_a:
        if module_path not in lora_b:
            raise RuntimeError(f"merge_peft_state_dict: found lora_A but no lora_B for {module_path}. "
                               f"State dict is corrupt or incomplete.")

        A = lora_a[module_path]  # [r, D]
        B = lora_b[module_path]  # [H, r]
        delta = (B @ A) * scaling  # [H, D]

        # Find the corresponding base weight
        base_key = module_path + ".base_layer.weight"
        if base_key not in base_weights:
            base_key = module_path + ".weight"

        if base_key in base_weights:
            base_weights[base_key] = base_weights[base_key] + delta.to(base_weights[base_key].dtype)

        else:
            raise RuntimeError(f"merge_peft_state_dict: no base weight found for {module_path}, "
                               f"LoRA delta would be silently dropped.")

    # 3. Remap names: strip peft_prefix and .base_layer suffix
    for peft_name, tensor in base_weights.items():
        if peft_name.startswith(peft_prefix):
            hf_name = peft_name[len(peft_prefix):]
        else:
            hf_name = peft_name

        hf_name = hf_name.replace(".base_layer.", ".")

        merged[hf_name] = tensor

    return merged


def barrier_with_error_check(succeeded, device, label):
    '''
        Error-propagating replacement for torch.distributed.barrier().

        Each rank publishes a 0/1 "I succeeded" flag and we all_reduce(MIN).
        If any rank failed, the reduced flag is 0 and every rank raises
        immediately, breaking the deadlock.
    '''
    if not torch.distributed.is_initialized():
        if not succeeded:
            raise RuntimeError(f"[{label}] Checkpoint operation failed on this process")
        return

    flag = torch.tensor([1 if succeeded else 0], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)

    if flag.item() == 0:
        raise RuntimeError(f"[{label}] Checkpoint operation failed on at least one rank "
                           f"(this is rank {torch.distributed.get_rank()}), all ranks aborting "
                           f"to prevent deadlock")


def resume_from_checkpoint(resume_path, model_engine, world_size, logger, zero_stage=None, model_dtype=None, use_peft=None, ref_model_name=None):
    '''
        Resume training from a previously saved checkpoint.
        Validates the checkpoint, loads ds engine state (weights + optimizer +
        scheduler), and restores per-rank RNG states.
        Returns (start_epoch, global_step).
    '''
    marker = os.path.join(resume_path, "CHECKPOINT_COMPLETE")
    if not os.path.exists(marker):
        raise FileNotFoundError(f"Checkpoint at {resume_path} is incomplete (missing CHECKPOINT_COMPLETE marker). "
                                f"This checkpoint was likely interrupted during writing and is not safe to resume from.")

    state_path = os.path.join(resume_path, "training_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"training_state.json not found in {resume_path}. "
                                f"Cannot determine epoch/step to resume from.")

    with open(state_path) as f:
        training_state = json.load(f)

    saved_epoch = training_state['epoch']
    global_step = training_state['global_step']

    # Validate world size matches
    if 'world_size' in training_state:
        saved_ws = training_state['world_size']
        if saved_ws != world_size:
            raise ValueError(f"Checkpoint was saved with {saved_ws} GPUs but current run uses {world_size}. "
                             f"DeepSpeed ZeRO optimizer state is partitioned by world size and cannot be resharded.")

    # Validate zeros stage matches, i.e., optimizer state format differs between stages
    if zero_stage is not None and 'zero_stage' in training_state:
        saved_zero = training_state['zero_stage']
        if saved_zero is not None and saved_zero != zero_stage:
            raise ValueError(f"Checkpoint was saved with ZeRO stage {saved_zero} but current config uses stage {zero_stage}. "
                             f"Optimizer state format is incompatible across ZeRO stages.")

    elif zero_stage is not None and 'zero_stage' not in training_state:
        logger.warning("[Resume] Checkpoint has no zero_stage metadata, skipping ZeRO stage validation.")

    # Validate model precision matches
    if model_dtype is not None and 'model_dtype' in training_state:
        saved_dtype = training_state['model_dtype']
        if saved_dtype is not None and saved_dtype != model_dtype:
            raise ValueError(f"Checkpoint was saved with dtype={saved_dtype} but current config uses dtype={model_dtype}. "
                             f"Precision mismatch can corrupt optimizer states.")

    elif model_dtype is not None and 'model_dtype' not in training_state:
        logger.warning("[Resume] Checkpoint has no model_dtype metadata, skipping precision validation.")

    # Validate PEFT mode matches, i.e., parameter structure differs
    if use_peft is not None and 'use_peft' in training_state:
        saved_peft = training_state['use_peft']
        if saved_peft is not None and saved_peft != use_peft:
            raise ValueError(f"Checkpoint was saved with use_peft={saved_peft} but current config uses use_peft={use_peft}. "
                             f"Parameter names and optimizer state structure are incompatible.")

    elif use_peft is not None and 'use_peft' not in training_state:
        logger.warning("[Resume] Checkpoint has no use_peft metadata, skipping PEFT validation.")

    # Validate ref model matches
    if ref_model_name is not None and 'ref_model_name' in training_state:
        saved_ref = training_state['ref_model_name']
        if saved_ref is not None and saved_ref != ref_model_name:
            raise ValueError(f"Checkpoint was saved with ref_model={saved_ref} but current config uses "
                             f"ref_model={ref_model_name}. ref model must be the same across resume.")

    elif ref_model_name is not None and 'ref_model_name' not in training_state:
        logger.warning("[Resume] Checkpoint has no ref_model_name metadata, skipping ref model validation.")

    logger.info(f"[Resume] Loading checkpoint from {resume_path} "
                f"(completed epoch {saved_epoch + 1}, global_step={global_step})")

    # Load DS engine state (model weights + optimizer + scheduler + RNG)
    engine_state_dir = os.path.join(resume_path, "ds_engine")
    load_path, client_state = model_engine.load_checkpoint(engine_state_dir, tag="policy")
    if load_path is None:
        raise FileNotFoundError(f"DeepSpeed failed to load policy checkpoint from {engine_state_dir}/policy. "
                                f"Directory may be missing or corrupt.")
    logger.info(f"[Resume] Policy engine state loaded from {load_path}")

    # Restore per-rank RNG states
    if client_state:
        if 'rng_python' in client_state:
            random.setstate(client_state['rng_python'])

        if 'rng_numpy' in client_state:
            np.random.set_state(client_state['rng_numpy'])
        if 'rng_torch_cpu' in client_state:
            torch.random.set_rng_state(client_state['rng_torch_cpu'])
        if 'rng_torch_cuda' in client_state and torch.cuda.is_available():
            torch.cuda.set_rng_state(client_state['rng_torch_cuda'])
        
        logger.info(f"[Resume] Restored RNG states")

    start_epoch = saved_epoch + 1
    logger.info(f"[Resume] Will resume from epoch {start_epoch + 1}")
    return start_epoch, global_step


def save_training_checkpoint(epoch, global_step, model_engine, tokenizer, model_path, peft_config, rank, world_size, logger, label, zero_stage=None, model_dtype=None, ref_model_name=None, save_ds_engine=True):
    '''
        Save a full training checkpoint: HF-compatible weights, model config,
        generation config, tokenizer, DeepSpeed engine state (optimizer/scheduler/RNG),
        training metadata, and a CHECKPOINT_COMPLETE crash-safety marker.
        Must be called on ALL ranks for ZeRO-3 correctness.
        Args:
            peft_config: object with .use_peft, .lora_alpha, .lora_rank attributes.
    '''
    # 1. Save HF-compatible weights (all ranks must participate for ZeRO-3 gather)
    raw_sd = gather_params_for_save(model_engine.module, rank)
    save_ok = True
    try:
        if rank == 0:
            os.makedirs(model_path, exist_ok=True)
            if peft_config.use_peft:
                merged_sd = merge_peft_state_dict(raw_state_dict=raw_sd,
                                                  lora_alpha=peft_config.lora_alpha,
                                                  lora_rank=peft_config.lora_rank)
                save_state_dict_sharded(state_dict=merged_sd, output_dir=model_path)
                print(f"[Alg:{label.upper()}][Rank {rank}] Saved merged PEFT model")

            else:
                save_state_dict_sharded(state_dict=raw_sd, output_dir=model_path)
                print(f"[Alg:{label.upper()}][Rank {rank}] Saved non-PEFT model")

    except Exception as e:
        save_ok = False
        logger.error(f"[Epoch {epoch+1}] Error saving model weights: {e}")

    barrier_with_error_check(succeeded=save_ok, device=model_engine.device, label=f"{label}_save_weights")

    # 2. Save model config, generation config, and tokenizer on rank 0
    save_ok = True
    try:
        if rank == 0:
            # When PEFT is active, model_engine.module is a PeftModel.
            # We must save the underlying base model's config so vllm can
            # load the merged checkpoint without peft specific fields.
            model_module = model_engine.module
            if peft_config.use_peft and hasattr(model_module, 'get_base_model'):
                base_config = model_module.get_base_model().config

            elif hasattr(model_module, 'config'):
                base_config = model_module.config

            else:
                base_config = None

            if base_config is not None:
                base_config.save_pretrained(model_path)
                logger.info(f"[Epoch {epoch+1}] Model config  saved")

            else:
                logger.info(f"[Epoch {epoch+1}] Could not find model config to save for model")
            
            # Save generation_config.json if available. This is needed by some hf pipelines or vllm.
            gen_cfg = getattr(model_module, 'generation_config', None)
            # For peft, generation_config lives on the base model.
            if gen_cfg is None and hasattr(model_module, 'get_base_model'):
                gen_cfg = getattr(model_module.get_base_model(), 'generation_config', None)

            if gen_cfg is not None:
                gen_cfg.save_pretrained(model_path)
                logger.info(f"[Epoch {epoch+1}] Generation config saved")
            
            tokenizer.save_pretrained(model_path)
            logger.info(f"[Epoch {epoch+1}] Tokenizer saved")

    except Exception as e:
        save_ok = False
        logger.error(f"[Epoch {epoch+1}] Error saving config/tokenizer: {e}")

    barrier_with_error_check(succeeded=save_ok, device=model_engine.device, label=f"{label}_save_config")

    # 3. Save DeepSpeed engine state (optimizer, scheduler, RNG) for resume
    if save_ds_engine:
        engine_state_dir = os.path.join(model_path, "ds_engine")
        if rank == 0:
            os.makedirs(engine_state_dir, exist_ok=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        client_state = {'rng_python': random.getstate(),
                        'rng_numpy': np.random.get_state(),
                        'rng_torch_cpu': torch.random.get_rng_state()}
        if torch.cuda.is_available():
            client_state['rng_torch_cuda'] = torch.cuda.get_rng_state()

        model_engine.save_checkpoint(engine_state_dir, tag="policy", client_state=client_state)
    else:
        logger.info(f"[Epoch {epoch+1}] Skipping DeepSpeed engine state save (save_ds_engine=False)")

    # 4. Training metadata and completion marker on rank 0
    if rank == 0:
        training_state = {'epoch': epoch,
                          'global_step': global_step,
                          'world_size': world_size,
                          'zero_stage': zero_stage,
                          'model_dtype': model_dtype,
                          'use_peft': peft_config.use_peft,
                          'peft_type': getattr(peft_config, 'peft_type', None) if peft_config.use_peft else None,
                          'ref_model_name': ref_model_name}

        state_file = os.path.join(model_path, "training_state.json")
        with open(state_file, "w") as f:
            json.dump(training_state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        marker_file = os.path.join(model_path, "CHECKPOINT_COMPLETE")
        with open(marker_file, "w") as f:
            f.write("")
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"[Epoch {epoch+1}] Checkpoint saved: {model_path}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def cleanup_incomplete_checkpoints(experiment_dir, rank, logger):
    '''
        Remove checkpoint directories that are missing the CHECKPOINT_COMPLETE marker.
        These are leftovers from crashed runs and are not safe to resume from.
        Must be called on all ranks (rank 0 does the deletion, then barrier).
    '''
    if rank == 0 and os.path.isdir(experiment_dir):
        for entry in os.listdir(experiment_dir):
            ckpt_path = os.path.join(experiment_dir, entry)

            if os.path.isdir(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "CHECKPOINT_COMPLETE")):
                logger.warning(f"Removing incomplete checkpoint: {ckpt_path}")
                shutil.rmtree(ckpt_path, ignore_errors=True)

    # Barrier must always be reached by all ranks, even when the directory
    # doesn't exist on some ranks, for example nfs propagation delay, to avoid deadlock.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
