import os
import json
import ray
from transformers import AutoTokenizer
from misc.utils import ray_get_with_timeout, get_experiment_dir_name

def setup_ray(ray_address):
    '''
       Initialize the Ray cluster and retrieve the driver node's IP address.
    '''
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)

    else:
        ray.init(ignore_reinit_error=True)

    try:
        # The IP is used as master_addr for deepspeed/pytorch distributed rendezvous.
        # rank 0 listens on this address and all other ranks connect to it to form the process group.
        master_addr = ray.util.get_node_ip_address()
    except Exception:
        # Fallback to localhost if we cannot get the IP (e.g. single node without network)
        print("Warning: Could not get master address, using localhost. This is fine for single-node but will fail for multi-node.")
        master_addr = "127.0.0.1"

    return master_addr

def load_tokenizer(model_name, trust_remote_code=False, rank=0):
    '''
       Load tokenizer from huggingface.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=trust_remote_code)

    # if pad token is not present, we use eos token as pad token
    if tokenizer.pad_token_id is None:
        print("Warning: Pad token is not present, using eos token as pad token")
        if getattr(tokenizer, 'eos_token', None) is not None:
            # prefer explicit token if available
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else:
            # fallback to eos token id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer

def load_checkpoint_for_resume(resume_path,
                               training_engines,
                               rollout_engines,
                               weight_sync_method,
                               logger,
                               sync_timeout,
                               save_timeout,
                               sync_fn=None,
                               refresh_fn=None):
    '''
       Resume training from a previously saved checkpoint.
       Loads deepspped engine state such as model weights, optimizer, scheduler, RNG
       on all training engine ranks, then syncs the policy to rollout engines.
       sync_fn: callable(training_engines, rollout_engines, version, logger, sync_timeout) -> bool
       refresh_fn: callable(rollout_engines, updated_policy_path, version, logger, sync_timeout)
       Returns (start_epoch, policy_version, global_step).
    '''
    # Validate checkpoint completeness
    marker = os.path.join(resume_path, "CHECKPOINT_COMPLETE")
    if not os.path.exists(marker):
        raise FileNotFoundError(f"Checkpoint at {resume_path} is incomplete (missing CHECKPOINT_COMPLETE marker). "
                                f"This checkpoint was likely interrupted during writing and is not safe to resume from.")

    # Load training metadata
    state_path = os.path.join(resume_path, "training_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"training_state.json not found in {resume_path}. "
                                f"Cannot determine epoch/version to resume from.")

    with open(state_path) as f:
        training_state = json.load(f)

    epoch = training_state['epoch']
    policy_version = training_state['policy_version']
    global_step = training_state['global_step']

    # Validate world size matches as ds optimizer state is partitioned
    # by world size and cannot be resharded across a different number of GPUs.
    if 'training_gpus' in training_state:
        saved_gpus   = training_state['training_gpus']
        current_gpus = len(training_engines)
        if saved_gpus != current_gpus:
            raise ValueError(f"Checkpoint was saved with {saved_gpus} training GPUs but "
                             f"current run uses {current_gpus}. DeepSpeed ZeRO optimizer state "
                             f"is partitioned by world size and cannot be resharded.")

    logger.info(f"[Resume] Loading checkpoint from {resume_path} "
                f"(epoch={epoch}, policy_version={policy_version}, global_step={global_step})")

    # Load ds engine state on all training actors (model + optimizer + scheduler + RNG)
    engine_state_dir = os.path.join(resume_path, "ds_engine")
    load_futures = [engine.load_engine_state.remote(engine_state_dir) for engine in training_engines]

    ray_get_with_timeout(refs=load_futures,
                         timeout=save_timeout,
                         description="load engine state for resume",
                         logger=logger)
    logger.info("[Resume] Engine state loaded on all training ranks")

    # Sync resumed policy to rollout engines so they match the training policy
    success = False
    if weight_sync_method == "direct" and sync_fn is not None:
        try:
            success = sync_fn(training_engines=training_engines,
                              rollout_engines=rollout_engines,
                              version=policy_version,
                              logger=logger,
                              sync_timeout=sync_timeout)
        except Exception as e:
            logger.warning(f"[Resume] Direct sync raised {e}, falling back to disk refresh")
            success = False

        if not success:
            logger.warning("[Resume] Direct sync failed, falling back to disk refresh")

    if not success and refresh_fn is not None:
        refresh_fn(rollout_engines=rollout_engines,
                   updated_policy_path=resume_path,
                   version=policy_version,
                   logger=logger,
                   sync_timeout=sync_timeout)

    logger.info(f"[Resume] Rollout engines synced to policy v{policy_version}")

    # Return next epoch (resume continues from epoch+1)
    return epoch + 1, policy_version, global_step

def save_checkpoint(epoch,
                    version,
                    global_step,
                    tokenizer,
                    training_engines,
                    checkpoint_dir,
                    experiment_id,
                    rank,
                    logger,
                    save_timeout,
                    save_ds_engine=True):
    '''
       Save model checkpoint. This must run on all ranks for ZeRO-3.
       Writes hf compatible weights and for vllm, ds engine state
       optimizer/scheduler for resume, training metadata, and a
       CHECKPOINT_COMPLETE marker for crash-safe.

       Args:
           save_ds_engine: When False, skip saving the DeepSpeed engine state
                           (optimizer/scheduler/RNG). The optimizer state is
                           typically several times larger than the raw weights,
                           so this saves significant disk space per checkpoint.
                           Set to False when training resume is not needed.
    '''
    # Note if multi-node cluster is used, checkpoint_dir must be on a shared
    # filesystem such as NFS, Lustre, etc. or each node writes to isolated local disk
    # and the checkpoint is silently incomplete.
    gpu_nodes = [n for n in ray.nodes() if n.get("Alive") and n.get("Resources", {}).get("GPU", 0) > 0]
    if len(gpu_nodes) > 1:
        logger.warning(f"Multi-node cluster detected ({len(gpu_nodes)} GPU nodes). "
                       f"Ensure checkpoint_dir is on a shared filesystem (NFS, Lustre, etc.) "
                       f"or checkpoints may be silently incomplete.")
    tag = f"iter{epoch+1:06d}_v{version:06d}"
    model_path = get_experiment_dir_name(output_dir=checkpoint_dir, tag=tag, experiment_id=experiment_id)
    logger.info(f"[Epoch {epoch+1}] Saving checkpoint to {model_path}")

    # Create the checkpoint dir on the driver first so all actors can write to it
    # immediately, avoiding mkdir races on slow NFS mounts.
    if rank == 0:
        os.makedirs(model_path, exist_ok=True)

    # 1. save HF-compatible weights and this must run on ALL ranks for zero-3 correctness.
    save_futures = []
    for engine in training_engines:
        save_futures.append(engine.save_checkpoint.remote(output_dir=model_path, tag=tag))

    # Wait for all saves to complete
    ray_get_with_timeout(refs=save_futures,
                         timeout=save_timeout,
                         description=f"checkpoint save (epoch {epoch+1})",
                         logger=logger)

    # 2. save DeepSpeed engine state so we can resume training later.
    # Directory creation happens inside save_engine_state on DS rank 0.
    engine_state_dir = os.path.join(model_path, "ds_engine")
    state_futures = [engine.save_engine_state.remote(engine_state_dir, save_ds_engine) for engine in training_engines]
    ray_get_with_timeout(refs=state_futures,
                         timeout=save_timeout,
                         description=f"engine state save (epoch {epoch+1})",
                         logger=logger)

    # 3. Driver-side files: Save the tokenizer only after all engine saves
    # finish to avoid racing with rank 0 writing config.json on shared filesystems.
    if rank == 0:
        tokenizer.save_pretrained(model_path)

        # Training metadata for resume
        training_state = {'epoch': epoch,
                          'policy_version': version,
                          'global_step': global_step,
                          'training_gpus': len(training_engines),
                         }
        state_file = os.path.join(model_path, "training_state.json")
        with open(state_file, "w") as f:
            json.dump(training_state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # On load, only checkpoints with this file are considered complete and safe to resume from.
        marker_file = os.path.join(model_path, "CHECKPOINT_COMPLETE")
        with open(marker_file, "w") as f:
            f.write("")
            f.flush()
            os.fsync(f.fileno())

    logger.info(f"[Epoch {epoch+1}] Checkpoint saved: {model_path}")
    return model_path