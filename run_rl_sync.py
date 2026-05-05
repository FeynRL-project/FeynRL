import os
import atexit
import numpy as np
import importlib
import ray
import time
import shutil

import configs.load as cfg
# imports local methods, classes, etc.
from misc.utils import load_algorithm, ray_get_with_timeout, set_random_seeds
from rollouts.replay_buffer import ReplayBuffer
from misc.logging import setup_logging, setup_tracker
from misc.setup_rl import load_tokenizer, save_checkpoint, load_checkpoint_for_resume, setup_ray
from core.rl_engines import (Algorithm_Registry,
                            create_training_engines,
                            create_rollout_engines,
                            create_rollout_dataloader,
                            collect_rollouts,
                            prepare_training_batches,
                            shard_and_put,
                            run_training_step,
                            sync_weights_direct,
                            refresh_rollout_engine,
                            reinit_nccl_weight_sync_group)
from rollouts.external_policy import ExternalPolicyEngine

def run_epoch_sync(epoch, training_engines, rollout_engines, rollout_dataloader,
                   replay_buffer, policy_version, rollout_policy_version, global_step,
                   n_samples, train_batch_size, steps_per_epoch, seed,
                   rollout_timeout, train_step_timeout, tracker, logger,
                   external_engines=None):
    '''
        Sequential epoch: [collect_rollouts] -> [prepare_training_batches] -> [train]]
    '''
    # 1. Reset replay buffer
    replay_buffer.reset()

    # 2. all engines must finish before we proceed. collect_rollouts is blocking call.
    logger.info(f"[Epoch {epoch+1}] Starting rollout generation...")
    rollout_dataloader.batch_sampler.set_epoch(epoch)
    rollout_metrics = collect_rollouts(dataloader=rollout_dataloader,
                                       rollout_engines=rollout_engines,
                                       epoch=epoch,
                                       policy_version=policy_version,
                                       replay_buffer=replay_buffer,
                                       n_samples=n_samples,
                                       logger=logger,
                                       rollout_timeout=rollout_timeout,
                                       external_engines=external_engines)

    # 3. Prepare training batches
    logger.info(f"[Epoch {epoch+1}] Replay buffer has {len(replay_buffer)} samples")
    train_start_time = time.time()
    num_engines      = len(training_engines)
    # shuffles the replay buffer globally and creates training batches
    train_batches_padded = prepare_training_batches(replay_buffer=replay_buffer,
                                                    batch_size=train_batch_size,
                                                    num_engines=num_engines,
                                                    seed=seed,
                                                    epoch=epoch)
    samples_per_engine = len(replay_buffer) // num_engines
    micro_per_engine   = len(train_batches_padded) // num_engines
    logger.info(f"[Epoch {epoch+1}] Training: "
                f"{len(replay_buffer)} replay samples / {num_engines} training engines "
                f"= {samples_per_engine} samples/engine / bsz={train_batch_size} "
                f"= {micro_per_engine} micro-batches/engine, "
                f"{steps_per_epoch} pass(es) over replay buffer")

    # while each engine gets same micro-batches per step, we shuffle them inside train_step of each engine
    shard_refs = shard_and_put(train_batches_padded, num_engines=num_engines)

    # 4. Training loop
    epoch_metrics = {}
    for step in range(steps_per_epoch):
        train_metrics = run_training_step(engines=training_engines,
                                          shard_refs=shard_refs,
                                          logger=logger,
                                          train_step_timeout=train_step_timeout)
        # collect the metrics
        for k, v in train_metrics.items():
            epoch_metrics.setdefault(k, []).append(v)

        global_step += 1
        if step % 10 == 0:
            metric_str = ", ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            logger.info(f"[Epoch {epoch+1}][Step {step+1}/{steps_per_epoch}] {metric_str}")

        if tracker:
            tracker.log_metrics({f"train/{k}": v for k, v in train_metrics.items()}, step=global_step)

    # 5. Post-training
    policy_version += 1

    return {'rollout_metrics': rollout_metrics,
            'epoch_metrics': epoch_metrics,
            'global_step': global_step,
            'policy_version': policy_version,
            'rollout_policy_version': rollout_policy_version,
            'train_step_count': steps_per_epoch,
            'train_time': time.time() - train_start_time,
            'sync_performed': False}

def main(args, config):
    '''
        This is the main entry point for the rl_sync training process.
    '''
    ########
    # 1. Miscellaneous setups
    ########
    # remember that main_rl.py is an orchestrator script,
    # not a distributed worker, so rank is always 0 here.
    rank = 0

    # Setup logging
    logger = setup_logging(rank=rank, log_level=args.log_level)
    logger.info(f"Starting RL training...")

    set_random_seeds(seed=config.run.seed)

    # setup remote experiment tracker
    tracker = setup_tracker(config=config, rank=rank)
    logger.info(f"Config loaded. experiment_id: {config.run.experiment_id}")

    # number of gpus for training which is used by deepspeed
    training_gpus = config.run.training_gpus
    # number of gpus for rollout generation which is used by vllm
    rollout_gpus  = config.run.rollout_gpus

    ########
    # 2. initialize ray
    ########
    logger.info(f"Initializing Ray ...")
    master_addr = setup_ray(ray_address=config.run.ray_address)
    logger.info(f"Ray initialized. Master address: {master_addr}")
    # registers ray.shutdown as a function to be called
    # automatically when the python process exits. Without it, orphaned ray
    # processes can linger and hold onto gpu memory after the script dies.
    atexit.register(ray.shutdown)

    cluster_gpus = ray.cluster_resources().get("GPU", 0)
    needed_gpus = training_gpus + rollout_gpus
    if needed_gpus > cluster_gpus:
        raise ValueError(f"Need {needed_gpus} GPUs (training={training_gpus} + rollout={rollout_gpus}) "
                         f"but Ray cluster only has {int(cluster_gpus)} GPUs")

    ########
    # 3. Initialize training engine
    ########
    logger.info(f"Loading training algorithm: {config.train.alg_name}")
    alg_class = load_algorithm(config.train.alg_name, Algorithm_Registry)

    training_engines = create_training_engines(params=config,
                                              alg=alg_class,
                                              world_size=training_gpus,
                                              master_addr=master_addr,
                                              master_port=config.run.ray_master_port)

    assert len(training_engines) == training_gpus, "Number of training engines does not match number of training gpus"
    logger.info(f"Created {len(training_engines)} training engine runners")

    # Synchronization barrier to prevent deepspeed rendezvous hang
    # wait for all training actors to finish initialization before proceeding
    logger.info("Waiting for all training engines to initialize...")

    init_timeout = config.run.init_timeout
    ready_checks = [engine.is_ready.remote() for engine in training_engines]
    ray_get_with_timeout(refs=ready_checks,
                         timeout=init_timeout,
                         description="training engine initialization",
                         logger=logger)
    logger.info("All training engines ready!")

    ########
    # 4. load tokenizer
    ########
    logger.info(f"Loading tokenizer from {config.model.name}")
    tokenizer = load_tokenizer(model_name=config.model.name,
                               trust_remote_code=config.model.trust_remote_code,
                               rank=rank)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, Pad token ID: {tokenizer.pad_token_id}")

    ########
    # 5. Initialize rollout engines
    ########
    logger.info("Setting up rollout engines...")
    reward_func_name = config.reward.reward_func if config.reward.reward_func else None
    if reward_func_name:
        reward_module = importlib.import_module("rewards." + reward_func_name)
        reward_fnc = getattr(reward_module, "compute_score")
        logger.info(f"Using reward function: {reward_func_name}")

    else:
        raise ValueError("Reward function not specified")

    ########
    # 5b. Load policy-mixing config and build external engines (sync engine only)
    ########
    mixing_cfg = cfg.load_mixing_config(
        config.policy_mixing.configs if config.policy_mixing else None
    )
    external_engines = []
    n_main_samples = config.rollout.n_samples

    if mixing_cfg.policies:
        n_ext_total = sum(p.n_samples for p in mixing_cfg.policies)
        n_main_samples = config.rollout.n_samples - n_ext_total
        if n_main_samples < 1:
            raise ValueError(
                f"Sum of external policy n_samples ({n_ext_total}) must be "
                f"< rollout.n_samples ({config.rollout.n_samples})"
            )
        for spec in mixing_cfg.policies:
            external_engines.append(ExternalPolicyEngine(
                base_url=spec.base_url,
                model=spec.model,
                n_samples=spec.n_samples,
                tokenizer=tokenizer,
                reward_func=reward_fnc,
                reward_broadcast=config.reward.broadcast,
                eos_id=tokenizer.eos_token_id,
                max_tokens=spec.max_tokens if spec.max_tokens is not None else config.rollout.max_tokens,
                temperature=spec.temperature if spec.temperature is not None else config.rollout.temperature,
                max_seq_len=config.data.max_seq_len,
                api_key=spec.api_key,
                max_concurrent=spec.max_concurrent,
                request_timeout=spec.request_timeout,
            ))
            logger.info(
                f"[PolicyMixing] External engine: model={spec.model} "
                f"base_url={spec.base_url} n_samples={spec.n_samples}"
            )
        logger.info(
            f"[PolicyMixing] {len(external_engines)} external engine(s), "
            f"main engine n_samples reduced: {config.rollout.n_samples} -> {n_main_samples}"
        )

    rollout_engines = create_rollout_engines(
        params=config,
        reward_fnc=reward_fnc,
        eos_id=tokenizer.eos_token_id,
        n_samples_override=n_main_samples if external_engines else None,
    )
    num_rollout_engines = len(rollout_engines)
    logger.info(f"Created {num_rollout_engines} rollout engines with TP={config.rollout.tensor_parallel_size}")

    ########
    # 6. Weight sync method (NCCL group initialized after resume in section 9b)
    ########
    weight_sync_method = config.run.weight_sync_method
    nccl_port = None
    nccl_sync_backend = None

    if weight_sync_method == "nccl":
        nccl_port = config.run.nccl_sync_port if config.run.nccl_sync_port else config.run.ray_master_port + 100
        nccl_sync_backend = config.run.nccl_sync_backend

    else:
        logger.info(f"Weight sync method is {weight_sync_method}")

    ########
    # 7. load the rollout dataloader
    ########
    logger.info(f"Loading rollout dataloader from {config.data.train_files_path}")
    rollout_dataloader = create_rollout_dataloader(params=config,
                                                  tokenizer=tokenizer,
                                                  num_rollout_engines=num_rollout_engines,
                                                  samples_per_epoch=config.rollout.rollout_samples_per_epoch)

    logger.info(f"Rollout dataloader with {len(rollout_dataloader)} batches/machine, "
                f"n_samples={config.rollout.n_samples} per prompt")

    # replay buffer size = rollout_samples_per_epoch (prompts) * n_samples (completions per prompt)
    replay_buffer = ReplayBuffer(pad_token_id=tokenizer.pad_token_id,
                                 max_seq_len=config.data.max_seq_len,
                                 )
    logger.info(f"Replay buffer initialized (max_seq_len={config.data.max_seq_len})")

    ########
    # 8. Some variables initialization
    ########
    number_of_epochs = config.train.total_number_of_epochs
    steps_per_epoch  = config.train.train_steps_per_epoch
    checkpoint_save_interval = config.run.checkpoint_save_interval if config.run.checkpoint_save_interval is not None else 1

    # Timeout settings (seconds) for ray.get() calls
    rollout_timeout    = config.run.rollout_timeout
    train_step_timeout = config.run.train_step_timeout
    save_timeout = config.run.save_timeout
    sync_timeout = config.run.sync_timeout

    ########
    # 9. Resume from checkpoint if requested and clean up incomplete checkpoint directories
    ########
    start_epoch = 0
    global_step = 0
    policy_version = 0
    rollout_policy_version = 0

    if args.resume_from:
        start_epoch, policy_version, global_step = load_checkpoint_for_resume(resume_path=args.resume_from,
                                                                              training_engines=training_engines,
                                                                              rollout_engines=rollout_engines,
                                                                              weight_sync_method=weight_sync_method,
                                                                              logger=logger,
                                                                              sync_timeout=sync_timeout,
                                                                              save_timeout=save_timeout,
                                                                              sync_fn=sync_weights_direct,
                                                                              refresh_fn=refresh_rollout_engine)
        rollout_policy_version = policy_version
        logger.info(f"Resuming from epoch {start_epoch+1}, policy_version={policy_version}, global_step={global_step}")

    ########
    # 9b. Initialize NCCL weight sync group (after resume, so engines are fresh)
    ########
    if weight_sync_method == "nccl":
        nccl_world_size, nccl_gname = reinit_nccl_weight_sync_group(training_engines=training_engines,
                                                                    rollout_engines=rollout_engines,
                                                                    master_addr=master_addr,
                                                                    nccl_port=nccl_port,
                                                                    tp_size=int(config.rollout.tensor_parallel_size),
                                                                    logger=logger,
                                                                    init_timeout=config.run.init_timeout,
                                                                    backend=nccl_sync_backend)
        logger.info(f"Weight sync: NCCL (port={nccl_port}, world_size={nccl_world_size}) with NCCL group name {nccl_gname}")

    # Clean up incomplete checkpoint directories from previous crashed runs.
    # Only directories missing the CHECKPOINT_COMPLETE marker are removed.
    experiment_dir = os.path.join(config.run.checkpoint_dir, config.run.experiment_id)
    if os.path.isdir(experiment_dir):
        for entry in os.listdir(experiment_dir):
            ckpt_path = os.path.join(experiment_dir, entry)

            if os.path.isdir(ckpt_path) and not os.path.exists(os.path.join(ckpt_path, "CHECKPOINT_COMPLETE")):
                logger.warning(f"Removing incomplete checkpoint: {ckpt_path}")
                shutil.rmtree(ckpt_path, ignore_errors=True)

    ########
    # 10. General logging printout before training-loop
    ########
    model_info       = ray.get(training_engines[0].get_model_info.remote())
    total_params     = model_info['total_params']
    trainable_params = model_info['trainable_params']
    frozen_params    = model_info['frozen_params']

    logger.info("=" * 50)
    logger.info(f"Starting training: {number_of_epochs} epochs, {steps_per_epoch} steps/epoch")
    logger.info(f"Training GPUs: {training_gpus}, Rollout GPUs: {rollout_gpus}")
    if model_info['peft_enabled']:
        logger.info(f"Model: {config.model.name} | PEFT: {model_info['peft_type']} | "
                    f"params: {total_params:,} total, {trainable_params:,} peft ({100*trainable_params/total_params:.2f}%), "
                    f"{frozen_params:,} frozen")

    else:
        logger.info(f"Model: {config.model.name} | PEFT: off | "
                    f"params: {total_params:,} total, {trainable_params:,} trainable")

    if 'value_total_params' in model_info:
        logger.info(f"Value model: {config.model.value_model or config.model.name} | "
                    f"params: {model_info['value_total_params']:,} total, {model_info['value_trainable_params']:,} trainable")

    logger.info(f"Weight sync method: {weight_sync_method}")
    logger.info(f"checkpoint_save_interval: {checkpoint_save_interval}")
    if args.resume_from:
        logger.info(f"Resuming from: {args.resume_from} (epoch {start_epoch+1}/{number_of_epochs})")

    logger.info("=" * 50)

    ########
    # 11. Training and rollout loop
    ########
    entire_training_start_time = time.time()

    for epoch in range(start_epoch, number_of_epochs):
        epoch_start_time = time.time()
        is_last_epoch = (epoch == number_of_epochs - 1)

        # Run epoch
        result = run_epoch_sync(epoch=epoch,
                                training_engines=training_engines,
                                rollout_engines=rollout_engines,
                                rollout_dataloader=rollout_dataloader,
                                replay_buffer=replay_buffer,
                                policy_version=policy_version,
                                rollout_policy_version=rollout_policy_version,
                                global_step=global_step,
                                n_samples=config.rollout.n_samples,
                                train_batch_size=config.train.train_batch_size_per_gpu,
                                steps_per_epoch=steps_per_epoch,
                                seed=config.run.seed,
                                rollout_timeout=rollout_timeout,
                                train_step_timeout=train_step_timeout,
                                tracker=tracker,
                                logger=logger,
                                external_engines=external_engines or None)

        # Unpack result
        global_step            = result['global_step']
        policy_version         = result['policy_version']
        rollout_metrics        = result['rollout_metrics']
        rollout_policy_version = result['rollout_policy_version']

        # Log rollout metrics
        time_str = f"time={rollout_metrics['rollout_time']:.2f}s"
        if 'rollout_time_with_overlap' in rollout_metrics:
            time_str += f" (wall_time={rollout_metrics['rollout_time_with_overlap']:.2f}s)"

        logger.info(f"[Epoch {epoch + 1}] Rollout complete: {rollout_metrics['total_samples_generated']} samples, "
                    f"avg_reward={rollout_metrics['avg_reward']:.4f}, reward_std={rollout_metrics['reward_std']:.4f}, "
                    f"reward_min={rollout_metrics['reward_min']:.4f}, reward_max={rollout_metrics['reward_max']:.4f}, "
                    f"frac_positive_reward={rollout_metrics['frac_positive_reward']:.4f}, "
                    f"avg_response_len={rollout_metrics['avg_response_len']:.1f}, "
                    f"response_len_std={rollout_metrics['response_len_std']:.1f}, "
                    f"min_response_len={rollout_metrics['min_response_len']:.1f}, "
                    f"max_response_len={rollout_metrics['max_response_len']:.1f}, "
                    f"truncated_ratio={rollout_metrics['truncated_ratio']:.4f}, "
                    f"seq_truncated_ratio={rollout_metrics['seq_truncated_ratio']:.4f}, "
                    f"eos_ratio={rollout_metrics['eos_ratio']:.4f}, "
                    f"mean_logprob={rollout_metrics['mean_logprob']:.4f}, "
                    f"unique_response_ratio={rollout_metrics['unique_response_ratio']:.4f}, "
                    f"{time_str}, tps={rollout_metrics['tokens_per_sec']:.2f}")

        if "_main" in rollout_metrics:
            m = rollout_metrics["_main"]
            logger.info(
                f"[Epoch {epoch + 1}] Rollout [main_policy]: {m['total_samples_generated']} samples, "
                f"avg_reward={m['avg_reward']:.4f}, reward_std={m['reward_std']:.4f}, "
                f"avg_response_len={m['avg_response_len']:.1f}, "
                f"truncated_ratio={m['truncated_ratio']:.4f}, "
                f"eos_ratio={m['eos_ratio']:.4f}, "
                f"mean_logprob={m['mean_logprob']:.4f}, "
                f"unique_response_ratio={m['unique_response_ratio']:.4f}"
            )
            e = rollout_metrics["_ext"]
            logger.info(
                f"[Epoch {epoch + 1}] Rollout [ext_policy]: {e['total_samples_generated']} samples, "
                f"avg_reward={e['avg_reward']:.4f}, reward_std={e['reward_std']:.4f}, "
                f"avg_response_len={e['avg_response_len']:.1f}, "
                f"truncated_ratio={e['truncated_ratio']:.4f}, "
                f"eos_ratio={e['eos_ratio']:.4f}, "
                f"mean_logprob={e['mean_logprob']:.4f}, "
                f"unique_response_ratio={e['unique_response_ratio']:.4f}"
            )

        if tracker:
            rollout_log = {
                "rollout/" + k: v
                for k, v in rollout_metrics.items()
                if not k.startswith("_")
            }
            if "_main" in rollout_metrics:
                rollout_log.update({"rollout/main/" + k: v for k, v in rollout_metrics["_main"].items()})
                rollout_log.update({"rollout/ext/"  + k: v for k, v in rollout_metrics["_ext"].items()})
            tracker.log_metrics(rollout_log, step=global_step)

        # Log training summary
        epoch_avg = {k: np.mean(v) for k, v in result['epoch_metrics'].items()}

        train_stats = ray.get(training_engines[0].get_training_stats.remote())
        current_lr = train_stats.get('lr', 0.0)
        gpu_mem_gb = train_stats.get('gpu_peak_mem_gb', 0.0)

        logger.info(f"[Epoch {epoch+1}] Training complete: {result['train_step_count']} steps, "
                    f"time={result['train_time']:.2f}s, "
                    f"avg_loss={epoch_avg.get('loss_total', 0.0):.4f}, "
                    f"avg_kl_ref={epoch_avg.get('kl_ref', 0.0):.4f}, "
                    f"avg_approx_kl={epoch_avg.get('approx_kl', 0.0):.6f}, "
                    f"lr={current_lr:.2e}, gpu_peak_mem={gpu_mem_gb:.2f}GB")

        if tracker:
            tracker.log_metrics({"train/epoch_time_sec": result['train_time'],
                                 "train/lr": current_lr,
                                 "train/gpu_peak_mem_gb": gpu_mem_gb,
                                }, step=global_step)

        # End-of-epoch weight sync
        sync_success = result['sync_performed']
        # End-of-epoch weight sync. Sync mode only supports "direct" or "disk"
        # (config validator forbids "nccl" because the non-Async vLLM engine
        # has no NCCL weight sync methods). For "disk", we fall through to the
        # disk save+refresh path below.
        if not sync_success and not is_last_epoch and weight_sync_method == "direct":
            logger.info(f"[Epoch {epoch+1}] Syncing weights directly to rollout engines "
                        f"(v{rollout_policy_version} -> v{policy_version})...")

            try:
                sync_success = sync_weights_direct(training_engines=training_engines,
                                                   rollout_engines=rollout_engines,
                                                   version=policy_version,
                                                   logger=logger,
                                                   sync_timeout=sync_timeout)
            except Exception as e:
                logger.warning(f"[Epoch {epoch+1}] Direct sync raised {e}, falling back to disk")
                sync_success = False

            if sync_success:
                rollout_policy_version = policy_version
                logger.info(f"[Epoch {epoch+1}] Direct sync successful")

        # Save checkpoint
        should_save_disk = (checkpoint_save_interval > 0 and
                           ((epoch + 1) % checkpoint_save_interval == 0 or is_last_epoch))

        # save to disk when:
        # 1. using disk-based sync (always need disk save for rollout refresh).
        # 2. all sync methods failed (need disk as last resort for rollout refresh).
        # 3. periodic/final checkpoint save.
        no_sync_succeeded = not result['sync_performed'] and not sync_success
        need_disk_for_rollout = (weight_sync_method == "disk" and not is_last_epoch) or \
                                (no_sync_succeeded and weight_sync_method in ("nccl", "direct", "disk") and not is_last_epoch)

        if need_disk_for_rollout or should_save_disk or is_last_epoch:
            model_path = save_checkpoint(epoch=epoch,
                                         version=policy_version,
                                         global_step=global_step,
                                         tokenizer=tokenizer,
                                         training_engines=training_engines,
                                         checkpoint_dir=config.run.checkpoint_dir,
                                         experiment_id=config.run.experiment_id,
                                         rank=rank,
                                         logger=logger,
                                         save_timeout=save_timeout)
            logger.info(f"[Epoch {epoch+1}] Saved disk checkpoint at {model_path}")

        # Disk-based rollout refresh
        if need_disk_for_rollout and not is_last_epoch:
            logger.info(f"[Epoch {epoch+1}] Refreshing rollout engines with new policy (version {policy_version})...")
            refresh_rollout_engine(rollout_engines=rollout_engines,
                                   updated_policy_path=model_path,
                                   version=policy_version,
                                   logger=logger,
                                   sync_timeout=sync_timeout)
            rollout_policy_version = policy_version
            logger.info(f"[Epoch {epoch+1}] Rollout engines refreshed")

            # Refresh destroys EngineCore workers, invalidating NCCL groups.
            # Re-initialize so subsequent NCCL syncs don't hang.
            if weight_sync_method == "nccl":
                logger.info(f"[Epoch {epoch+1}] Re-initializing NCCL weight sync group after engine refresh...")
                reinit_nccl_weight_sync_group(training_engines=training_engines,
                                              rollout_engines=rollout_engines,
                                              master_addr=master_addr,
                                              nccl_port=nccl_port,
                                              tp_size=int(config.rollout.tensor_parallel_size),
                                              logger=logger,
                                              init_timeout=config.run.init_timeout,
                                              backend=nccl_sync_backend)

        # NCCL sync metrics
        if tracker:
            tracker.log_metrics({"nccl/policy_version": policy_version,
                                 "nccl/rollout_policy_version": rollout_policy_version,
                                 "nccl/policy_lag": policy_version - rollout_policy_version,
                                 "nccl/sync_success": 1 if sync_success else 0,
                                }, step=global_step)

        epoch_time = time.time() - epoch_start_time
        logger.info(f"[Epoch {epoch+1}] Complete! Total epoch time: {epoch_time:.2f}s")
        logger.info("=" * 50)

    ########
    # 12. Cleanup
    ########
    if tracker:
        tracker.finish()

    entire_training_time = time.time() - entire_training_start_time
    logger.info(f"Training completed successfully! Total time: {entire_training_time:.2f}s ({entire_training_time/3600:.2f}h)")

    # Tear down NCCL weight sync groups if they were initialized
    if weight_sync_method == "nccl":
        logger.info("[Cleanup] Closing NCCL weight sync groups...")
        try:
            ray.get(training_engines[0].close_weight_nccl_group.remote())

        except Exception as e:
            logger.warning(f"[Cleanup] Failed to close training NCCL group: {e}")

        for eng in rollout_engines:
            try:
                ray.get(eng.close_nccl_group.remote())
            except Exception:
                pass

    # Clean up process groups before ray tears down actors.
    shutdown_futures = [engine.shutdown.remote() for engine in training_engines]
    try:
        ray.get(shutdown_futures, timeout=30)

    except Exception:
        pass

    ray.shutdown()
    logger.info("Done!")