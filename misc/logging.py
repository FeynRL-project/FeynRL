import os
import logging
import mlflow

def setup_logging(rank: int, log_level: str = "INFO", exp_name: str = "") -> logging.Logger:
    '''
        Setup logging configuration. Only rank 0 logs to console to avoid duplicate messages.
    '''
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Format with timestamp and rank
    formatter = logging.Formatter(
        fmt=f"[%(asctime)s][Rank {rank}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (only rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def setup_mlflow(config, tracking_uri: str, rank: int):
    '''
        Setup MLflow tracking. Only rank 0 logs to MLflow.
    '''
    if rank != 0:
        return None

    # Set tracking URI
    #tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Set experiment name
    experiment_name = config.run.project_name
    mlflow.set_experiment(experiment_name)

    # Start run
    run = mlflow.start_run(run_name=config.run.experiment_id)

    # Log base config parameters (common to SL and RL)
    mlflow.log_params({
        "alg_name": config.train.alg_name,
        "model_name": config.model.name,
        "learning_rate": config.train.lr,
        "train_batch_size_per_gpu": config.train.train_batch_size_per_gpu,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
        "total_epochs": config.train.total_number_of_epochs,
        "seed": config.run.seed,
    })

    # Log method-specific step config
    if config.run.method == "sl":
        mlflow.log_params({
            "micro_batches_per_epoch": config.train.micro_batches_per_epoch,
        })
    elif config.run.method == "rl":
        mlflow.log_params({
            "train_steps_per_epoch": config.train.train_steps_per_epoch,
            "n_samples": config.rollout.n_samples,
            "max_tokens": config.rollout.max_tokens,
            "kl_coeff": config.train.kl_coeff,
            "clip_low": config.train.clip_low,
            "clip_high": config.train.clip_high,
            "entropy_coeff": config.train.entropy_coeff,
            "training_gpus": config.run.training_gpus,
            "rollout_gpus": config.run.rollout_gpus,
        })

    return run