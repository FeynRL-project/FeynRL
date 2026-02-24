import os
import atexit
import logging
from misc.trackers import get_tracker

def setup_logging(rank: int, log_level: str = "INFO", exp_name: str = "") -> logging.Logger:
    '''
        Setup logging configuration. Only rank 0 logs to console to avoid duplicate messages.
    '''
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    # Prevent logs from bubbling up to root logger where Ray/DeepSpeed might double-print them
    logger.propagate = False

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

def setup_tracker(config, rank: int):
    '''
        Setup experiment tracking using ExperimentTracker interface
        so we can switch between different tracking backends easily.
    '''
    tracker = get_tracker(config, rank)
    if tracker is not None:
        # Best-effort cleanup on exceptions/early exits. finish() is idempotent.
        atexit.register(tracker.finish)
    return tracker
