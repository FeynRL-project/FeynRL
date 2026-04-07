import argparse
import configs.load as cfg

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./config/rl_args.yaml", help="config file")
    parser.add_argument("--experiment_id", type=str, default="run_1", help="experiment id")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging level")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a ckp to resume training. It must contain a CHECKPOINT_COMPLETE marker.")
    args = parser.parse_args()

    config = cfg.load_and_verify(method="rl",
                                 input_yaml=args.config_file,
                                 experiment_id=args.experiment_id,
                                 rank=0)

    # Dispatch to async (overlap) or sync engine based on config.
    if config.overlap and config.overlap.enabled:
        from run_rl_async import main
        main(args, config)

    else:
        from run_rl_sync import main
        main(args, config)
