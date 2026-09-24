import logging
import wandb
import os

from cs336_basics.utils.config_utils import TrainingConfig

def setup_logging(config: TrainingConfig) -> None:
    # Setup logging configuration
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config.log_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")
    run = wandb.init(project="cs336_basics",
                     config=config.model_dump(),
                     name=f"run_nlayer_{config.num_layers}_nheads_{config.num_heads}_{wandb.util.generate_id()}",
    )
    return run
    



def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
    