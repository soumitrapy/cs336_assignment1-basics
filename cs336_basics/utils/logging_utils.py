import logging
import wandb

from cs336_basics.utils.config_utils import TrainingConfig

def setup_logging(config: TrainingConfig) -> None:
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(config.logging_path),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")

    # Setup WandB configuration
    wandb.init(project="cs336_basics", 
               config=config.model_dump(),
               name=f"run_nlayer_{config.num_layers}_nheads_{config.num_heads}_{wandb.util.generate_id()}",
               )
    



def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
    