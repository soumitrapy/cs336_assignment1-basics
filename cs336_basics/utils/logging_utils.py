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

def setup_wandb(config: TrainingConfig) -> tuple[wandb.Run, wandb.Artifact]:
    run = wandb.init(project=config.project,
                     entity=config.entity,
                     name=config.run_name,
                     config=config.model_dump(),
    )
    run.define_metric("train/step")
    run.define_metric("train/loss", step_metric="train/step")
    run.define_metric("train/perplexity", step_metric="train/step")
    run.define_metric("train/grad_norm", step_metric="train/step")
    run.define_metric("train/learning_rate", step_metric="train/step")
    run.define_metric("val/loss", step_metric="train/step")
    run.define_metric("val/perplexity", step_metric="train/step")

    artifact = wandb.Artifact(name = f"{config.run_name}_checkpoints",
                             type = "model",
    )
    return run, artifact

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
    