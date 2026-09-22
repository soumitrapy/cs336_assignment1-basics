import yaml
from pydantic import BaseModel, model_validator

class TrainingConfig(BaseModel):
    #------ Dataset configuration ------
    train_path: str = "data/TinyStoriesV2-GPT4-train-tokenized.bin"
    valid_path: str = "data/TinyStoriesV2-GPT4-valid-tokenized.bin"
    #------ Model configuration ------
    vocab_size: int = 50257
    context_len: int = 1024
    d_model: int = 1600
    num_heads: int = 25
    d_ff: int = 4288
    num_layers: int = 48
    rope_theta: float = 10000.0
    #------ Optimization configuration ------
    lr: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    #------ LR Scheduler configuration ------
    min_lr: float = 1e-5
    warmup_step: int = 1e2
    final_step: int = 1e4
    #------ Training configuration ------
    n_steps: int = 10000
    batch_size: int = 3

    #------ Validation configuration ------
    val_interval: int = 1000
    val_steps: int = 100
    #full_val_interval: int = 10000
    #full_val_steps: int = 1000

    #------ Checkpoint configuration ------
    checkpoint_dir: str = "checkpoints/testing"
    initial_checkpoint: str | None = None
    checkpoint_interval: int = 1000
    #------ Logging configuration ------
    log_interval: int = 100
    logging_path: str = "logs/training.log"
    #------ Miscellaneous configuration ------
    device: str = "cpu"
    seed: int = 42

    @model_validator(mode="before")
    @classmethod
    def auto_cast_types(cls, data):
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if field_name not in data:
                continue
            value = data[field_name]
            if isinstance(value, str):
                try:
                    target_type = field_info.annotation
                    if target_type is float:
                        data[field_name] = float(value)
                    elif target_type is int:
                        data[field_name] = int(float(value))
                except (ValueError, TypeError):
                    raise ValueError(f"Cannot convert value '{value}' of field '{field_name}' to {target_type}")
        return data

def update_config(config: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and key in config:
            update_config(config[key], value)
        else:
            config[key] = value
def load_config(config_path: str, updates: dict = dict()) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    update_config(config, updates)
    return config
