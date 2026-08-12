import yaml

def update_config(config: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and key in config:
            update_config(config[key], value)
        else:
            config[key] = value


def load_config(config_path: str, updates: dict) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    update_config(config, updates)
    return config
