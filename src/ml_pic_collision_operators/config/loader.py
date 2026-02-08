import yaml
from ml_pic_collision_operators.config.schema import MainConfig


def load_config(path: str) -> tuple[MainConfig, dict]:
    with open(path, "r") as f:
        config_raw = yaml.safe_load(f)
    if config_raw is None:
        raise Exception(f"Empty config file provided: {path}")
    return MainConfig.model_validate(config_raw), config_raw
