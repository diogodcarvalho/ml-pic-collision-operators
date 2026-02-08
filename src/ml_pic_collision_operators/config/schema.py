import yaml
from typing import Literal, Optional

from ml_pic_collision_operators.config.utils import StrictBaseModel
from ml_pic_collision_operators.config.train import TrainConfig
from ml_pic_collision_operators.config.test import TestConfig


class MainConfig(StrictBaseModel):
    mode: Literal["train", "test"]
    train: Optional[TrainConfig] = None
    test: Optional[TestConfig] = None


def load_config(path: str) -> tuple[MainConfig, dict]:
    with open(path, "r") as f:
        config_raw = yaml.safe_load(f)
    if config_raw is None:
        raise Exception(f"Empty config file provided: {path}")
    return MainConfig.model_validate(config_raw), config_raw
