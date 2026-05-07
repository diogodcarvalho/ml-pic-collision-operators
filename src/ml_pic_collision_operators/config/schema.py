from typing import Literal

from pydantic import model_validator

from ml_pic_collision_operators.config.utils import StrictBaseModel
from ml_pic_collision_operators.config.train import TrainConfig
from ml_pic_collision_operators.config.test import TestConfig


class MainConfig(StrictBaseModel):
    mode: Literal["train", "test"]
    train: TrainConfig | None = None
    test: TestConfig | None = None

    @model_validator(mode="after")
    def check_mode_config_present(self):
        if self.mode == "train" and self.train is None:
            raise ValueError("train config must be provided when mode='train'")
        if self.mode == "test" and self.test is None:
            raise ValueError("test config must be provided when mode='test'")
        return self
