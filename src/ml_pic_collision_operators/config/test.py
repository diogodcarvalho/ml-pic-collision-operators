from enum import Enum
from pydantic import Field, model_validator
from typing import Literal, Any

from ml_pic_collision_operators.config.utils import StrictBaseModel


class MLflowModelConfig(StrictBaseModel):
    type: Literal["mlflow"]
    experiment_name: str
    run_name: str
    fname: str
    change_params: dict[str, Any] | None = None


class HDFModelConfig(StrictBaseModel):
    type: Literal["hdf"]
    hdf_file: str
    params: dict[str, Any] | None = None
    change_params: dict[str, Any] | None = None


class TestDataConfig(StrictBaseModel):
    folders: list[str]
    step_size: int = 1
    conditioners: list[dict[str, Any]] | None = None
    include_time: bool = False

    @model_validator(mode="after")
    def check_fields(self):
        if not self.folders:
            raise ValueError("folders must not be empty")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.conditioners is not None and len(self.conditioners) != len(self.folders):
            raise ValueError("conditioners and folders must have the same length")
        return self


class TestMetric(str, Enum):
    mse = "mse"
    l1 = "l1"
    l2 = "l2"
    l1_norm = "l1_norm"
    l2_norm = "l2_norm"


class PlotSliceConfig(StrictBaseModel):
    axis: int
    index: int


class TestConfig(StrictBaseModel):
    mode: Literal["rollout"]
    data: TestDataConfig
    model: MLflowModelConfig | HDFModelConfig = Field(discriminator="type")
    video: bool = False
    video_fps: int = 10
    metrics: set[TestMetric] = set(TestMetric)
    plot_slice: PlotSliceConfig | None = None
