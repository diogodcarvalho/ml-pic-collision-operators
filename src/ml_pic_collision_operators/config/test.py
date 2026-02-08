from enum import Enum
from pydantic import Field
from typing import Literal, Dict, Any, List

from ml_pic_collision_operators.config.utils import StrictBaseModel


class MLflowModelConfig(StrictBaseModel):
    type: Literal["mlflow"]
    experiment_name: str
    run_name: str
    fname: str
    change_params: Dict[str, Any] | None = None


class HDFModelConfig(StrictBaseModel):
    type: Literal["hdf"]
    hdf_file: str
    params: Dict[str, Any] | None = None
    change_params: Dict[str, Any] | None = None


class TestDataConfig(StrictBaseModel):
    folders: list[str]
    step_size: float = 1.0
    conditioners: List[Dict[str, Any]] | None = None
    include_time: bool = False


class TestMetric(str, Enum):
    mse = "mse"
    l1 = "l1"
    l2 = "l2"
    l1_norm = "l1_norm"
    l2_norm = "l2_norm"


class TestConfig(StrictBaseModel):
    mode: Literal["rollout"]
    data: TestDataConfig
    model: MLflowModelConfig | HDFModelConfig = Field(discriminator="type")
    video: bool = False
    video_fps: int = 10
    metrics: set[TestMetric] = set(TestMetric)
