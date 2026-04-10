from pydantic import model_validator
from typing import Any, Literal, Generic, TypeVar

from ml_pic_collision_operators.config.utils import StrictBaseModel

FrequencyType = TypeVar("FrequencyType")


class TrainDataConfig(StrictBaseModel):
    folders: list[str]
    train_valid_ratio: float = 1.0
    conditioners: list[dict[str, Any]] | None = None
    include_time: bool = False


class TemporalUnrolligStageConfig(StrictBaseModel):
    unrolling_steps: int
    epochs: int
    lr: float | None = None


class ToggleWithFrequencyCallback(StrictBaseModel, Generic[FrequencyType]):
    enabled: bool = False
    frequency: FrequencyType | None = None

    @model_validator(mode="after")
    def check_frequency(cls, values):
        if values.enabled and values.frequency is None:
            raise ValueError("frequency must be set when enabled=True")
        return values


class FixedFrequencyCallback(StrictBaseModel):
    enabled: bool = False
    use_best_model: bool = False


class TrainCallbackConfig(StrictBaseModel):
    log_best_model: ToggleWithFrequencyCallback[
        Literal["always", "stage_end", "train_end"]
    ] = ToggleWithFrequencyCallback(enabled=True, frequency="stage_end")
    log_best_stage_model: FixedFrequencyCallback = FixedFrequencyCallback(enabled=True)
    plot_best_stage_model: FixedFrequencyCallback = FixedFrequencyCallback(enabled=True)
    plot_best_final_model: FixedFrequencyCallback = FixedFrequencyCallback(enabled=True)
    log_model: ToggleWithFrequencyCallback[int] = ToggleWithFrequencyCallback(
        enabled=False
    )
    plot_model: ToggleWithFrequencyCallback[int] = ToggleWithFrequencyCallback(
        enabled=False
    )
    plot_model_start: FixedFrequencyCallback = FixedFrequencyCallback(enabled=True)


class LossConfig(StrictBaseModel):
    name: Literal["mae", "mse"]
    mode: Literal["accumulated", "last"]
    reg_first_deriv: float = 0.0
    reg_second_deriv: float = 0.0


class TrainConfig(StrictBaseModel):
    random_seed: int = 42
    device: str = "cuda"
    mode: Literal["temporal_unrolling"]
    model_cls: str
    model_cls_kwargs: dict[str, Any] = {}
    data: TrainDataConfig
    dataset_cls: str
    dataset_cls_kwargs: dict[str, Any] = {}
    dataloader_cls: str | None = None
    dataloader_cls_kwargs: dict[str, Any] = {}
    optimizer_cls: str = "torch.optim.Adam"
    optimizer_cls_kwargs: dict[str, Any] = {}
    temporal_unrolling_stages: dict[str, TemporalUnrolligStageConfig]
    loss: LossConfig
    callbacks: TrainCallbackConfig | None = None
