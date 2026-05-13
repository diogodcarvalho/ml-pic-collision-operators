import numpy as np
from pathlib import Path
from typing import Any

from .temporal_unrolled import TemporalUnrolledDataset
from ml_pic_collision_operators.datasets.dataset_utils import DatasetItem


class TemporalUnrolledwConditionersDataset(TemporalUnrolledDataset):

    def __init__(
        self,
        folder: str | Path,
        conditioners: dict[str, Any] | None = None,
        include_time: bool = False,
        i_start: int = 0,
        i_end: int = -1,
        step_size: int = 1,
        extra_cells: int = 0,
        temporal_unroll_steps: int = 1,
    ):
        super().__init__(
            folder=folder,
            i_start=i_start,
            i_end=i_end,
            extra_cells=extra_cells,
            step_size=step_size,
            temporal_unroll_steps=temporal_unroll_steps,
        )
        self.include_time = include_time
        self.conditioners = conditioners
        if conditioners is None:
            self.conditioners_array = np.array([], dtype=self._dtype)
        else:
            self.conditioners_array = np.stack(
                [float(v) for k, v in conditioners.items()], dtype=self._dtype
            )

    @property
    def conditioners_size(self):
        return self.conditioners_array.shape[-1] + self.include_time

    def __getitem__(self, idx: int) -> DatasetItem:
        inputs = self._load_inputs(idx)
        targets = self._load_targets(idx)

        conditioners = self.conditioners_array
        if self.include_time:
            time_value = np.array([self.dt * idx], dtype=self._dtype)
            conditioners = np.concatenate([conditioners, time_value], axis=0)

        return DatasetItem(
            inputs=inputs, targets=targets, dt=self.dt, conditioners=conditioners
        )
