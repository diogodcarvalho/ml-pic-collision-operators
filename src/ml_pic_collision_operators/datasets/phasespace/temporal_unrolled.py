import numpy as np
from pathlib import Path

from .base import BaseDataset
from ml_pic_collision_operators.datasets.dataset_utils import DatasetItem


class TemporalUnrolledDataset(BaseDataset):

    def __init__(
        self,
        folder: str | Path,
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
            step_size=step_size,
            extra_cells=extra_cells,
        )
        self.temporal_unroll_steps = temporal_unroll_steps

    def __len__(self) -> int:
        return self.i_end - self.i_start - self.step_size * self.temporal_unroll_steps

    def _load_inputs(self, idx: int) -> np.ndarray:
        return self._load_file(idx, normalized=True)

    def _load_targets(self, idx: int) -> np.ndarray:
        return np.stack(
            [
                self._load_file(idx + (ts + 1) * self.step_size, normalized=True)
                for ts in range(self.temporal_unroll_steps)
            ],
            axis=0,
            dtype=self._dtype,
        )

    def __getitem__(self, idx: int) -> DatasetItem:
        inputs = self._load_inputs(idx)
        targets = self._load_targets(idx)
        return DatasetItem(inputs=inputs, targets=targets, dt=self.dt)
