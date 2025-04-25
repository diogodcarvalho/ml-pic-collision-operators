import numpy as np
from pathlib import Path

from .base import BaseDataset
from typing import Any


class BasewConditionersDataset(BaseDataset):

    def __init__(
        self,
        folder: str | Path,
        conditioners: dict[str, Any],
        i_start: int = 0,
        i_end: int = -1,
        step_size: int = 1,
    ):
        super().__init__(
            folder=folder,
            i_start=i_start,
            i_end=i_end,
            step_size=step_size,
        )
        self.conditioners = conditioners
        self.conditioners_array = np.stack(
            [float(v) for k, v in conditioners.items()], dtype=self._dtype
        )

    @property
    def conditioners_size(self):
        return self.conditioners_array.shape[-1]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        inputs = self._load_file(idx, normalized=True)
        targets = self._load_file(idx + self.step_size, normalized=True)

        return (inputs, targets, self.dt, self.conditioners_array)
