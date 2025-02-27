import numpy as np
from pathlib import Path

from .temporal_unrolled import TemporalUnrolledDataset
from typing import Any


def conditioner_numerical(conditioner_type, conditioner_value):
    if conditioner_type == "ppc":
        return float(conditioner_value)


class TemporalUnrolledwConditionersDataset(TemporalUnrolledDataset):

    def __init__(
        self,
        folder: str | Path,
        conditioners: dict[str, Any],
        i_start: int = 0,
        i_end: int = -1,
        step_size: int = 1,
        temporal_unroll_steps: int = 1,
    ):
        super().__init__(
            folder=folder,
            i_start=i_start,
            i_end=i_end,
            step_size=step_size,
            temporal_unroll_steps=temporal_unroll_steps,
        )
        self.conditioners = conditioners
        self.conditioners_array = np.stack(
            [conditioner_numerical(k, v) for k, v in conditioners.items()]
        )

    @property
    def conditioners_size(self):
        return self.conditioners_array.shape[-1]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = self._load_inputs(idx)
        targets = self._load_targets(idx)
        return inputs, targets, self.conditioners_array
