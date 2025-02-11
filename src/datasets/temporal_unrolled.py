import numpy as np
from pathlib import Path

from .base import BaseDataset


class TemporalUnrolledDataset(BaseDataset):

    def __init__(
        self,
        folder: str | Path,
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
        )
        self.temporal_unroll_steps = temporal_unroll_steps

    def __len__(self) -> int:
        return (
            self.info["i_end"]
            - self.info["i_start"]
            - self.step_size * self.temporal_unroll_steps
        )

    def _load_inputs(self, idx: int) -> np.ndarray:
        return self._load_file(idx, normalized=True)

    def _load_targets(self, idx: int) -> np.ndarray:
        return np.stack(
            [
                self._load_file(idx + (ts + 1) * self.step_size, normalized=True)
                for ts in range(self.temporal_unroll_steps)
            ],
            axis=0,
        )

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = self._load_inputs(idx)
        targets = self._load_targets(idx)
        return inputs, targets
