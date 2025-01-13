import numpy as np
from pathlib import Path

from .base import BaseDataset


class TemporalUnrolledDataset(BaseDataset):
    def __init__(
        self, folder: str | Path, step_size: int = 1, temporal_unroll_steps: int = 1
    ):
        super().__init__(folder=folder, step_size=step_size)
        self.temporal_unroll_steps = temporal_unroll_steps

    def __len__(self) -> int:
        return self.info["i_end"] - self.step_size * self.temporal_unroll_steps

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = self._load_file(idx, normalized=True)
        targets = np.stack(
            [
                self._load_file(idx + (ts + 1) * self.step_size, normalized=True)
                for ts in range(self.temporal_unroll_steps)
            ],
            axis=0,
        )
        return inputs, targets
