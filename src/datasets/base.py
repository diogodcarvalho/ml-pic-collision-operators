import re
import yaml
import torch
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path


class BaseDataset(Dataset):

    def __init__(
        self, folder: str | Path, i_start: int = 0, i_end: int = -1, step_size: int = 1
    ):
        super().__init__()
        self.folder = Path(folder)
        self.step_size = step_size

        with open(self.folder / "args.yaml", "r") as f:
            self.info = yaml.safe_load(f)

        self.i_start = max(int(self.info["i_start"]), i_start)
        if i_end == -1:
            i_end = len(list(self.folder.glob("*.npy")))
        if self.info["i_end"] == -1:
            i_end_info = len(list(self.folder.glob("*.npy")))
        else:
            i_end_info = int(self.info["info"])
        self.i_end = min(i_end, i_end_info)

        self.n_particles = np.sum(self._load_file(0, normalized=False))
        self.dt = float(self.info["dt"])

        self.grid_ndims = int(self._load_file(0).ndim)
        self.grid_size = self._load_file(0).shape
        self.grid_range = self.info["v_range"]
        self.grid_range_c = self.info["v_range_c"]
        self.grid_units = re.sub(r"_(\w+)", r"_{{\1}}", self.info["v_range_units"])
        self.grid_dx = [
            (self.grid_range[2 * i + 1] - self.grid_range[2 * i]) / self.grid_size[i]
            for i in range(self.grid_ndims)
        ]

    def _load_file(self, i: int, normalized: bool = True) -> np.ndarray:
        if not isinstance(i, int):
            raise KeyError(
                f"Can only access file with integer index, request: {i} ({type(i)})"
            )
        i += self.i_start
        data = np.load(self.folder / f"{i:06d}.npy")
        if normalized:
            data /= self.n_particles
        return data

    def __len__(self) -> int:
        return self.i_end - self.i_start - self.step_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        inputs = self._load_file(idx, normalized=True)
        targets = self._load_file(idx + self.step_size, normalized=True)

        return inputs, targets, self.dt
