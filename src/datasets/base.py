import re
import yaml
import torch
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path


class BaseDataset(Dataset):

    def __init__(
        self,
        folder: str | Path,
        i_start: int = 0,
        i_end: int = -1,
        step_size: int = 1,
        extra_cells: int = 0,
    ):
        super().__init__()

        self._dtype = (
            np.float32 if torch.get_default_dtype() == torch.float32 else np.float64
        )

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

        self.extra_cells = extra_cells
        self.n_particles = np.sum(self._load_file(0, normalized=False, pad=False))
        self.dt = float(self.info["dt"])

        self.grid_ndims = int(self._load_file(0, pad=False).ndim)
        self.grid_units = re.sub(r"_(\w+)", r"_{{\1}}", self.info["v_range_units"])

        self.original_grid_size = self._load_file(0, pad=False).shape
        self.original_grid_range = self.info["v_range"]
        self.original_grid_range_c = self.info["v_range_c"]
        print(self.original_grid_range)

        self.grid_dx = [
            (self.original_grid_range[2 * i + 1] - self.original_grid_range[2 * i])
            / self.original_grid_size[i]
            for i in range(self.grid_ndims)
        ]
        self.grid_dx_c = [
            (self.original_grid_range_c[2 * i + 1] - self.original_grid_range_c[2 * i])
            / self.original_grid_size[i]
            for i in range(self.grid_ndims)
        ]

        self.grid_size = list(self.original_grid_size)
        self.grid_range = list(self.original_grid_range)
        self.grid_range_c = list(self.original_grid_range_c)

        if self.extra_cells != 0:
            for i in range(self.grid_ndims):
                self.grid_size[i] += 2 * extra_cells
                self.grid_range[2 * i] -= extra_cells * self.grid_dx[i]
                self.grid_range[2 * i + 1] += extra_cells * self.grid_dx[i]
                self.grid_range_c[2 * i] -= extra_cells * self.grid_dx_c[i]
                self.grid_range_c[2 * i + 1] += extra_cells * self.grid_dx_c[i]

    def _load_file(
        self, i: int, normalized: bool = True, pad: bool = True
    ) -> np.ndarray:
        if not isinstance(i, int):
            raise KeyError(
                f"Can only access file with integer index, request: {i} ({type(i)})"
            )
        i += self.i_start
        data = np.load(self.folder / f"{i:06d}.npy").astype(self._dtype)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        if normalized:
            data /= self.n_particles
        if pad:
            data = np.pad(data, self.extra_cells)

        return data

    def __len__(self) -> int:
        return self.i_end - self.i_start - self.step_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        inputs = self._load_file(idx, normalized=True)
        targets = self._load_file(idx + self.step_size, normalized=True)

        return inputs, targets, self.dt
