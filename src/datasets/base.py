import yaml
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self, folder: str | Path, step_size: int = 1):
        super().__init__()
        self.folder = Path(folder)
        self.step_size = step_size

        with open(self.folder / "f/args.yaml", "r") as f:
            self.info = yaml.safe_load(f)

        if self.info["i_end"] == -1:
            self.info["i_end"] = len(list(self.folder.glob("f/*.npy")))

        self.info["n_samples"] = np.sum(self._load_file(0, normalized=False))

    @property
    def grid_ndims(self) -> int:
        return len(self.info["bin_range"]) // 2

    @property
    def grid_size(self) -> tuple[int, ...]:
        return tuple([self.info["n_bins"]] * self.grid_ndims)

    @property
    def grid_dx(self):
        return [
            (self.info["bin_range"][2 * i + 1] - self.info["bin_range"][2 * i])
            / self.grid_size[i]
            for i in range(self.grid_ndims)
        ]

    def _load_file(self, i: int, normalized: bool = True) -> np.ndarray:
        data = np.load(self.folder / f"f/{i:06d}.npy")
        if normalized:
            data /= self.info["n_samples"]
        return data

    def __len__(self) -> int:
        return self.info["i_end"] - self.step_size

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        inputs = self._load_file(idx, normalized=True)
        targets = self._load_file(idx + self.step_size, normalized=True)

        return inputs, targets
