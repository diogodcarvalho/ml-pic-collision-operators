import re
import yaml
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset

from ml_pic_collision_operators.datasets.dataset_utils import DatasetItem


class BaseTracksDataset(Dataset):
    """Loads sampled particle tracks as ordered particle clouds.

    Folder must contain:

    - ``<folder>/<idx:06d>.h5`` with one file per simulation dump, the
        integer ``idx`` being the simulation step.
    - ``<folder>/args.yaml`` metadata file with entries ``dt``, ``i_start``,
        ``i_end`` and ``v_units``. ``dt`` is assumed to be in units of
        ``1 / omega_p``, matching the convention used by the gridded
        phase-space datasets.

    Each h5 file is a pandas DataFrame (PyTables) whose index is the int32 particle tag
    and whose columns are the recorded phase-space quantities. All columns are loaded
    and particles are aligned across timesteps by sorting on tag.
    """

    kind: str = "tracks"

    def __init__(
        self,
        folder: str | Path,
        i_start: int = 0,
        i_end: int = -1,
        step_size: int = 1,
        mode: str = "train",
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
        n_files = len(list(self.folder.glob("*.h5")))
        if i_end == -1:
            i_end = self.i_start + n_files
        if self.info["i_end"] == -1:
            i_end_info = self.i_start + n_files
        else:
            i_end_info = int(self.info["i_end"])
        self.i_end = min(i_end, i_end_info)

        self.dt = float(self.info["dt"])
        self.grid_units = re.sub(r"_(\w+)", r"_{{\1}}", self.info["v_units"])

        df0 = pd.read_hdf(self.folder / f"{self.i_start:06d}.h5")
        self.coords = tuple(df0.columns)
        self.n_particles = int(df0.shape[0])
        self.phase_space_ndims = len(self.coords)

        if mode not in ("train", "test"):
            raise ValueError(f"Invalid mode {mode}. Must be 'train' or 'test'.")
        self.mode = mode

    def _load_file(self, i: int) -> np.ndarray:
        if not isinstance(i, int):
            raise KeyError(
                f"Can only access file with integer index, request: {i} ({type(i)})"
            )
        i += self.i_start
        df = pd.read_hdf(self.folder / f"{i:06d}.h5")
        if df.shape[0] != self.n_particles:
            raise ValueError(
                f"{self.folder / f'{i:06d}.h5'}: particle count {df.shape[0]} "
                f"!= reference {self.n_particles}"
            )
        return df.sort_index().to_numpy(dtype=self._dtype)

    def __len__(self) -> int:
        n = self.i_end - self.i_start
        if self.mode == "train":
            return n - self.step_size
        return (n - 1) // self.step_size

    def __getitem__(self, idx: int) -> DatasetItem:
        if self.mode == "test":
            idx *= self.step_size
        inputs = self._load_file(idx)
        targets = self._load_file(idx + self.step_size)
        return DatasetItem(inputs=inputs, targets=targets, dt=self.dt)
