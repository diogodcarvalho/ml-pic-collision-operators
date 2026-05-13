import numpy as np
from dataclasses import dataclass


@dataclass
class DatasetItem:
    inputs: np.ndarray
    targets: np.ndarray
    dt: float
    conditioners: np.ndarray | None = None
