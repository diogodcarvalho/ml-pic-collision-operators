import torch
import numpy as np
import torch.nn as nn
import copy
from src.models.fp2d.base import FokkerPlanck2DBase


class FokkerPlanck2D(FokkerPlanck2DBase):
    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        ensure_non_negative_f: bool = True,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
        )
        self.A = nn.Parameter(torch.zeros((2, grid_size[0], grid_size[1])))
        self.B = nn.Parameter(torch.zeros((3, grid_size[0], grid_size[1])))

    @property
    def A_grid(self) -> torch.Tensor:
        return self.A

    @property
    def B_grid(self) -> torch.Tensor:
        return self.B

    def load_from_numpy(self, A: np.ndarray, B: np.ndarray):
        assert A.shape == self.A.shape
        assert B.shape == self.B.shape
        with torch.no_grad():
            A = torch.Tensor(A).type_as(self.A)
            B = torch.Tensor(B).type_as(self.B)
            cloned_model = copy.deepcopy(self)  # Create a new instance
            cloned_model.A.copy_(A)
            cloned_model.B.copy_(B)
        return cloned_model
