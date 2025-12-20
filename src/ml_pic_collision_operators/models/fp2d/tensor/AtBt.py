import torch
import torch.nn as nn
from ml_pic_collision_operators.models.fp2d.tensor.base import FokkerPlanck2DTensorBase


class FokkerPlanck2D_AtBt(FokkerPlanck2DTensorBase):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_B=ensure_non_negative_B,
            guard_cells=guard_cells,
            includes_symmetry=True,
        )
        self.A = nn.Parameter(torch.zeros((1, grid_size[0], grid_size[1])))
        self.B = nn.Parameter(torch.zeros((2, grid_size[0], grid_size[1])))

    @property
    def A_grid(self) -> torch.Tensor:
        return torch.stack([self.A[0], self.A[0].T], dim=0)

    @property
    def B_grid(self) -> torch.Tensor:
        return torch.stack([self.B[0], self.B[0].T, self.B[1]], dim=0)
