import torch
import torch.nn as nn
from src.models.fp2d.tensor.base import FokkerPlanck2DTensorBase


class FokkerPlanck2D_AsymBsym(FokkerPlanck2DTensorBase):

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
        shape = (grid_size[0] // 2 + grid_size[0] % 2, grid_size[1])
        self.A = nn.Parameter(torch.zeros((1, *shape)))
        self.B = nn.Parameter(torch.zeros((2, *shape)))

    @property
    def A_grid(self) -> torch.Tensor:
        Ax = torch.concatenate(
            [self.A[0], -torch.flip(self.A[0], dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        return torch.stack([Ax, Ax.T], dim=0)

    @property
    def B_grid(self) -> torch.Tensor:
        Bxx = torch.concatenate(
            [self.B[0], torch.flip(self.B[0], dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        Bxy = torch.concatenate(
            [self.B[1], -torch.flip(self.B[1], dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        return torch.stack([Bxx, Bxx.T, Bxy], dim=0)
