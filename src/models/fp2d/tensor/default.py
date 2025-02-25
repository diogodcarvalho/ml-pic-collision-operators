import torch
import torch.nn as nn

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
