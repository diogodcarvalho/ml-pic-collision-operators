import torch
import torch.nn as nn

from typing import Callable

from src.models.fp2d.base import FokkerPlanck2DBase
from src.models.utils.nn import MLP
from src.utils import class_from_str


class FokkerPlanck2DNN(FokkerPlanck2DBase):
    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        ensure_non_negative_f: bool = True,
        normalize_v_grid: bool = True,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
        )

        if isinstance(activation, str):
            activation = class_from_str(activation)

        self.Ax = MLP(2, 1, depth, width_size, activation, use_bias, use_final_bias)
        self.Ay = MLP(2, 1, depth, width_size, activation, use_bias, use_final_bias)
        self.Bxx = MLP(2, 1, depth, width_size, activation, use_bias, use_final_bias)
        self.Byy = MLP(2, 1, depth, width_size, activation, use_bias, use_final_bias)
        self.Bxy = MLP(2, 1, depth, width_size, activation, use_bias, use_final_bias)
        self._init_v_grid(normalize_v_grid)

    def _init_v_grid(self, normalize: bool):
        vx = torch.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size[0] + 1
        )[:-1]
        vy = torch.linspace(
            self.grid_range[2], self.grid_range[3], self.grid_size[1] + 1
        )[:-1]
        vx += self.dx[0] / 2.0
        vy += self.dx[1] / 2.0
        if normalize:
            vx /= torch.std(vx)
            vy /= torch.std(vy)
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    @property
    def A_grid(self):
        v = self.v_grid.detach()
        Ax = self.Ax(v)
        Ay = self.Ay(v)
        A_grid = torch.cat([Ax.T, Ay.T], dim=0)
        A_grid = A_grid.view(2, *self.grid_size)
        return A_grid

    @property
    def B_grid(self):
        v = self.v_grid.detach()
        Bxx = self.Bxx(v)
        Byy = self.Byy(v)
        Bxy = self.Bxy(v)
        B_grid = torch.cat([Bxx.T, Byy.T, Bxy.T], dim=0)
        B_grid = B_grid.view(3, *self.grid_size)
        return B_grid
