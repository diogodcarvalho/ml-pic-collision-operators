import torch
import torch.nn as nn

from typing import Callable

from src.models.fp2d.nn.conditioned.base import FokkerPlanck2DNNBaseConditioned
from src.models.utils.nn import MLP


class FokkerPlanck2DNNConditioned_AtBt(FokkerPlanck2DNNBaseConditioned):
    """
    This model parametrizes A_x, B_xx and Bxy using independet (equivalent) MLPs:

        A_x(vx, vy) = MPL_A_x(vx, vy, c)
        B_xx(vx, vy) = MPL_B_xx(vx, vy, c)
        B_xy(vx, vy) = MPL_B_xy(vx, vy, c)

    and enforces that:

        A_y = A_x^T
        B_yy = B_xx^T
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        conditioners_size: int,
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
            conditioners_size=conditioners_size,
            ensure_non_negative_f=ensure_non_negative_f,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            normalize_v_grid=normalize_v_grid,
            includes_symmetry=True,
        )

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        conditioners_size: int,
    ):

        self.Ax = MLP(
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
        )
        self.Bxx = MLP(
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
        )
        self.Bxy = MLP(
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
        )

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

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        # (batch_size * grid_size**2, 2 + C)
        inputs = self._prepare_input(conditioners)
        # (batch_size * grid_size**2, 1)
        Ax = self.Ax(inputs)
        # (batch_size, 1, grid_size, grid_size)
        Ax = Ax.view(conditioners.shape[0], 1, *self.grid_size)
        # (batch_size, 2, grid_size, grid_size)
        A_grid = torch.cat([Ax, Ax.transpose(2, 3)], dim=1)
        return A_grid

    def B_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_input(conditioners)
        Bxx = self.Bxx(inputs)
        Bxy = self.Bxy(inputs)
        Bxx = Bxx.view(conditioners.shape[0], 1, *self.grid_size)
        Bxy = Bxy.view(conditioners.shape[0], 1, *self.grid_size)
        B_grid = torch.cat([Bxx, Bxx.transpose(2, 3), Bxy], dim=1)
        return B_grid
