import torch
import torch.nn as nn

from typing import Callable

from src.models.fp2d.nn.default import FokkerPlanck2DNNBase
from src.models.utils.nn import MLP


class FokkerPlanck2DNN(FokkerPlanck2DNNBase):
    """
    This model parametrizes A and B using 5 independet (equivalent) MLPs:

        A_i(vx, vy) = MPL_A_i(vx, vy)
        B_ij(vx, vy) = MPL_B_ij(vx, vy)

    No symmetries are enforced.
    """

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
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        normalize_v_grid: bool = True,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
            normalize_v_grid=normalize_v_grid,
        )

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        self.Ax = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Ay = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Bxx = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Byy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Bxy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
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

    @property
    def A_grid(self) -> torch.Tensor:
        inputs = self.v_grid.detach()
        Ax = self.Ax(inputs)
        Ay = self.Ay(inputs)
        A_grid = torch.cat([Ax, Ay], dim=0)
        A_grid = A_grid.view(2, *self.grid_size)
        return A_grid

    @property
    def B_grid(self) -> torch.Tensor:
        inputs = self.v_grid.detach()
        Bxx = self.Bxx(inputs)
        Byy = self.Byy(inputs)
        Bxy = self.Bxy(inputs)
        B_grid = torch.cat([Bxx, Byy, Bxy], dim=0)
        B_grid = B_grid.view(3, *self.grid_size)
        return B_grid
