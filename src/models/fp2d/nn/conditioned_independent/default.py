import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from src.models.fp2d.nn.conditioned_independent import (
    FokkerPlanck2DNNBaseConditionedIndependent,
)
from src.models.utils.nn import MLP


class FokkerPlanck2DNNConditionedIndependent(
    FokkerPlanck2DNNBaseConditionedIndependent
):
    """
    This model parametrizes A and B using 6 independet MLPs:

        A_i(vx, vy) = MPL_A_i(vx, vy) * MLP_C(c)
        B_ij(vx, vy) = MPL_B_ij(vx, vy) * MLP_C(c)

    No symmetries along v are enforced.
    Models assumes that A/B depend separabily on (vx,vy) and the conditioners c.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        conditioners_size: int,
        depth: int,
        width_size: int,
        c_depth: int,
        c_width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        c_activation: Callable | str = nn.ReLU,
        c_use_bias: bool = True,
        c_use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        normalize_v_grid: bool = True,
        conditioners_min_values: list[float] | np.ndarray | None = None,
        conditioners_max_values: list[float] | np.ndarray | None = None,
        normalize_conditioners: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            conditioners_size=conditioners_size,
            ensure_non_negative_f=ensure_non_negative_f,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            c_depth=c_depth,
            c_width_size=c_width_size,
            c_activation=c_activation,
            c_use_bias=c_use_bias,
            c_use_final_bias=c_use_final_bias,
            batch_norm=batch_norm,
            normalize_v_grid=normalize_v_grid,
            conditioners_min_values=conditioners_min_values,
            conditioners_max_values=conditioners_max_values,
            normalize_conditioners=normalize_conditioners,
        )

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        conditioners_size: int,
        c_depth: int,
        c_width_size: int,
        c_activation: Callable,
        c_use_bias: bool,
        c_use_final_bias: bool,
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
        self.C = MLP(
            conditioners_size,
            1,
            c_depth,
            c_width_size,
            c_activation,
            c_use_bias,
            c_use_final_bias,
            batch_norm,
        )

    def _init_v_grid(self, normalize: bool):
        vx = torch.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size[0] + 1
        )[:-1]
        vy = torch.linspace(
            self.grid_range[2], self.grid_range[3], self.grid_size[1] + 1
        )[:-1]
        vx += self.grid_dx[0] / 2.0
        vy += self.grid_dx[1] / 2.0
        if normalize:
            vx /= torch.std(vx)
            vy /= torch.std(vy)
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        # (batch_size, conditioners_size)
        assert conditioners.ndim == 2
        v = self.v_grid.detach()
        # (grid_size**2, 1)
        Ax = self.Ax(v)
        # print("1", Ax.shape)
        Ay = self.Ay(v)
        # (2, grid_size**2)
        A_grid = torch.cat([Ax.T, Ay.T], dim=0)
        # print("2", Ax.shape)
        # (1, 2, grid_size, grid_size)
        A_grid = A_grid.view(1, 2, *self.grid_size)
        # print("3", A_grid.shape)
        # (batch_size, 1)
        C = self.C(conditioners.detach())
        # print("4", C.shape)
        # (batch_size, 1, 1, 1)
        C = C.unsqueeze(2).unsqueeze(3)
        # print("5", C.shape)
        # (batch_size, 2, grid_size, grid_size)
        A_grid = A_grid * C
        # print("6", A_grid.shape)
        return A_grid

    def B_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        v = self.v_grid.detach()
        Bxx = self.Bxx(v)
        Byy = self.Byy(v)
        Bxy = self.Bxy(v)
        B_grid = torch.cat([Bxx.T, Byy.T, Bxy.T], dim=0)
        B_grid = B_grid.view(1, 3, *self.grid_size)
        C = self.C(conditioners.detach())
        C = C.unsqueeze(2).unsqueeze(3)
        return B_grid * C
