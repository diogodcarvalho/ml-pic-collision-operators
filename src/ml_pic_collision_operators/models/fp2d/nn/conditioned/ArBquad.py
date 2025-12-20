import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.conditioned.base import (
    FokkerPlanck2DNNBaseConditioned,
)
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2DNNConditioned_ArBquad(FokkerPlanck2DNNBaseConditioned):
    """
    This model parametrizes |A|, B_xx and B_xy using independet (equivalent) MLPs:

        |A|(vx, vy) = MLP_A(||v||, c)
        B_xx(vx, vy) = MLP_B_xx(|vx|, |vy|, c)
        B_xy(vx, vy) = - sign(vx) * sign(vy) * MLP_B_xy(|vx|, |vy|, c)

    and enforces that:

        A_x = |A| * cos(theta)
        A_y = |A| * sin(theta) (equivalent to A_x^T)
        B_yy = B_xx^T
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
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        normalize_v_grid: bool = True,
        conditioners_min_values: list[float] | np.ndarray | None = None,
        conditioners_max_values: list[float] | np.ndarray | None = None,
        normalize_conditioners: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            conditioners_size=conditioners_size,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_B=ensure_non_negative_B,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
            normalize_v_grid=normalize_v_grid,
            conditioners_min_values=conditioners_min_values,
            conditioners_max_values=conditioners_max_values,
            normalize_conditioners=normalize_conditioners,
            guard_cells=guard_cells,
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
        batch_norm: bool,
    ):
        self.A = MLP(
            1 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )
        self.Bxx = MLP(
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )
        self.Bxy = MLP(
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )

    def _init_v_grid(self, normalize: bool):
        # bin center positions
        vx, vy = self._default_vx_vy(normalize)

        # this one is needed for A
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.vr_grid = nn.Buffer(torch.sqrt(VX**2 + VY**2).reshape(-1, 1))
        # precompute angles of v=(vx,vy) with respect to x-axis
        theta = torch.arctan2(VY, VX)
        self.cos_theta = nn.Buffer(torch.cos(theta))

        # this one is needed for Bxx and Bxy
        # keep only half the grid along x and y
        vx = vx[self.grid_size[0] // 2 :]
        vy = vy[self.grid_size[0] // 2 :]
        # create meshgrid
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        # (batch_size * grid_size**2, 1 + C)
        inputs = self._prepare_input(conditioners, self.vr_grid)
        # (batch_size * grid_size**2, 1)
        A = self.A(inputs)
        # (batch_size, grid_size, grid_size)
        A = A.view(conditioners.shape[0], self.grid_size[0], self.grid_size[1])
        # (batch_size, grid_size, grid_size)
        # for some reason with ddp need the clone here
        Ax = A * self.cos_theta.clone()
        # (batch_size, 2, grid_size, grid_size)
        A_grid = torch.stack([Ax, Ax.transpose(1, 2)], dim=1)
        return A_grid

    def B_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_input(conditioners)
        Bxx = self.Bxx(inputs)
        Bxy = self.Bxy(inputs)
        Bxx = Bxx.view(
            conditioners.shape[0],
            1,
            self.grid_size[0] // 2 + self.grid_size[0] % 2,
            self.grid_size[1] // 2 + self.grid_size[1] % 2,
        )
        Bxy = Bxy.view(
            conditioners.shape[0],
            1,
            self.grid_size[0] // 2 + self.grid_size[0] % 2,
            self.grid_size[1] // 2 + self.grid_size[1] % 2,
        )
        Bxx = torch.cat(
            [Bxx, torch.flip(Bxx, dims=(2,))[:, :, self.grid_size[0] % 2 :]],
            dim=2,
        )
        Bxx = torch.cat(
            [Bxx, torch.flip(Bxx, dims=(3,))[:, :, :, self.grid_size[1] % 2 :]],
            dim=3,
        )
        Bxy = torch.cat(
            [Bxy, -torch.flip(Bxy, dims=(2,))[:, :, self.grid_size[0] % 2 :]],
            dim=2,
        )
        Bxy = torch.cat(
            [Bxy, -torch.flip(Bxy, dims=(3,))[:, :, :, self.grid_size[1] % 2 :]],
            dim=3,
        )
        B_grid = torch.cat([Bxx, Bxx.transpose(2, 3), Bxy], dim=1)
        return B_grid
