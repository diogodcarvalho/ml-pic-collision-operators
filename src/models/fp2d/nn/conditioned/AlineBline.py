import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from src.models.fp2d.nn.conditioned.base import FokkerPlanck2DNNBaseConditioned
from src.models.utils.nn import MLP


class FokkerPlanck2DNNConditioned_AlineBline(FokkerPlanck2DNNBaseConditioned):
    """
    This model parametrizes A_x, B_xx and Bxy using independet (equivalent) MLPs:

        A_x(vx, vy) = - sign(vx) * MPL_A_x(|vx|, c)
        B_xx(vx, vy) = MPL_B_xx(|vx|, c)
        B_xy(vx, vy) = - sign(vx) * sign(vy) * MPL_B_xy(|vx|, |vy|, c)

    and enforces that:

        A_y = A_x^T
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
        self.Ax = MLP(
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
            1 + conditioners_size,
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

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        # (batch_size * (grid_size//2 + grid_size%2), 1 + C)
        inputs = self._prepare_input(conditioners, self.vx)
        # print("i", inputs.shape)
        # (batch_size * (grid_size//2 + grid_size%2), 1)
        Ax = self.Ax(inputs)
        # print("1", Ax.shape)
        # (batch_size, 1, grid_size//2 + grid_size%2, grid_size//2 + grid_size%2)
        Ax = Ax.view(
            conditioners.shape[0],
            1,
            self.grid_size[0] // 2 + self.grid_size[0] % 2,
            1,
        )
        # print("2", Ax.shape)
        Ax = Ax.repeat(1, 1, 1, self.grid_size[1] // 2 + self.grid_size[1] % 2)
        # print("3", Ax.shape)
        # (batch_size, 1, grid_size, grid_size//2 + grid_size%2)
        Ax = torch.cat(
            [Ax, -torch.flip(Ax, dims=(2,))[:, :, self.grid_size[0] % 2 :]],
            dim=2,
        )
        # print("4", Ax.shape)
        # (batch_size, 1, grid_size, grid_size)
        Ax = torch.cat(
            [Ax, torch.flip(Ax, dims=(3,))[:, :, :, self.grid_size[1] % 2 :]],
            dim=3,
        )
        # print("5", Ax.shape)
        # (batch_size, 2, grid_size, grid_size)
        A_grid = torch.cat([Ax, Ax.transpose(2, 3)], dim=1)
        # #print("5", A_grid.shape)
        return A_grid

    def B_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_input(conditioners, self.vx)
        Bxx = self.Bxx(inputs)
        inputs = self._prepare_input(conditioners)
        Bxy = self.Bxy(inputs)
        Bxx = Bxx.view(
            conditioners.shape[0],
            1,
            self.grid_size[0] // 2 + self.grid_size[0] % 2,
            1,
        )
        Bxx = Bxx.repeat(1, 1, 1, self.grid_size[1] // 2 + self.grid_size[1] % 2)
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
