import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.conditioned.base import (
    FokkerPlanck2D_NNConditioned_Base,
)
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NNConditioned_AD_T(FokkerPlanck2D_NNConditioned_Base):
    """Conditioned Fokker-Planck 2D NN Model with Transposed Symmetry.

    This model parametrizes A_x, D_xx and D_xy using independent (equivalent) MLPs:

        A_x(vx, vy, c) = MLP_A_x(vx, vy, c)
        D_xx(vx, vy, c) = MLP_D_xx(vx, vy, c)
        D_xy(vx, vy, c) = MLP_D_xy(vx, vy, c)

    and enforces that:

        A_y = A_x^T
        D_yy = D_xx^T
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
        ensure_non_negative_D: bool = False,
        normalize_v_grid: bool = True,
        conditioners_min_values: list[float] | np.ndarray | None = None,
        conditioners_max_values: list[float] | np.ndarray | None = None,
        normalize_conditioners: bool = False,
        guard_cells: bool = False,
        operator_is_time_dependent: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            conditioners_size=conditioners_size,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
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
            operator_is_time_dependent=operator_is_time_dependent,
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
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )
        self.Dxx = MLP(
            2 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )
        self.Dxy = MLP(
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
        vx, vy = self._default_vx_vy(normalize)
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        # (batch_size * grid_size**2, 2 + C)
        inputs = self._prepare_input(conditioners, self.v_grid.data)
        # (batch_size * grid_size**2, 1)
        Ax = self.Ax(inputs)
        # (batch_size, 1, grid_size, grid_size)
        Ax = Ax.view(conditioners.shape[0], 1, *self.grid_size)
        # (batch_size, 2, grid_size, grid_size)
        A_grid = torch.cat([Ax, Ax.transpose(2, 3)], dim=1)
        return A_grid

    def D_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_input(conditioners, self.v_grid.data)
        Dxx = self.Dxx(inputs)
        Dxy = self.Dxy(inputs)
        Dxx = Dxx.view(conditioners.shape[0], 1, *self.grid_size)
        Dxy = Dxy.view(conditioners.shape[0], 1, *self.grid_size)
        D_grid = torch.cat([Dxx, Dxx.transpose(2, 3), Dxy], dim=1)
        return D_grid
