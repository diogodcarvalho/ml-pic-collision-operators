import torch
import torch.nn as nn

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.default import FokkerPlanck2D_NN_Base
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NN_AD(FokkerPlanck2D_NN_Base):
    """Fokker-Planck 2D Neural Network Model.

    This model parametrizes A and D using 5 independent (equivalent) MLPs:

        A_i(vx, vy) = MLP_A_i(vx, vy)
        D_ij(vx, vy) = MLP_D_ij(vx, vy)

    No symmetries are enforced.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_D: bool = False,
        normalize_v_grid: bool = True,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
            normalize_v_grid=normalize_v_grid,
            guard_cells=guard_cells,
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
        self.Dxx = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dyy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dxy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    def _init_v_grid(self, normalize: bool):
        vx, vy = self._default_vx_vy(normalize)
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    @property
    def A_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        inputs = self.v_grid.data
        Ax = self.Ax(inputs)
        Ay = self.Ay(inputs)
        A_grid = torch.cat([Ax, Ay], dim=0)
        A_grid = A_grid.view(2, *self.grid_size)
        return A_grid

    @property
    def D_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        inputs = self.v_grid.data
        Dxx = self.Dxx(inputs)
        Dyy = self.Dyy(inputs)
        Dxy = self.Dxy(inputs)
        D_grid = torch.cat([Dxx, Dyy, Dxy], dim=0)
        D_grid = D_grid.view(3, *self.grid_size)
        return D_grid
