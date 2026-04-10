import torch
import torch.nn as nn

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.default import FokkerPlanck2D_NN_Base
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NN_AD_Sym(FokkerPlanck2D_NN_Base):
    """Fokker-Planck 2D Neural Network Model with Axis (Anti-)Symmetry.

    This model parametrizes A_x, D_xx and D_xy using independent (equivalent) MLPs:

        A_x(vx, vy) = - sign(vx) * MLP_A_x(|vx|, vy)
        D_xx(vx, vy) = MLP_D_xx(|vx|, vy)
        D_xy(vx, vy) = - sign(vx) * MLP_D_xy(|vx|, vy)

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
            includes_symmetry=True,
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
        self.Dxx = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dxy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    def _init_v_grid(self, normalize: bool):
        # bin center positions
        vx, vy = self._default_vx_vy(normalize)
        # keep only half the grid along x
        vx = vx[self.grid_size[0] // 2 :]
        # create meshgrid
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    @property
    def A_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        # ((grid_size//2 + grid_size%2) * grid_size, 2)
        inputs = self.v_grid.data
        # ((grid_size//2 + grid_size%2) * grid_size, 1)
        Ax = self.Ax(inputs)
        # (grid_size//2 + grid_size%2, grid_size)
        Ax = Ax.view(self.grid_size[0] // 2 + self.grid_size[0] % 2, self.grid_size[1])
        # (grid_size, grid_size)
        Ax = torch.cat(
            [Ax, -torch.flip(Ax, dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        # (2, grid_size, grid_size)
        A_grid = torch.stack([Ax, Ax.T], dim=0)
        return A_grid

    @property
    def D_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        inputs = self.v_grid.data
        Dxx = self.Dxx(inputs)
        Dxy = self.Dxy(inputs)
        Dxx = Dxx.view(
            self.grid_size[0] // 2 + self.grid_size[0] % 2, self.grid_size[1]
        )
        Dxy = Dxy.view(
            self.grid_size[0] // 2 + self.grid_size[0] % 2, self.grid_size[1]
        )
        Dxx = torch.cat(
            [Dxx, torch.flip(Dxx, dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        Dxy = torch.cat(
            [Dxy, -torch.flip(Dxy, dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        D_grid = torch.stack([Dxx, Dxx.T, Dxy], dim=0)
        return D_grid
