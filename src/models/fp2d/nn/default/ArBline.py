import torch
import torch.nn as nn

from typing import Callable

from src.models.fp2d.nn.default import FokkerPlanck2DNNBase
from src.models.utils.nn import MLP


class FokkerPlanck2DNN_ArBline(FokkerPlanck2DNNBase):
    """
    This model parametrizes |A|, B_xx and B_xy using independet (equivalent) MLPs:

        |A|(vx, vy) = MPL_A(||v||)
        B_xx(vx, vy) = MPL_B_xx(|vx|)
        B_xy(vx, vy) = - sign(vx) * sign(vy) * MPL_B_xy(|vx|, |vy|)

    and enforces that:

        A_x = |A| * cos(theta)
        A_y = |A| * sin(theta) (equivalent to A_x^T)
        B_yy = B_xx^T
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
            includes_symmetry=True,
        )

    def _init_v_grid(self, normalize: bool):
        # bin center positions
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
        # for Bxx
        self.vx = nn.Buffer(vx.unsqueeze(1))
        # for Bxy
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        self.A = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Bxx = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Bxy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    @property
    def A_grid(self) -> torch.Tensor:
        # (grid_size**2, 1)
        inputs = self.vr_grid.detach()
        # (grid_size**2, 1)
        A = self.A(inputs)
        # (grid_sixe, grid_size)
        A = A.view(self.grid_size[0], self.grid_size[1])
        # (grid_sixe, grid_size)
        Ax = A * self.cos_theta
        # (2, grid_size, grid_size)
        A_grid = torch.stack([Ax, Ax.T], dim=0)
        return A_grid

    @property
    def B_grid(self) -> torch.Tensor:
        Bxx = self.Bxx(self.vx.detach())
        Bxy = self.Bxy(self.v_grid.detach())
        Bxx = Bxx.repeat(1, self.grid_size[1] // 2 + self.grid_size[1] % 2)
        Bxy = Bxy.view(
            self.grid_size[0] // 2 + self.grid_size[0] % 2,
            self.grid_size[1] // 2 + self.grid_size[1] % 2,
        )
        Bxx = torch.cat(
            [Bxx, torch.flip(Bxx, dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        Bxx = torch.cat(
            [Bxx, torch.flip(Bxx, dims=(1,))[:, self.grid_size[1] % 2 :]],
            dim=1,
        )
        Bxy = torch.cat(
            [Bxy, -torch.flip(Bxy, dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        Bxy = torch.cat(
            [Bxy, -torch.flip(Bxy, dims=(1,))[:, self.grid_size[1] % 2 :]],
            dim=1,
        )
        B_grid = torch.stack([Bxx, Bxx.T, Bxy], dim=0)
        return B_grid
