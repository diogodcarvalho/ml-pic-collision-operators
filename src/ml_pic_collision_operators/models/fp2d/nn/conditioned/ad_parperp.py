import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.conditioned.base import (
    FokkerPlanck2D_NNConditioned_Base,
)
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NNConditioned_AD_ParPerp(FokkerPlanck2D_NNConditioned_Base):
    """Conditioned Fokker-Planck 2D NN Model with Parallel-Perpendicular Symmetry.

    This model parametrizes A_par, D_par, D_perp using independent (equivalent) MLPs:

        A_par(vx, vy, c) = MLP_A(||v||, c)
        D_par(vx, vy, c) = MLP_D_parr(||v||, c)
        D_perp(vx, vy, c) = MLP_D_perp(||v||, c)

    and enforces that:

        A_x = A_par * cos(theta)
        A_y = A_par * sin(theta) (equivalent to A_x^T)
        D_xx = D_par * cos(theta)^2 + D_perp * sin(theta)^2
        D_yy = D_par * sin(theta)^2 + D_perp * cos(theta)^2
        D_xy = (D_par - D_perp) * sin(theta) * cos(theta)
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
        self.Dpar = MLP(
            1 + conditioners_size,
            1,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )
        self.Dperp = MLP(
            1 + conditioners_size,
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
        self.sin_theta = nn.Buffer(torch.sin(theta))
        # force cos(vx=0,vy=0) and sin(vx=0,vy=0) = sqrt(2) / 2 to ensure model
        # learns that A(vx=0,vy=0) = 0 and Dpar(vx=0,vy=0) = Dperp(vx=0,vy=0)
        # otherwise, atan2 sets cos=1 and sin=0
        if self.grid_size[0] % 2:
            self.cos_theta[self.grid_size[0] // 2, self.grid_size[0] // 2] = (
                np.sqrt(2) / 2
            )
            self.sin_theta[self.grid_size[0] // 2, self.grid_size[0] // 2] = (
                np.sqrt(2) / 2
            )

    @property
    def vr_axis(self) -> torch.Tensor:
        vr = torch.unique(self.vr_grid.detach())
        if self.normalize_v_grid:
            vr, _ = self._denormalize_v(vr, torch.zeros_like(vr))
        return vr

    def Apar_real(self, conditioners: torch.Tensor) -> torch.Tensor:
        if conditioners.ndim == 1:
            conditioners = conditioners.unsqueeze(0)
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        inputs = self._prepare_input(
            conditioners, torch.unique(self.vr_grid.detach()).reshape(-1, 1)
        )
        Apar = self.A(inputs).detach().cpu().numpy() * self.grid_dx[0]
        return Apar.reshape(conditioners.shape[0], -1)

    def Dpar_real(self, conditioners: torch.Tensor) -> torch.Tensor:
        if conditioners.ndim == 1:
            conditioners = conditioners.unsqueeze(0)
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        inputs = self._prepare_input(
            conditioners, torch.unique(self.vr_grid.detach()).reshape(-1, 1)
        )
        Dpar = self.Dpar(inputs).detach().cpu().numpy() * self.grid_dx[0] ** 2
        return Dpar.reshape(conditioners.shape[0], -1)

    def Dperp_real(self, conditioners: torch.Tensor) -> torch.Tensor:
        if conditioners.ndim == 1:
            conditioners = conditioners.unsqueeze(0)
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        inputs = self._prepare_input(
            conditioners,
            torch.unique(self.vr_grid.detach()).reshape(-1, 1),
        )
        Dperp = self.Dperp(inputs).detach().cpu().numpy() * self.grid_dx[0] ** 2
        return Dperp.reshape(conditioners.shape[0], -1)

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        # (batch_size * grid_size**2, 1 + C)
        inputs = self._prepare_input(conditioners, self.vr_grid.data)
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

    def D_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        inputs = self._prepare_input(conditioners, self.vr_grid.data)

        Dpar = self.Dpar(inputs)
        Dperp = self.Dperp(inputs)

        Dpar = Dpar.view(conditioners.shape[0], *self.grid_size)
        Dperp = Dperp.view(conditioners.shape[0], *self.grid_size)

        cos = self.cos_theta.clone()
        sin = self.sin_theta.clone()

        Dxx = Dpar * cos**2 + Dperp * sin**2
        Dyy = Dpar * sin**2 + Dperp * cos**2
        Dxy = (Dpar - Dperp) * sin * cos

        D_grid = torch.stack([Dxx, Dyy, Dxy], dim=1)
        return D_grid
