import torch
import torch.nn as nn
import numpy as np
from ml_pic_collision_operators.models.fp2d.tensor.time_dependent.base import (
    FokkerPlanck2D_Tensor_Base_TimeDependent,
)
from ml_pic_collision_operators.models.utils import torch_interpolate


class FokkerPlanck2D_Tensor_TimeDependent_AD_ParPerp(
    FokkerPlanck2D_Tensor_Base_TimeDependent
):
    """Time-Dependent Fokker-Planck 2D Tensor Model with Parallel-Perpendicular Symmetry.

    This model parametrizes A_par, D_par, and D_perp using 3 independent Tensors:

        A_par(t, ||v||) of shape (n_t, n_radial)
        D_par(t, ||v||) of shape (n_t, n_radial)
        D_perp(t, ||v||) of shape (n_t, n_radial)

    and enforces that:

        A_x = A_par * cos(theta)
        A_y = A_par * sin(theta) (equivalent to A_x^T)
        D_xx = D_par * cos(theta)^2 + D_perp * sin(theta)^2
        D_yy = D_par * sin(theta)^2 + D_perp * cos(theta)^2
        D_xy = (D_par - D_perp) * sin(theta) * cos(theta)

    Values of A_par(t), D_par(t), and D_perp(t) are defined on a 1D grid of velocity
    magnitudes (n_radial,), and are interpolated to the 2D velocity grid using the
    velocity magnitude at the center of each grid point.

    Linear interpolation is used to compute coefficients for t-values not stored in the
    tensor time grid of size (n_t,).
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        grid_size_t: int,
        grid_dt: float,
        n_radial: int = -1,
        n_t: int = -1,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_D: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            grid_size_t=grid_size_t,
            grid_dt=grid_dt,
            n_t=n_t,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
            guard_cells=guard_cells,
            includes_symmetry=True,
        )
        if n_radial == -1:
            self.n_radial = grid_size[0] // 2 + grid_size[0] % 2
        else:
            self.n_radial = n_radial
        self._init_params_dict.update(
            {
                "n_radial": n_radial,
            }
        )

        self.A = nn.Parameter(torch.zeros((self.n_t, self.n_radial)))
        self.Dpar = nn.Parameter(torch.zeros((self.n_t, self.n_radial)))
        self.Dperp = nn.Parameter(torch.zeros((self.n_t, self.n_radial)))

        # maximum |v| (diagonal)
        r_max = np.sqrt(self.grid_range[1] ** 2 + self.grid_range[3] ** 2)
        # velocity magnitude along axis in which self.A is defined
        self.vr_axis = nn.Buffer(torch.linspace(0, r_max, self.n_radial))

        # get velocities at bin centers
        vx = torch.linspace(*self.grid_range[:2], self.grid_size[0] + 1)[:-1]
        vy = torch.linspace(*self.grid_range[2:], self.grid_size[1] + 1)[:-1]
        vx += grid_dx[0] / 2.0
        vy += grid_dx[1] / 2.0
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        # velocity magnitude at bin centers
        self.vr_grid = nn.Buffer(torch.sqrt(VX**2 + VY**2).flatten())
        # get angle of v=(vx,vy) with respect to x-axis
        theta = torch.atan2(VY, VX)
        self.cos_theta = nn.Buffer(torch.cos(theta))
        self.sin_theta = nn.Buffer(torch.sin(theta))
        # force cos(vx=0,vy=0) and sin(vx=0,vy=0) = sqrt(2) / 2 to ensure model
        # learns that A(vx=0,vy=0) = 0 and Dpar(vx=0,vy=0) = Dperp(vx=0,vy=0)
        # otherwise, atan2 sets cos=1 and sin=0
        if grid_size[0] % 2:
            self.cos_theta[grid_size[0] // 2, grid_size[0] // 2] = np.sqrt(2) / 2
            self.sin_theta[grid_size[0] // 2, grid_size[0] // 2] = np.sqrt(2) / 2

    def Apar_real(self, t: torch.Tensor) -> np.ndarray:
        if self.n_t == self.grid_size_t:
            A = self.A[self._it(t)]
        else:
            A = self._t_interpolate(self.A, t)
        return A.detach().cpu().numpy() * self.grid_dx[0]

    def Dpar_real(self, t: torch.Tensor) -> np.ndarray:
        if self.n_t == self.grid_size_t:
            Dpar = self.Dpar[self._it(t)]
        else:
            Dpar = self._t_interpolate(self.Dpar, t)
        return Dpar.detach().cpu().numpy() * self.grid_dx[0] ** 2

    def Dperp_real(self, t: torch.Tensor) -> np.ndarray:
        if self.n_t == self.grid_size_t:
            Dperp = self.Dperp[self._it(t)]
        else:
            Dperp = self._t_interpolate(self.Dperp, t)
        return Dperp.detach().cpu().numpy() * self.grid_dx[0] ** 2

    def A_grid(self, t: torch.Tensor) -> torch.Tensor:
        vr_grid = self.vr_grid.unsqueeze(0).repeat(t.shape[0], 1)
        vr_axis = self.vr_axis.unsqueeze(0).repeat(t.shape[0], 1)
        if self.n_t == self.grid_size_t:
            A = torch_interpolate(vr_grid, vr_axis, self.A[self._it(t)][:, 0])
        else:
            A = self._t_interpolate(self.A, t)
            A = torch_interpolate(vr_grid, vr_axis, A, dim=1)
        A = A.reshape((t.shape[0], *self.grid_size))
        Ax = A * self.cos_theta
        Ay = A * self.sin_theta
        return torch.stack([Ax, Ay], dim=1)

    def D_grid(self, t: torch.Tensor) -> torch.Tensor:
        vr_grid = self.vr_grid.unsqueeze(0).repeat(t.shape[0], 1)
        vr_axis = self.vr_axis.unsqueeze(0).repeat(t.shape[0], 1)
        if self.n_t == self.grid_size_t:
            Dpar = torch_interpolate(vr_grid, vr_axis, self.Dpar[self._it(t)][:, 0])
            Dperp = torch_interpolate(vr_grid, vr_axis, self.Dperp[self._it(t)][:, 0])
        else:
            Dpar = self._t_interpolate(self.Dpar, t)
            Dperp = self._t_interpolate(self.Dperp, t)
            Dpar = torch_interpolate(vr_grid, vr_axis, Dpar, dim=1)
            Dperp = torch_interpolate(vr_grid, vr_axis, Dperp, dim=1)

        Dpar = Dpar.reshape((t.shape[0], *self.grid_size))
        Dperp = Dperp.reshape((t.shape[0], *self.grid_size))

        Dxx = Dpar * self.cos_theta**2 + Dperp * self.sin_theta**2
        Dyy = Dpar * self.sin_theta**2 + Dperp * self.cos_theta**2
        Dxy = (Dpar - Dperp) * self.cos_theta * self.sin_theta

        return torch.stack([Dxx, Dyy, Dxy], dim=1)

    def get_first_deriv_norm(self) -> torch.Tensor:
        # only in time
        return (
            torch.mean(torch.abs(self.A[1:] - self.A[:-1]))
            + torch.mean(torch.abs(self.Dpar[1:] - self.Dpar[:-1]))
            + torch.mean(torch.abs(self.Dperp[1:] - self.Dperp[:-1]))
        )

    def get_second_deriv_norm(self) -> torch.Tensor:
        # only in time
        return torch.sqrt(
            torch.mean(torch.square(self.A[2:] - 2 * self.A[1:-1] + self.A[:-2]))
            + torch.mean(
                torch.square(self.Dpar[2:] - 2 * self.Dpar[1:-1] + self.Dpar[:-2])
            )
            + torch.mean(
                torch.square(self.Dperp[2:] - 2 * self.Dperp[1:-1] + self.Dperp[:-2])
            )
            + 1e-10  # have to add to have gradients defined at initialization
        )
