import torch
import torch.nn as nn
import numpy as np

from ml_pic_collision_operators.models.fp2d.tensor.default import (
    FokkerPlanck2D_Tensor_Base,
)
from ml_pic_collision_operators.models.utils import torch_interpolate


class FokkerPlanck2D_Tensor_AD_ParPerp(FokkerPlanck2D_Tensor_Base):
    """Fokker-Planck 2D Tensor Model with Parallel-Perpendicular Symmetry.

    This model parametrizes A_par, B_par, and B_perp using 3 independent Tensors:

        A_par(||v||) of shape (n_radial,)
        B_par(||v||) of shape (n_radial,)
        B_perp(||v||) of shape (n_radial,)

    and enforces that:

        A_x = A_par * cos(theta)
        A_y = A_par * sin(theta) (equivalent to A_x^T)
        B_xx = B_par * cos(theta)^2 + B_perp * sin(theta)^2
        B_yy = B_par * sin(theta)^2 + B_perp * cos(theta)^2
        B_xy = (B_par - B_perp) * sin(theta) * cos(theta)

    Values of A_par, B_par, and B_perp are defined on a 1D grid of velocity magnitudes
    (n_radial,), and are interpolated to the 2D velocity grid using the velocity
    magnitude at the center of each grid point.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        n_radial: int = -1,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_B=ensure_non_negative_B,
            guard_cells=guard_cells,
            includes_symmetry=True,
        )

        if n_radial == -1:
            self.n_radial = grid_size[0] // 2 + grid_size[0] % 2
        else:
            self.n_radial = n_radial
        self._init_params_dict.update({"n_radial": n_radial})

        self.A = nn.Parameter(torch.zeros((self.n_radial)))
        self.Bpar = nn.Parameter(torch.zeros((self.n_radial)))
        self.Bperp = nn.Parameter(torch.zeros((self.n_radial)))

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
        # learns that A(vx=0,vy=0) = 0 and Bpar(vx=0,vy=0) = Bperp(vx=0,vy=0)
        # otherwise, atan2 sets cos=1 and sin=0
        if grid_size[0] % 2:
            self.cos_theta[grid_size[0] // 2, grid_size[0] // 2] = np.sqrt(2) / 2
            self.sin_theta[grid_size[0] // 2, grid_size[0] // 2] = np.sqrt(2) / 2

    @property
    def Apar_real(self) -> np.ndarray:
        return self.A[0].detach().cpu().numpy() * self.grid_dx[0]

    @property
    def Bpar_real(self) -> np.ndarray:
        return self.Bpar[0].detach().cpu().numpy() * self.grid_dx[0] ** 2

    @property
    def Bperp_real(self) -> np.ndarray:
        return self.Bperp[0].detach().cpu().numpy() * self.grid_dx[0] ** 2

    @property
    def A_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        A = torch_interpolate(self.vr_grid.data, self.vr_axis.data, self.A)
        A = A.reshape(*self.grid_size)
        Ax = A * self.cos_theta.data
        Ay = A * self.sin_theta.data
        return torch.stack([Ax, Ay], dim=0)

    @property
    def B_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        Bpar = torch_interpolate(self.vr_grid.data, self.vr_axis.data, self.Bpar)
        Bperp = torch_interpolate(self.vr_grid.data, self.vr_axis.data, self.Bperp)

        Bpar = Bpar.reshape(*self.grid_size)
        Bperp = Bperp.reshape(*self.grid_size)

        Bxx = Bpar * self.cos_theta.data**2 + Bperp * self.sin_theta.data**2
        Byy = Bpar * self.sin_theta.data**2 + Bperp * self.cos_theta.data**2
        Bxy = (Bpar - Bperp) * self.cos_theta.data * self.sin_theta.data

        return torch.stack([Bxx, Byy, Bxy], dim=0)
