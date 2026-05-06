import torch
import torch.nn as nn
import numpy as np

from ml_pic_collision_operators.models.fp3d.base import (
    FokkerPlanck3D_Base,
)
from ml_pic_collision_operators.models.utils import torch_interpolate


class FokkerPlanck3D_Tensor_AD_ParPerp(FokkerPlanck3D_Base):
    """Fokker-Planck 3D Tensor Model with Parallel-Perpendicular Symmetry.

    This model parametrizes A_par, D_par, and D_perp as 1D radial profiles.

    The boundary conditions at v=0, A_par(0) = 0 and D_par(0) = D_perp(0), are
    enforced *by construction* by omitting the v=0 entry from the trainable
    tensors where it is fixed:

        Trainable parameters (state_dict):
            _Apar  of shape (n_radial - 1,)  — A_par at vr_axis[1:]
            Dpar   of shape (n_radial,)      — D_par at vr_axis (full)
            _Dperp of shape (n_radial - 1,)  — D_perp at vr_axis[1:]

        Full radial profiles (properties):
            Apar  = cat([0], _Apar)            so Apar(0) = 0
            Dpar  = Dpar                       (already full)
            Dperp = cat([Dpar[:1], _Dperp])    so Dperp(0) = Dpar(0)

    With theta the azimuth from +x in the xy-plane and phi the polar angle from +z,
    the radial unit vector is r̂ = (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)),
    the model enforces (writing delta = D_par - D_perp):

        A_x = A_par * sin(phi) * cos(theta)
        A_y = A_par * sin(phi) * sin(theta)
        A_z = A_par * cos(phi)
        D_xx = D_perp + delta * sin(phi)^2 * cos(theta)^2
        D_yy = D_perp + delta * sin(phi)^2 * sin(theta)^2
        D_zz = D_perp + delta * cos(phi)^2
        D_xy = delta * sin(phi)^2 * cos(theta) * sin(theta)
        D_xz = delta * sin(phi) * cos(phi) * cos(theta)
        D_yz = delta * sin(phi) * cos(phi) * sin(theta)

    Profiles are interpolated to the 3D velocity grid using the velocity magnitude
    at the center of each grid point.
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        grid_range: tuple[float, float, float, float, float, float],
        grid_dx: tuple[float, float, float],
        grid_units: str,
        n_radial: int = -1,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_D: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
            guard_cells=guard_cells,
            includes_symmetry=True,
        )

        if n_radial == -1:
            self.n_radial = grid_size[0] // 2 + grid_size[0] % 2
        else:
            self.n_radial = n_radial
        assert self.n_radial >= 2, "n_radial must be >= 2"
        self._init_params_dict.update({"n_radial": n_radial})

        # _Apar and _Dperp omit the v=0 entry — see class docstring
        self._Apar = nn.Parameter(torch.zeros(self.n_radial - 1))
        self.Dpar = nn.Parameter(torch.zeros(self.n_radial))
        self._Dperp = nn.Parameter(torch.zeros(self.n_radial - 1))

        # maximum |v| (diagonal of the 3D box)
        r_max = np.sqrt(
            self.grid_range[1] ** 2 + self.grid_range[3] ** 2 + self.grid_range[5] ** 2
        )
        # velocity magnitude along axis in which self.A is defined
        self.vr_axis = nn.Buffer(torch.linspace(0, r_max, self.n_radial))

        # get velocities at bin centers
        vx = torch.linspace(*self.grid_range[:2], self.grid_size[0] + 1)[:-1]
        vy = torch.linspace(*self.grid_range[2:4], self.grid_size[1] + 1)[:-1]
        vz = torch.linspace(*self.grid_range[4:6], self.grid_size[2] + 1)[:-1]
        vx += grid_dx[0] / 2.0
        vy += grid_dx[1] / 2.0
        vz += grid_dx[2] / 2.0
        VX, VY, VZ = torch.meshgrid(vx, vy, vz, indexing="ij")
        # velocity magnitude at bin centers
        self.vr_grid = nn.Buffer(torch.sqrt(VX**2 + VY**2 + VZ**2).flatten())
        # get angle of v=(vx,vy) with respect to x-axis
        theta = torch.atan2(VY, VX)
        self.cos_theta = nn.Buffer(torch.cos(theta))
        self.sin_theta = nn.Buffer(torch.sin(theta))
        # polar angle from +z axis: r̂ = (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi))
        phi = torch.atan2(torch.sqrt(VX**2 + VY**2), VZ)
        self.cos_phi = nn.Buffer(torch.cos(phi))
        self.sin_phi = nn.Buffer(torch.sin(phi))

    @property
    def Apar(self) -> torch.Tensor:
        return torch.cat([torch.zeros(1, device=self._Apar.device), self._Apar])

    @property
    def Dperp(self) -> torch.Tensor:
        return torch.cat([self.Dpar[:1], self._Dperp])

    @property
    def Apar_real(self) -> np.ndarray:
        return self.Apar.detach().cpu().numpy() * self.grid_dx[0]

    @property
    def Dpar_real(self) -> np.ndarray:
        return self.Dpar.detach().cpu().numpy() * self.grid_dx[0] ** 2

    @property
    def Dperp_real(self) -> np.ndarray:
        return self.Dperp.detach().cpu().numpy() * self.grid_dx[0] ** 2

    @property
    def A_grid(self) -> torch.Tensor:
        Apar = torch_interpolate(self.vr_grid.data, self.vr_axis.data, self.Apar)
        Apar = Apar.reshape(*self.grid_size)
        Ax = Apar * self.cos_theta.data * self.sin_phi.data
        Ay = Apar * self.sin_theta.data * self.sin_phi.data
        Az = Apar * self.cos_phi.data
        return torch.stack([Ax, Ay, Az], dim=0)

    @property
    def D_grid(self) -> torch.Tensor:
        # accessing .data avoids need for clone() and backprop errors in DDP
        Dpar = torch_interpolate(self.vr_grid.data, self.vr_axis.data, self.Dpar)
        Dperp = torch_interpolate(self.vr_grid.data, self.vr_axis.data, self.Dperp)

        Dpar = Dpar.reshape(*self.grid_size)
        Dperp = Dperp.reshape(*self.grid_size)

        delta = Dpar - Dperp
        cos_theta = self.cos_theta.data
        sin_theta = self.sin_theta.data
        cos_phi = self.cos_phi.data
        sin_phi = self.sin_phi.data

        Dxx = Dperp + delta * sin_phi**2 * cos_theta**2
        Dyy = Dperp + delta * sin_phi**2 * sin_theta**2
        Dzz = Dperp + delta * cos_phi**2
        Dxy = delta * sin_phi**2 * cos_theta * sin_theta
        Dxz = delta * sin_phi * cos_phi * cos_theta
        Dyz = delta * sin_phi * cos_phi * sin_theta

        return torch.stack([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], dim=0)
