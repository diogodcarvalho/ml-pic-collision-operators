import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from ml_pic_collision_operators.models.fp3d.nn.base import FokkerPlanck3D_NN_Base
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck3D_NN_AD_ParPerp(FokkerPlanck3D_NN_Base):
    """Fokker-Planck 3D Neural Network Model with Parallel-Perpendicular Symmetry.

    This model parametrizes A_par, D_perp, and delta = (D_par - D_perp) using independent MLPs
    that take the radial speed ||v|| as input.

    Trainable modules are:

        Apar_over_v   — MLP(1→1), represents A_par / ||v||
        Dperp         — MLP(1→1), represents D_perp directly
        delta_over_v  — MLP(1→1), represents (D_par - D_perp) / ||v||

    D_par is derived from:

        D_perp(v) = Dperp(||v||)
        D_par(v)  = Dperp(||v||) + ||v|| * delta_over_v(||v||)

    The boundary conditions at v=0 are enforced by construction via multiplicative factors:

        A_par(0) = 0             via  A_par(v)  = ||v|| * Apar_over_v(||v||)
        D_par(0) = D_perp(0)     via  delta(v)  = ||v|| * delta_over_v(||v||)

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
    """

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        grid_range: tuple[float, float, float, float, float, float],
        grid_dx: tuple[float, float, float],
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
        # A_par(v) = ||v|| * Apar_over_v(||v||)
        self.Apar_over_v = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        # D_perp(v) = Dperp(||v||)
        self.Dperp = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        # D_par(v) = Dperp(v) + ||v|| * delta_over_v(||v||)
        self.delta_over_v = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    def _init_v_grid(self, normalize: bool):
        vx, vy, vz = self._default_vx_vy_vz(normalize)
        VX, VY, VZ = torch.meshgrid(vx, vy, vz, indexing="ij")
        self.vr_grid = nn.Buffer(torch.sqrt(VX**2 + VY**2 + VZ**2).reshape(-1, 1))
        # azimuthal angle in xy-plane
        theta = torch.atan2(VY, VX)
        self.cos_theta = nn.Buffer(torch.cos(theta))
        self.sin_theta = nn.Buffer(torch.sin(theta))
        # polar angle from +z axis
        phi = torch.atan2(torch.sqrt(VX**2 + VY**2), VZ)
        self.cos_phi = nn.Buffer(torch.cos(phi))
        self.sin_phi = nn.Buffer(torch.sin(phi))

    @property
    def vr_axis(self) -> torch.Tensor:
        vr = torch.unique(self.vr_grid.detach().squeeze())
        if self.normalize_v_grid:
            vr, _, _ = self._denormalize_v(
                vr, torch.zeros_like(vr), torch.zeros_like(vr)
            )
        return vr

    @property
    def Apar_real(self) -> np.ndarray:
        vr = torch.unique(self.vr_grid.detach()).reshape(-1, 1)
        return (vr * self.Apar_over_v(vr)).detach().cpu().numpy() * self.grid_dx[0]

    @property
    def Dpar_real(self) -> np.ndarray:
        vr = torch.unique(self.vr_grid.detach()).reshape(-1, 1)
        Dpar = self.Dperp(vr) + vr * self.delta_over_v(vr)
        return Dpar.detach().cpu().numpy() * self.grid_dx[0] ** 2

    @property
    def Dperp_real(self) -> np.ndarray:
        vr = torch.unique(self.vr_grid.detach()).reshape(-1, 1)
        return self.Dperp(vr).detach().cpu().numpy() * self.grid_dx[0] ** 2

    @property
    def A_grid(self) -> torch.Tensor:
        vr = self.vr_grid.data
        # ||v|| * Apar_over_v(||v||) enforces A_par(0) = 0
        Apar = (vr * self.Apar_over_v(vr)).view(*self.grid_size)
        Ax = Apar * self.cos_theta.data * self.sin_phi.data
        Ay = Apar * self.sin_theta.data * self.sin_phi.data
        Az = Apar * self.cos_phi.data
        return torch.stack([Ax, Ay, Az], dim=0)

    @property
    def D_grid(self) -> torch.Tensor:
        vr = self.vr_grid.data
        Dperp = self.Dperp(vr).view(*self.grid_size)
        # ||v|| * delta_over_v(||v||) enforces D_par(0) = D_perp(0)
        delta_flat = vr * self.delta_over_v(vr)
        delta = delta_flat.view(*self.grid_size)

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
