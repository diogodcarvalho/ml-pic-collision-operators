import torch
import torch.nn as nn
import numpy as np
from src.models.fp2d.tensor.conditioned.base import FokkerPlanck2DBaseTime
from src.models.utils import torch_interpolate


class FokkerPlanck2DTime_ABparperp(FokkerPlanck2DBaseTime):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        grid_size_dt: float,
        grid_dt: float,
        n_radial: int = -1,
        n_t: int = -1,
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
        self.grid_dt = grid_dt
        self.grid_size_dt = grid_size_dt
        if n_t == -1:
            self.n_t = grid_size_dt
        else:
            self.n_t = n_t
            self._t_axis = nn.Buffer(
                torch.linspace(0, grid_dt * self.grid_size_dt, self.n_t)
            )
        if n_radial == -1:
            self.n_radial = grid_size[0] // 2 + grid_size[0] % 2
        else:
            self.n_radial = n_radial
        self._init_params_dict.update(
            {
                "grid_dt": grid_dt,
                "grid_size_dt": grid_size_dt,
                "n_radial": n_radial,
                "n_t": n_t,
            }
        )

        self.A = nn.Parameter(torch.zeros((self.n_t, self.n_radial)))
        self.Bpar = nn.Parameter(torch.zeros((self.n_t, self.n_radial)))
        self.Bperp = nn.Parameter(torch.zeros((self.n_t, self.n_radial)))

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

    def _it(self, t: torch.Tensor) -> int:
        return (t / self.grid_dt).to(torch.int64)

    def _t_interpolate(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t_aux = t.unsqueeze(1)
        else:
            t_aux = t
        return torch_interpolate(
            t_aux.repeat(1, self.A.shape[1]),
            self._t_axis.unsqueeze(1).repeat(1, self.A.shape[1]),
            X,
            dim=0,
        )

    def Apar_real(self, t: torch.Tensor) -> np.ndarray:
        if self.n_t == self.grid_size_dt:
            A = self.A[self._it(t)]
        else:
            A = self._t_interpolate(self.A, t)
        return A.detach().cpu().numpy() * self.grid_dx[0]

    def Bpar_real(self, t: torch.Tensor) -> np.ndarray:
        if self.n_t == self.grid_size_dt:
            Bpar = self.Bpar[self._it(t)]
        else:
            Bpar = self._t_interpolate(self.Bpar, t)
        return Bpar.detach().cpu().numpy() * self.grid_dx[0] ** 2

    def Bperp_real(self, t: torch.Tensor) -> np.ndarray:
        if self.n_t == self.grid_size_dt:
            Bperp = self.Bpar[self._it(t)]
        else:
            Bperp = self._t_interpolate(self.Bperp, t)
        return Bperp.detach().cpu().numpy() * self.grid_dx[0] ** 2

    def A_grid(self, t: torch.Tensor) -> torch.Tensor:
        vr_grid = self.vr_grid.unsqueeze(0).repeat(t.shape[0], 1)
        vr_axis = self.vr_axis.unsqueeze(0).repeat(t.shape[0], 1)
        if self.n_t == self.grid_size_dt:
            A = torch_interpolate(vr_grid, vr_axis, self.A[self._it(t)][:, 0])
        else:
            A = self._t_interpolate(self.A, t)
            A = torch_interpolate(vr_grid, vr_axis, A, dim=1)
        A = A.reshape((t.shape[0], *self.grid_size))
        Ax = A * self.cos_theta
        Ay = A * self.sin_theta
        return torch.stack([Ax, Ay], dim=1)

    def B_grid(self, t: torch.Tensor) -> torch.Tensor:
        vr_grid = self.vr_grid.unsqueeze(0).repeat(t.shape[0], 1)
        vr_axis = self.vr_axis.unsqueeze(0).repeat(t.shape[0], 1)
        if self.n_t == self.grid_size_dt:
            Bpar = torch_interpolate(vr_grid, vr_axis, self.Bpar[self._it(t)][:, 0])
            Bperp = torch_interpolate(vr_grid, vr_axis, self.Bperp[self._it(t)][:, 0])
        else:
            Bpar = self._t_interpolate(self.Bpar, t)
            Bperp = self._t_interpolate(self.Bperp, t)
            Bpar = torch_interpolate(vr_grid, vr_axis, Bpar, dim=1)
            Bperp = torch_interpolate(vr_grid, vr_axis, Bperp, dim=1)

        Bpar = Bpar.reshape((t.shape[0], *self.grid_size))
        Bperp = Bperp.reshape((t.shape[0], *self.grid_size))

        Bxx = Bpar * self.cos_theta**2 + Bperp * self.sin_theta**2
        Byy = Bpar * self.sin_theta**2 + Bperp * self.cos_theta**2
        Bxy = (Bpar - Bperp) * self.cos_theta * self.sin_theta

        return torch.stack([Bxx, Byy, Bxy], dim=1)
