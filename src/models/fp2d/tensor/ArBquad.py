import torch
import torch.nn as nn
import numpy as np
from src.models.fp2d.tensor.base import FokkerPlanck2DTensorBase
from src.models.utils import torch_interpolate


class FokkerPlanck2D_ArBquad(FokkerPlanck2DTensorBase):

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

        shape = (
            grid_size[0] // 2 + grid_size[0] % 2,
            grid_size[0] // 2 + grid_size[0] % 2,
        )

        if n_radial == -1:
            self.n_radial = shape[0]
        else:
            self.n_radial = n_radial
        self._init_params_dict.update({"n_radial": n_radial})

        self.A = nn.Parameter(torch.zeros((1, self.n_radial)))
        self.B = nn.Parameter(torch.zeros((2, *shape)))

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
        # force cos(vx=0,vy=0) = sin(vx=0,vy=0) = 1 to ensure model
        # learns that A(vx=0,vy=0) = 0
        # otherwise, atan2 sets cos=1 and sin=0
        if grid_size[0] % 2:
            self.cos_theta[grid_size[0] // 2, grid_size[0] // 2] = 1.0
            self.sin_theta[grid_size[0] // 2, grid_size[0] // 2] = 1.0

    @property
    def A_grid(self) -> torch.Tensor:
        Ar = torch_interpolate(self.vr_grid, self.vr_axis, self.A[0])
        Ar = Ar.reshape(*self.grid_size)
        Ax = Ar * self.cos_theta
        Ay = Ar * self.sin_theta
        return torch.stack([Ax, Ay], dim=0)

    @property
    def B_grid(self) -> torch.Tensor:
        Bxx = torch.concatenate(
            [self.B[0], torch.flip(self.B[0], dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        Bxx = torch.concatenate(
            [Bxx, torch.flip(Bxx, dims=(1,))[:, self.grid_size[0] % 2 :]],
            dim=1,
        )
        Bxy = torch.concatenate(
            [self.B[1], -torch.flip(self.B[1], dims=(0,))[self.grid_size[0] % 2 :]],
            dim=0,
        )
        Bxy = torch.concatenate(
            [Bxy, -torch.flip(Bxy, dims=(1,))[:, self.grid_size[0] % 2 :]],
            dim=1,
        )
        return torch.stack([Bxx, Bxx.T, Bxy], dim=0)
