import torch
import torch.nn as nn
import numpy as np

from typing import Any

from ml_pic_collision_operators.models.utils import torch_interpolate_uniform_firstdim
from ml_pic_collision_operators.models.fp2d.fp2d_utils import fp2d_step, plot_operator


class FokkerPlanck2D_Tensor_Base_TimeDependent(nn.Module):
    """Base class for Fokker-Planck 2D Tensor models with time-dependence.

    Child class should implement:
        `A_grid` and `B_grid` - properties that compute the A and B coefficients on the
            velocity grid for a given time t.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        grid_size_t: int,
        grid_dt: float,
        n_t: int = -1,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        includes_symmetry: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__()
        assert len(grid_size) == 2
        if includes_symmetry:
            assert grid_size[0] == grid_size[1]
            assert grid_range[0] == grid_range[2]
            assert grid_range[1] == grid_range[3]
            assert grid_dx[0] == grid_dx[1]

        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.grid_size_t = grid_size_t
        self.grid_dt = grid_dt
        self.guard_cells = guard_cells
        self.ensure_non_negative_f = ensure_non_negative_f
        self.ensure_non_negative_B = ensure_non_negative_B

        if n_t == -1:
            self.n_t = grid_size_t
        else:
            self.n_t = n_t
            self._t_axis = nn.Buffer(
                torch.linspace(0, grid_dt * self.grid_size_t, self.n_t)
            )

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "grid_size_t": grid_size_t,
            "grid_dt": grid_dt,
            "n_t": n_t,
            "ensure_non_negative_f": ensure_non_negative_f,
            "ensure_non_negative_B": ensure_non_negative_B,
            "guard_cells": guard_cells,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    def change_attribute(self, attr_name: str, attr_value: Any):
        if attr_name in [
            "ensure_non_negative_f",
            "ensure_non_negative_B",
            "guard_cells",
        ]:
            setattr(self, attr_name, attr_value)
        elif hasattr(self, attr_name):
            raise ValueError(
                f"Can not change attribute: {attr_name} after initialization"
            )
        else:
            raise KeyError(f"{type(self)} does not have attribute: {attr_name}")

    def _it(self, t: torch.Tensor) -> torch.Tensor:
        return (torch.round(t / self.grid_dt)).to(torch.int64)

    def _t_interpolate(self, F: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch_interpolate_uniform_firstdim(
            x=t.flatten(),
            x0=float(self._t_axis[0]),
            dx=float(self._t_axis[1] - self._t_axis[0]),
            f=F,
            extrapolate="linear",
        )

    def A_grid(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def B_grid(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def A_grid_real(self, t: torch.Tensor) -> np.ndarray:
        return np.array(self.A_grid(t).detach().cpu().numpy()) * np.array(
            self.grid_dx
        ).reshape((1, 2, 1, 1))

    def B_grid_real(self, t: torch.Tensor) -> np.ndarray:
        B = self.B_grid(t).detach().cpu()
        if self.ensure_non_negative_B:
            B[:, :2] = torch.clamp(B[:, :2], min=0)
        return np.array(B.numpy()) * np.array(
            [self.grid_dx[0] ** 2, self.grid_dx[1] ** 2, np.prod(self.grid_dx)]
        ).reshape((1, 3, 1, 1))

    def plot(self, t: torch.Tensor, save_to: str | None = None, show: bool = True):
        plot_operator(
            A=self.A_grid_real(t),
            B=self.B_grid_real(t),
            grid_range=self.grid_range,
            grid_units=self.grid_units,
            save_to=save_to,
            show=show,
        )

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t_unique, reverse_indices = torch.unique(t, return_inverse=True, dim=0)
        A = self.A_grid(t_unique)
        B = self.B_grid(t_unique)

        if self.ensure_non_negative_B:
            B[:, :2] = torch.clamp(B[:, :2], min=0)
        A = A[reverse_indices]
        B = B[reverse_indices]

        return fp2d_step(
            A=A,
            B=B,
            f=f,
            dt=dt,
            guard_cells=self.guard_cells,
            ensure_non_negative_f=self.ensure_non_negative_f,
        )
