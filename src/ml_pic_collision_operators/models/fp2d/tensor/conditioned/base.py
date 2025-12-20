import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from typing import Any
from ml_pic_collision_operators.models.utils import torch_interpolate_uniform_firstdim


class FokkerPlanck2DBaseTime(nn.Module):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        grid_size_dt: float,
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

        self.A = None
        self.B = None
        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.grid_size_dt = grid_size_dt
        self.grid_dt = grid_dt
        self.guard_cells = guard_cells
        self.ensure_non_negative_f = ensure_non_negative_f
        self.ensure_non_negative_B = ensure_non_negative_B

        if n_t == -1:
            self.n_t = grid_size_dt
        else:
            self.n_t = n_t
            self._t_axis = nn.Buffer(
                torch.linspace(0, grid_dt * self.grid_size_dt, self.n_t)
            )

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "grid_size_dt": grid_size_dt,
            "grid_dt": grid_dt,
            "n_t": n_t,
            "ensure_non_negative_f": ensure_non_negative_f,
            "ensure_non_negative_B": ensure_non_negative_B,
            "guard_cells": guard_cells,
        }

    def _grad(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        if self.guard_cells:
            return torch.gradient(f, dim=axis)[0]
        else:
            return torch.gradient(f, dim=axis, edge_order=2)[0]

    def _grad2(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        grad2f = torch.roll(f, -1, axis) - 2 * f + torch.roll(f, 1, axis)

        if self.guard_cells:
            return grad2f

        if axis == 1:
            # left x-boundary
            grad2f[:, 0] = 2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]
            # right x-boundary
            grad2f[:, -1] = 2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]
        elif axis == 2:
            # left y-boundary
            grad2f[:, :, 0] = (
                2 * f[:, :, 0] - 5 * f[:, :, 1] + 4 * f[:, :, 2] - f[:, :, 3]
            )
            # right y-boundary
            grad2f[:, :, -1] = (
                2 * f[:, :, -1] - 5 * f[:, :, -2] + 4 * f[:, :, -3] - f[:, :, -4]
            )
        else:
            raise ValueError(f"Invalid axis: {axis}")

        return grad2f

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    def _it(self, t: torch.Tensor) -> int:
        return (torch.round(t / self.grid_dt)).to(torch.int64)

    def _t_interpolate(self, X: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch_interpolate_uniform_firstdim(
            t=t.flatten(),
            t0=float(self._t_axis[0]),
            dt=float(self._t_axis[1] - self._t_axis[0]),
            X=X,
            extrapolate="linear",
        )

    def A_grid(self, t: torch.Tensor) -> np.ndarray:
        A = self._t_interpolate(self.A, t)
        return A

    def B_grid(self, t: torch.Tensor) -> np.ndarray:
        return self._t_interpolate(self.B, t)

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

    def load_from_numpy(self, A: np.ndarray, B: np.ndarray):
        assert A.shape == (self.n_t, 2, *self.grid_size)
        assert B.shape == (self.n_t, 3, *self.grid_size)
        with torch.no_grad():
            A = torch.Tensor(A).to(torch.float32)
            B = torch.Tensor(B).to(torch.float32)
            cloned_model = copy.deepcopy(self)  # Create a new instance
            cloned_model.A = A
            cloned_model.B = B
        return cloned_model

    def plot(
        self, conditioners: torch.Tensor, save_to: str | None = None, show: bool = True
    ):
        fig = plt.figure(figsize=(12, 2.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3], figure=fig, wspace=0.4)

        gs_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.2)
        ax0 = fig.add_subplot(gs_A[0])
        ax1 = fig.add_subplot(gs_A[1])

        gs_B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.2)
        ax2 = fig.add_subplot(gs_B[0])
        ax3 = fig.add_subplot(gs_B[1])
        ax4 = fig.add_subplot(gs_B[2])

        ax = [ax0, ax1, ax2, ax3, ax4]

        A_grid = self.A_grid_real(conditioners)
        B_grid = self.B_grid_real(conditioners)
        Ax = A_grid[0]
        Ay = A_grid[1]
        Bxx = B_grid[0]
        Byy = B_grid[1]
        Bxy = B_grid[2]

        kwargs = {
            "origin": "lower",
            "extent": self.grid_range,
            "interpolation": None,
        }

        kwargs_A = dict(kwargs)
        kwargs_A["vmin"] = -np.max(np.abs(Ax))
        kwargs_A["vmax"] = np.max(np.abs(Ax))
        kwargs_A["cmap"] = "bwr"

        im0 = ax0.imshow(Ax.T, **kwargs_A)
        ax0.set_title(r"$\mathbf{A}_x$")

        im1 = ax1.imshow(Ay.T, **kwargs_A)
        ax1.set_title(r"$\mathbf{A}_y$")

        cbaxes_A = ax1.inset_axes([1.05, 0, 0.05, 1])
        cbar = fig.colorbar(im1, cax=cbaxes_A, orientation="vertical")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_ylabel(f"$[{self.grid_units[1:-1]}\omega_p]$")

        kwargs_B = dict(kwargs)
        kwargs_B["vmin"] = -np.max(np.abs(Bxx))
        kwargs_B["vmax"] = np.max(np.abs(Bxx))
        kwargs_B["cmap"] = "BrBG"

        im2 = ax2.imshow(Bxx.T, **kwargs_B)
        ax2.set_title(r"$\mathbf{B}_{xx}$")

        im3 = ax3.imshow(Byy.T, **kwargs_B)
        ax3.set_title(r"$\mathbf{B}_{yy}$")

        im4 = ax4.imshow(Bxy.T, **kwargs_B)
        ax4.set_title(r"$\mathbf{B}_{xy}$")

        cbaxes_B = ax4.inset_axes([1.05, 0, 0.05, 1])
        cbar = fig.colorbar(im4, cax=cbaxes_B, orientation="vertical")
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_ylabel(f"$[{self.grid_units[1:-1]}^2\omega_p]$")

        xlabel = "$v_x[v_{th}]$"
        ylabel = "$v_y[v_{th}]$"
        plt.setp(ax, xlabel=xlabel)
        ax0.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        plt.setp(ax, xticks=[self.grid_range[0], 0, self.grid_range[1]])
        plt.setp(ax, yticks=[self.grid_range[2], 0, self.grid_range[3]])
        for a in [ax1, ax3, ax4]:
            a.set_yticklabels([])

        if save_to is not None:
            plt.savefig(save_to, dpi=300)
        if show:
            plt.show()
        plt.close()

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
        conditioners: torch.Tensor,
    ) -> torch.Tensor:
        c_unique, reverse_indices = torch.unique(
            conditioners, return_inverse=True, dim=0
        )
        A = self.A_grid(c_unique)
        B = self.B_grid(c_unique)

        if self.ensure_non_negative_B:
            B[:, :2] = torch.clamp(B[:, :2], min=0)
        A = A[reverse_indices]
        B = B[reverse_indices]
        # from here on is the same as before
        Af = A * f.unsqueeze(1)
        Bf = B * f.unsqueeze(1)

        if self.guard_cells:
            Af = F.pad(Af, (1, 1, 1, 1), "constant", 0)
            Bf = F.pad(Bf, (1, 1, 1, 1), "constant", 0)

        gradv_Af = self._grad(Af[:, 0], axis=1) + self._grad(Af[:, 1], axis=2)
        gradvv_Bf = (
            self._grad2(Bf[:, 0], 1)
            + self._grad2(Bf[:, 1], 2)
            + self._grad(self._grad(Bf[:, 2], 2), 1)
            + self._grad(self._grad(Bf[:, 2], 1), 2)
        )

        if self.guard_cells:
            gradv_Af = gradv_Af[:, 1:-1, 1:-1]
            gradvv_Bf = gradvv_Bf[:, 1:-1, 1:-1]

        df = -gradv_Af + gradvv_Bf / 2.0
        if isinstance(dt, torch.Tensor):
            f = f + df * dt.unsqueeze(1).unsqueeze(2)
        else:
            f = f + df * dt
        if self.ensure_non_negative_f:
            f = torch.clamp(f, min=0)
        return f
