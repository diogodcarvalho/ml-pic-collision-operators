import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class FokkerPlanck2DBase(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        ensure_non_negative_f: bool = True,
    ):
        super().__init__()
        assert len(grid_size) == 2
        assert len(grid_range) == 4
        assert len(grid_dx) == 2
        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.ensure_non_negative_f = ensure_non_negative_f

    def _grad(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.gradient(f, dim=axis, edge_order=2)[0]

    def _grad2(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        grad2f = torch.roll(f, -1, axis) - 2 * f + torch.roll(f, 1, axis)
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
    def A_grid(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def B_grid(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def A_grid_real(self) -> np.ndarray:
        return np.array(self.A_grid.detach().cpu().numpy()) * np.array(
            self.grid_dx
        ).reshape((2, 1, 1))

    @property
    def B_grid_real(self) -> np.ndarray:
        return np.array(self.B_grid.detach().cpu().numpy()) * np.array(
            [self.grid_dx[0] ** 2, self.grid_dx[1] ** 2, np.prod(self.grid_dx)]
        ).reshape((3, 1, 1))

    def load_from_numpy(self, A: np.ndarray, B: np.ndarray) -> "FokkerPlanck2DBase":
        raise NotImplementedError

    def plot(self, save_to: str | None = None):
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

        A_grid = self.A_grid_real
        B_grid = self.B_grid_real
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

        xlabel = f"$v_x{self.grid_units}$"
        ylabel = f"$v_y{self.grid_units}$"
        plt.setp(ax, xlabel=xlabel)
        ax0.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        plt.setp(ax, xticks=[self.grid_range[0], 0, self.grid_range[1]])
        plt.setp(ax, yticks=[self.grid_range[2], 0, self.grid_range[3]])
        for a in [ax1, ax3, ax4]:
            a.set_yticklabels([])

        if save_to is not None:
            plt.savefig(save_to, dpi=300)
        plt.show()
        plt.close()

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        Af = self.A_grid.unsqueeze(0) * f.unsqueeze(1)
        Bf = self.B_grid.unsqueeze(0) * f.unsqueeze(1)
        gradv_Af = self._grad(Af[:, 0], axis=1) + self._grad(Af[:, 1], axis=2)
        gradvv_Bf = (
            self._grad2(Bf[:, 0], 1)
            + self._grad2(Bf[:, 1], 2)
            + self._grad(self._grad(Bf[:, 2], 2), 1)
            + self._grad(self._grad(Bf[:, 2], 1), 2)
        )
        df = -gradv_Af + gradvv_Bf / 2.0
        if isinstance(dt, torch.Tensor):
            f = f + df * dt.unsqueeze(1).unsqueeze(2)
        else:
            f = f + df * dt

        if self.ensure_non_negative_f:
            f = torch.clamp(f, min=0)
        return f
