import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec


class FokkerPlanck2DBase(eqx.Module):

    grid_size: tuple[int, int]
    grid_range: tuple[float, float]
    dx: tuple[float, float]
    ensure_non_negative_f: bool

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        ensure_non_negative_f: bool = True,
    ):
        assert len(grid_size) == 2
        self.dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.ensure_non_negative_f = ensure_non_negative_f

    def _grad(self, f: jax.Array, axis: int) -> jax.Array:
        # note: division by dx not used to avoid problems with numerical precision
        return (jnp.roll(f, -1, axis) - jnp.roll(f, 1, axis)) / 2.0

    def _grad2(self, f: jax.Array, axis: int) -> jax.Array:
        # note: division by dx^2 not used to avoid problems with numerical precision
        return jnp.roll(f, -1, axis) - 2 * f + jnp.roll(f, 1, axis)

    @property
    def A_grid(self) -> jax.Array:
        raise NotImplementedError

    @property
    def B_grid(self) -> jax.Array:
        raise NotImplementedError

    @property
    def A_grid_real(self) -> np.ndarray:
        # multiply by the resolution so that A/B have the right units
        return np.array(self.A_grid) * np.array(self.dx).reshape((2, 1, 1))

    @property
    def B_grid_real(self) -> np.ndarray:
        # multiply by the resolution so that A/B have the right units
        return np.array(self.B_grid) * np.array(
            [self.dx[0] ** 2, self.dx[1] ** 2, np.prod(self.dx)]
        ).reshape((3, 1, 1))

    def load_from_numpy(self, A: np.ndarray, B: np.ndarray) -> "FokkerPlanck2DBase":
        raise NotImplementedError

    def plot(self, save_to: str | None = None):
        fig = plt.figure(figsize=(12, 2.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3], figure=fig, wspace=0.4)

        # GridSpec for Ax and Ay
        gs_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.2)
        ax0 = fig.add_subplot(gs_A[0])
        ax1 = fig.add_subplot(gs_A[1])

        # GridSpec for Bxx, Byy, Bxy
        gs_B = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.2)
        ax2 = fig.add_subplot(gs_B[0])
        ax3 = fig.add_subplot(gs_B[1])
        ax4 = fig.add_subplot(gs_B[2])

        # Collect all axes
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

        xlabel = "$v_x[c]$"
        ylabel = "$v_y[c]$"
        plt.setp(ax, xlabel=xlabel)
        ax0.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        for a in [ax1, ax3, ax4]:
            a.set_yticklabels([])

        if save_to is not None:
            plt.savefig(save_to, dpi=300)
        plt.show()
        plt.close()

    def __call__(self, f: jax.Array) -> jax.Array:
        Af = self.A_grid * f
        Bf = self.B_grid * f
        gradv_Af = self._grad(Af[0], axis=0) + self._grad(Af[1], axis=1)
        gradvv_Bf = (
            self._grad2(Bf[0], 0)
            + self._grad2(Bf[1], 1)
            + self._grad(self._grad(Bf[2], 1), 0)
            + self._grad(self._grad(Bf[2], 0), 1)
        )
        df = -gradv_Af + gradvv_Bf / 2.0
        f += df
        if self.ensure_non_negative_f:
            f = jnp.maximum(f, 0)
        return f
