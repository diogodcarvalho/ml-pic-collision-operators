import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec


class FokkerPlanck2D(eqx.Module):

    grid_size: tuple[int, int]
    grid_range: tuple[float, float]
    dx: tuple[float, float]
    ensure_non_negative_f: bool
    A: jax.Array
    B: jax.Array

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
        self.A = jnp.zeros((2, grid_size[0], grid_size[1]))
        self.B = jnp.zeros((2, 2, grid_size[0], grid_size[1]))

    def _grad(self, f: jax.Array, axis: int) -> jax.Array:
        return (jnp.roll(f, -1, axis) - jnp.roll(f, 1, axis)) / (2 * self.dx[axis])

    def _grad2(self, f: jax.Array, axis: int) -> jax.Array:
        return (jnp.roll(f, -1, axis) - 2 * f + jnp.roll(f, 1, axis)) / (
            self.dx[axis] ** 2
        )

    def plot_not_so_pretty(self, save_to: str | None = None):
        fig, ax = plt.subplots(1, 6, figsize=(12, 4))

        Ax = np.array(self.A[0])
        Ay = np.array(self.A[1])
        Bxx = np.array(self.B[0, 0])
        Bxy = np.array(self.B[0, 1])
        Byx = np.array(self.B[1, 0])
        Byy = np.array(self.B[1, 1])

        kwargs = {
            "origin": "lower",
            "extent": self.grid_range,
            "interpolation": None,
        }

        kwargs_A = dict(kwargs)
        kwargs_A["vmin"] = -np.max(np.abs(Ax))
        kwargs_A["vmax"] = np.max(np.abs(Ax))
        kwargs_A["cmap"] = "bwr"

        im0 = ax[0].imshow(Ax.T, **kwargs_A)
        ax[0].set_title(r"$\mathbf{A}_x$")

        im1 = ax[1].imshow(Ay.T, **kwargs_A)
        ax[1].set_title(r"$\mathbf{A}_y$")

        cbaxes = ax[1].inset_axes([1.05, 0, 0.05, 1])
        cbar = fig.colorbar(im1, cax=cbaxes, orientation="vertical")

        kwargs_B = dict(kwargs)
        kwargs_B["vmin"] = -np.max(np.abs(Bxx))
        kwargs_B["vmax"] = np.max(np.abs(Bxx))
        kwargs_B["cmap"] = "BrBG"

        im2 = ax[2].imshow(Bxx.T, **kwargs_B)
        ax[2].set_title(r"$\mathbf{B}_{xx}$")

        im3 = ax[3].imshow(Byy.T, **kwargs_B)
        ax[3].set_title(r"$\mathbf{B}_{yy}$")

        im4 = ax[4].imshow(Bxy.T, **kwargs_B)
        ax[4].set_title(r"$\mathbf{B}_{xy}$")

        im5 = ax[5].imshow(Byx.T, **kwargs_B)
        ax[5].set_title(r"$\mathbf{B}_{yx}$")

        cbaxes = ax[5].inset_axes([1.05, 0, 0.05, 1])
        cbar = fig.colorbar(im5, cax=cbaxes, orientation="vertical")

        xlabel = "$v1[c]$"
        ylabel = "$v2[c]$"
        plt.setp(ax, xlabel=xlabel)
        ax[0].set_ylabel(ylabel)
        ax[2].set_ylabel(ylabel)

        for a in [ax[1], *ax[3:]]:
            a.set_yticklabels([])

        # plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to, dpi=200)
        plt.show()
        plt.close()

    def plot(self, save_to: str | None = None):
        fig = plt.figure(figsize=(12, 2.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 4], figure=fig, wspace=0.4)

        # GridSpec for Ax and Ay
        gs_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.2)
        ax0 = fig.add_subplot(gs_A[0])
        ax1 = fig.add_subplot(gs_A[1])

        # GridSpec for Bxx, Bxy, Byx, Byy
        gs_B = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.2)
        ax2 = fig.add_subplot(gs_B[0])
        ax3 = fig.add_subplot(gs_B[1])
        ax4 = fig.add_subplot(gs_B[2])
        ax5 = fig.add_subplot(gs_B[3])

        # Collect all axes
        ax = [ax0, ax1, ax2, ax3, ax4, ax5]

        Ax = np.array(self.A[0])
        Ay = np.array(self.A[1])
        Bxx = np.array(self.B[0, 0])
        Bxy = np.array(self.B[0, 1])
        Byx = np.array(self.B[1, 0])
        Byy = np.array(self.B[1, 1])

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
        fig.colorbar(im1, cax=cbaxes_A, orientation="vertical")

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

        im5 = ax5.imshow(Byx.T, **kwargs_B)
        ax5.set_title(r"$\mathbf{B}_{yx}$")

        cbaxes_B = ax5.inset_axes([1.05, 0, 0.05, 1])
        fig.colorbar(im5, cax=cbaxes_B, orientation="vertical")

        xlabel = "$v1[c]$"
        ylabel = "$v2[c]$"
        plt.setp(ax, xlabel=xlabel)
        ax0.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

        for a in [ax1, ax3, ax4, ax5]:
            a.set_yticklabels([])

        if save_to is not None:
            plt.savefig(save_to, dpi=300)
        plt.show()
        plt.close()

    def get_first_deriv_norm(self) -> jax.Array:
        return (
            jnp.mean(jnp.abs(self.A[:, 1:] - self.A[:, :-1]))  # dAdx
            + jnp.mean(jnp.abs(self.A[:, :, 1:] - self.A[:, :, :-1]))  # dAdy
            + jnp.mean(jnp.abs(self.B[:, :, 1:] - self.B[:, :, :-1]))  # dBdx
            + jnp.mean(jnp.abs(self.B[:, :, :, 1:] - self.B[:, :, :, :-1]))  # dBdy
        )

    def __call__(self, f: jax.Array) -> jax.Array:
        Af = self.A * f
        Bf = self.B * f
        gradv_Af = self._grad(Af[0], axis=0) + self._grad(Af[1], axis=1)
        gradvv_Bf = (
            self._grad2(Bf[0, 0], 0)
            + self._grad2(Bf[1, 1], 1)
            + self._grad(self._grad(Bf[0, 1], 1), 0)
            + self._grad(self._grad(Bf[1, 0], 0), 1)
        )
        df = -gradv_Af + gradvv_Bf / 2.0
        f += df
        if self.ensure_non_negative_f:
            f = jnp.maximum(f, 0)
        return f
