import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
import numpy as np


class FokkerPlanck2D(eqx.Module):

    grid_size: tuple[int, int]
    dx: tuple[float, float]
    A: jax.Array
    B: jax.Array

    def __init__(self, grid_size: tuple[int, int], grid_dx: tuple[float, float]):
        assert len(grid_size) == 2
        self.dx = grid_dx
        self.grid_size = grid_size
        self.A = jnp.zeros((2, grid_size[0], grid_size[1]))
        self.B = jnp.zeros((2, 2, grid_size[0], grid_size[1]))

    def _grad(self, f: jax.Array, axis: int) -> jax.Array:
        return (jnp.roll(f, -1, axis) - jnp.roll(f, 1, axis)) / (2 * self.dx[axis])

    def _grad2(self, f: jax.Array, axis: int) -> jax.Array:
        return (jnp.roll(f, -1, axis) - 2 * f + jnp.roll(f, 1, axis)) / (
            self.dx[axis] ** 2
        )

    def plot(self, save_to: str | None = None):
        fig, ax = plt.subplots(1, 6, figsize=(24, 4))

        Ax = np.array(self.A[0])
        Ay = np.array(self.A[1])
        Bxx = np.array(self.B[0, 0])
        Bxy = np.array(self.B[0, 1])
        Byx = np.array(self.B[1, 0])
        Byy = np.array(self.B[1, 1])

        kwargs = {
            "origin": "lower",
            #    "extent": np.array(self.dx) * np.self.grid_size / 0.1,
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

        plt.tight_layout()
        if save_to is not None:
            plt.savefig(save_to, dpi=200)
        plt.show()

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
        return f + df
