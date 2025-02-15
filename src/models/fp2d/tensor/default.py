import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from src.models.fp2d.base import FokkerPlanck2DBase


class FokkerPlanck2D(FokkerPlanck2DBase):

    A: jax.Array
    B: jax.Array

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        ensure_non_negative_f: bool = True,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
        )
        self.A = jnp.zeros((2, grid_size[0], grid_size[1]))  # [Ax, Ay]
        self.B = jnp.zeros((3, grid_size[0], grid_size[1]))  # [Bxx, Byy, Bxy]

    @property
    def A_grid(self) -> jax.Array:
        return self.A

    @property
    def B_grid(self) -> jax.Array:
        return self.B

    def load_from_numpy(self, A: np.ndarray, B: np.ndarray) -> "FokkerPlanck2D":
        assert A.shape == self.A.shape
        assert B.shape == self.B.shape
        new_model = eqx.tree_at(lambda m: m.A, self, jnp.array(A))
        new_model = eqx.tree_at(lambda m: m.B, new_model, jnp.array(B))
        return new_model

    def get_first_deriv_norm(self) -> jax.Array:
        return (
            jnp.mean(jnp.abs(self.A[:, 1:] - self.A[:, :-1]))  # dAdx
            + jnp.mean(jnp.abs(self.A[:, :, 1:] - self.A[:, :, :-1]))  # dAdy
            + jnp.mean(jnp.abs(self.B[:, 1:] - self.B[:, :-1]))  # dBdx
            + jnp.mean(jnp.abs(self.B[:, :, 1:] - self.B[:, :, :-1]))  # dBdy
        )

    def get_second_deriv_norm(self) -> jax.Array:
        return (
            jnp.mean(
                jnp.abs(self.A[:, 2:] - 2 * self.A[:, 1:-1] + self.A[:, :-2])
            )  # dAdx2
            + jnp.mean(
                jnp.abs(self.A[:, :, 2:] - 2 * self.A[:, :, 1:-1] + self.A[:, :, :-2])
            )  # dAdy2
            + jnp.mean(
                jnp.abs(self.B[:, 2:] - 2 * self.B[:, 1:-1] + self.B[:, :-2])
            )  # dBdx2
            + jnp.mean(
                jnp.abs(self.B[:, :, 2:] - 2 * self.B[:, :, 1:-1] + self.B[:, :, :-2])
            )  # dBdy2
        )

    def __repr__(self):
        return (
            f"FokkerPlanck2D(A=Array{self.A.shape}, B=Array{self.B.shape}, "
            + f"grid_size={self.grid_size}, grid_range={self.grid_range}, "
            + f"dx={self.dx}, ensure_non_negative_f={self.ensure_non_negative_f})"
        )
