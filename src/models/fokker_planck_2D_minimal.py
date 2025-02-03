import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .fokker_planck_2D_base import FokkerPlanck2DBase


class FokkerPlanck2DMinimal(FokkerPlanck2DBase):

    A_r: jax.Array
    b: jax.Array

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
        assert grid_size[0] == grid_size[1]
        assert grid_range[0] == grid_range[2]
        assert grid_range[1] == grid_range[3]
        assert grid_dx[0] == grid_dx[1]
        self.A_r = jnp.arange(grid_size[0] // 2 + grid_size[0] % 2)[::-1].reshape(-1, 1)
        self.b = jnp.zeros(1)

    @property
    def A_grid(self) -> jax.Array:
        A_grid = self.A_r * jnp.ones((1, self.grid_size[1]))
        A_grid = jnp.concatenate(
            [A_grid, -jnp.flip(A_grid, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        A_grid = jnp.stack([A_grid, A_grid.T], axis=0)
        return A_grid

    @property
    def B_grid(self) -> jax.Array:
        return self.b * jnp.ones((3, *self.grid_size))

    def load_from_numpy(
        self, A_r: np.ndarray, b: np.ndarray
    ) -> "FokkerPlanck2DMinimal":
        assert A_r.shape == self.A.shape
        assert b.shape == self.B.shape
        new_model = eqx.tree_at(lambda m: m.A_r, self, jnp.array(A_r))
        new_model = eqx.tree_at(lambda m: m.b, new_model, jnp.array(b))
        return new_model

    def __repr__(self):
        return (
            f"FokkerPlanck2DMinimal(A_r=Array{self.A_r.shape}, b=Array{self.b.shape},"
            + f"grid_size={self.grid_size}, grid_range={self.grid_range},"
            + f"dx={self.dx}, ensure_non_negative_f={self.ensure_non_negative_f})"
        )
