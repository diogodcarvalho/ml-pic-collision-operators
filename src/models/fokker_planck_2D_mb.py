import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .fokker_planck_2D_base import FokkerPlanck2DBase


class FokkerPlanck2Dmb(FokkerPlanck2DBase):

    m: jax.Array
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
        self.m = jnp.ones(1)
        self.b = jnp.ones(1)

    @property
    def A_grid(self) -> jax.Array:
        # compute along a line (A = 0 at center)
        # (grid_size,)
        A_line = self.m * (jnp.arange(self.grid_size[0]) - self.grid_size[0] // 2)
        # stack equal lines
        # (grid_size, grid_size)
        A_grid = A_line.reshape((self.grid_size[0], 1)) * jnp.ones(
            (1, self.grid_size[1])
        )
        # stack equivalent y dimension
        # (2, grid_size, grid_size)
        A_grid = jnp.stack([A_grid, A_grid.T], axis=0)
        return A_grid

    @property
    def B_grid(self) -> jax.Array:
        Bxx = self.b * jnp.ones(self.grid_size)
        B_grid = jnp.stack([Bxx, Bxx, jnp.zeros_like(Bxx)], axis=0)
        return B_grid

    def load_from_numpy(self, m: np.ndarray, b: np.ndarray) -> "FokkerPlanck2Dmb":
        assert m.shape == self.m.shape
        assert b.shape == self.b.shape
        new_model = eqx.tree_at(lambda m: m.m, self, jnp.array(m))
        new_model = eqx.tree_at(lambda m: m.b, new_model, jnp.array(b))
        return new_model

    def __repr__(self):
        return (
            f"FokkerPlanck2Dmb(m=Array{self.m.shape}, b=Array{self.b.shape},"
            + f"grid_size={self.grid_size}, grid_range={self.grid_range},"
            + f"dx={self.dx}, ensure_non_negative_f={self.ensure_non_negative_f})"
        )
