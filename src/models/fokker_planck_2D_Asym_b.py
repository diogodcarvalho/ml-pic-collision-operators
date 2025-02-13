import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .fokker_planck_2D_base import FokkerPlanck2DBase


class FokkerPlanck2D_Asym_b(FokkerPlanck2DBase):
    """
    Fokker-Planck model whish assumes that:
        1. Ax does not depend on vy
        2. Ax is anti-symmetric along vx=0
        3. Ay = Ax^T
        4. Bxx = Byy = constant.
        5. Bxy = 0
    """

    Asym: jax.Array
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
        self.Asym = jnp.arange(grid_size[0] // 2 + grid_size[0] % 2)[::-1].reshape(
            -1, 1
        )
        self.b = jnp.ones(1)

    @property
    def A_grid(self) -> jax.Array:
        # assumes Ax doest not depend on vy
        # (grid_size/2, grid_size)
        A_grid = self.Asym * jnp.ones((1, self.grid_size[1]))
        # assume reflection symmetry along vx=0
        # (grid_size, grid_size)
        A_grid = jnp.concatenate(
            [A_grid, -jnp.flip(A_grid, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        # assume Ay is transposed of Ax
        # (2, grid_size, grid_size)
        A_grid = jnp.stack([A_grid, A_grid.T], axis=0)
        return A_grid

    @property
    def B_grid(self) -> jax.Array:
        # assume Bxx=Byy=constant and Bxy=0
        # (3, grid_size, grid_size)
        B_grid = jnp.stack(
            [
                self.b * jnp.ones(self.grid_size),
                self.b * jnp.ones(self.grid_size),
                jnp.zeros_like(self.grid_size),
            ],
            axis=0,
        )
        return B_grid

    def load_from_numpy(
        self, Asym: np.ndarray, b: np.ndarray
    ) -> "FokkerPlanck2D_Asym_b":
        assert Asym.shape == self.Asym.shape
        assert b.shape == self.b.shape
        new_model = eqx.tree_at(lambda m: m.Asym, self, jnp.array(Asym))
        new_model = eqx.tree_at(lambda m: m.b, new_model, jnp.array(b))
        return new_model

    def __repr__(self):
        return (
            f"FokkerPlanck2D_Asym_b(Asym=Array{self.Asym.shape}, b=Array{self.b.shape},"
            + f"grid_size={self.grid_size}, grid_range={self.grid_range},"
            + f"dx={self.dx}, ensure_non_negative_f={self.ensure_non_negative_f})"
        )
