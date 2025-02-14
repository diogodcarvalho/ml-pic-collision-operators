import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from .fokker_planck_2D_base import FokkerPlanck2DBase


class FokkerPlanck2D_Aquad_Bquad(FokkerPlanck2DBase):
    """
    Fokker-Planck model whish assumes that:
        1. Ax is anti-symmetric along vx=0 and symmetric along vy=0
        2. Ay = Ax^T
        3. Bxx is symmetric along vx=0 and vy=0
        4. Byy = Bxx^T
        5. Bxy is anti-symmetric along vx=0 and vy=0 OR zero
    """

    A: jax.Array
    B: jax.Array
    Bxy: jax.Array
    zero_Bxy: bool

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        ensure_non_negative_f: bool = True,
        zero_Bxy: bool = True,
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
        self.zero_Bxy = zero_Bxy

        shape = (
            grid_size[0] // 2 + grid_size[0] % 2,
            grid_size[0] // 2 + grid_size[0] % 2,
        )
        self.A = jnp.zeros(shape)
        self.B = jnp.zeros(shape)
        self.Bxy = jnp.zeros(shape)

    @property
    def A_grid(self) -> jax.Array:
        Ax = jnp.concatenate(
            [self.A, -jnp.flip(self.A, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        Ax = jnp.concatenate(
            [Ax, jnp.flip(Ax, axis=1)[:, self.grid_size[0] % 2 :]],
            axis=1,
        )
        return jnp.stack([Ax, Ax.T], axis=0)

    @property
    def B_grid(self) -> jax.Array:
        Bxx = jnp.concatenate(
            [self.B, jnp.flip(self.B, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        Bxx = jnp.concatenate(
            [Bxx, jnp.flip(Bxx, axis=1)[:, self.grid_size[0] % 2 :]],
            axis=1,
        )
        if self.zero_Bxy:
            Bxy = jnp.zeros_like(Bxx)
        else:
            Bxy = jnp.concatenate(
                [self.Bxy, -jnp.flip(self.Bxy, axis=0)[self.grid_size[0] % 2 :]],
                axis=0,
            )
            Bxy = jnp.concatenate(
                [Bxy, -jnp.flip(Bxy, axis=1)[:, self.grid_size[0] % 2 :]],
                axis=1,
            )
        return jnp.stack([Bxx, Bxx.T, Bxy], axis=0)

    def load_from_numpy(
        self, A: np.ndarray, B: np.ndarray
    ) -> "FokkerPlanck2D_Aquad_Bquad":
        assert A.shape == self.A.shape
        assert B.shape == self.B.shape
        new_model = eqx.tree_at(lambda m: m.A, self, jnp.array(A))
        new_model = eqx.tree_at(lambda m: m.B, new_model, jnp.array(B))
        return new_model

    def __repr__(self):
        return (
            f"FokkerPlanck2D_Aquad_Bquad(A=Array{self.A.shape}, B=Array{self.B.shape}, "
            + f"Bxy=Array{self.Bxy.shape}, zero_Bxy={self.zero_Bxy}, "
            + f"grid_size={self.grid_size}, grid_range={self.grid_range}, "
            + f"dx={self.dx}, ensure_non_negative_f={self.ensure_non_negative_f})"
        )
