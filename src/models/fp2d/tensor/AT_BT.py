import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from src.models.fp2d.base import FokkerPlanck2DBase


class FokkerPlanck2D_AT_BT(FokkerPlanck2DBase):
    """
    Fokker-Planck model whish assumes that:
        1. Ax = Ay^T
        2. Bxx = Byy^T
        3. Bxy is free OR zero
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
        self.A = jnp.zeros(self.grid_size)
        self.B = jnp.zeros(self.grid_size)
        self.Bxy = jnp.zeros(self.grid_size)

    @property
    def A_grid(self) -> jax.Array:
        return jnp.stack([self.A, self.A.T], axis=0)

    @property
    def B_grid(self) -> jax.Array:
        if self.zero_Bxy:
            return jnp.stack([self.B, self.B.T, jnp.zeros_like(self.B)], axis=0)
        else:
            return jnp.stack([self.B, self.B.T, self.Bxy], axis=0)
