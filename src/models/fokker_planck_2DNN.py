import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from typing import Callable

from .fokker_planck_2D_base import FokkerPlanck2DBase


class FokkerPlanck2DNN(FokkerPlanck2DBase):
    """
    This model parametrizes A, B using an MLP i.e.

    A(v1,v2) = MLP_A(v1,v2) # 2 outputs
    B(v1,v2) = MLP_B(v1,v2) # 3 outputs

    No constraints are applied.
    """

    A: eqx.Module
    B: eqx.Module
    v_grid: jax.Array

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        depth: int,
        width_size: int,
        activation: Callable = jax.nn.relu,
        use_bias: bool = True,
        use_final_bias: bool = True,
        random_seed: int = 42,
        ensure_non_negative_f: bool = True,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
        )

        key = jax.random.PRNGKey(random_seed)

        if isinstance(activation, str):
            activation = eval(activation)
        print(activation)

        self.A = eqx.nn.MLP(
            in_size=2,
            out_size=2,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )
        self.B = eqx.nn.MLP(
            in_size=2,
            out_size=3,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        vx = jnp.linspace(*self.grid_range[:2], self.grid_size[0])
        vy = jnp.linspace(*self.grid_range[2:], self.grid_size[1])
        print(vx.shape, vy.shape)
        self.v_grid = jnp.stack(jnp.meshgrid(vx, vy, indexing="ij"), axis=-1).reshape(
            -1, 2
        )

        print(self.v_grid.shape)

    @property
    def A_grid(self) -> jax.Array:
        A_grid = jax.vmap(self.A)(self.v_grid).T
        A_grid = A_grid.reshape(2, *self.grid_size)
        return A_grid

    @property
    def B_grid(self) -> jax.Array:
        B_grid = jax.vmap(self.B)(self.v_grid).T
        B_grid = B_grid.reshape(3, *self.grid_size)
        return B_grid

    def __repr__(self):
        return (
            f"FokkerPlanck2DNN(A={self.A}, B={self.B}, "
            + f"grid_size={self.grid_size}, grid_range={self.grid_range}, "
            + f"dx={self.dx}, ensure_non_negative_f={self.ensure_non_negative_f})"
        )
