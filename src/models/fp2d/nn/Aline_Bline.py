import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Callable

from src.models.fp2d.nn import FokkerPlanck2DNNBase


class FokkerPlanck2DNN_Aline_Bline(FokkerPlanck2DNNBase):
    """
    This model parametrizes each A_i, B_ij using an independet (equivalent) MLPs:

        A_i(vx, vy) = MPL_A_i(vx, vy)
        B_ij(vx, vy) = MPL_B_ij(vx, vy)

    No constraints are applied.
    """

    # A: tuple[eqx.Module, eqx.Module]
    # B: tuple[eqx.Module, eqx.Module, eqx.Module]
    A: eqx.Module
    B: eqx.Module

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
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            random_seed=random_seed,
            ensure_non_negative_f=ensure_non_negative_f,
        )

    def _init_NN(
        self,
        depth: int,
        width_size,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        key: jax.random.KeyArray,
    ):
        self.A = eqx.nn.MLP(
            in_size="scalar",
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self.B = eqx.nn.MLP(
            in_size="scalar",
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

    def _init_v_grid(self):
        # bin center positions
        vx = jnp.linspace(*self.grid_range[:2], self.grid_size[0], endpoint=False)
        vx += self.dx[0] / 2.0
        self.v_grid = vx[self.grid_size[0] // 2 :]

    @property
    def A_grid(self) -> jax.Array:
        v = jax.lax.stop_gradient(self.v_grid)
        Ax = jax.vmap(self.A)(v)
        Ax = jnp.concatenate(
            [Ax, -jnp.flip(Ax, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        Ax = Ax * jnp.ones((1, self.grid_size[1]))
        return jnp.stack([Ax, Ax.T], axis=0)

    @property
    def B_grid(self) -> jax.Array:
        v = jax.lax.stop_gradient(self.v_grid)
        Bxx = jax.vmap(self.B)(v)
        Bxx = jnp.concatenate(
            [Bxx, jnp.flip(Bxx, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        Bxx = Bxx * jnp.ones((1, self.grid_size[1]))
        return jnp.stack([Bxx, Bxx.T, jnp.zeros_like(Bxx)], axis=0)
