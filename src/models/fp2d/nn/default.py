import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Callable

from src.models.fp2d.nn import FokkerPlanck2DNNBase


class FokkerPlanck2DNN(FokkerPlanck2DNNBase):
    """
    This model parametrizes each A_i, B_ij using an independet (equivalent) MLPs:

        A_i(vx, vy) = MPL_A_i(vx, vy)
        B_ij(vx, vy) = MPL_B_ij(vx, vy)

    No constraints are applied.
    """

    # A: tuple[eqx.Module, eqx.Module]
    # B: tuple[eqx.Module, eqx.Module, eqx.Module]
    Ax: eqx.Module
    Ay: eqx.Module
    Bxx: eqx.Module
    Byy: eqx.Module
    Bxy: eqx.Module

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
        normalize_v_grid: bool = True,
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
            normalize_v_grid=normalize_v_grid,
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
        self.Ax = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self.Ay = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self.Bxx = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self.Byy = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self.Bxy = eqx.nn.MLP(
            in_size=2,
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

    def _init_v_grid(self, normalize: bool):
        # bin center positions
        vx = jnp.linspace(*self.grid_range[:2], self.grid_size[0], endpoint=False)
        vy = jnp.linspace(*self.grid_range[2:], self.grid_size[1], endpoint=False)
        vx += self.dx[0] / 2.0
        vy += self.dx[1] / 2.0
        if normalize:
            vx /= jnp.std(vx)
            vy /= jnp.std(vy)
        VX, VY = jnp.meshgrid(vx, vy, indexing="ij")
        self.v_grid = jnp.stack([VX.flatten(), VY.flatten()], axis=-1)

    @property
    def A_grid(self) -> jax.Array:
        v = jax.lax.stop_gradient(self.v_grid)
        Ax = jax.vmap(self.Ax)(v)
        Ay = jax.vmap(self.Ay)(v)
        A_grid = jnp.concatenate([Ax.T, Ay.T], axis=0)
        A_grid = A_grid.reshape(2, *self.grid_size)
        return A_grid

    @property
    def B_grid(self) -> jax.Array:
        v = jax.lax.stop_gradient(self.v_grid)
        Bxx = jax.vmap(self.Bxx)(v)
        Byy = jax.vmap(self.Byy)(v)
        Bxy = jax.vmap(self.Bxy)(v)
        B_grid = jnp.concatenate([Bxx.T, Byy.T, Bxy.T], axis=0)
        B_grid = B_grid.reshape(3, *self.grid_size)
        return B_grid
