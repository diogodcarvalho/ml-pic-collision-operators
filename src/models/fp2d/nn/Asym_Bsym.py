import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Callable

from src.models.fp2d.nn import FokkerPlanck2DNNBase


class FokkerPlanck2DNN_Asym_Bsym(FokkerPlanck2DNNBase):
    """
    This model parametrizes A_x, B_xx and Bxy using a independet (equivalent) MLPs:

        A_x(vx, vy) = MPL_A_x(vx, vy)
        B_xx(vx, vy) = MPL_B_xx(vx, vy)
        B_xy(vx, vy) = MPL_B_xy(vx, vy)

    and enforces that:
        A_y = A_x^T
        B_yy = B_xx^T
    """

    A: eqx.Module
    B: tuple[eqx.Module, eqx.Module]

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
            in_size=2,
            out_size=1,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self.B = [
            eqx.nn.MLP(
                in_size=2,
                out_size=1,
                depth=depth,
                width_size=width_size,
                activation=activation,
                use_bias=use_bias,
                use_final_bias=use_final_bias,
                key=key,
            )
            for _ in range(2)
        ]

    def _init_v_grid(self):
        assert self.grid_size[0] == self.grid_size[1]
        assert self.grid_range[0] == self.grid_range[2]
        assert self.grid_range[1] == self.grid_range[3]
        assert self.grid_dx[0] == self.grid_dx[1]
        vx = jnp.linspace(*self.grid_range[:2], self.grid_size[0])
        vy = jnp.linspace(*self.grid_range[2:], self.grid_size[1])
        self.v_grid = jnp.stack(jnp.meshgrid(vx, vy, indexing="ij"), axis=-1).reshape(
            -1, 2
        )

    @property
    def A_grid(self) -> jax.Array:
        Ax = jax.vmap(self.A[0])(self.v_grid).T
        Ax = Ax.reshape(1, *self.grid_size)
        A_grid = jnp.concatenate([Ax, Ax.T], axis=0)
        return A_grid

    @property
    def B_grid(self) -> jax.Array:
        Bxx = jax.vmap(self.B[0])(self.v_grid).T
        Bxy = jax.vmap(self.B[2])(self.v_grid).T
        B_xx = B_xx.reshape(1, *self.grid_size)
        B_xy = B_xy.reshape(1, *self.grid_size)
        B_grid = jnp.concatenate([Bxx, Bxx.T, Bxy], axis=0)
        return B_grid
