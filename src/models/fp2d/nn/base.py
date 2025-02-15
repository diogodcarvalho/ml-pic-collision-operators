import jax
import equinox as eqx

from typing import Callable

from src.models.fp2d.base import FokkerPlanck2DBase


class FokkerPlanck2DNNBase(FokkerPlanck2DBase):

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

        if isinstance(activation, str):
            activation = eval(activation)

        key = jax.random.PRNGKey(random_seed)

        self.v_grid = None

        self._init_NN(
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

        self._init_v_grid()

    def _init_NN(
        self,
        depth: int,
        width_size,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        key: jax.random.KeyArray,
    ):
        raise NotImplementedError

    def _init_v_grid(self):
        raise NotImplementedError

    def _update_grid(self, grid_size, grid_range, grid_dx) -> "FokkerPlanck2DNNBase":
        new_model = eqx.tree_at(lambda m: m.grid_size, self, grid_size)
        new_model = eqx.tree_at(lambda m: m.grid_range, new_model, grid_range)
        new_model = eqx.tree_at(lambda m: m.grid_range, new_model, grid_dx)
        new_model._init_v_grid()
        return new_model
