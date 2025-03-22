import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from src.models.fp2d.base_conditioned import FokkerPlanck2DBaseConditioned
from src.models.utils.nn import MLP
from src.utils import class_from_str


class FokkerPlanck2DNNBaseConditioned(FokkerPlanck2DBaseConditioned):
    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        conditioners_size: int,
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        normalize_v_grid: bool = True,
        conditioners_min_values: np.ndarray | None = None,
        conditioners_max_values: np.ndarray | None = None,
        normalize_conditioners: bool = False,
        includes_symmetry: bool = False,
    ):

        if includes_symmetry:
            assert grid_size[0] == grid_size[1]
            assert grid_range[0] == grid_range[2]
            assert grid_range[1] == grid_range[3]
            assert grid_dx[0] == grid_dx[1]

        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            conditioners_size=conditioners_size,
            ensure_non_negative_f=ensure_non_negative_f,
            conditioners_min_values=conditioners_min_values,
            conditioners_max_values=conditioners_max_values,
            normalize_conditioners=normalize_conditioners,
        )

        if isinstance(activation, str):
            activation = class_from_str(activation)

        self._init_NN(
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            conditioners_size=conditioners_size,
            batch_norm=batch_norm,
        )
        self._init_v_grid(normalize_v_grid)

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        conditioners_size: int,
        batch_norm: bool,
    ):
        raise NotImplementedError

    def _init_v_grid(self, normalize: bool):
        raise NotImplementedError

    def _prepare_input(
        self, conditioners: torch.Tensor, v: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert conditioners.ndim == 2
        # ex. dimensions assume v_grid is full grid (but code also works for other cases)
        if v is None:
            # (grid_size**2, 2)
            v = self.v_grid
        else:
            # (any, any)
            assert v.ndim == 2
        # shape hints assume v = self.v_grid
        # (1, grid_size**2, 2)
        v = v.unsqueeze(0).detach()
        # (batch_size, 1, C)
        c = conditioners.unsqueeze(1).detach()
        # (batch_size, grid_size**2, 2)
        v = v.repeat(c.shape[0], 1, 1)
        # (batch_size, grid_size**2, C)
        c = c.repeat(1, v.shape[1], 1)
        # (batch_size, grid_size**2, 2 + C)
        inputs = torch.cat([v, c], axis=-1)
        return inputs.reshape(-1, v.shape[-1] + conditioners.shape[-1])
