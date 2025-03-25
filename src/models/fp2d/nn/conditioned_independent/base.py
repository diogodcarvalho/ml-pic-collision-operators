import torch
import torch.nn as nn
import numpy as np

from typing import Callable

from src.models.fp2d.base_conditioned import FokkerPlanck2DBaseConditioned
from src.utils import class_from_str


class FokkerPlanck2DNNBaseConditionedIndependent(FokkerPlanck2DBaseConditioned):
    """Base model to estabilish common structure of Fokker Planck 2D NN ConditionedIndependent classes."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        conditioners_size: int,
        depth: int,
        width_size: int,
        c_depth: int,
        c_width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        c_activation: Callable | str = nn.ReLU,
        c_use_bias: bool = True,
        c_use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        normalize_v_grid: bool = True,
        conditioners_min_values: list[float] | np.ndarray | None = None,
        conditioners_max_values: list[float] | np.ndarray | None = None,
        normalize_conditioners: bool = False,
        includes_symmetry: bool = False,
    ):

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
            includes_symmetry=includes_symmetry,
        )

        new_params = {
            "depth": depth,
            "width_size": width_size,
            "activation": activation,
            "use_bias": use_bias,
            "use_final_bias": use_final_bias,
            "c_depth": c_depth,
            "c_width_size": c_width_size,
            "c_activation": c_activation,
            "c_use_bias": c_use_bias,
            "c_use_final_bias": c_use_final_bias,
            "batch_norm": batch_norm,
            "normalize_v_grid": normalize_v_grid,
        }

        self._init_params_dict.update(new_params)

        if isinstance(activation, str):
            activation = class_from_str(activation)

        if isinstance(c_activation, str):
            c_activation = class_from_str(c_activation)

        self._init_NN(
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            conditioners_size=conditioners_size,
            c_depth=c_depth,
            c_width_size=c_width_size,
            c_activation=c_activation,
            c_use_bias=c_use_bias,
            c_use_final_bias=c_use_final_bias,
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
        c_depth: int,
        c_width_size: int,
        c_activation: Callable,
        c_use_bias: bool,
        c_use_final_bias: bool,
        batch_norm: bool,
    ):
        raise NotImplementedError

    def _default_vx_vy(self, normalize: bool):
        vx = torch.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size[0] + 1
        )[:-1]
        vy = torch.linspace(
            self.grid_range[2], self.grid_range[3], self.grid_size[1] + 1
        )[:-1]
        vx += self.grid_dx[0] / 2.0
        vy += self.grid_dx[1] / 2.0
        if normalize:
            vx = 2 * (vx - torch.min(vx)) / (torch.max(vx) - torch.min(vx)) - 1
            vy = 2 * (vy - torch.min(vy)) / (torch.max(vy) - torch.min(vy)) - 1
        return vx, vy

    def _init_v_grid(self, normalize: bool):
        raise NotImplementedError
