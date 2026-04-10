import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Callable

from ml_pic_collision_operators.utils import class_from_str
from ml_pic_collision_operators.models.k2d.base import K2D_Base


class K2D_NN_Base(K2D_Base):
    """Base class for NN Integro-Differential Operators in 2D.

    Child classes need to implement :
        K - property that returns the operator computed on the grid with shape
            (2, kernel_size, kernel_size, grid_size_x, grid_size_y)
        _init_NN - NN initialization function
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        normalize_v_grid: bool = True,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
        gradient_scheme: str = "forward",
        includes_symmetry: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            ensure_non_negative_f=ensure_non_negative_f,
            includes_symmetry=includes_symmetry,
            gradient_scheme=gradient_scheme,
        )

        self._init_params_dict.update(
            {
                "depth": depth,
                "width_size": width_size,
                "activation": activation,
                "use_bias": use_bias,
                "use_final_bias": use_final_bias,
                "batch_norm": batch_norm,
                "normalize_v_grid": normalize_v_grid,
            }
        )

        if isinstance(activation, str):
            activation = class_from_str(activation)

        self._init_NN(
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
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
        vx, vy = self._default_vx_vy(normalize)
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))
