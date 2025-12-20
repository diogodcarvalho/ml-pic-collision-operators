import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from ml_pic_collision_operators.models.op2d.nn.gradient_kxy import (
    Operator2DNN_Gradient_Kxy,
)
from ml_pic_collision_operators.models.utils import MLP


class Operator2DNN_Gradient_Kxyt(Operator2DNN_Gradient_Kxy):

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
        zero_kernel_indices: list[tuple[int, int]] = None,
    ):

        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
            normalize_v_grid=normalize_v_grid,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            ensure_non_negative_f=ensure_non_negative_f,
        )

        self.zero_kernel_indices = zero_kernel_indices
        self._init_params_dict.update({"zero_kernel_indices": zero_kernel_indices})

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        self.Kx = MLP(
            2,
            self.kernel_size**2,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )

    def _get_kernels(self):

        kernels_x = self.Kx(self.v_grid.detach())

        if self.zero_kernel_indices is not None:
            for i, j in self.zero_kernel_indices:
                kernels_x[:, self.kernel_size * i + j] = 0.0

        kernels_y = (
            kernels_x.reshape(*self.grid_size, self.kernel_size, self.kernel_size)
            .permute(1, 0, 3, 2)
            .reshape(-1, self.kernel_size**2)
        )

        return kernels_x.T, kernels_y.T
