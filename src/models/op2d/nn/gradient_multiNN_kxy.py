import torch
import torch.nn as nn
from typing import Callable

from src.models.op2d.nn.gradient_kxy import Operator2DNN_Gradient_Kxy
from src.models.utils import MLP


class Operator2DNN_Gradient_MultiNN_Kxy(Operator2DNN_Gradient_Kxy):

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

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        self.Kx = nn.ModuleList(
            [
                MLP(
                    2,
                    1,
                    depth,
                    width_size,
                    activation,
                    use_bias,
                    use_final_bias,
                    batch_norm,
                )
                for k in range(self.kernel_size**2)
            ]
        )
        self.Ky = nn.ModuleList(
            [
                MLP(
                    2,
                    1,
                    depth,
                    width_size,
                    activation,
                    use_bias,
                    use_final_bias,
                    batch_norm,
                )
                for k in range(self.kernel_size**2)
            ]
        )

    def _get_kernels(self):
        kernels_x = torch.concatenate(
            [K(self.v_grid.detach()) for K in self.Kx], axis=-1
        )
        kernels_y = torch.concatenate(
            [K(self.v_grid.detach()) for K in self.Ky], axis=-1
        )
        return kernels_x.T, kernels_y.T
