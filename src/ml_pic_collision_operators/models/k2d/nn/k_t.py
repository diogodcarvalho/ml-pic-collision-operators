import torch
import torch.nn as nn
from typing import Callable

from ml_pic_collision_operators.models.k2d.nn.base import K2D_NN_Base
from ml_pic_collision_operators.models.utils import MLP


class K2D_NN_T(K2D_NN_Base):
    """Integro-Differential 2D NN Operator.

    Parameterizes K using a NN:
        K(vx, vy) = NN(vx, vy)
    where K(vx, vy) has shape (2, kernel_size, kernel_size).
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
        gradient_scheme: str = "forward",
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
            gradient_scheme=gradient_scheme,
            includes_symmetry=False,
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

    @property
    def K(self) -> torch.Tensor:
        Kx = self.Kx(self.v_grid.data)
        Kx = Kx.reshape(*self.grid_size, self.kernel_size, self.kernel_size)
        Kx = Kx.permute(2, 3, 1, 0)
        Ky = Kx.permute(1, 0, 3, 2)
        return torch.stack([Kx, Ky], dim=0)
