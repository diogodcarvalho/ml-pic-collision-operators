import torch
import torch.nn as nn

from ml_pic_collision_operators.models.op2d.base_gradient_kxy import (
    Operator2DBase_Gradient_Kxy,
)


class Operator2DTensor_Gradient_Kxyt(Operator2DBase_Gradient_Kxy):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
        gradient_order: int = 2,
        zero_kernel_indices: list[tuple[int, int]] = None,
    ):

        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            ensure_non_negative_f=ensure_non_negative_f,
            gradient_order=gradient_order,
            zero_kernel_indices=zero_kernel_indices,
            includes_symmetry=True,
        )

        self.Kx = nn.Parameter(
            torch.zeros(
                (self.kernel_size, self.kernel_size, grid_size[0], grid_size[1])
            )
        )

    def _get_kernels_full(self) -> torch.Tensor:
        return self.Kx, self.Kx.permute(1, 0, 3, 2)
