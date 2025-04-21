import torch
import torch.nn as nn

from src.models.op2d.base import Operator2DBase


class Operator2DTensor(Operator2DBase):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
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
            zero_kernel_indices=zero_kernel_indices,
            includes_symmetry=False,
        )

        self.K = nn.Parameter(
            torch.zeros(
                (self.kernel_size, self.kernel_size, grid_size[0], grid_size[1])
            )
        )

    def _get_kernels_full(self):
        return self.K
