import torch
import torch.nn as nn

from src.models.op2d.base_gradient_kxy import Operator2DBase_Gradient_Kxy


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
            includes_symmetry=True,
        )

        self.zero_kernel_indices = zero_kernel_indices
        self._init_params_dict.update({"zero_kernel_indices": zero_kernel_indices})

        self.Kx = nn.Parameter(
            torch.zeros(
                (self.kernel_size, self.kernel_size, grid_size[0], grid_size[1])
            )
        )

    def _get_kernels(self) -> torch.Tensor:
        if self.zero_kernel_indices is not None:
            Kx = self.Kx.clone()
            for i, j in self.zero_kernel_indices:
                Kx[i, j] = 0.0
        else:
            Kx = self.Kx
        Ky = Kx.permute(1, 0, 3, 2)
        Kx = Kx.reshape(self.kernel_size**2, self.grid_size[0] * self.grid_size[1])
        Ky = Ky.reshape(self.kernel_size**2, self.grid_size[0] * self.grid_size[1])
        return Kx, Ky
