import torch
import torch.nn as nn

from ml_pic_collision_operators.models.k2d.base import K2D_Base


class K2D_Tensor_T(K2D_Base):
    """Integro-Differential 2D Tensor Operator with transposed symmetry.

    Parameterizes Kx as a tensor of shape
    (kernel_size, kernel_size, grid_size_x, grid_size_y) and enforces that
    Ky(vx, vy, l, m) = Kx(vy, vx, m, l)
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
        gradient_scheme: str = "forward",
    ):

        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            ensure_non_negative_f=ensure_non_negative_f,
            gradient_scheme=gradient_scheme,
            includes_symmetry=True,
        )

        self.Kx = nn.Parameter(
            torch.zeros(
                (self.kernel_size, self.kernel_size, grid_size[0], grid_size[1])
            )
        )

    @property
    def K(self) -> torch.Tensor:
        return torch.stack([self.Kx, self.Kx.permute(1, 0, 3, 2)], dim=0)
