import torch
import torch.nn as nn

from ml_pic_collision_operators.models.k2d.base import K2D_Base


class K2D_Tensor(K2D_Base):
    """Integro-Differential 2D Tensor Operator.

    Parameterizes K as a tensor of shape
        (2, kernel_size, kernel_size, grid_size_x, grid_size_y)
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
            includes_symmetry=False,
        )

        # Can't name it K because of the property name
        self.K_ = nn.Parameter(
            torch.zeros((2, self.kernel_size, self.kernel_size, *grid_size))
        )

    @property
    def K(self) -> torch.Tensor:
        # The [:] necessary to return a view so that a Tensor is returned and
        # self.K_ is not cached as a nn.Parameter.
        return self.K_[:]
