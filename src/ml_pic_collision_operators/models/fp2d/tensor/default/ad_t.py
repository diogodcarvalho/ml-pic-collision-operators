import torch
import torch.nn as nn

from ml_pic_collision_operators.models.fp2d.tensor.default import (
    FokkerPlanck2D_Tensor_Base,
)


class FokkerPlanck2D_Tensor_AD_T(FokkerPlanck2D_Tensor_Base):
    """Fokker-Planck 2D Tensor Model with Transposed Symmetry.

    This model parametrizes Ax, Dxx, and Dxy using 3 independent Tensors:

        A_x(vx, vy) of shape (grid_size_x, grid_size_y)
        D_xx(vx, vy) of shape (grid_size_x, grid_size_y)
        D_xy(vx, vy) of shape (grid_size_x, grid_size_y)

    and enforces that:
        A_y = A_x^T
        D_yy = D_xx^T
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_D: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
            guard_cells=guard_cells,
            includes_symmetry=True,
        )
        self.Ax = nn.Parameter(torch.zeros((grid_size[0], grid_size[1])))
        self.Dxx = nn.Parameter(torch.zeros((grid_size[0], grid_size[1])))
        self.Dxy = nn.Parameter(torch.zeros((grid_size[0], grid_size[1])))

    @property
    def A_grid(self) -> torch.Tensor:
        return torch.stack([self.Ax, self.Ax.T], dim=0)

    @property
    def D_grid(self) -> torch.Tensor:
        return torch.stack([self.Dxx, self.Dxx.T, self.Dxy], dim=0)
