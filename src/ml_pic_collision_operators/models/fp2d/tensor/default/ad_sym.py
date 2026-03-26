import torch
import torch.nn as nn

from ml_pic_collision_operators.models.fp2d.tensor.default import (
    FokkerPlanck2D_Tensor_Base,
)


class FokkerPlanck2D_Tensor_AD_Sym(FokkerPlanck2D_Tensor_Base):
    """Fokker-Planck 2D Tensor Model with Axis (Anti-)Symmetry.

    This model parametrizes Ax, Bxx, and Bxy for v_x <= 0 using 3 independent Tensors:

        A_x_half(vx, vy) of shape (grid_size_x / 2, grid_size_y)
        B_xx_half(vx, vy) of shape (grid_size_x / 2, grid_size_y)
        B_xy_half(vx, vy) of shape (grid_size_x / 2, grid_size_y)

    and enforces that:
        A_x(-vx, vy) = -A_x(vx, vy)
        B_xx(-vx, vy) = B_xx(vx, vy)
        B_xy(-vx, vy) = -B_xy(vx, vy)
        A_y = A_x^T
        B_yy = B_xx^T

    If the grid size is uneven along vx, the middle slice (vx = 0) is included in the
    half tensors.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_B=ensure_non_negative_B,
            guard_cells=guard_cells,
            includes_symmetry=True,
        )
        # Only keep half the grid for vx. Other half is determined by the symmetry
        shape = (grid_size[0] // 2 + grid_size[0] % 2, grid_size[1])
        self.Ax_half = nn.Parameter(torch.zeros(shape))
        self.Bxx_half = nn.Parameter(torch.zeros(shape))
        self.Bxy_half = nn.Parameter(torch.zeros(shape))

    @property
    def A_grid(self) -> torch.Tensor:
        Ax = torch.concatenate(
            [
                self.Ax_half,
                -torch.flip(self.Ax_half, dims=(0,))[self.grid_size[0] % 2 :],
            ],
            dim=0,
        )
        return torch.stack([Ax, Ax.T], dim=0)

    @property
    def B_grid(self) -> torch.Tensor:
        Bxx = torch.concatenate(
            [
                self.Bxx_half,
                torch.flip(self.Bxx_half, dims=(0,))[self.grid_size[0] % 2 :],
            ],
            dim=0,
        )
        Bxy = torch.concatenate(
            [
                self.Bxy_half,
                -torch.flip(self.Bxy_half, dims=(0,))[self.grid_size[0] % 2 :],
            ],
            dim=0,
        )
        return torch.stack([Bxx, Bxx.T, Bxy], dim=0)
