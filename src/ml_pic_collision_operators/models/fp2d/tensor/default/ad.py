import torch
import numpy as np
import torch.nn as nn
import copy

from ml_pic_collision_operators.models.fp2d.tensor.default import (
    FokkerPlanck2D_Tensor_Base,
)


class FokkerPlanck2D_Tensor_AD(FokkerPlanck2D_Tensor_Base):
    """Fokker-Planck 2D Tensor Model.

    This model parametrizes A and D using 2 independent Tensors:

        A(vx, vy) of shape (2, grid_size_x, grid_size_y)
        D(vx, vy) of shape (3, grid_size_x, grid_size_y)

    No symmetries are enforced.
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
        )
        self.A = nn.Parameter(torch.zeros((2, grid_size[0], grid_size[1])))
        self.D = nn.Parameter(torch.zeros((3, grid_size[0], grid_size[1])))

    @property
    def A_grid(self) -> torch.Tensor:
        return self.A

    @property
    def D_grid(self) -> torch.Tensor:
        return self.D

    def load_from_numpy(
        self, A: np.ndarray, D: np.ndarray
    ) -> "FokkerPlanck2D_Tensor_AD":
        assert A.shape == self.A.shape
        assert D.shape == self.D.shape
        with torch.no_grad():
            A_torch = torch.Tensor(A).type_as(self.A)
            D_torch = torch.Tensor(D).type_as(self.D)
            # Create a new instance
            cloned_model = copy.deepcopy(self)
            cloned_model.A.copy_(A_torch)
            cloned_model.D.copy_(D_torch)
        return cloned_model
