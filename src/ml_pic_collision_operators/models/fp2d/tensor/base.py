import torch
import numpy as np
import copy
from ml_pic_collision_operators.models.fp2d.base import FokkerPlanck2DBase


class FokkerPlanck2DTensorBase(FokkerPlanck2DBase):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        guard_cells: bool = False,
        includes_symmetry: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_B=ensure_non_negative_B,
            includes_symmetry=includes_symmetry,
            guard_cells=guard_cells,
        )
        self.A = None
        self.B = None

    def load_from_numpy(self, A: np.ndarray, B: np.ndarray):
        assert A.shape == self.A.shape
        assert B.shape == self.B.shape
        with torch.no_grad():
            A = torch.Tensor(A).type_as(self.A)
            B = torch.Tensor(B).type_as(self.B)
            cloned_model = copy.deepcopy(self)  # Create a new instance
            cloned_model.A.copy_(A)
            cloned_model.B.copy_(B)
        return cloned_model
