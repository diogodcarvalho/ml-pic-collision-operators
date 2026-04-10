import torch
import numpy as np
import copy

from ml_pic_collision_operators.models.fp2d.base import FokkerPlanck2D_Base


class FokkerPlanck2D_Tensor_Base(FokkerPlanck2D_Base):
    """Base class for Fokker-Planck 2D Tensor models.

    For now this is only a placeholder with the same properties as FokkerPlanck2D_Base
    but it can be used in the future to add common methods or properties for all
    tensor-based models.
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
        includes_symmetry: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
            includes_symmetry=includes_symmetry,
            guard_cells=guard_cells,
        )
