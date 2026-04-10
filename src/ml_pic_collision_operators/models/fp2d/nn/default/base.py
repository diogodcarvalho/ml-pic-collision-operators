import torch
import torch.nn as nn

from typing import Callable

from ml_pic_collision_operators.models.fp2d.base import FokkerPlanck2D_Base
from ml_pic_collision_operators.utils import class_from_str


class FokkerPlanck2D_NN_Base(FokkerPlanck2D_Base):
    """Base class for Fokker-Planck 2D NN models.

    Child class should implement:
        `_init_NN` - method to define the architecture of the neural networks used to
            parametrize A and B.
        `_init_v_grid` - method to define the velocity grid used for the input of the
            neural networks. Different models may use different velocity grids
            depending on the implemented symmetries.
        `A_grid` and `B_grid` - properties that compute the A and B coefficients on the
            velocity grid.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        normalize_v_grid: bool = True,
        includes_symmetry: bool = False,
        guard_cells: bool = False,
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

        new_params = {
            "depth": depth,
            "width_size": width_size,
            "activation": activation,
            "use_bias": use_bias,
            "use_final_bias": use_final_bias,
            "batch_norm": batch_norm,
            "normalize_v_grid": normalize_v_grid,
        }
        self._init_params_dict.update(new_params)

        if isinstance(activation, str):
            activation = class_from_str(activation)

        self._init_NN(
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
        )

        self.normalize_v_grid = normalize_v_grid
        self.normalize_vx_min = torch.nan
        self.normalize_vx_max = torch.nan
        self.normalize_vy_min = torch.nan
        self.normalize_vy_max = torch.nan
        self._init_v_grid(normalize_v_grid)

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        raise NotImplementedError

    def _default_vx_vy(self, normalize: bool):
        vx = torch.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size[0] + 1
        )[:-1]
        vy = torch.linspace(
            self.grid_range[2], self.grid_range[3], self.grid_size[1] + 1
        )[:-1]
        vx += self.grid_dx[0] / 2.0
        vy += self.grid_dx[1] / 2.0
        if normalize:
            self.normalize_vx_min = float(torch.min(vx))
            self.normalize_vx_max = float(torch.max(vx))
            self.normalize_vy_min = float(torch.min(vy))
            self.normalize_vy_max = float(torch.max(vy))
            vx = 2 * (vx - torch.min(vx)) / (torch.max(vx) - torch.min(vx)) - 1
            vy = 2 * (vy - torch.min(vy)) / (torch.max(vy) - torch.min(vy)) - 1
        return vx, vy

    def _init_v_grid(self, normalize: bool):
        raise NotImplementedError

    def _normalize_v(
        self, vx: torch.Tensor, vy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_v_grid:
            vx = (
                2
                * (vx - self.normalize_vx_min)
                / (self.normalize_vx_max - self.normalize_vx_min)
                - 1
            )
            vy = (
                2
                * (vy - self.normalize_vy_min)
                / (self.normalize_vy_max - self.normalize_vy_min)
                - 1
            )
        return vx, vy

    def _denormalize_v(
        self, vx: torch.Tensor, vy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_v_grid:
            vx = (vx + 1) / 2 * (
                self.normalize_vx_max - self.normalize_vx_min
            ) + self.normalize_vx_min
            vy = (vy + 1) / 2 * (
                self.normalize_vy_max - self.normalize_vy_min
            ) + self.normalize_vy_min
        return vx, vy
