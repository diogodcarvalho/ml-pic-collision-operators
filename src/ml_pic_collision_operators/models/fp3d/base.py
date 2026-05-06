import torch
import torch.nn as nn
import numpy as np

from typing import Any

from ml_pic_collision_operators.models.fp3d.fp3d_utils import (
    fp3d_step,
    plot_operator_3D,
)


class FokkerPlanck3D_Base(nn.Module):
    """Base class to estabilish common structure of Fokker-Planck 3D models.

    It can be used for both Tensor and NN models without conditioning / time dependence.

    This class should not be used directly, but should be inherited by specific models
    that need to implement the properties:

        `A_grid` - method to compute the A coefficient on the velocity grid.
        `D_grid` - method to compute the D coefficient on the velocity grid.

    Whose returned arrays should be of shape:
        A - (3, grid_size_x, grid_size_y, grid_size_z)
        D - (6, grid_size_x, grid_size_y, grid_size_z)
    """

    # A and D depend only on model parameters — safe to cache across rollout steps.
    operator_is_time_dependent: bool = False

    def __init__(
        self,
        grid_size: tuple[int, int, int],
        grid_range: tuple[float, float, float, float, float, float],
        grid_dx: tuple[float, float, float],
        grid_units: str,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_D: bool = False,
        includes_symmetry: bool = False,
        guard_cells: bool = True,
    ):
        super().__init__()
        assert len(grid_size) == 3
        assert len(grid_range) == 6
        assert len(grid_dx) == 3
        if includes_symmetry:
            assert grid_size[0] == grid_size[1]
            assert grid_size[0] == grid_size[2]
            assert grid_range[0] == grid_range[2]
            assert grid_range[0] == grid_range[4]
            assert grid_range[1] == grid_range[3]
            assert grid_range[1] == grid_range[5]
            assert grid_dx[0] == grid_dx[1]
            assert grid_dx[0] == grid_dx[2]

        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.ensure_non_negative_f = ensure_non_negative_f
        self.ensure_non_negative_D = ensure_non_negative_D
        self.guard_cells = guard_cells
        self._operator_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "ensure_non_negative_f": ensure_non_negative_f,
            "ensure_non_negative_D": ensure_non_negative_D,
            "guard_cells": guard_cells,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    @property
    def A_grid(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def D_grid(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def A_grid_real(self) -> np.ndarray:
        return np.array(self.A_grid.detach().cpu().numpy()) * np.array(
            self.grid_dx
        ).reshape((3, 1, 1, 1))

    @property
    def D_grid_real(self) -> np.ndarray:
        D = self.D_grid.detach().cpu()
        if self.ensure_non_negative_D:
            D[:3] = torch.clamp(D[:3], min=0)
        return np.array(D.numpy()) * np.array(
            [
                self.grid_dx[0] ** 2,
                self.grid_dx[1] ** 2,
                self.grid_dx[2] ** 2,
                self.grid_dx[0] * self.grid_dx[1],
                self.grid_dx[0] * self.grid_dx[2],
                self.grid_dx[1] * self.grid_dx[2],
            ]
        ).reshape((6, 1, 1, 1))

    def change_attribute(self, attr_name: str, attr_value: Any):
        if attr_name in [
            "ensure_non_negative_f",
            "ensure_non_negative_D",
            "guard_cells",
        ]:
            setattr(self, attr_name, attr_value)
        elif hasattr(self, attr_name):
            raise ValueError(
                f"Can not change attribute: {attr_name} after initialization"
            )
        else:
            raise KeyError(f"{type(self)} does not have attribute: {attr_name}")

    def plot(self, save_to: str | None = None, show: bool = True):
        plot_operator_3D(
            A=self.A_grid_real,
            D=self.D_grid_real,
            grid_range=self.grid_range,
            grid_units=self.grid_units,
            save_to=save_to,
            show=show,
        )

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
        use_cached_operator: bool = False,
    ) -> torch.Tensor:
        if use_cached_operator and self._operator_cache is not None:
            A, D = self._operator_cache
        else:
            A = self.A_grid
            D = self.D_grid
            if self.ensure_non_negative_D:
                D = torch.cat([torch.clamp(D[:3], min=0), D[3:]], dim=0)
            self._operator_cache = (A, D)

        return fp3d_step(
            A=A,
            D=D,
            f=f,
            dt=dt,
            guard_cells=self.guard_cells,
            ensure_non_negative_f=self.ensure_non_negative_f,
        )
