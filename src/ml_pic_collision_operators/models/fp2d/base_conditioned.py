import torch
import torch.nn as nn
import numpy as np

from typing import Any

from ml_pic_collision_operators.models.fp2d.fp2d_utils import fp2d_step, plot_operator


class FokkerPlanck2D_Base_Conditioned(nn.Module):
    """Base class to estabilish common structure of Conditioned Fokker-Planck 2D models.

    For now, only used for NN models with conditioning.

    This class should not be used directly, but should be inherited by specific models
    that need to implement the functions:
        `A_grid` - method to compute the A coefficient on the velocity grid for given conditioners.
        `B_grid` - method to compute the B coefficient on the velocity grid for given conditioners.

    Whose returned arrays should be of shape:
        A - (2, grid_size_x, grid_size_y, len_conditioners_batch)
        B - (3, grid_size_x, grid_size_y, len_conditioners_batch)
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        conditioners_size: int,
        conditioners_min_values: list[float] | np.ndarray | None = None,
        conditioners_max_values: list[float] | np.ndarray | None = None,
        normalize_conditioners: bool = False,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_B: bool = False,
        includes_symmetry: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__()
        assert len(grid_size) == 2
        if includes_symmetry:
            assert grid_size[0] == grid_size[1]
            assert grid_range[0] == grid_range[2]
            assert grid_range[1] == grid_range[3]
            assert grid_dx[0] == grid_dx[1]

        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.conditioners_size = conditioners_size
        self.ensure_non_negative_f = ensure_non_negative_f
        self.ensure_non_negative_B = ensure_non_negative_B
        self.normalize_conditioners = normalize_conditioners
        self.guard_cells = guard_cells

        if self.normalize_conditioners:
            if conditioners_min_values is None or conditioners_max_values is None:
                raise ValueError(
                    "conditioners_min_values and conditioners_max_values must be"
                    " provided if normalize_conditioners is True"
                )
            if len(conditioners_min_values) != conditioners_size:
                raise ValueError(
                    "conditioners_min_values must have the same length as conditioners_size"
                )
            if len(conditioners_max_values) != conditioners_size:
                raise ValueError(
                    "conditioners_max_values must have the same length as conditioners_size"
                )

            self.register_buffer(
                "conditioners_min_values",
                torch.Tensor(conditioners_min_values).unsqueeze(0),
            )
            self.register_buffer(
                "conditioners_max_values",
                torch.Tensor(conditioners_max_values).unsqueeze(0),
            )
            aux = np.array(conditioners_max_values) - np.array(conditioners_min_values)
            # avoids division by zero
            aux[aux == 0.0] = 1.0
            self.register_buffer(
                "conditioners_scale_values", torch.Tensor(aux).unsqueeze(0)
            )
            # for serialization to work they have to be list
            if isinstance(conditioners_min_values, np.ndarray):
                conditioners_min_values = conditioners_min_values.tolist()
            if isinstance(conditioners_max_values, np.ndarray):
                conditioners_max_values = conditioners_max_values.tolist()

        else:
            self.conditioners_min_values = None
            self.conditioners_max_values = None
            self.conditioners_scale_values = None

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "conditioners_size": conditioners_size,
            "conditioners_min_values": conditioners_min_values,
            "conditioners_max_values": conditioners_max_values,
            "normalize_conditioners": normalize_conditioners,
            "ensure_non_negative_f": ensure_non_negative_f,
            "ensure_non_negative_B": ensure_non_negative_B,
            "guard_cells": guard_cells,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def _normalize_conditioners(self, c: torch.Tensor):
        """Normalizes conditioners to be between [-1,1]."""
        if self.conditioners_min_values is None:
            raise ValueError(
                "conditioners_min_values must be defined to normalize conditioners"
            )
        if self.conditioners_scale_values is None:
            raise ValueError(
                "conditioners_scale_values must be defined to normalize conditioners"
            )
        return (
            2 * (c - self.conditioners_min_values) / self.conditioners_scale_values
        ) - 1

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def B_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def A_grid_real(self, conditioners: torch.Tensor) -> np.ndarray:
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        return np.array(self.A_grid(conditioners).detach().cpu().numpy()[0]) * np.array(
            self.grid_dx
        ).reshape((2, 1, 1))

    def B_grid_real(self, conditioners: torch.Tensor) -> np.ndarray:
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        B = self.B_grid(conditioners).detach().cpu()
        if self.ensure_non_negative_B:
            B[:2] = torch.clamp(B[:2], min=0)
        return np.array(B.numpy()[0]) * np.array(
            [self.grid_dx[0] ** 2, self.grid_dx[1] ** 2, np.prod(self.grid_dx)]
        ).reshape((3, 1, 1))

    def change_attribute(self, attr_name: str, attr_value: Any):
        if attr_name in [
            "ensure_non_negative_f",
            "ensure_non_negative_B",
            "guard_cells",
        ]:
            setattr(self, attr_name, attr_value)
        elif hasattr(self, attr_name):
            raise ValueError(
                f"Can not change attribute: {attr_name} after initialization"
            )
        else:
            raise KeyError(f"{type(self)} does not have attribute: {attr_name}")

    def plot(
        self, conditioners: torch.Tensor, save_to: str | None = None, show: bool = True
    ):
        plot_operator(
            A=self.A_grid_real(conditioners),
            B=self.B_grid_real(conditioners),
            grid_range=self.grid_range,
            grid_units=self.grid_units,
            save_to=save_to,
            show=show,
        )

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
        conditioners: torch.Tensor,
    ) -> torch.Tensor:
        # We only need to apply NNs to unique conditioners.
        # Saves a lot of time and memory
        c_unique, reverse_indices = torch.unique(
            conditioners, return_inverse=True, dim=0
        )
        if self.normalize_conditioners:
            c_unique = self._normalize_conditioners(c_unique)
        A = self.A_grid(c_unique)
        B = self.B_grid(c_unique)

        if self.ensure_non_negative_B:
            B[:2] = torch.clamp(B[:2], min=0)

        A = A[reverse_indices]
        B = B[reverse_indices]

        return fp2d_step(
            A=A,
            B=B,
            f=f,
            dt=dt,
            guard_cells=self.guard_cells,
            ensure_non_negative_f=self.ensure_non_negative_f,
        )
