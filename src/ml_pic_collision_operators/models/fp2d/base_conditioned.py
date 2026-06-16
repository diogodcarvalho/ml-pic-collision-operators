import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod

from ml_pic_collision_operators.models.fp2d.fp2d_utils import fp2d_step, plot_operator
from ml_pic_collision_operators.models.utils import (
    SupportsAttributeChange,
    grid_shape_checks,
)


class FokkerPlanck2D_Base_Conditioned(nn.Module, ABC, SupportsAttributeChange):
    """Base class to estabilish common structure of Conditioned Fokker-Planck 2D models.

    For now, only used for NN models with conditioning.

    This class should not be used directly, but should be inherited by specific models
    that need to implement the functions:
        `A_grid` - method to compute the A coefficient on the velocity grid for given conditioners.
        `D_grid` - method to compute the D coefficient on the velocity grid for given conditioners.

    Whose returned arrays should be of shape:
        A - (2, grid_size_x, grid_size_y, len_conditioners_batch)
        D - (3, grid_size_x, grid_size_y, len_conditioners_batch)
    """

    _mutable_attrs = frozenset(
        {"ensure_non_negative_f", "ensure_non_negative_D", "guard_cells"}
    )

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
        ensure_non_negative_D: bool = False,
        includes_symmetry: bool = False,
        guard_cells: bool = False,
        operator_is_time_dependent: bool = False,
    ):
        super().__init__()
        grid_shape_checks(
            ndim=2,
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            includes_simmetry=includes_symmetry,
        )

        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.conditioners_size = conditioners_size
        self.ensure_non_negative_f = ensure_non_negative_f
        self.ensure_non_negative_D = ensure_non_negative_D
        self.normalize_conditioners = normalize_conditioners
        self.guard_cells = guard_cells
        # For conditioned models we assume by default that it is not time-dependence.
        # But this can be overridden if time is a conditioner.
        self.operator_is_time_dependent = operator_is_time_dependent
        self._operator_cache: tuple[torch.Tensor, torch.Tensor] | None = None

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
            "ensure_non_negative_D": ensure_non_negative_D,
            "guard_cells": guard_cells,
            "operator_is_time_dependent": operator_is_time_dependent,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    def _normalize_conditioners(self, c: torch.Tensor):
        """Normalizes conditioners to be between [-1,1]."""
        # asserts needed for mypy. __init__ should never allow them to be triggered.
        assert isinstance(self.conditioners_min_values, torch.Tensor)
        assert isinstance(self.conditioners_scale_values, torch.Tensor)
        return (
            2 * (c - self.conditioners_min_values) / self.conditioners_scale_values
        ) - 1

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    @abstractmethod
    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def D_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def A_grid_processed(self, conditioners: torch.Tensor) -> torch.Tensor:
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        return self.A_grid(conditioners)

    def D_grid_processed(self, conditioners: torch.Tensor) -> torch.Tensor:
        if self.normalize_conditioners:
            conditioners = self._normalize_conditioners(conditioners)
        D = self.D_grid(conditioners)
        if self.ensure_non_negative_D:
            D = torch.cat([torch.clamp(D[:, :2], min=0), D[:, 2:]], dim=1)
        return D

    def A_grid_real(self, conditioners: torch.Tensor) -> np.ndarray:
        A = self.A_grid_processed(conditioners).detach().cpu()
        return np.array(A.numpy()) * np.array(self.grid_dx).reshape((1, 2, 1, 1))

    def D_grid_real(self, conditioners: torch.Tensor) -> np.ndarray:
        D = self.D_grid_processed(conditioners).detach().cpu()
        return np.array(D.numpy()) * np.array(
            [self.grid_dx[0] ** 2, self.grid_dx[1] ** 2, np.prod(self.grid_dx)]
        ).reshape((1, 3, 1, 1))

    def plot(
        self, conditioners: torch.Tensor, save_to: str | None = None, show: bool = True
    ):
        if conditioners.ndim == 1:
            # Add batch dimension
            conditioners = conditioners.unsqueeze(0)
        elif conditioners.ndim != 2 or (
            conditioners.ndim == 2 and conditioners.shape[0] != 1
        ):
            raise ValueError(
                "Plot function only accepts conditioners arrays of shape (1, C) or (C,)."
                f" Received {conditioners.shape}."
            )

        with torch.no_grad():
            A = self.A_grid_real(conditioners)[0]
            D = self.D_grid_real(conditioners)[0]

        plot_operator(
            A=A,
            D=D,
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
        use_cached_operator: bool = False,
    ) -> torch.Tensor:
        if use_cached_operator and self._operator_cache is not None:
            A, D = self._operator_cache
        else:
            # We only need to apply NNs to unique conditioners.
            # Saves a lot of time and memory
            c_unique, reverse_indices = torch.unique(
                conditioners, return_inverse=True, dim=0
            )
            A = self.A_grid_processed(c_unique)
            D = self.D_grid_processed(c_unique)
            A = A[reverse_indices]
            D = D[reverse_indices]
            self._operator_cache = (A, D)

        return fp2d_step(
            A=A,
            D=D,
            f=f,
            dt=dt,
            guard_cells=self.guard_cells,
            ensure_non_negative_f=self.ensure_non_negative_f,
        )
