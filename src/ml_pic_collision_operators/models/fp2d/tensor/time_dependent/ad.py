import copy
import torch
import torch.nn as nn
import numpy as np

from ml_pic_collision_operators.models.fp2d.tensor.time_dependent.base import (
    FokkerPlanck2D_Tensor_Base_TimeDependent,
)


class FokkerPlanck2D_Tensor_TimeDependent_AD(FokkerPlanck2D_Tensor_Base_TimeDependent):
    """Time-Dependent Fokker-Planck 2D Tensor Model.

    This model parametrizes A and D using 2 independent Tensors:

        A(t, vx, vy) of shape (2, n_t, grid_size_x, grid_size_y)
        D(t, vx, vy) of shape (3, n_t, grid_size_x, grid_size_y)

    No symmetries are enforced.

    Linear interpolation is used to compute coefficients for t-values not stored in the
    tensor time grid of size (n_t,).
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        grid_size_t: int,
        grid_dt: float,
        n_t: int = -1,
        ensure_non_negative_f: bool = True,
        ensure_non_negative_D: bool = False,
        guard_cells: bool = False,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            grid_size_t=grid_size_t,
            grid_dt=grid_dt,
            n_t=n_t,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
            guard_cells=guard_cells,
            includes_symmetry=False,
        )
        self.A = nn.Parameter(torch.zeros((self.n_t, 2, *self.grid_size)))
        self.D = nn.Parameter(torch.zeros((self.n_t, 3, *self.grid_size)))

    def A_grid(self, t: torch.Tensor) -> torch.Tensor:
        if self.n_t == self.grid_size_t:
            return self.A[self._it(t)]
        else:
            return self._t_interpolate(self.A, t)

    def D_grid(self, t: torch.Tensor) -> torch.Tensor:
        if self.n_t == self.grid_size_t:
            return self.D[self._it(t)]
        else:
            return self._t_interpolate(self.D, t)

    def A_grid_real(self, t: torch.Tensor) -> np.ndarray:
        return np.array(self.A_grid(t).detach().cpu().numpy()) * np.array(
            self.grid_dx
        ).reshape((1, 2, 1, 1))

    def D_grid_real(self, t: torch.Tensor) -> np.ndarray:
        D = self.D_grid(t).detach().cpu()
        if self.ensure_non_negative_D:
            D[:, :2] = torch.clamp(D[:, :2], min=0)
        return np.array(D.numpy()) * np.array(
            [self.grid_dx[0] ** 2, self.grid_dx[1] ** 2, np.prod(self.grid_dx)]
        ).reshape((1, 3, 1, 1))

    def load_from_numpy(self, A: np.ndarray, D: np.ndarray):
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
