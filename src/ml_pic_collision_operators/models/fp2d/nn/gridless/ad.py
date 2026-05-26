import torch
import torch.nn as nn
from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.gridless.base import (
    FokkerPlanck2D_NN_Gridless_Base,
)
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NN_Gridless_AD(FokkerPlanck2D_NN_Gridless_Base):
    """Gridless Fokker-Planck 2D Neural Network Model.

    This model parametrizes A and D using 5 independent (equivalent) MLPs:

        A_i(vx, vy) = MLP_A_i(vx, vy)
        D_ij(vx, vy) = MLP_D_ij(vx, vy)

    No symmetries are enforced.

    Mirrors `FokkerPlanck2D_NN_AD` but without any operator velocity grid.
    """

    def __init__(
        self,
        v_range_norm: tuple[float, float, float, float],
        v_units: str,
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        ensure_non_negative_D: bool = False,
        eps_psd: float = 1e-8,
    ):
        super().__init__(
            v_range_norm=v_range_norm,
            v_units=v_units,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
            ensure_non_negative_D=ensure_non_negative_D,
            includes_symmetry=False,
            eps_psd=eps_psd,
        )

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        self.Ax = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Ay = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dxx = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dyy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dxy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    def A_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        xn = self._normalize_v(v)
        Ax = self.Ax(xn)
        Ay = self.Ay(xn)
        return torch.cat([Ax, Ay], dim=-1)

    def D_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        xn = self._normalize_v(v)
        Dxx = self.Dxx(xn)
        Dyy = self.Dyy(xn)
        Dxy = self.Dxy(xn)
        if self.ensure_non_negative_D:
            Dxx = torch.clamp(Dxx, min=0)
            Dyy = torch.clamp(Dyy, min=0)
        return torch.cat([Dxx, Dyy, Dxy], dim=-1)
