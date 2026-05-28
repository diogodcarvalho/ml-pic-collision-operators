import torch
import torch.nn as nn

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.gridless.base import (
    FokkerPlanck2D_NN_Gridless_Base,
)
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NN_Gridless_AD_T(FokkerPlanck2D_NN_Gridless_Base):
    """Gridless Fokker-Planck 2D NN Model with Transposed Symmetry.

    This model parametrizes A_x, D_xx and D_xy using independent (equivalent) MLPs:

        A_x(vx, vy) = MLP_A_x(vx, vy)
        D_xx(vx, vy) = MLP_D_xx(vx, vy)
        D_xy(vx, vy) = MLP_D_xy(vx, vy)

    and enforces that:

        A_y(vx, vy) = A_x(vy, vx)
        D_yy(vx, vy) = D_xx(vy, vx)

    Mirrors `FokkerPlanck2D_NN_AD_T` but without any operator velocity grid.
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
            eps_psd=eps_psd,
            includes_symmetry=True,
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
        self.Dxx = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        self.Dxy = MLP(
            2, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    def A_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        vn = self._normalize_v(v)
        vn_T = torch.stack([vn[..., 1], vn[..., 0]], dim=-1)
        Ax = self.Ax(vn)
        Ay = self.Ax(vn_T)
        return torch.cat([Ax, Ay], dim=-1)

    def D_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        vn = self._normalize_v(v)
        vn_T = torch.stack([vn[..., 1], vn[..., 0]], dim=-1)
        Dxx = self.Dxx(vn)
        Dyy = self.Dxx(vn_T)
        Dxy = self.Dxy(vn)
        if self.ensure_non_negative_D:
            Dxx = torch.clamp(Dxx, min=0)
            Dyy = torch.clamp(Dyy, min=0)
        return torch.cat([Dxx, Dyy, Dxy], dim=-1)
