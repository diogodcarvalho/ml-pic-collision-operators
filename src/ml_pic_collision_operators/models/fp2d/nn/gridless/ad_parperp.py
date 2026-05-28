import torch
import torch.nn as nn

from typing import Callable

from ml_pic_collision_operators.models.fp2d.nn.gridless.base import (
    FokkerPlanck2D_NN_Gridless_Base,
)
from ml_pic_collision_operators.models.utils.nn import MLP


class FokkerPlanck2D_NN_Gridless_AD_ParPerp(FokkerPlanck2D_NN_Gridless_Base):
    """Gridless Fokker-Planck 2D NN Model with Parallel-Perpendicular Symmetry.

    This model parametrizes A_par, D_perp, and delta = D_par - D_perp using
    three independent MLPs:

        A_par(v)  = ‖v‖ · Apar_over_v(‖v‖)
        D_perp(v) = MLP_Dperp(‖v‖)
        delta(v)  = ‖v‖ · MLP_delta_over_v(‖v‖)

    The multiplicative ‖v‖ factors enforce the isotropy boundary conditions
    A_par(0) = 0 and D_par(0) = D_perp(0) by construction.

    With `v_x, v_y` denoting the components of the normalized velocity `v_n`:

        A_x  = v_x · Apar_over_v(‖v‖)
        A_y  = v_y · Apar_over_v(‖v‖)
        D_xx = D_perp(‖v‖) + (v_x^2  / ‖v‖) · delta_over_v(‖v‖)
        D_yy = D_perp(‖v‖) + (v_y^2  / ‖v‖) · delta_over_v(‖v‖)
        D_xy = (v_x v_y / ‖v‖) · delta_over_v(‖v‖)

    The symmetry implemented is equivalent to the one in `FokkerPlanck2D_NN_AD_ParPerp`
    but is enforced differently (and without any operator velocity grid).
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
        eps_v_origin: float = 1e-8,
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
            includes_symmetry=True,
            eps_psd=eps_psd,
        )
        self.eps_v_origin = eps_v_origin
        self._init_params_dict["eps_v_origin"] = eps_v_origin

    def _init_NN(
        self,
        depth: int,
        width_size: int,
        activation: Callable,
        use_bias: bool,
        use_final_bias: bool,
        batch_norm: bool,
    ):
        # A_par(v) = ||v_n|| · Apar_over_v(||v_n||); enforces A_par(0) = 0
        self.Apar_over_v = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        # D_perp(v) = Dperp(||v_n||) directly
        self.Dperp = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )
        # delta(v) = ||v_n|| · delta_over_v(||v_n||); enforces D_par(0) = D_perp(0)
        self.delta_over_v = MLP(
            1, 1, depth, width_size, activation, use_bias, use_final_bias, batch_norm
        )

    def _vxy_to_vr(
        self, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized (v_x, v_y, ‖v‖) with smooth ‖v‖ at the origin."""
        v_n = self._normalize_v(v)
        vx_n = v_n[..., 0:1]
        vy_n = v_n[..., 1:2]
        # +eps_v_origin^2 inside the sqrt keeps gradients finite at v=0
        vr_n = torch.sqrt(vx_n * vx_n + vy_n * vy_n + self.eps_v_origin**2)
        return vx_n, vy_n, vr_n

    def A_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        vx_n, vy_n, vr_n = self._vxy_to_vr(v)
        apar_over_v = self.Apar_over_v(vr_n)
        Ax = vx_n * apar_over_v
        Ay = vy_n * apar_over_v
        return torch.cat([Ax, Ay], dim=-1)

    def D_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        vx_n, vy_n, vr_n = self._vxy_to_vr(v)
        Dperp = self.Dperp(vr_n)
        delta_coef = self.delta_over_v(vr_n) / vr_n
        Dxx = Dperp + vx_n * vx_n * delta_coef
        Dyy = Dperp + vy_n * vy_n * delta_coef
        Dxy = vx_n * vy_n * delta_coef
        if self.ensure_non_negative_D:
            Dxx = torch.clamp(Dxx, min=0)
            Dyy = torch.clamp(Dyy, min=0)
        return torch.cat([Dxx, Dyy, Dxy], dim=-1)
