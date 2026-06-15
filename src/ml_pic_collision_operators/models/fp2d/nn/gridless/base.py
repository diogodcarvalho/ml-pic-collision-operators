import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Callable

from ml_pic_collision_operators.models.fp2d.fp2d_utils import (
    fp2d_sde_step,
    plot_operator,
)
from ml_pic_collision_operators.utils import class_from_str


class FokkerPlanck2D_NN_Gridless_Base(nn.Module, ABC):
    """Base class for gridless Fokker-Planck 2D NN models.

    Gridless models parametrize the advection `A(v)` and diffusion `D(v)` without
    any operator velocity grid. They are queried directly at particle velocities:

        A_at_points_real(x) -> (B, N, 2)  physical advection
        D_at_points_real(x) -> (B, N, 3)  physical diffusion components
                                          stacked as [Dxx, Dyy, Dxy]

    Child class should implement:

        `_init_NN` - method to define the architecture of the neural networks used to
            parametrize A and D.
        `A_at_points_real` - physical advection at particle velocities.
        `D_at_points_real` - physical diffusion components at particle velocities;
            must apply the `ensure_non_negative_D` clamp.
    """

    operator_is_time_dependent: bool = False

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
        includes_symmetry: bool = False,
        eps_psd: float = 1e-8,
    ):
        super().__init__()
        if len(v_range_norm) != 4:
            raise ValueError(
                f"v_range_norm must have 4 entries "
                f"(vx_min, vx_max, vy_min, vy_max); got {v_range_norm}"
            )
        if includes_symmetry:
            if (
                v_range_norm[0] != -v_range_norm[1]
                or v_range_norm[2] != -v_range_norm[3]
            ):
                raise ValueError(
                    "includes_symmetry=True requires v_range_norm to be axis-symmetric "
                    "(vx_min == -vx_max and vy_min == -vy_max)"
                )
            if v_range_norm[1] != v_range_norm[3]:
                raise ValueError(
                    "includes_symmetry=True requires equal x and y range magnitudes "
                    f"(vx_max={v_range_norm[1]} != vy_max={v_range_norm[3]}). "
                )

        self.v_range_norm = v_range_norm
        self.v_units = v_units
        self.ensure_non_negative_D = ensure_non_negative_D
        self.eps_psd = eps_psd

        self._init_params_dict = {
            "v_range_norm": v_range_norm,
            "v_units": v_units,
            "depth": depth,
            "width_size": width_size,
            "activation": activation,
            "use_bias": use_bias,
            "use_final_bias": use_final_bias,
            "batch_norm": batch_norm,
            "ensure_non_negative_D": ensure_non_negative_D,
            "eps_psd": eps_psd,
        }

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

        self.normalize_v_min = nn.Buffer(
            torch.Tensor([v_range_norm[0], v_range_norm[2]])
        )
        self.normalize_v_max = nn.Buffer(
            torch.Tensor([v_range_norm[1], v_range_norm[3]])
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    @abstractmethod
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

    def _normalize_v(self, v: torch.Tensor) -> torch.Tensor:
        v_min = self.normalize_v_min
        v_max = self.normalize_v_max
        v = 2 * (v - v_min) / (v_max - v_min) - 1
        return v

    def _denormalize_v(self, v: torch.Tensor) -> torch.Tensor:
        v_min = self.normalize_v_min
        v_max = self.normalize_v_max
        v = (v + 1) / 2 * (v_max - v_min) + v_min
        return v

    @abstractmethod
    def A_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        """Advection components `[Ax, Ay](v)` at particle velocities, shape (B, N, 2)."""
        raise NotImplementedError

    @abstractmethod
    def D_at_points_real(self, v: torch.Tensor) -> torch.Tensor:
        """Diffusion components `[Dxx, Dyy, Dxy](v)` at particle velocities, shape (B, N, 3).
        Child classes must honor `ensure_non_negative_D`.
        """
        raise NotImplementedError

    def plot(
        self,
        save_to: str | None = None,
        show: bool = True,
        v_range: tuple[float, float, float, float] | None = None,
        grid_size: tuple[int, int] = (51, 51),
    ):
        if v_range is None:
            v_range = self.v_range_norm
        vx_min, vx_max, vy_min, vy_max = v_range

        vx = torch.linspace(vx_min, vx_max, grid_size[0] + 1, device=self.device)[:-1]
        vy = torch.linspace(vy_min, vy_max, grid_size[1] + 1, device=self.device)[:-1]
        dx_plot = (vx_max - vx_min) / grid_size[0]
        dy_plot = (vy_max - vy_min) / grid_size[1]
        vx = vx + dx_plot / 2.0
        vy = vy + dy_plot / 2.0
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        pts = torch.stack([VX.flatten(), VY.flatten()], dim=-1).unsqueeze(0)

        with torch.no_grad():
            A = (
                self.A_at_points_real(pts)
                .reshape(grid_size[0], grid_size[1], 2)
                .permute(2, 0, 1)
            )
            D = (
                self.D_at_points_real(pts)
                .reshape(grid_size[0], grid_size[1], 3)
                .permute(2, 0, 1)
            )

        plot_operator(
            A=A.detach().cpu().numpy(),
            D=D.detach().cpu().numpy(),
            grid_range=(vx_min, vx_max, vy_min, vy_max),
            grid_units=self.v_units,
            save_to=save_to,
            show=show,
        )

    def forward(
        self,
        v: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        """Performs a single Euler-Maruyama step of the model SDE."""
        A = self.A_at_points_real(v)
        D = self.D_at_points_real(v)
        return fp2d_sde_step(A=A, D=D, v=v, dt=dt, eps_psd=self.eps_psd)
