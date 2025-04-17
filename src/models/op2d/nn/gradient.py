import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from src.models.op2d.nn.base import Operator2DNNBase
from src.models.utils import MLP


class Operator2DNN_Gradient(Operator2DNNBase):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        depth: int,
        width_size: int,
        activation: Callable | str = nn.ReLU,
        use_bias: bool = True,
        use_final_bias: bool = True,
        batch_norm: bool = False,
        normalize_v_grid: bool = True,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
    ):

        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            grid_units=grid_units,
            depth=depth,
            width_size=width_size,
            activation=activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            batch_norm=batch_norm,
            normalize_v_grid=normalize_v_grid,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            ensure_non_negative_f=ensure_non_negative_f,
            includes_symmetry=False,
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
        self.K = MLP(
            2,
            self.kernel_size**2,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )

    def _init_v_grid(self, normalize: bool):
        vx, vy = self._default_vx_vy(normalize)
        VX, VY = torch.meshgrid(vx, vy, indexing="ij")
        self.v_grid = nn.Buffer(torch.stack([VX.flatten(), VY.flatten()], dim=-1))

    def _get_kernels(self):
        kernels = self.K(self.v_grid.detach())
        return kernels.T

    def _grad(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.gradient(f, dim=axis)[0]

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:

        if self.padding_mode == "zeros":
            f_padded = F.pad(f.unsqueeze(1), self.pad_size, "constant", 0)
        else:
            f_padded = F.pad(f.unsqueeze(1), self.pad_size, mode=self.padding_mode)
        # print("fp", f_padded.shape)
        # extract patches using unfold
        patches = F.unfold(f_padded, self.kernel_size, stride=1)
        # print("p", patches.shape)
        # compute kernels
        kernels = self._get_kernels()
        # print("k", kernels.shape)
        # # apply convolution using einsum
        df = torch.einsum("bkv,kv->bv", patches, kernels)
        # print("df", df.shape)
        df = df.reshape(f.shape)
        df = F.pad(df, (1, 1, 1, 1), "constant", 0)
        df = self._grad(df, axis=1) + self._grad(df, axis=2)
        df = df[:, 1:-1, 1:-1]

        # advance in time
        if isinstance(dt, torch.Tensor):
            f = f + df * dt.unsqueeze(1).unsqueeze(2)
        else:
            f = f + df * dt
        if self.ensure_non_negative_f:
            f = torch.clamp(f, min=0)
        return f
