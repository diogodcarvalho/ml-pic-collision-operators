import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from src.models.op2d.base import Operator2DBase
from src.models.utils import MLP


class Operator2DNN_Gradient_Kxy(Operator2DBase):

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
        self.Kx = MLP(
            2,
            self.kernel_size**2,
            depth,
            width_size,
            activation,
            use_bias,
            use_final_bias,
            batch_norm,
        )
        self.Ky = MLP(
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
        kernels_x = self.Kx(self.v_grid.detach())
        kernels_y = self.Ky(self.v_grid.detach())
        return kernels_x.T, kernels_y.T

    def _grad(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        return torch.gradient(f, dim=axis)[0]

    def plot(self, save_to: str | None = None):

        Kx, Ky = self._get_kernels()
        Kx = Kx.detach().cpu().numpy()
        Ky = Ky.detach().cpu().numpy()
        # print(K.shape)
        Kx = Kx.reshape(self.kernel_size, self.kernel_size, *self.grid_size)
        Ky = Ky.reshape(self.kernel_size, self.kernel_size, *self.grid_size)
        # print(K.shape)

        # Define shared imshow kwargs
        imshow_kwargs = {
            "origin": "lower",
            "vmax": np.max(np.abs(np.concatenate([Kx, Ky]))),
            "vmin": -np.max(np.abs(np.concatenate([Kx, Ky]))),
            "extent": self.grid_range,
            "cmap": "bwr",
            "interpolation": None,
        }

        xlabel = f"$v_x{self.grid_units}$"
        ylabel = f"$v_y{self.grid_units}$"

        if self.kernel_size > 1:
            # Create figure and axes
            fig, axes = plt.subplots(
                self.kernel_size,
                2 * self.kernel_size,
                figsize=(16, 8),
                sharex=True,
                sharey=True,
            )

            # Plot each subplot
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    im = axes[self.kernel_size - j - 1, i].imshow(
                        Kx[i, j].T, **imshow_kwargs
                    )

                    im = axes[self.kernel_size - j - 1, i + self.kernel_size].imshow(
                        Ky[i, j].T, **imshow_kwargs
                    )

                    # Remove x ticks and labels except for the bottom row
                    if i < self.kernel_size - 2:
                        axes[i, j].set_xticklabels([])
                        axes[i, j].set_xticks([])

                    # Remove y ticks and labels except for the leftmost column
                    if j > 0:
                        axes[i, j].set_yticklabels([])
                        axes[i, j].set_yticks([])

            plt.setp(axes[-1], xlabel=xlabel)
            plt.setp(axes[:, 0], ylabel=ylabel)

            # Add single colorbar
            cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.02, pad=0.05)

        else:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(16, 8),
                sharex=True,
                sharey=True,
            )
            axes[0].imshow(Kx[0, 0], **imshow_kwargs)
            im = axes[1].imshow(Ky[0, 0], **imshow_kwargs)
            plt.setp(axes, xlabel=xlabel)
            axes[0].set_ylabel(ylabel)
            axes[1].set_yticklabels([])
            axes[1].set_yticks([])
            cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.02, pad=0.05)

        if save_to is not None:
            plt.savefig(save_to)
        plt.show()
        plt.close()

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
        kx, ky = self._get_kernels()
        # print("k", kernels.shape)
        # # apply convolution using einsum
        fkx = torch.einsum("bkv,kv->bv", patches, kx)
        fky = torch.einsum("bkv,kv->bv", patches, ky)
        # print("df", df.shape)
        fkx = fkx.reshape(f.shape)
        fky = fky.reshape(f.shape)
        fkx = F.pad(fkx, (1, 1, 1, 1), "constant", 0)
        fky = F.pad(fky, (1, 1, 1, 1), "constant", 0)
        df = self._grad(fkx, axis=1) + self._grad(fky, axis=2)
        df = df[:, 1:-1, 1:-1]

        # advance in time
        if isinstance(dt, torch.Tensor):
            f = f + df * dt.unsqueeze(1).unsqueeze(2)
        else:
            f = f + df * dt
        if self.ensure_non_negative_f:
            f = torch.clamp(f, min=0)
        return f
