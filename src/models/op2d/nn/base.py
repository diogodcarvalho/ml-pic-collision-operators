import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Callable

from src.utils import class_from_str


class Operator2DNNBase(nn.Module):

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
        includes_symmetry: bool = False,
    ):
        super().__init__()
        assert len(grid_size) == 2
        assert len(grid_range) == 4
        assert len(grid_dx) == 2
        if includes_symmetry:
            assert grid_size[0] == grid_size[1]
            assert grid_range[0] == grid_range[2]
            assert grid_range[1] == grid_range[3]
            assert grid_dx[0] == grid_dx[1]

        self.grid_dx = grid_dx
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.grid_units = grid_units
        self.ensure_non_negative_f = ensure_non_negative_f
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.pad_size = (self.kernel_size // 2, max(0, (self.kernel_size - 1) // 2)) * 2

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "ensure_non_negative_f": ensure_non_negative_f,
            "kernel_size": kernel_size,
            "depth": depth,
            "width_size": width_size,
            "activation": activation,
            "use_bias": use_bias,
            "use_final_bias": use_final_bias,
            "batch_norm": batch_norm,
            "normalize_v_grid": normalize_v_grid,
            "padding_mode": padding_mode,
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

        self._init_v_grid(normalize_v_grid)

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

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

    def _default_vx_vy(self, normalize: bool):
        vx = torch.linspace(
            self.grid_range[0], self.grid_range[1], self.grid_size[0] + 1
        )[:-1]
        vy = torch.linspace(
            self.grid_range[2], self.grid_range[3], self.grid_size[1] + 1
        )[:-1]
        vx += self.grid_dx[0] / 2.0
        vy += self.grid_dx[1] / 2.0
        if normalize:
            vx = 2 * (vx - torch.min(vx)) / (torch.max(vx) - torch.min(vx)) - 1
            vy = 2 * (vy - torch.min(vy)) / (torch.max(vy) - torch.min(vy)) - 1
        return vx, vy

    def _init_v_grid(self, normalize: bool):
        raise NotImplementedError

    def _get_kernels(self) -> torch.Tensor:
        raise NotImplementedError

    def plot(self, save_to: str | None = None):

        # Generate sample data
        K = self._get_kernels().detach().cpu().numpy()
        # print(K.shape)
        K = K.reshape(self.kernel_size, self.kernel_size, *self.grid_size)
        # print(K.shape)

        # Define shared imshow kwargs
        imshow_kwargs = {
            "origin": "lower",
            "vmax": np.max(np.abs(K)),
            "vmin": -np.max(np.abs(K)),
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
                self.kernel_size,
                figsize=(8, 8),
                sharex=True,
                sharey=True,
            )

            # Plot each subplot
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    im = axes[self.kernel_size - j - 1, i].imshow(
                        K[i, j].T, **imshow_kwargs
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
            plt.imshow(K[0, 0], **imshow_kwargs)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.colorbar()

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
        kernels = self._get_kernels()
        # print("k", kernels.shape)
        # # apply convolution using einsum
        df = torch.einsum("bkv,kv->bv", patches, kernels)
        # print("df", df.shape)
        df = df.reshape(f.shape)

        # advance in time
        if isinstance(dt, torch.Tensor):
            f = f + df * dt.unsqueeze(1).unsqueeze(2)
        else:
            f = f + df * dt
        if self.ensure_non_negative_f:
            f = torch.clamp(f, min=0)
        return f
