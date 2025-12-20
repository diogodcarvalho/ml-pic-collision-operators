import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Operator2DBase_Gradient_Kxy(nn.Module):

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
        includes_symmetry: bool = False,
        gradient_order: int = 2,
        zero_kernel_indices: list[tuple[int, int]] = None,
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
        self.gradient_order = gradient_order
        self.zero_kernel_indices = zero_kernel_indices

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "ensure_non_negative_f": ensure_non_negative_f,
            "kernel_size": kernel_size,
            "padding_mode": padding_mode,
            "gradient_order": gradient_order,
            "zero_kernel_indices": zero_kernel_indices,
        }

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    def _get_kernels_full(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _get_kernels(self) -> tuple[torch.Tensor, torch.Tensor]:
        Kx, Ky = self._get_kernels_full()
        if self.zero_kernel_indices is not None:
            Kx = Kx.clone()
            Ky = Ky.clone()
            for i, j in self.zero_kernel_indices:
                Kx[i, j] = 0
                Ky[j, i] = 0
        return Kx, Ky

    def _grad(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        if self.gradient_order == -1:
            return torch.roll(f, -1, dims=axis) - f
        if self.gradient_order == 1:
            return f - torch.roll(f, 1, dims=axis)
        elif self.gradient_order == 2:
            return torch.gradient(f, dim=axis)[0]
        else:
            raise NotImplementedError("gradient_order must be -1, 1 or 2")

    def plot(self, save_to: str | None = None):

        Kx, Ky = self._get_kernels()
        Kx = Kx.detach().cpu().numpy()
        Ky = Ky.detach().cpu().numpy()

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
        # extract patches using unfold
        patches = F.unfold(f_padded, self.kernel_size, stride=1)
        # compute kernels
        kx, ky = self._get_kernels()
        kx = kx.reshape(self.kernel_size**2, -1)
        ky = ky.reshape(self.kernel_size**2, -1)
        # apply convolution using einsum
        fkx = torch.einsum("bkv,kv->bv", patches, kx)
        fky = torch.einsum("bkv,kv->bv", patches, ky)
        fkx = fkx.reshape(f.shape)
        fky = fky.reshape(f.shape)
        # compute gradient
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
