import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class GradientScheme(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    CENTERED = "centered"


class K2D_Base(nn.Module):
    """Base class for General Integro-Differential Operator in 2D.

    Operator evolves a distribution function f according to:
        (df(vx,vy)/dt) = grad_v \cdot (K(vx,vy) * f(vx,vy))
    where `K` is the learned kernel operator and * is a cross-correlation operation.

    Child classes need to implement:
        K - property that returns the operator computed on the grid with shape
            (2, kernel_size, kernel_size, grid_size_x, grid_size_y)
    """

    # K depends only on model parameters — safe to cache across rollout steps.
    operator_is_step_invariant: bool = True

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float, float, float],
        grid_dx: tuple[float, float],
        grid_units: str,
        kernel_size: int,
        padding_mode: str = "zeros",
        ensure_non_negative_f: bool = True,
        gradient_scheme: str = "forward",
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
        self.gradient_scheme = GradientScheme(gradient_scheme)
        self._operator_cache: torch.Tensor | None = None

        self._init_params_dict = {
            "grid_dx": grid_dx,
            "grid_size": grid_size,
            "grid_range": grid_range,
            "grid_units": grid_units,
            "ensure_non_negative_f": ensure_non_negative_f,
            "kernel_size": kernel_size,
            "padding_mode": padding_mode,
            "gradient_scheme": gradient_scheme,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def init_params_dict(self) -> dict:
        return self._init_params_dict

    @property
    def K(self) -> torch.Tensor:
        # Must output shape (2, kernel_size, kernel_size, grid_size_x, grid_size_y)
        raise NotImplementedError

    @property
    def K_real(self) -> np.ndarray:
        return self.K.detach().cpu().numpy() * np.array(self.grid_dx).reshape(
            (2, 1, 1, 1, 1)
        )

    def _grad(self, f: torch.Tensor, axis: int) -> torch.Tensor:
        if self.gradient_scheme == GradientScheme.FORWARD:
            return torch.roll(f, -1, dims=axis) - f
        if self.gradient_scheme == GradientScheme.BACKWARD:
            return f - torch.roll(f, 1, dims=axis)
        elif self.gradient_scheme == GradientScheme.CENTERED:
            return torch.gradient(f, dim=axis)[0]
        else:
            raise NotImplementedError(
                "gradient_scheme must be one of "
                f"{', '.join([member.value for member in GradientScheme])}"
            )

    def plot(self, save_to: str | None = None):

        K = self.K_real
        Kx = K[0]
        Ky = K[1]

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
            # Create figure and axes with a small extra gap between the Kx and Ky halves.
            fig = plt.figure(figsize=(8, 4))
            width_ratios = [1] * self.kernel_size + [0.2] + [1] * self.kernel_size
            gs = fig.add_gridspec(
                self.kernel_size,
                2 * self.kernel_size + 1,
                width_ratios=width_ratios,
                hspace=0.1,
                wspace=0.1,
            )
            axes = np.empty((self.kernel_size, 2 * self.kernel_size), dtype=object)

            # Plot each subplot
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    row = self.kernel_size - j - 1
                    axes[row, i] = fig.add_subplot(gs[row, i])
                    axes[row, i].imshow(Kx[i, j].T, **imshow_kwargs)

                    axes[row, i + self.kernel_size] = fig.add_subplot(
                        gs[row, i + self.kernel_size + 1]
                    )
                    im = axes[row, i + self.kernel_size].imshow(
                        Ky[i, j].T, **imshow_kwargs
                    )

                    # Remove x ticks and labels except for the bottom row
                    if row != self.kernel_size - 1:
                        axes[row, i].set_xticklabels([])
                        axes[row, i].set_xticks([])
                        axes[row, i + self.kernel_size].set_xticklabels([])
                        axes[row, i + self.kernel_size].set_xticks([])

                    # Remove y ticks and labels except for the leftmost column of each half
                    if i > 0:
                        axes[row, i].set_yticklabels([])
                        axes[row, i].set_yticks([])
                        axes[row, i + self.kernel_size].set_yticklabels([])
                        axes[row, i + self.kernel_size].set_yticks([])

            plt.setp(axes[-1], xlabel=xlabel)
            plt.setp(axes[:, 0], ylabel=ylabel)

            # Add titles over the two halves
            axes[0, 0].set_title(r"$\mathbf{K}_x$", fontsize=16, pad=12)
            axes[0, self.kernel_size].set_title(r"$\mathbf{K}_y$", fontsize=16, pad=12)

            # Add single colorbar
            cbar = fig.colorbar(
                im, ax=axes.ravel().tolist(), location="right", fraction=0.02, pad=0.05
            )
            cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
            cbar.ax.set_ylabel(f"$[{self.grid_units[1:-1]}\omega_p]$")
        else:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(8, 4),
                sharex=True,
                sharey=True,
            )
            axes[0].imshow(Kx[0, 0], **imshow_kwargs)
            im = axes[1].imshow(Ky[0, 0], **imshow_kwargs)
            axes[0].set_title(r"$\mathbf{K}_x$", fontsize=16, pad=12)
            axes[1].set_title(r"$\mathbf{K}_y$", fontsize=16, pad=12)
            plt.setp(axes, xlabel=xlabel)
            axes[0].set_ylabel(ylabel)
            axes[1].set_yticklabels([])
            axes[1].set_yticks([])
            cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.02, pad=0.05)
            cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
            cbar.ax.set_ylabel(f"$[{self.grid_units[1:-1]}\omega_p]$")

        if save_to is not None:
            plt.savefig(save_to)
        plt.show()
        plt.close()

    def forward(
        self,
        f: torch.Tensor,
        dt: torch.Tensor | float,
        use_cached_operator: bool = False,
    ) -> torch.Tensor:

        if self.padding_mode == "zeros":
            f_padded = F.pad(f.unsqueeze(1), self.pad_size, "constant", 0)
        else:
            f_padded = F.pad(f.unsqueeze(1), self.pad_size, mode=self.padding_mode)
        # Extract patches using unfold
        patches = F.unfold(f_padded, self.kernel_size, stride=1)
        # Compute kernels
        if use_cached_operator and self._operator_cache is not None:
            K = self._operator_cache
        else:
            K = self.K
            self._operator_cache = K
        kx = K[0]
        ky = K[1]
        kx = kx.reshape(self.kernel_size**2, -1)
        ky = ky.reshape(self.kernel_size**2, -1)
        # Apply convolution using einsum
        fkx = torch.einsum("bkv,kv->bv", patches, kx)
        fky = torch.einsum("bkv,kv->bv", patches, ky)
        fkx = fkx.reshape(f.shape)
        fky = fky.reshape(f.shape)
        # Compute gradient
        fkx = F.pad(fkx, (1, 1, 1, 1), "constant", 0)
        fky = F.pad(fky, (1, 1, 1, 1), "constant", 0)
        df = self._grad(fkx, axis=1) + self._grad(fky, axis=2)
        df = df[:, 1:-1, 1:-1]

        # Advance in time
        if isinstance(dt, torch.Tensor):
            f = f + df * dt.unsqueeze(1).unsqueeze(2)
        else:
            f = f + df * dt
        if self.ensure_non_negative_f:
            f = torch.clamp(f, min=0)
        return f
