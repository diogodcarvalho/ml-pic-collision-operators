import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


def _grad(f: torch.Tensor, axis: int, guard_cells: bool) -> torch.Tensor:
    """Computes the first derivative of f along axis with/out guard cells."""
    if guard_cells:
        return torch.gradient(f, dim=axis)[0]
    else:
        return torch.gradient(f, dim=axis, edge_order=2)[0]


def _grad2(f: torch.Tensor, axis: int, guard_cells: bool) -> torch.Tensor:
    """Computes the second derivative of f along axis with/out guard cells."""
    grad2f = torch.roll(f, -1, axis) - 2 * f + torch.roll(f, 1, axis)

    if guard_cells:
        return grad2f

    if axis == 1:
        # left x-boundary
        grad2f[:, 0] = 2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]
        # right x-boundary
        grad2f[:, -1] = 2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]
    elif axis == 2:
        # left y-boundary
        grad2f[:, :, 0] = 2 * f[:, :, 0] - 5 * f[:, :, 1] + 4 * f[:, :, 2] - f[:, :, 3]
        # right y-boundary
        grad2f[:, :, -1] = (
            2 * f[:, :, -1] - 5 * f[:, :, -2] + 4 * f[:, :, -3] - f[:, :, -4]
        )
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return grad2f


def fp2d_step(
    A: torch.Tensor,
    D: torch.Tensor,
    f: torch.Tensor,
    dt: torch.Tensor | float,
    guard_cells: bool,
    ensure_non_negative_f: bool,
) -> torch.Tensor:

    Af = A * f.unsqueeze(1)
    Df = D * f.unsqueeze(1)

    if guard_cells:
        Af = F.pad(Af, (1, 1, 1, 1), "constant", 0)
        Df = F.pad(Df, (1, 1, 1, 1), "constant", 0)

    gradv_Af = _grad(Af[:, 0], 1, guard_cells) + _grad(Af[:, 1], 2, guard_cells)
    gradvv_Df = (
        _grad2(Df[:, 0], 1, guard_cells)
        + _grad2(Df[:, 1], 2, guard_cells)
        + _grad(_grad(Df[:, 2], 2, guard_cells), 1, guard_cells)
        + _grad(_grad(Df[:, 2], 1, guard_cells), 2, guard_cells)
    )

    if guard_cells:
        gradv_Af = gradv_Af[:, 1:-1, 1:-1]
        gradvv_Df = gradvv_Df[:, 1:-1, 1:-1]

    df = -gradv_Af + gradvv_Df / 2.0

    if isinstance(dt, torch.Tensor):
        f = f + df * dt.unsqueeze(1).unsqueeze(2)
    else:
        f = f + df * dt

    if ensure_non_negative_f:
        f = torch.clamp(f, min=0)
    return f


def plot_operator(
    A: np.ndarray,
    D: np.ndarray,
    grid_range: tuple[float, float, float, float],
    grid_units: str,
    save_to: str | None = None,
    show: bool = True,
):
    """Plots the 2D Fokker-Planck Operator.

    Args:
        A: Advection tensor (2, Nx, Ny)
        D: Diffusion tensor (3, Nx, Ny)
        grid_range: Grid range (xmin, xmax, ymin, ymax)
        grid_units: Grid units (e.g. "[v_th]", "[c]", etc.)
        save_to: Save path for the plot. If None, the plot is not saved.
        show: Whether to show the plot.
    """
    fig = plt.figure(figsize=(12, 2.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3], figure=fig, wspace=0.4)

    gs_A = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.2)
    ax0 = fig.add_subplot(gs_A[0])
    ax1 = fig.add_subplot(gs_A[1])

    gs_D = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1], wspace=0.2)
    ax2 = fig.add_subplot(gs_D[0])
    ax3 = fig.add_subplot(gs_D[1])
    ax4 = fig.add_subplot(gs_D[2])

    ax = [ax0, ax1, ax2, ax3, ax4]

    Ax = A[0]
    Ay = A[1]
    Dxx = D[0]
    Dyy = D[1]
    Dxy = D[2]

    kwargs = {
        "origin": "lower",
        "extent": grid_range,
        "interpolation": None,
    }

    kwargs_A = dict(kwargs)
    kwargs_A["vmin"] = -np.max(np.abs(A))
    kwargs_A["vmax"] = np.max(np.abs(A))
    kwargs_A["cmap"] = "bwr"

    im0 = ax0.imshow(Ax.T, **kwargs_A)  # type: ignore[arg-type]
    ax0.set_title(r"$\mathbf{A}_x$")

    im1 = ax1.imshow(Ay.T, **kwargs_A)  # type: ignore[arg-type]
    ax1.set_title(r"$\mathbf{A}_y$")

    cbaxes_A = ax1.inset_axes((1.05, 0, 0.05, 1))
    cbar = fig.colorbar(im1, cax=cbaxes_A, orientation="vertical")
    cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
    cbar.ax.set_ylabel(f"$[{grid_units[1:-1]}\omega_p]$")

    kwargs_D = dict(kwargs)
    kwargs_D["vmin"] = -np.max(np.abs(D))
    kwargs_D["vmax"] = np.max(np.abs(D))
    kwargs_D["cmap"] = "BrBG"

    im2 = ax2.imshow(Dxx.T, **kwargs_D)  # type: ignore[arg-type]
    ax2.set_title(r"$\mathbf{D}_{xx}$")

    im3 = ax3.imshow(Dyy.T, **kwargs_D)  # type: ignore[arg-type]
    ax3.set_title(r"$\mathbf{D}_{yy}$")

    im4 = ax4.imshow(Dxy.T, **kwargs_D)  # type: ignore[arg-type]
    ax4.set_title(r"$\mathbf{D}_{xy}$")

    cbaxes_D = ax4.inset_axes((1.05, 0, 0.05, 1))
    cbar = fig.colorbar(im4, cax=cbaxes_D, orientation="vertical")
    cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
    cbar.ax.set_ylabel(f"$[{grid_units[1:-1]}^2\omega_p]$")

    xlabel = f"$v_x{grid_units}$"
    ylabel = f"$v_y{grid_units}$"
    plt.setp(ax, xlabel=xlabel)
    ax0.set_ylabel(ylabel)
    ax2.set_ylabel(ylabel)

    plt.setp(ax, xticks=[grid_range[0], 0, grid_range[1]])
    plt.setp(ax, yticks=[grid_range[2], 0, grid_range[3]])
    for a in [ax1, ax3, ax4]:
        a.set_yticklabels([])

    if save_to is not None:
        plt.savefig(save_to, dpi=300)
    if show:
        plt.show()
    plt.close()
