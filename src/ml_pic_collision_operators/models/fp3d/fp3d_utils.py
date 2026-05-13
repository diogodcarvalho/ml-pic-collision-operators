import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from ml_pic_collision_operators.models.fp2d.fp2d_utils import _grad, _grad2


def _grad2_3D(f: torch.Tensor, axis: int, guard_cells: bool) -> torch.Tensor:
    """Computes the second derivative of f along axis with/out guard cells."""

    if axis != 3 or guard_cells:
        return _grad2(f, axis, guard_cells)

    grad2f = torch.roll(f, -1, axis) - 2 * f + torch.roll(f, 1, axis)
    # left z-boundary
    grad2f[:, :, :, 0] = (
        2 * f[:, :, :, 0] - 5 * f[:, :, :, 1] + 4 * f[:, :, :, 2] - f[:, :, :, 3]
    )
    # right z-boundary
    grad2f[:, :, :, -1] = (
        2 * f[:, :, :, -1] - 5 * f[:, :, :, -2] + 4 * f[:, :, :, -3] - f[:, :, :, -4]
    )
    return grad2f


def fp3d_step(
    A: torch.Tensor,
    D: torch.Tensor,
    f: torch.Tensor,
    dt: torch.Tensor | float,
    guard_cells: bool,
    ensure_non_negative_f: bool,
) -> torch.Tensor:
    """Perfoms a 3D Fokker-Planck Update step for a batch of data.

    Args:
        A: Advection tensor ([B,] 3, Nx, Ny, Nz)
        D: Diffusion tensor ([B,] Nx, Ny, Nz)
        f: Distribution function with shape (B, Nx, Ny, Nz)
        dt: Time-step, float or tensor with shape (B,)
        guard_cells: If True, add guard cells before gradient calculations
        ensure_non_negative_f: If True, forces f > 0 after FP update

    Returns:
        torch.Tensor: updated f with shape (B, Nx, Ny, Nz)
    """

    # (B, 3, Nx, Ny, Nz)
    Af = A * f.unsqueeze(1)
    # (B, 6, Nx, Ny, Nz)
    Df = D * f.unsqueeze(1)

    if guard_cells:
        # (B, 3, Nx + 2, Ny + 2, Nz + 2)
        Af = F.pad(Af, (1,) * 6, "constant", 0)
        Df = F.pad(Df, (1,) * 6, "constant", 0)

    # (B, Nx [+ 2], Ny [+ 2], Nz [+ 2])
    gradv_Af = (
        _grad(Af[:, 0], 1, guard_cells)
        + _grad(Af[:, 1], 2, guard_cells)
        + _grad(Af[:, 2], 3, guard_cells)
    )
    gradvv_Df = (
        _grad2_3D(Df[:, 0], 1, guard_cells)
        + _grad2_3D(Df[:, 1], 2, guard_cells)
        + _grad2_3D(Df[:, 2], 3, guard_cells)
        + _grad(_grad(Df[:, 3], 2, guard_cells), 1, guard_cells)
        + _grad(_grad(Df[:, 3], 1, guard_cells), 2, guard_cells)
        + _grad(_grad(Df[:, 4], 3, guard_cells), 1, guard_cells)
        + _grad(_grad(Df[:, 4], 1, guard_cells), 3, guard_cells)
        + _grad(_grad(Df[:, 5], 3, guard_cells), 2, guard_cells)
        + _grad(_grad(Df[:, 5], 2, guard_cells), 3, guard_cells)
    )
    df = -gradv_Af + gradvv_Df / 2.0

    if guard_cells:
        # (B, Nx, Ny, Nz)
        df = df[:, 1:-1, 1:-1, 1:-1]

    # (B, Nx, Ny, Nz)
    if isinstance(dt, torch.Tensor):
        f = f + df * dt.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        f = f + df * dt

    if ensure_non_negative_f:
        f = torch.clamp(f, min=0)
    return f


def plot_operator_3D(
    A: np.ndarray,
    D: np.ndarray,
    grid_range: tuple[float, float, float, float, float, float],
    grid_units: str,
    save_to: str | None = None,
    show: bool = True,
):
    """Plots the 3D Fokker-Planck Operator on the central z-slice.

    Args:
        A: Advection tensor (3, Nx, Ny, Nz)
        D: Diffusion tensor (6, Nx, Ny, Nz)
        grid_range: Grid range (xmin, xmax, ymin, ymax, zmin, zmax)
        grid_units: Grid units (e.g. "[v_th]", "[c]", etc.)
        save_to: Save path for the plot. If None, the plot is not saved.
        show: Whether to show the plot.
    """
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.4, hspace=0.45)

    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    z_mid = A.shape[3] // 2
    Ax = A[0, :, :, z_mid]
    Ay = A[1, :, :, z_mid]
    Az = A[2, :, :, z_mid]
    Dxx = D[0, :, :, z_mid]
    Dyy = D[1, :, :, z_mid]
    Dzz = D[2, :, :, z_mid]
    Dxy = D[3, :, :, z_mid]
    Dxz = D[4, :, :, z_mid]
    Dyz = D[5, :, :, z_mid]

    kwargs = {
        "origin": "lower",
        "extent": grid_range[:4],
        "interpolation": None,
    }

    kwargs_A = dict(kwargs)
    kwargs_A["vmin"] = -np.max(np.abs(A))
    kwargs_A["vmax"] = np.max(np.abs(A))
    kwargs_A["cmap"] = "bwr"

    kwargs_D = dict(kwargs)
    kwargs_D["vmin"] = -np.max(np.abs(D))
    kwargs_D["vmax"] = np.max(np.abs(D))
    kwargs_D["cmap"] = "BrBG"

    images = [
        (axes[0], Ax, r"$\boldsymbol{A}_x$", kwargs_A),
        (axes[1], Ay, r"$\boldsymbol{A}_y$", kwargs_A),
        (axes[2], Az, r"$\boldsymbol{A}_z$", kwargs_A),
        (axes[3], Dxx, r"$\boldsymbol{D}_{xx}$", kwargs_D),
        (axes[4], Dyy, r"$\boldsymbol{D}_{yy}$", kwargs_D),
        (axes[5], Dzz, r"$\boldsymbol{D}_{zz}$", kwargs_D),
        (axes[6], Dxy, r"$\boldsymbol{D}_{xy}$", kwargs_D),
        (axes[7], Dxz, r"$\boldsymbol{D}_{xz}$", kwargs_D),
        (axes[8], Dyz, r"$\boldsymbol{D}_{yz}$", kwargs_D),
    ]

    for ax, array, title, plot_kwargs in images:
        im = ax.imshow(array.T, **plot_kwargs)  # type: ignore[arg-type]
        ax.set_title(title)
        cbaxes = ax.inset_axes((1.05, 0, 0.05, 1))
        cbar = fig.colorbar(im, cax=cbaxes, orientation="vertical")
        cbar.formatter.set_powerlimits((0, 0))  # type: ignore[attr-defined]
        units_exp = "^2" if plot_kwargs is kwargs_D else ""
        cbar.ax.set_ylabel(rf"$[{grid_units[1:-1]}{units_exp}\omega_p]$")

    xlabel = f"$v_x{grid_units}$"
    ylabel = f"$v_y{grid_units}$"
    plt.setp(axes, xlabel=xlabel)
    plt.setp(axes, ylabel=ylabel)
    plt.setp(axes, xticks=[grid_range[0], 0, grid_range[1]])
    plt.setp(axes, yticks=[grid_range[2], 0, grid_range[3]])

    if save_to is not None:
        plt.savefig(save_to, dpi=300)
    if show:
        plt.show()
    plt.close()
