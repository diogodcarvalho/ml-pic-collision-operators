import torch


# code by MoritzLange
# https://github.com/pytorch/pytorch/issues/50334
def torch_interpolate(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    dim: int = -1,
    extrapolate: str = "constant",
) -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == "constant":
        # Pad m and b to get constant values outside of xp range
        m = torch.cat(
            [torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1
        )
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    elif extrapolate == "linear":
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)
    else:
        raise ValueError(
            f'Extrapolation type not implemented: {extrapolate} (use "constant" or "linear")'
        )

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)


import torch


def torch_interpolate2d(
    x: torch.Tensor,
    y: torch.Tensor,
    xp: torch.Tensor,
    yp: torch.Tensor,
    fp: torch.Tensor,
    extrapolate: str = "constant",
) -> torch.Tensor:
    """Two-dimensional bilinear interpolation on a rectilinear grid, with optional extrapolation.

    Args:
        x: X-coordinates where to interpolate, shape (...).
        y: Y-coordinates where to interpolate, shape (...).
        xp: Sorted 1D tensor of x sample locations, shape (Nx,).
        yp: Sorted 1D tensor of y sample locations, shape (Ny,).
        fp: Values on the grid, shape (..., Ny, Nx).
        extrapolate: How to handle points outside the grid:
            - 'linear': Linearly extrapolate using edge slopes.
            - 'constant': Clamp to boundary values.

    Returns:
        Interpolated values at (x, y), same shape as x and y.
    """
    if xp.ndim != 1 or yp.ndim != 1:
        raise ValueError("xp and yp must be 1D and sorted increasing.")
    if fp.shape[-2:] != (len(yp), len(xp)):
        raise ValueError("fp must have shape (..., Ny, Nx).")

    # Compute indices of the cell in which (x, y) lies
    ix = torch.searchsorted(xp, x) - 1
    iy = torch.searchsorted(yp, y) - 1

    if extrapolate == "constant":
        ix = torch.clamp(ix, 0, len(xp) - 2)
        iy = torch.clamp(iy, 0, len(yp) - 2)
    elif extrapolate == "linear":
        ix = torch.clamp(ix, -1, len(xp) - 2)
        iy = torch.clamp(iy, -1, len(yp) - 2)
    else:
        raise ValueError(f"Unknown extrapolation mode: {extrapolate}")

    # Get surrounding grid points
    x0 = xp[ix]
    x1 = xp[ix + 1].clamp_max(xp[-1])
    y0 = yp[iy]
    y1 = yp[iy + 1].clamp_max(yp[-1])

    # Normalize interpolation weights
    tx = (x - x0) / (x1 - x0 + 1e-12)
    ty = (y - y0) / (y1 - y0 + 1e-12)

    # Gather corner values
    def gather(ix_offset, iy_offset):
        gx = torch.clamp(ix + ix_offset, 0, len(xp) - 1)
        gy = torch.clamp(iy + iy_offset, 0, len(yp) - 1)
        return fp[..., gy, gx]

    f00 = gather(0, 0)
    f10 = gather(1, 0)
    f01 = gather(0, 1)
    f11 = gather(1, 1)

    # Bilinear interpolation formula
    fxy = (
        f00 * (1 - tx) * (1 - ty)
        + f10 * tx * (1 - ty)
        + f01 * (1 - tx) * ty
        + f11 * tx * ty
    )

    if extrapolate == "constant":
        # Clamp output outside domain
        fxy = torch.where(
            (x < xp[0]) | (x > xp[-1]) | (y < yp[0]) | (y > yp[-1]),
            f00,  # nearest edge value
            fxy,
        )

    return fxy


def torch_interpolate_uniform_firstdim(X, t, t0, dt, extrapolate="constant"):
    """
    X: (T, ...)
    t: (...,) or broadcastable
    t0: float
    dt: float
    """

    T = X.shape[0]

    # fractional index along the first dimension
    idx = (t - t0) / dt

    if extrapolate == "constant":
        idx = idx.clamp(0.0, T - 1.0000001)
    elif extrapolate != "linear":
        raise ValueError("extrapolate must be 'constant' or 'linear'")

    # compute integer bounds
    i0 = idx.floor().long()
    i1 = i0 + 1

    if extrapolate == "linear":
        i0 = i0.clamp(0, T - 2)
        i1 = i1.clamp(1, T - 1)

    w = (idx - i0).unsqueeze(0)  # broadcast across X's trailing dims

    # Advanced indexing: X[i0,...] and X[i1,...].
    # This uses real slicing; no gather kernels.
    X0 = X[i0]  # shape (..., *shape)
    X1 = X[i1]

    return X0 + w * (X1 - X0)
