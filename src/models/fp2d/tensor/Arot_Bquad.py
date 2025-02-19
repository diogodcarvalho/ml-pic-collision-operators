import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from src.models.fp2d.base import FokkerPlanck2DBase


class FokkerPlanck2D_Arot_Bquad(FokkerPlanck2DBase):
    """
    Fokker-Planck model whish assumes that:
        1. Ax = |A_r|*cos(theta)
        2. Ay = |A_r|*sin(theta)
        3. Bxx is symmetric along vx=0 and vy=0
        4. Byy = Bxx^T
        5. Bxy is anti-symmetric along vx=0 and vy=0 OR zero
    """

    A: jax.Array
    B: jax.Array
    Bxy: jax.Array
    zero_Bxy: bool

    n_radial: int
    vr_axis: jax.Array
    vr_grid: jax.Array
    cos_theta: jax.Array
    sin_theta: jax.Array

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_range: tuple[float, float],
        grid_dx: tuple[float, float],
        ensure_non_negative_f: bool = True,
        zero_Bxy: bool = True,
        n_radial: int = -1,
    ):
        super().__init__(
            grid_size=grid_size,
            grid_range=grid_range,
            grid_dx=grid_dx,
            ensure_non_negative_f=ensure_non_negative_f,
        )
        assert grid_size[0] == grid_size[1]
        assert grid_range[0] == grid_range[2]
        assert grid_range[1] == grid_range[3]
        assert grid_range[0] == -grid_range[1]
        assert grid_dx[0] == grid_dx[1]
        self.zero_Bxy = zero_Bxy

        shape = (
            grid_size[0] // 2 + grid_size[0] % 2,
            grid_size[0] // 2 + grid_size[0] % 2,
        )
        self.B = jnp.zeros(shape)
        self.Bxy = jnp.zeros(shape)

        if n_radial == -1:
            self.n_radial = shape[0]
        else:
            self.n_radial = n_radial

        self.A = jnp.zeros((self.n_radial, 1))
        # velocity magnitude along axis in which self.A is defined
        self.vr_axis = jnp.linspace(0, self.grid_range[1], self.n_radial)

        # get velocities at bin centers
        vx = jnp.linspace(*self.grid_range[:2], self.grid_size[0], endpoint=False)
        vy = jnp.linspace(*self.grid_range[2:], self.grid_size[1], endpoint=False)
        vx += self.dx[0] / 2.0
        vy += self.dx[1] / 2.0
        VX, VY = jnp.meshgrid(vx, vy, indexing="ij")
        # velocity magnitude at bin centers
        self.vr_grid = jnp.sqrt(VX**2 + VY**2)

        # get angle of v=(vx,vy) with respect to x-axis
        theta = np.arctan2(VY, VX)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # force cos(vx=0,vy=0) = sin(vx=0,vy=0) = 1 to ensure model
        # learns that A(vx=0,vy=0) = 0
        # otherwise, atan2 sets cos=1 and sin=0
        if grid_size[0] % 2:
            cos_theta[grid_size[0] // 2, grid_size[0] // 2] = 1.0
            sin_theta[grid_size[0] // 2, grid_size[0] // 2] = 1.0

        self.cos_theta = jnp.array(cos_theta)
        self.sin_theta = jnp.array(sin_theta)

    @property
    def A_grid(self) -> jax.Array:
        Ar = jnp.interp(
            jax.lax.stop_gradient(self.vr_grid),
            jax.lax.stop_gradient(self.vr_axis),
            self.A[:, 0],
            right="extrapolate",
        )
        Ax = Ar * jax.lax.stop_gradient(self.cos_theta)
        Ay = Ar * jax.lax.stop_gradient(self.sin_theta)
        return jnp.stack([Ax, Ay], axis=0)

    @property
    def B_grid(self) -> jax.Array:
        Bxx = jnp.concatenate(
            [self.B, jnp.flip(self.B, axis=0)[self.grid_size[0] % 2 :]],
            axis=0,
        )
        Bxx = jnp.concatenate(
            [Bxx, jnp.flip(Bxx, axis=1)[:, self.grid_size[0] % 2 :]],
            axis=1,
        )
        if self.zero_Bxy:
            Bxy = jnp.zeros_like(Bxx)
        else:
            Bxy = jnp.concatenate(
                [self.Bxy, -jnp.flip(self.Bxy, axis=0)[self.grid_size[0] % 2 :]],
                axis=0,
            )
            Bxy = jnp.concatenate(
                [Bxy, -jnp.flip(Bxy, axis=1)[:, self.grid_size[0] % 2 :]],
                axis=1,
            )
        return jnp.stack([Bxx, Bxx.T, Bxy], axis=0)
