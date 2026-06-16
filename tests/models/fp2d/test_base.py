import numpy as np
import torch

from ml_pic_collision_operators.models.fp2d.base import FokkerPlanck2D_Base

_BASE_KWARGS = dict(
    grid_size=(3, 3),
    grid_range=(-1.0, 1.0, -1.0, 1.0),
    grid_dx=(0.5, 0.5),
    grid_units="[c]",
)


class _Dummy(FokkerPlanck2D_Base):
    """Concrete probe whose A/D grids are supplied directly."""

    def __init__(
        self,
        A: torch.Tensor = torch.zeros(2, 3, 3),
        D: torch.Tensor = torch.zeros(3, 3, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._A = A
        self._D = D

    @property
    def A_grid(self) -> torch.Tensor:
        return self._A

    @property
    def D_grid(self) -> torch.Tensor:
        return self._D


class TestFokkerPlanck2D_Base:

    def test_D_grid_processed_clamp(self):
        D = torch.stack(
            [
                torch.linspace(-1.0, 1.0, 9).reshape(3, 3),
                torch.full((3, 3), -0.5),
                torch.linspace(-2.0, -1.0, 9).reshape(3, 3),
            ]
        )
        model = _Dummy(D=D, ensure_non_negative_D=True, **_BASE_KWARGS)
        processed = model.D_grid_processed
        assert torch.allclose(processed[0], torch.clamp(D[0], min=0))
        assert torch.allclose(processed[1], torch.zeros(3, 3))
        assert torch.allclose(processed[2], D[2])

    def test_D_grid_processed_no_clamp(self):
        D = torch.full((3, 3, 3), -1.0)
        model = _Dummy(D=D, ensure_non_negative_D=False, **_BASE_KWARGS)
        assert torch.allclose(model.D_grid_processed, D)

    def test_D_grid_real_uses_processed(self):
        D = torch.full((3, 3, 3), -1.0)
        model = _Dummy(D=D, ensure_non_negative_D=True, **_BASE_KWARGS)
        real = model.D_grid_real
        assert np.allclose(real[0], 0.0)
        assert np.allclose(real[1], 0.0)
        assert np.allclose(real[2], -1.0 * np.prod(_BASE_KWARGS["grid_dx"]))

    def test_AD_grid_real_scales_by_dx(self):
        kwargs = {**_BASE_KWARGS, "grid_dx": (0.5, 0.25)}
        A = torch.randn(2, 3, 3)
        D = torch.rand(3, 3, 3)
        model = _Dummy(A, D, **kwargs)
        assert np.allclose(
            model.A_grid_real, A.numpy() * np.array([0.5, 0.25]).reshape((2, 1, 1))
        )
        assert np.allclose(
            model.D_grid_real,
            D.numpy() * np.array([0.25, 0.0625, 0.125]).reshape((3, 1, 1)),
        )

    def test_forward_uses_cached_operator(self):
        kwargs = {**_BASE_KWARGS, "grid_size": (5, 5)}
        A = torch.zeros(2, 5, 5)
        D = torch.full((3, 5, 5), 0.1)
        model = _Dummy(A, D, ensure_non_negative_f=False, **kwargs)
        f = torch.arange(25.0).reshape(1, 5, 5)

        out = model.forward(f, dt=0.1)  # populates the cache
        model._D = torch.full((3, 5, 5), 5.0)  # change the underlying operator

        cached = model.forward(f, dt=0.1, use_cached_operator=True)
        fresh = model.forward(f, dt=0.1, use_cached_operator=False)
        assert torch.allclose(cached, out)
        assert not torch.allclose(fresh, out)
