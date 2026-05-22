import pytest
import torch
import numpy as np

from ml_pic_collision_operators.models import FokkerPlanck2D_Tensor_AD_ParPerp

_GRID_SIZE = (5, 5)
_GRID_RANGE = (-1.0, 1.0, -1.0, 1.0)
_GRID_DX = (0.4, 0.4)
_GRID_UNITS = "[c]"


def _make_model(n_radial: int = -1) -> FokkerPlanck2D_Tensor_AD_ParPerp:
    return FokkerPlanck2D_Tensor_AD_ParPerp(
        grid_size=_GRID_SIZE,
        grid_range=_GRID_RANGE,
        grid_dx=_GRID_DX,
        grid_units=_GRID_UNITS,
        n_radial=n_radial,
    )


class TestFokkerPlanck2D_Tensor_AD_ParPerp:

    def test_AD_real_return_full_axis_values(self):
        m = _make_model(n_radial=6)
        a_vals = torch.linspace(0.1, 1.0, m.n_radial)
        dpar_vals = torch.linspace(0.2, 2.0, m.n_radial)
        dperp_vals = torch.linspace(0.3, 3.0, m.n_radial)
        with torch.no_grad():
            m.A.copy_(a_vals)
            m.Dpar.copy_(dpar_vals)
            m.Dperp.copy_(dperp_vals)
        assert m.Apar_real.shape == (m.n_radial,)
        assert np.allclose(m.Apar_real, a_vals.numpy() * _GRID_DX[0])
        assert np.allclose(m.Dpar_real, dpar_vals.numpy() * _GRID_DX[0] ** 2)
        assert np.allclose(m.Dperp_real, dperp_vals.numpy() * _GRID_DX[0] ** 2)

    def test_n_radial_default_and_override(self):
        default_expected = _GRID_SIZE[0] // 2 + _GRID_SIZE[0] % 2
        assert _make_model().n_radial == default_expected
        assert _make_model(n_radial=12).n_radial == 12

    def test_vr_axis_spans_zero_to_diagonal(self):
        # vr_axis must reach the grid's diagonal magnitude so interpolation
        # covers every |v| at bin centers, including the corners.
        m = _make_model()
        diagonal = np.sqrt(_GRID_RANGE[1] ** 2 + _GRID_RANGE[3] ** 2)
        assert m.vr_axis[0].item() == pytest.approx(0.0)
        assert m.vr_axis[-1].item() == pytest.approx(diagonal)

    def test_A_grid_matches_par_perp_decomposition(self):
        # testing for constant A
        m = _make_model()
        a_val = 2.0
        with torch.no_grad():
            m.A.fill_(a_val)
        A = m.A_grid
        assert A.shape == (2, *_GRID_SIZE)
        assert torch.allclose(A[0], a_val * m.cos_theta)
        assert torch.allclose(A[1], a_val * m.sin_theta)

    def test_D_grid_matches_par_perp_decomposition(self):
        m = _make_model()
        # different constant values to ensure off-diagonal term is non-zero
        dpar_val, dperp_val = 1.0, 0.25
        with torch.no_grad():
            m.Dpar.fill_(dpar_val)
            m.Dperp.fill_(dperp_val)
        D = m.D_grid
        assert D.shape == (3, *_GRID_SIZE)
        c2 = m.cos_theta**2
        s2 = m.sin_theta**2
        cs = m.cos_theta * m.sin_theta
        assert torch.allclose(D[0], dpar_val * c2 + dperp_val * s2)
        assert torch.allclose(D[1], dpar_val * s2 + dperp_val * c2)
        assert torch.allclose(D[2], (dpar_val - dperp_val) * cs)

    def test_origin_cos_sin_overridden_for_odd_grid(self):
        # For odd grid_size the central bin sits at (vx=vy=0) where atan2 is
        # ambiguous; cos and sin are forced to sqrt(2)/2 so the model is pushed
        # to learn A(0)=0 and Dpar(0)=Dperp(0) instead of an arbitrary axis.
        m = _make_model()
        c = _GRID_SIZE[0] // 2
        assert m.cos_theta[c, c].item() == pytest.approx(np.sqrt(2) / 2)
        assert m.sin_theta[c, c].item() == pytest.approx(np.sqrt(2) / 2)
