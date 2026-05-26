import torch

from ml_pic_collision_operators.models import FokkerPlanck2D_NN_Gridless_AD

_V_RANGE = (-0.5, 0.5, -0.5, 0.5)
_V_UNITS = "[c]"
_DEPTH = 2
_WIDTH = 8
_B = 2
_N = 7


def _make_model(**overrides) -> FokkerPlanck2D_NN_Gridless_AD:
    kwargs = dict(
        v_range_norm=_V_RANGE,
        v_units=_V_UNITS,
        depth=_DEPTH,
        width_size=_WIDTH,
    )
    kwargs.update(overrides)
    return FokkerPlanck2D_NN_Gridless_AD(**kwargs)  # type: ignore[arg-type]


def _sample_v(batch: int = _B, n: int = _N) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, n, 2) * 0.1


class TestFokkerPlanck2D_NN_Gridless_AD:

    def test_AD_output_shapes(self):
        m = _make_model()
        v = _sample_v()
        assert m.A_at_points_real(v).shape == (_B, _N, 2)
        assert m.D_at_points_real(v).shape == (_B, _N, 3)

    def test_ensure_non_negative_D_clamps_only_diagonal(self):
        m_free = _make_model(ensure_non_negative_D=False)
        m_clamp = _make_model(ensure_non_negative_D=True)
        # Share NN weights so the models are identical.
        m_clamp.load_state_dict(m_free.state_dict())
        v = torch.linspace(-0.4, 0.4, 64).unsqueeze(-1).repeat(1, 2).unsqueeze(0)
        D_free = m_free.D_at_points_real(v)
        D_clamp = m_clamp.D_at_points_real(v)
        # Ensure randomly-initialized model hits negatives.
        # Otherwise the test does not matter.
        assert (D_free[..., 0] < 0).any() or (D_free[..., 1] < 0).any()
        # This is the actual test
        assert (D_clamp[..., 0] >= 0).all() and (D_clamp[..., 1] >= 0).all()
        assert torch.allclose(D_clamp[..., 2], D_free[..., 2])

    def test_forward_SDE_step_advances_and_backprops(self):
        m = _make_model()
        v = _sample_v()
        dt = torch.full((_B,), 0.1)
        v_new = m(v, dt)
        assert v_new.shape == v.shape and torch.isfinite(v_new).all()
        v_new.sum().backward()
        for head in (m.Ax, m.Ay, m.Dxx, m.Dyy, m.Dxy):
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()
            )

    def test_init_params_dict_round_trips_through_init(self):
        m = _make_model()
        # leaking init_params_dict would break checkpoint reload.
        assert "includes_symmetry" not in m.init_params_dict
        FokkerPlanck2D_NN_Gridless_AD(**m.init_params_dict)
