import pytest
import torch
import torch.nn as nn

from ml_pic_collision_operators.models.fp2d.nn.gridless.base import (
    FokkerPlanck2D_NN_Gridless_Base,
)

_V_RANGE = (-0.5, 0.5, -0.5, 0.5)
_V_RANGE_ANISO = (-0.3, 0.7, -0.5, 0.5)
_V_UNITS = "[c]"
_DEPTH = 2
_WIDTH = 8


class _DummyGridless(FokkerPlanck2D_NN_Gridless_Base):
    """Minimal concrete subclass so non-abstract base behaviors can be exercised."""

    def _init_NN(
        self, depth, width_size, activation, use_bias, use_final_bias, batch_norm
    ):
        pass

    def A_at_points_real(self, v):
        return v

    def D_at_points_real(self, v):
        return v


def _make_dummy(**overrides) -> _DummyGridless:
    kwargs = dict(
        v_range_norm=_V_RANGE,
        v_units=_V_UNITS,
        depth=_DEPTH,
        width_size=_WIDTH,
    )
    kwargs.update(overrides)
    return _DummyGridless(**kwargs)  # type: ignore[arg-type]


class TestFokkerPlanck2D_NN_Gridless_Base:

    def test_base_is_abstract(self):
        # the base declares abstract methods, so it cannot be instantiated directly
        with pytest.raises(TypeError, match="abstract"):
            FokkerPlanck2D_NN_Gridless_Base(
                v_range_norm=_V_RANGE,
                v_units=_V_UNITS,
                depth=_DEPTH,
                width_size=_WIDTH,
            )

    def test_v_range_norm_must_have_four_entries(self):
        with pytest.raises(ValueError, match="v_range_norm must have 4 entries"):
            _make_dummy(v_range_norm=(-1.0, 1.0, -1.0))

    def test_includes_symmetry_rejects_non_axis_symmetric_range(self):
        with pytest.raises(ValueError, match="axis-symmetric"):
            _make_dummy(v_range_norm=_V_RANGE_ANISO, includes_symmetry=True)

    def test_includes_symmetry_rejects_anisotropic_magnitudes(self):
        with pytest.raises(ValueError, match="equal x and y range"):
            _make_dummy(v_range_norm=(-1.0, 1.0, -0.5, 0.5), includes_symmetry=True)

    def test_normalize_v_maps_endpoints_to_1(self):
        m = _make_dummy()
        corners = torch.tensor([[_V_RANGE[0], _V_RANGE[2]], [_V_RANGE[1], _V_RANGE[3]]])
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        assert torch.allclose(m._normalize_v(corners), expected)

    def test_normalize_and_denormalize_is_identity(self):
        m = _make_dummy(v_range_norm=_V_RANGE_ANISO)
        v = torch.tensor([[0.1, -0.2], [-0.25, 0.4]])
        assert torch.allclose(m._denormalize_v(m._normalize_v(v)), v, atol=1e-6)

    def test_init_params_dict_round_trips_through_init(self):
        m = _make_dummy(
            activation="torch.nn.ReLU", batch_norm=True, ensure_non_negative_D=True
        )
        rebuilt = _DummyGridless(**m.init_params_dict)
        assert rebuilt.init_params_dict == m.init_params_dict
