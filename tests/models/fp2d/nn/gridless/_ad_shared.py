import torch
from typing import Any


class GridlessADSharedTests:
    """Shared tests for all (non-base) gridless FP2D NN models.

    Subclasses inherit the shared tests below and should set MODEL_CLS and HEAD_ATTRS.

    Class name avoids the 'Test' prefix so pytest does not collect it directly.
    """

    MODEL_CLS: Any = None  # replace with model class to be tested.
    HEAD_ATTRS: tuple = ()  # replace with sub-module names whose gradients are checked

    # These should not need to be overridden by subclasses, but can be if needed.
    V_RANGE_NORM = (-2.0, 2.0, -2.0, 2.0)
    V_UNITS = "[c]"
    DEPTH = 2
    WIDTH_SIZE = 8
    _B = 2
    _N = 7
    _ATOL = 1e-8

    @staticmethod
    def _swap_axes(pts: torch.Tensor) -> torch.Tensor:
        return torch.stack([pts[..., 1], pts[..., 0]], dim=-1)

    def _make_model(self, **overrides):
        kwargs = dict(
            v_range_norm=self.V_RANGE_NORM,
            v_units=self.V_UNITS,
            depth=self.DEPTH,
            width_size=self.WIDTH_SIZE,
        )
        kwargs.update(overrides)
        return self.MODEL_CLS(**kwargs)  # type: ignore[arg-type]

    def _sample_v(self, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(self._B, self._N, 2) * 0.1

    def _force_negative_outputs(self, model):
        # set the final layer weights and biases to produce negative outputs for all heads
        # useful for testing clamping behavior when ensure_non_negative_D=True
        with torch.no_grad():
            for attr in self.HEAD_ATTRS:
                mlp = getattr(model, attr)
                mlp.model[-1].weight.zero_()
                mlp.model[-1].bias.fill_(-1.0)

    def test_A_output_shape(self):
        A = self._make_model().A_at_points_real(self._sample_v())
        assert A.shape == (self._B, self._N, 2)

    def test_D_output_shape(self):
        D = self._make_model().D_at_points_real(self._sample_v())
        assert D.shape == (self._B, self._N, 3)

    def test_ensure_non_negative_D_clamps_only_diagonal(self):
        m_free = self._make_model(ensure_non_negative_D=False)
        self._force_negative_outputs(m_free)
        m_clamp = self._make_model(ensure_non_negative_D=True)
        m_clamp.load_state_dict(m_free.state_dict())
        pts = self._sample_v()
        D_free = m_free.D_at_points_real(pts)
        D_clamp = m_clamp.D_at_points_real(pts)
        assert (D_free[..., 0] < 0).all() and (D_free[..., 1] < 0).all()
        assert (D_clamp[..., 0] >= 0).all() and (D_clamp[..., 1] >= 0).all()
        assert torch.equal(D_clamp[..., 2], D_free[..., 2])

    def test_forward_backprops_through_all_heads(self):
        m = self._make_model()
        v_new = m(self._sample_v(), dt=0.1)
        v_new.sum().backward()
        for attr in self.HEAD_ATTRS:
            head = getattr(m, attr)
            assert any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()
            )

    def test_forward_shape_matches_input(self):
        model = self._make_model()
        pts = self._sample_v()
        out = model.forward(pts, dt=0.1)
        assert out.shape == pts.shape
        assert out.dtype == pts.dtype

    def test_init_params_dict_roundtrip(self):
        model = self._make_model()
        rebuilt = self.MODEL_CLS(**model.init_params_dict)
        assert isinstance(rebuilt, self.MODEL_CLS)
        assert rebuilt.v_range_norm == self.V_RANGE_NORM

    def test_init_params_dict_excludes_includes_symmetry(self):
        # includes_symmetry is an internal base-class flag; leaking it would
        # break checkpoint reload since concrete __init__ signatures don't accept it.
        assert "includes_symmetry" not in self._make_model().init_params_dict
