import pytest
import numpy as np
import torch

from ml_pic_collision_operators.models.fp2d.base_conditioned import (
    FokkerPlanck2D_Base_Conditioned,
)

_CONDITIONED_KWARGS = dict(
    grid_size=(3, 3),
    grid_range=(-1.0, 1.0, -1.0, 1.0),
    grid_dx=(0.5, 0.5),
    grid_units="[c]",
    conditioners_size=2,
)


class _DummyConditioned(FokkerPlanck2D_Base_Conditioned):
    """Concrete probe whose A/D are constant over the grid but proportional to the first
    conditioner received, so whether normalization was applied is observable downstream.
    """

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        val = conditioners[:, 0].reshape(-1, 1, 1, 1)
        return val * torch.ones(conditioners.shape[0], 2, *self.grid_size)

    def D_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        val = conditioners[:, 0].reshape(-1, 1, 1, 1)
        return val * torch.ones(conditioners.shape[0], 3, *self.grid_size)


class TestFokkerPlanck2D_Base_Conditioned:

    def test_normalize_requires_min_and_max(self):
        with pytest.raises(ValueError, match="must be"):
            _DummyConditioned(**_CONDITIONED_KWARGS, normalize_conditioners=True)

    def test_min_values_length_must_match_conditioners_size(self):
        with pytest.raises(ValueError, match="conditioners_min_values"):
            _DummyConditioned(
                **_CONDITIONED_KWARGS,
                normalize_conditioners=True,
                conditioners_min_values=[0.0],
                conditioners_max_values=[1.0, 1.0],
            )

    def test_max_values_length_must_match_conditioners_size(self):
        with pytest.raises(ValueError, match="conditioners_max_values"):
            _DummyConditioned(
                **_CONDITIONED_KWARGS,
                normalize_conditioners=True,
                conditioners_min_values=[0.0, 0.0],
                conditioners_max_values=[1.0],
            )

    def test_normalize_conditioners(self):
        model = _DummyConditioned(
            **_CONDITIONED_KWARGS,
            normalize_conditioners=True,
            conditioners_min_values=[0.0, -2.0],
            conditioners_max_values=[1.0, 2.0],
        )
        # pick min, max, middle points
        c = torch.tensor([[0.0, -2.0], [1.0, 2.0], [0.5, 0.0]])
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
        assert torch.allclose(model._normalize_conditioners(c), expected)

    def test_AD_grid_real_normalizes_before_evaluating(self):
        c = torch.tensor([[3.0, 0.0]])
        norm = _DummyConditioned(
            **_CONDITIONED_KWARGS,
            normalize_conditioners=True,
            conditioners_min_values=[0.0, 0.0],
            conditioners_max_values=[4.0, 4.0],
        )
        plain = _DummyConditioned(**_CONDITIONED_KWARGS, normalize_conditioners=False)
        c_norm = norm._normalize_conditioners(c)
        # the normalize path matches the plain path evaluated at the manually-normalized
        # input, and differs from evaluating at the raw (un-normalized) input
        assert np.allclose(norm.A_grid_real(c), plain.A_grid_real(c_norm))
        assert np.allclose(norm.D_grid_real(c), plain.D_grid_real(c_norm))
        assert not np.allclose(norm.A_grid_real(c), plain.A_grid_real(c))
        assert not np.allclose(norm.D_grid_real(c), plain.D_grid_real(c))
