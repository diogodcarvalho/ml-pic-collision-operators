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
    conditioner received.
    """

    def A_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        val = conditioners[:, 0].reshape(-1, 1, 1, 1)
        return val * torch.ones(conditioners.shape[0], 2, *self.grid_size)

    def D_grid(self, conditioners: torch.Tensor) -> torch.Tensor:
        val = conditioners[:, 0].reshape(-1, 1, 1, 1)
        return val * torch.ones(conditioners.shape[0], 3, *self.grid_size)


class TestFokkerPlanck2D_Base_Conditioned:

    def test_normalize_conditioners_requires_min_and_max(self):
        with pytest.raises(ValueError, match="must be provided"):
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

    def test_conditioners_min_max_np_array_stored_as_lists(self):
        model = _DummyConditioned(
            **_CONDITIONED_KWARGS,
            normalize_conditioners=True,
            conditioners_min_values=np.array([0.0, -2.0]),
            conditioners_max_values=np.array([1.0, 2.0]),
        )
        params = model.init_params_dict
        # np arrays are converted to plain lists so the params dict stays serializable
        assert isinstance(params["conditioners_min_values"], list)
        assert isinstance(params["conditioners_max_values"], list)
        assert params["conditioners_min_values"] == [0.0, -2.0]
        assert params["conditioners_max_values"] == [1.0, 2.0]

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

    def test_D_grid_processed_clamp(self):
        model = _DummyConditioned(**_CONDITIONED_KWARGS, ensure_non_negative_D=True)
        # negative first conditioner makes every D component negative
        c = torch.tensor([[-1.0, 0.0]])
        raw = model.D_grid(c)
        processed = model.D_grid_processed(c)
        assert torch.all(processed[:, :2] >= 0)
        # off-diagonal component keeps its (negative) value
        assert torch.allclose(processed[:, 2], raw[:, 2])

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

    def test_AD_grid_real_scales_by_dx(self):
        kwargs = {**_CONDITIONED_KWARGS, "grid_dx": (0.5, 0.25)}  # distinct per axis
        model = _DummyConditioned(**kwargs)
        c = torch.tensor([[2.0, 0.0], [3.0, 0.0]])  # batch of 2
        A_real = model.A_grid_real(c)
        D_real = model.D_grid_real(c)
        assert A_real.shape == (2, 2, *kwargs["grid_size"])
        assert D_real.shape == (2, 3, *kwargs["grid_size"])
        assert np.allclose(
            A_real,
            model.A_grid(c).numpy() * np.array([0.5, 0.25]).reshape((1, 2, 1, 1)),
        )
        assert np.allclose(
            D_real,
            model.D_grid(c).numpy()
            * np.array([0.25, 0.0625, 0.125]).reshape((1, 3, 1, 1)),
        )

    @pytest.mark.parametrize(
        "conditioners",
        [
            torch.tensor([1.0, 0.0]),  # unbatched (C,)
            torch.tensor([[1.0, 0.0]]),  # single-row batch (1, C)
        ],
        ids=["1d", "2d_single_row"],
    )
    def test_plot_accepts_single_conditioner(self, conditioners, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "ml_pic_collision_operators.models.fp2d.base_conditioned.plot_operator",
            lambda **kwargs: calls.append(kwargs),
        )
        model = _DummyConditioned(**_CONDITIONED_KWARGS)
        model.plot(conditioners, show=False)
        # the batch dimension is dropped before plotting a single operator
        assert len(calls) == 1
        assert calls[0]["A"].shape == (2, *_CONDITIONED_KWARGS["grid_size"])
        assert calls[0]["D"].shape == (3, *_CONDITIONED_KWARGS["grid_size"])

    @pytest.mark.parametrize(
        "conditioners",
        [
            torch.tensor([[1.0, 0.0], [2.0, 0.0]]),  # multi-row batch (B>1, C)
            torch.zeros(1, 1, 2),  # too many dimensions
        ],
        ids=["multi_row_batch", "3d"],
    )
    def test_plot_rejects_bad_shapes(self, conditioners):
        model = _DummyConditioned(**_CONDITIONED_KWARGS)
        with pytest.raises(ValueError, match="shape"):
            model.plot(conditioners, show=False)

    def test_forward_reconstructs_per_conditioner(self):
        # a 5x5 grid gives the boundary stencils enough points without guard cells
        kwargs = {**_CONDITIONED_KWARGS, "grid_size": (5, 5)}
        model = _DummyConditioned(**kwargs, ensure_non_negative_f=False)
        f = torch.arange(25.0).reshape(1, 5, 5).repeat(3, 1, 1)
        # rows 0 and 2 share a conditioner, so unique/reverse_indices must map them
        # back to the same operator output, while row 1 differs
        c = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        out = model.forward(f, dt=0.1, conditioners=c)
        assert torch.allclose(out[0], out[2])
        assert not torch.allclose(out[0], out[1])

    def test_forward_uses_cached_operator(self):
        kwargs = {**_CONDITIONED_KWARGS, "grid_size": (5, 5)}
        model = _DummyConditioned(**kwargs, ensure_non_negative_f=False)
        f = torch.arange(25.0).reshape(1, 5, 5)
        c = torch.tensor([[1.0, 0.0]])

        out = model.forward(f, dt=0.1, conditioners=c)  # populates the cache
        new_c = torch.tensor([[5.0, 0.0]])
        cached = model.forward(f, dt=0.1, conditioners=new_c, use_cached_operator=True)
        fresh = model.forward(f, dt=0.1, conditioners=new_c, use_cached_operator=False)
        assert torch.allclose(cached, out)
        assert not torch.allclose(fresh, out)
