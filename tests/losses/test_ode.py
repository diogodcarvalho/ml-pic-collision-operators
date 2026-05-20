import pytest
import torch
import torch.nn as nn

from ml_pic_collision_operators.dataloaders import BatchDatasetItem
from ml_pic_collision_operators.losses import generate_ode_loss_fn

_DT_VAL = 0.1
_UNROLLING_STEPS = 3


class _DummyModel(nn.Module):
    """Deterministic placeholder model: y_pred = x + dt * scale.
    Each forward call's kwargs are appended to `self.calls` for inspection.
    """

    def __init__(self, scale: float = 1.0, time_dependent: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.operator_is_time_dependent = time_dependent
        self.calls: list[dict] = []

    def forward(self, x, dt, conditioners=None, **kwargs):
        self.calls.append({"conditioners": conditioners, **kwargs})
        dt_b = dt.view(-1, *([1] * (x.dim() - 1)))
        return x + dt_b * self.scale


def _make_batch(
    B: int = 2,
    unrolling_steps: int = _UNROLLING_STEPS,
    dt_val: float = _DT_VAL,
    conditioners: bool = False,
) -> BatchDatasetItem:
    torch.manual_seed(0)
    inputs = torch.randn(B, 4, 2)
    targets = torch.randn(B, unrolling_steps, 4, 2)
    dt = torch.full((B,), dt_val)
    cond = torch.randn(B, 3) if conditioners else None
    return BatchDatasetItem(inputs=inputs, targets=targets, dt=dt, conditioners=cond)


class TestODELoss:

    def test_rejects_unknown_loss_mode(self):
        with pytest.raises(ValueError, match="loss mode"):
            generate_ode_loss_fn("mse", "bogus", unrolling_steps=1)

    def test_rejects_unknown_loss_name(self):
        with pytest.raises(ValueError, match="loss function"):
            generate_ode_loss_fn("bogus", "accumulated", unrolling_steps=1)

    @pytest.mark.parametrize("bad_steps", [0, -1])
    def test_rejects_non_positive_unrolling_steps(self, bad_steps):
        with pytest.raises(ValueError, match="unrolling_steps"):
            generate_ode_loss_fn("mse", "accumulated", unrolling_steps=bad_steps)

    @pytest.mark.parametrize(
        "loss_name, reduce_fn",
        [("mse", torch.square), ("mae", torch.abs)],
    )
    def test_accumulated_matches_manual_rollout(self, loss_name, reduce_fn):
        scale = 1.0
        model = _DummyModel(scale=scale)
        batch = _make_batch()
        loss_fn = generate_ode_loss_fn(
            loss_name, "accumulated", unrolling_steps=_UNROLLING_STEPS
        )
        loss = loss_fn(model, batch)

        y_pred = batch.inputs.clone()
        expected = torch.zeros(())
        for step in range(_UNROLLING_STEPS):
            y_pred = y_pred + _DT_VAL * scale
            expected = expected + reduce_fn(batch.targets[:, step] - y_pred).mean()
        expected = expected / _UNROLLING_STEPS
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_accumulated_with_unrolling_steps_1_is_single_step_loss(self):
        model = _DummyModel()
        batch = _make_batch(unrolling_steps=1)
        loss_fn = generate_ode_loss_fn("mse", "accumulated", unrolling_steps=1)
        loss = loss_fn(model, batch)
        expected = ((batch.targets[:, 0] - (batch.inputs + _DT_VAL)) ** 2).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_loss_propagates_gradient_to_model(self):
        model = _DummyModel()
        batch = _make_batch()
        loss_fn = generate_ode_loss_fn(
            "mse", "accumulated", unrolling_steps=_UNROLLING_STEPS
        )
        loss_fn(model, batch).backward()
        assert model.scale.grad is not None and model.scale.grad.abs().item() > 0

    def test_last_uses_only_final_step(self):
        model = _DummyModel()
        batch = _make_batch()
        loss_fn = generate_ode_loss_fn("mse", "last", unrolling_steps=_UNROLLING_STEPS)
        loss = loss_fn(model, batch)

        y_final = batch.inputs + _UNROLLING_STEPS * _DT_VAL
        expected = ((batch.targets[:, _UNROLLING_STEPS - 1] - y_final) ** 2).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_time_independent_operator_caches_after_first_step(self):
        model = _DummyModel()
        batch = _make_batch()
        loss_fn = generate_ode_loss_fn(
            "mse", "accumulated", unrolling_steps=_UNROLLING_STEPS
        )
        loss_fn(model, batch)
        assert [c["use_cached_operator"] for c in model.calls] == [False] + [True] * (
            _UNROLLING_STEPS - 1
        )

    def test_time_dependent_operator_omits_cache_kwarg(self):
        model = _DummyModel(time_dependent=True)
        batch = _make_batch()
        loss_fn = generate_ode_loss_fn(
            "mse", "accumulated", unrolling_steps=_UNROLLING_STEPS
        )
        loss_fn(model, batch)
        for c in model.calls:
            assert "use_cached_operator" not in c

    def test_conditioners_are_forwarded(self):
        model = _DummyModel()
        batch = _make_batch(conditioners=True)
        loss_fn = generate_ode_loss_fn(
            "mse", "accumulated", unrolling_steps=_UNROLLING_STEPS
        )
        loss_fn(model, batch)
        for c in model.calls:
            assert c["conditioners"] is batch.conditioners
