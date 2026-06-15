import pytest
import torch
import torch.nn as nn

from ml_pic_collision_operators.dataloaders import BatchDatasetItem
from ml_pic_collision_operators.losses import (
    MonomialTestFunctions,
    generate_weak_sde_loss_fn,
)
from ml_pic_collision_operators.losses.weak_sde import _single_step_residual
from ml_pic_collision_operators.models.fp2d.nn.gridless import (
    FokkerPlanck2D_NN_Gridless_Base,
)

_DT = 0.1
_UNROLLING_STEPS = 1
_ATOL = 1e-8


class _ConstADModel(FokkerPlanck2D_NN_Gridless_Base):
    """Minimal gridless FP2D model with constant, learnable A and D."""

    def __init__(self, A=(0.0, 0.0), D=(0.0, 0.0, 0.0)):
        # Bypasses the base __init__ so no neural network is built + don't have to pass
        # all the input args.
        # Only `A_at_points_real`, `D_at_points_real`, and isinstance (for type check)
        # are really needed to test the weak-SDE loss implementation.
        nn.Module.__init__(self)
        self.A_const = nn.Parameter(torch.tensor(A, dtype=torch.float64))
        self.D_const = nn.Parameter(torch.tensor(D, dtype=torch.float64))

    def _init_NN(self, *args, **kwargs):
        # required concrete override of the abstract base. unused: __init__ is bypassed
        pass

    def A_at_points_real(self, v):
        return self.A_const.to(v).expand_as(v)

    def D_at_points_real(self, v):
        return self.D_const.to(v).expand(*v.shape[:-1], 3)


def _make_batch(
    B: int = 2,
    N: int = 32,
    unrolling_steps: int = _UNROLLING_STEPS,
    dt: float = _DT,
    A: tuple[float, float] = (0.0, 0.0),
) -> BatchDatasetItem:
    """Build a deterministic batch where v_next = v_t + (k+1)·dt·A."""
    torch.manual_seed(0)
    A_t = torch.tensor(A)
    inputs = torch.randn(B, N, 2) * 0.1
    targets = torch.stack(
        [inputs + (k + 1) * dt * A_t for k in range(unrolling_steps)], dim=1
    )
    dt_b = torch.full((B,), dt)
    return BatchDatasetItem(inputs=inputs, targets=targets, dt=dt_b, conditioners=None)


class TestSingleStepResidual:

    def test_single_step_residual_shape(self):
        tf = MonomialTestFunctions(n_dims=2, degree=2)
        B, N = 3, 7
        v_t = torch.randn(B, N, 2)
        v_next = torch.randn_like(v_t)
        dt = torch.full((B,), _DT)
        res = _single_step_residual(_ConstADModel(), tf, v_t, v_next, dt)
        assert res.shape == (B, tf.n_functions)

    def test_residual_is_zero_when_drift_matches_for_linear_phi(self):
        A = (0.3, -0.2)
        batch = _make_batch(A=A)
        model = _ConstADModel(A=A)
        tf = MonomialTestFunctions(n_dims=2, degree=1)
        res = _single_step_residual(
            model, tf, batch.inputs, batch.targets[:, 0], batch.dt
        )
        assert torch.allclose(res, torch.zeros_like(res), atol=_ATOL)

    def test_residual_picks_up_diffusion_for_quadratic_phi(self):
        # With v_t = v_next and A=0, per-φ residual = -dt · (1/2) D : <∇²φ>_n.
        Dxx, Dyy, Dxy = 0.4, 0.7, 0.1
        v_t = torch.randn(2, 16, 2) * 0.1
        model = _ConstADModel(D=(Dxx, Dyy, Dxy))
        tf = MonomialTestFunctions(n_dims=2, degree=2)
        alphas = tf.alpha.tolist()
        dt = torch.full((2,), _DT)

        res = _single_step_residual(model, tf, v_t, v_t.clone(), dt)
        # For φ = v_x² the Hessian is constant [[2,0],[0,0]] giving -dt·Dxx.
        assert torch.allclose(
            res[:, alphas.index([2, 0])],
            torch.full_like(res[:, 0], -_DT * Dxx),
            atol=_ATOL,
        )
        # v_y² gives -dt·Dyy
        assert torch.allclose(
            res[:, alphas.index([0, 2])],
            torch.full_like(res[:, 0], -_DT * Dyy),
            atol=_ATOL,
        )
        # v_x·v_y gives -dt·Dxy
        assert torch.allclose(
            res[:, alphas.index([1, 1])],
            torch.full_like(res[:, 0], -_DT * Dxy),
            atol=_ATOL,
        )


class TestWeakSDELoss:

    def test_rejects_unknown_loss_mode(self):
        with pytest.raises(ValueError, match="loss_mode"):
            generate_weak_sde_loss_fn(
                MonomialTestFunctions(n_dims=2, degree=2), "mse", "bogus", 1
            )

    def test_rejects_unknown_loss_name(self):
        with pytest.raises(ValueError, match="loss_name"):
            generate_weak_sde_loss_fn(
                MonomialTestFunctions(n_dims=2, degree=2), "bogus", "accumulated", 1
            )

    @pytest.mark.parametrize("bad_steps", [0, -1])
    def test_rejects_non_positive_unrolling_steps(self, bad_steps):
        with pytest.raises(ValueError, match="unrolling_steps"):
            generate_weak_sde_loss_fn(
                MonomialTestFunctions(n_dims=2, degree=2),
                "mse",
                "accumulated",
                bad_steps,
            )

    def test_rejects_batch_with_mismatched_unrolling_steps(self):
        loss_fn = generate_weak_sde_loss_fn(
            MonomialTestFunctions(n_dims=2, degree=2), "mse", "accumulated", 1
        )
        batch = _make_batch(unrolling_steps=2)
        with pytest.raises(ValueError, match="targets.shape"):
            loss_fn(_ConstADModel(), batch)

    def test_unrolling_steps_greater_than_1_not_implemented(self):
        loss_fn = generate_weak_sde_loss_fn(
            MonomialTestFunctions(n_dims=2, degree=2), "mse", "accumulated", 2
        )
        with pytest.raises(NotImplementedError):
            loss_fn(_ConstADModel(), _make_batch(unrolling_steps=2))

    def test_loss_propagates_gradient_to_both_A_and_D(self):
        # Quadratic monomials turn on both the advection term (via grad_prev) and
        # the diffusion term (via hess_prev), so both A and D should receive
        # non-zero gradients.
        model = _ConstADModel(A=(0.1, -0.2), D=(0.05, 0.05, 0.01))
        batch = _make_batch()
        tf = MonomialTestFunctions(n_dims=2, degree=2)
        loss_fn = generate_weak_sde_loss_fn(tf, "mse", "accumulated", 1)
        loss_fn(model, batch).backward()
        assert model.A_const.grad is not None
        assert model.A_const.grad.abs().sum().item() > 0
        assert model.D_const.grad is not None
        assert model.D_const.grad.abs().sum().item() > 0
