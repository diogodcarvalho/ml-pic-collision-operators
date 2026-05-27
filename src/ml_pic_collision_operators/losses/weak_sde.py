from typing import Callable

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_pic_collision_operators.dataloaders import BatchDatasetItem
from ml_pic_collision_operators.losses.test_functions import TestFunction
from ml_pic_collision_operators.models.fp2d.nn.gridless import (
    FokkerPlanck2D_NN_Gridless_Base,
)


def _single_step_residual(
    model: FokkerPlanck2D_NN_Gridless_Base,
    test_function: TestFunction,
    v_t: torch.Tensor,
    v_next: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """One-step Itô-Doeblin weak residual on ground-truth particle statistics.

    For each test function φ in the family, the residual r is

        r = ⟨φ(v_next) - φ(v_t)⟩_n  -  dt · ⟨A·∇φ + (1/2) D:∇²φ⟩_n (v_t)

    with the contraction expanded for 2D as

        (1/2) D:∇²φ = (1/2) (Dxx·∂²_xx φ + Dyy·∂²_yy φ + 2·Dxy·∂²_xy φ),

    No Brownian noise is drawn (better gradients for single step rollout).

    Args:
        model: gridless FP2D NN model.
        test_function: provides (φ, ∇φ, ∇²φ) at particle velocities.
        v_t: (B, N, 2) ground-truth particle velocities at time t.
        v_next: (B, N, 2) ground-truth particle velocities at time t + dt.
        dt: (B,) time step per batch item.

    Returns:
        Residual of shape (B, n_test_functions).
    """
    # F == n_test_functions
    # (B, N, F), (B, N, F, 2) and (B, N, F, 2, 2)
    phi_prev, grad_prev, hess_prev = test_function.evaluate(v_t)
    phi_next = test_function.evaluate_phi(v_next)

    # (B, N, 2)
    A = model.A_at_points_real(v_t)
    # (B, N, 3)
    D = model.D_at_points_real(v_t)

    # A · ∇φ  ->  (B, N, F)
    A_dot_grad = (A.unsqueeze(2) * grad_prev).sum(dim=-1)

    # (1/2) D:∇²φ  ->  (B, N, F)
    Dxx = D[..., 0:1]
    Dyy = D[..., 1:2]
    Dxy = D[..., 2:3]
    hxx = hess_prev[..., 0, 0]
    hyy = hess_prev[..., 1, 1]
    hxy = hess_prev[..., 0, 1]
    D_dotdot_hess = 0.5 * (Dxx * hxx + Dyy * hyy + 2.0 * Dxy * hxy)

    # (B, N, F)
    Lphi = A_dot_grad + D_dotdot_hess
    # (B, F)
    return (phi_next - phi_prev).mean(dim=1) - dt.view(-1, 1) * Lphi.mean(dim=1)


def generate_weak_sde_loss_fn(
    test_function: TestFunction,
    loss_name: str = "mse",
    loss_mode: str = "accumulated",
    unrolling_steps: int = 1,
) -> Callable[[nn.Module, BatchDatasetItem], torch.Tensor]:
    """Build a weak-form SDE loss function.

    Dispatches on `unrolling_steps`:

    - `unrolling_steps == 1`: Itô-Doeblin one-step residual on the ground-truth particle
      pair. The model contributes A and D evaluated at ground-truth particles, contracted
      analytically with ∇φ and ∇²φ from the test-function family. No model-side
      Brownian noise is drawn, so gradients are free of SDE-rollout shot noise.
      `loss_mode` is a no-op here (a single residual is produced).

    - `unrolling_steps > 1`: Not implemented. TODO

    Args:
        test_function: closed-form test-function family (provides φ, ∇φ, ∇²φ).
        loss_name: 'mae' or 'mse'.
        loss_mode: 'accumulated' (mean error over rollout) or 'last' (only last step).
            Only meaningful when `unrolling_steps > 1`.
        unrolling_steps: number of rollout steps.

    Returns:
        A `(model, batch) -> scalar` loss callable.
    """

    if loss_name not in ("mae", "mse"):
        raise ValueError(f"loss_name must be 'mae' or 'mse', got {loss_name}")
    if loss_mode not in ("accumulated", "last"):
        raise ValueError(f"loss_mode must be 'accumulated' or 'last', got {loss_mode}")
    if unrolling_steps < 1:
        raise ValueError(f"unrolling_steps must be >= 1, got {unrolling_steps}")

    reduce_fn = (lambda r: r.abs()) if loss_name == "mae" else (lambda r: r.square())

    def _assert_unrolling_step_match(batch: BatchDatasetItem) -> None:
        unrolling_steps_batch = batch.targets.shape[1]
        if unrolling_steps_batch != unrolling_steps:
            raise ValueError(
                f"loss was built with unrolling_steps={unrolling_steps} "
                f"but received a batch with targets.shape[1]={unrolling_steps_batch}. "
                "The dataset's `temporal_unroll_steps` must match the loss's "
                "`unrolling_steps`."
            )

    def loss_fn_single_step(model: nn.Module, batch: BatchDatasetItem) -> torch.Tensor:
        _assert_unrolling_step_match(batch)
        _m = model.module if isinstance(model, DDP) else model
        assert isinstance(_m, FokkerPlanck2D_NN_Gridless_Base)
        # batch.targets has shape (B, 1, N, 2) when unrolling_steps == 1
        v_next = batch.targets[:, 0]
        residual = _single_step_residual(
            _m, test_function, batch.inputs, v_next, batch.dt
        )
        return reduce_fn(residual).mean()

    def loss_fn_rollout(model: nn.Module, batch: BatchDatasetItem) -> torch.Tensor:
        raise NotImplementedError

    return loss_fn_single_step if unrolling_steps == 1 else loss_fn_rollout
