import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Callable

from ml_pic_collision_operators.dataloaders import BatchDatasetItem


def generate_ode_loss_fn(
    loss_name: str,
    loss_mode: str,
    unrolling_steps: int = 1,
) -> Callable[[nn.Module, BatchDatasetItem], torch.Tensor]:
    """Generates ODE loss functions.

    Used for temporal unrolling training with phase space models.

    Args:
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
        raise ValueError(f"unrolling_steps must be >= 1, got {unrolling_steps}.")

    def _assert_unrolling_step_match(batch: BatchDatasetItem) -> None:
        unrolling_steps_batch = batch.targets.shape[1]
        if unrolling_steps_batch != unrolling_steps:
            raise ValueError(
                f"loss was built with unrolling_steps={unrolling_steps} "
                f"but received a batch with targets.shape[1]={unrolling_steps_batch}. "
                "The dataset's `temporal_unroll_steps` must match the loss's "
                "`unrolling_steps`."
            )

    def single_step_loss_fn(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        error = y - y_pred
        if loss_name == "mae":
            return torch.mean(torch.abs(error))
        return torch.mean(torch.square(error))

    def loss_accumulated(model: nn.Module, batch: BatchDatasetItem) -> torch.Tensor:
        _assert_unrolling_step_match(batch)
        loss = torch.zeros((), device=batch.inputs.device)
        y_pred = batch.inputs.clone()
        _m = model.module if isinstance(model, DDP) else model
        cacheable = not _m.operator_is_time_dependent
        for step in range(unrolling_steps):
            kwargs = {"use_cached_operator": step > 0} if cacheable else {}
            if batch.conditioners is None:
                y_pred = model(y_pred, batch.dt, **kwargs)
            else:
                y_pred = model(y_pred, batch.dt, batch.conditioners, **kwargs)
            loss = loss + single_step_loss_fn(batch.targets[:, step], y_pred)
        loss = loss / unrolling_steps
        return loss

    def loss_last(model: nn.Module, batch: BatchDatasetItem) -> torch.Tensor:
        _assert_unrolling_step_match(batch)
        y_pred = batch.inputs.clone()
        _m = model.module if isinstance(model, DDP) else model
        cacheable = not _m.operator_is_time_dependent
        for step in range(unrolling_steps):
            kwargs = {"use_cached_operator": step > 0} if cacheable else {}
            if batch.conditioners is None:
                y_pred = model(y_pred, batch.dt, **kwargs)
            else:
                y_pred = model(y_pred, batch.dt, batch.conditioners, **kwargs)
        loss = single_step_loss_fn(batch.targets[:, step], y_pred)
        return loss

    return loss_accumulated if loss_mode == "accumulated" else loss_last
