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
        loss_name: Name of the loss function to use. Valid options are 'mae' and 'mse'.
        loss_mode: Mode of loss accumulation. Valid options are 'accumulated' and 'last'.
        unrolling_steps: Number of temporal unrolling steps.

    Returns:
        A `(model, batch) -> scalar` loss callable.
    """

    if loss_name not in ("mae", "mse"):
        raise ValueError(
            f"Unknown loss function: {loss_name}. Valid options are 'mae' and 'mse'."
        )
    if unrolling_steps < 1:
        raise ValueError(
            f"unrolling_steps must be >= 1, got {unrolling_steps}."
        )

    def single_step_loss_fn(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        error = y - y_pred
        if loss_name == "mae":
            return torch.mean(torch.abs(error))
        return torch.mean(torch.square(error))

    def loss_accumulated(model: nn.Module, batch: BatchDatasetItem) -> torch.Tensor:
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

    if loss_mode == "accumulated":
        return loss_accumulated
    elif loss_mode == "last":
        return loss_last
    else:
        raise ValueError(
            f"Unknown loss mode: {loss_mode}. Valid options are 'accumulated' and 'last'."
        )
