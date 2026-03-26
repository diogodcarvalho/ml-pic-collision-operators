import os
import numpy as np
import mlflow
import tempfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data import ConcatDataset, random_split
from typing import Any, Callable

import ml_pic_collision_operators.logging as logging
import ml_pic_collision_operators.utils as utils
from ml_pic_collision_operators.config.train import TrainConfig, TrainCallbackConfig
from ml_pic_collision_operators.datasets import BaseDataset, BasewConditionersDataset
from ml_pic_collision_operators.dataloaders import BaseDataLoader
from ml_pic_collision_operators.models import (
    FPModelType,
    FokkerPlanck2D_Base,
    FokkerPlanck2DBaseConditioned,
)


def _plot_loss(run_id: str):
    """Plot training and validation loss curves from MLflow metrics.

    Args:
        run_id: MLflow run ID to retrieve metrics from.
    """
    epochs, train_loss = logging.get_mlflow_metric_history("train_loss", run_id)
    _, valid_loss = logging.get_mlflow_metric_history("valid_loss", run_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plt.figure(figsize=(5, 4))
        plt.plot(epochs, train_loss, label="Train")
        if not np.all(valid_loss == 0):
            plt.plot(epochs, valid_loss, label="Valid")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0, epochs[-1])
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(tmp_dir, "loss.png")
        plt.savefig(fname, dpi=200)
        mlflow.log_artifact(fname, artifact_path="train_img")
        plt.show()
        plt.close()


def _plot_loss_with_regularization(run_id: str):
    """Plot training and validation loss curves with regularization from MLflow metrics.

    Args:
        run_id: MLflow run ID to retrieve metrics from.
    """
    epochs, train_loss = logging.get_mlflow_metric_history("train_loss", run_id)
    epochs, train_loss_data = logging.get_mlflow_metric_history(
        "train_loss_data", run_id
    )
    epochs, train_loss_reg = logging.get_mlflow_metric_history("train_loss_reg", run_id)
    _, valid_loss = logging.get_mlflow_metric_history("valid_loss", run_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plt.figure(figsize=(5, 4))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, train_loss_data, label="Train-Data")
        plt.plot(epochs, train_loss_reg, label="Train-Reg")
        if not np.all(valid_loss == 0):
            plt.plot(epochs, valid_loss, label="Valid")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0, epochs[-1])
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(tmp_dir, "loss_w_reg.png")
        plt.savefig(fname, dpi=200)
        mlflow.log_artifact(fname, artifact_path="train_img")
        plt.show()
        plt.close()


def _initialize_datasets(
    dataset_cls: str,
    folders: list[str],
    temporal_unroll_steps: int,
    dataset_cls_kwargs: dict[str, Any] = {},
    conditioners: list[dict[str, Any]] | None = None,
) -> list[BaseDataset] | list[BasewConditionersDataset]:
    """Initialize datasets from folders and dataset class.

    Args:
        dataset_cls: Dataset class name as string.
        folders: List of folders containing the datasets.
        temporal_unroll_steps: Number of temporal unrolling steps used for training.
        dataset_cls_kwargs: Keyword arguments for the dataset class.
        conditioners: Optional list of conditioners dictionaries (one per dataset) to be
            passed to the dataset class. If None, no conditioners are passed.

    Returns:
        datasets: List of initialized dataset objects.
    """
    dataset_cls = utils.class_from_str(
        dataset_cls, "ml_pic_collision_operators.datasets"
    )
    if conditioners is None:
        datasets = [
            dataset_cls(
                folder=f,
                temporal_unroll_steps=temporal_unroll_steps,
                **dataset_cls_kwargs,
            )
            for f in folders
        ]
    else:
        datasets = [
            dataset_cls(
                folder=f,
                conditioners=c,
                temporal_unroll_steps=temporal_unroll_steps,
                **dataset_cls_kwargs,
            )
            for f, c in zip(folders, conditioners)
        ]
    for i in range(1, len(datasets)):
        # All datasets must share same spatial dimensions and units
        assert datasets[0].grid_size == datasets[i].grid_size
        assert np.equal(datasets[0].grid_range, datasets[i].grid_range).all()
        assert np.equal(datasets[0].grid_dx, datasets[i].grid_dx).all()
        assert datasets[0].grid_units == datasets[i].grid_units

    return datasets


def _initialze_dataloader(
    dataset: ConcatDataset | BaseDataset | BasewConditionersDataset,
    dataloader_cls: str | None = None,
    dataloader_cls_kwargs: dict[str, Any] = {},
    device: str | int | None = None,
) -> BaseDataLoader | list[list[torch.Tensor]]:
    """Initializes dataloader object.

    Args:
        dataset: Dataset object to initialize dataloader for.
        dataloader_cls: Dataloader class name as string. If None, the entire dataset is
            loaded into memory and returned as a list of batches. If not None, the
            dataloader class is initialized with dataset and dataloader_cls_kwargs.
        dataloader_cls_kwargs: Keyword arguments for the dataloader class.
        device: Device to load the data to when using default BaseDataLoader. Ignored if
            dataloader_cls is not None.

    Returns:
        dataloader: Initialized dataloader object or list of batches loaded into memory.
    """
    if dataloader_cls is None:
        dataloader = BaseDataLoader(
            dataset,
            batch_size=len(dataset),
        )
        dataloader = next(iter(dataloader))
        dataloader = [[dataloader[i].to(device) for i in range(len(dataloader))]]
    else:
        d_cls = utils.class_from_str(
            dataloader_cls, "ml_pic_collision_operators.dataloaders"
        )
        dataloader = d_cls(dataset, **dataloader_cls_kwargs)

    return dataloader


def _do_train_valid_split(
    datasets: list[Any], train_valid_ratio: float
) -> tuple[list[Any], list[Any]]:
    """Randomly splits datasets into train and validation sets.

    Args:
        datasets: List of datasets to be split.
        train_valid_ratio: Ratio of train set size to total dataset size. Must be
            between 0 and 1. If train_valid_ratio is 1, no validation set is created.

    Returns:
        train_datasets: List of training datasets.
        valid_datasets: List of validation datasets.
    """
    train_datasets = []
    valid_datasets = []
    train_valid_ratio = train_valid_ratio
    if train_valid_ratio < 0 or train_valid_ratio > 1:
        raise ValueError("train_valid_ratio must be between 0 and 1.")
    elif train_valid_ratio == 1.0:
        train_datasets = datasets
    else:
        for i in range(len(datasets)):
            train_size = int(len(datasets[i]) * train_valid_ratio)
            valid_size = len(datasets[i]) - train_size
            train_portion, valid_portion = random_split(
                datasets[i], [train_size, valid_size]
            )
            train_datasets.append(train_portion)
            valid_datasets.append(valid_portion)
    return train_datasets, valid_datasets


def _initialize_model(
    model_cls_str: str,
    model_cls_kwargs: dict[str, Any],
    datasets: list[BaseDataset] | list[BasewConditionersDataset],
    device: str,
    compile_model: bool,
) -> tuple[FPModelType, dict[str, Any]]:
    """Initialize model for non-DDP training.

    Args:
        model_cls_str: Model class name as string.
        model_cls_kwargs: Keyword arguments for the model class.
        datasets: List of datasets used for training (used for inferring model grid
            parameters).
        deviice: Device to initialize the model on.
        compile_model: Whether to compile the model with torch.compile().

    Returns:
        model: Initialized model.
        model_kwargs: Dictionary of model keyword arguments used for initialization
            (useful for logging).
    """
    model_kwargs = {
        "grid_size": datasets[0].grid_size,
        "grid_range": datasets[0].grid_range,
        "grid_dx": datasets[0].grid_dx,
        "grid_units": datasets[0].grid_units,
    }

    if isinstance(datasets[0], BasewConditionersDataset):
        model_kwargs["conditioners_size"] = datasets[0].conditioners_size
        if model_cls_kwargs.get("normalize_conditioners", False):
            c_values = []
            for i in range(len(datasets)):
                d_aux = datasets[i]
                assert isinstance(d_aux, BasewConditionersDataset)
                c_values.append(d_aux.conditioners_array)
            c_values = np.stack(c_values, axis=0)
            model_kwargs["conditioners_min_values"] = np.min(c_values, axis=0)
            model_kwargs["conditioners_max_values"] = np.max(c_values, axis=0)
    # if datasets[0].include_time:
    #     model_kwargs["conditioners_size"] = 1
    #     model_kwargs["conditioners_min_values"] = np.array([0.0])
    #     model_kwargs["conditioners_max_values"] = np.array(
    #         [datasets[0].dt * len(datasets[0])]
    #     )
    if model_cls_str == "FokkerPlanck2DTime_ABparperp":
        model_kwargs["grid_size_dt"] = len(datasets[0])  # type: ignore
        model_kwargs["grid_dt"] = datasets[0].dt  # type: ignore
        del model_kwargs["conditioners_size"]
        del model_kwargs["conditioners_min_values"]
        del model_kwargs["conditioners_max_values"]

    model_cls = utils.class_from_str(model_cls_str, "ml_pic_collision_operators.models")
    model_kwargs = model_kwargs | model_cls_kwargs
    model = model_cls(**model_kwargs)
    model = model.to(device)
    if compile_model:
        model = torch.compile(model)

    return model, model_kwargs


def _initialize_model_ddp(
    model_cls_str: str,
    model_cls_kwargs: dict[str, Any],
    datasets: list[BaseDataset] | list[BasewConditionersDataset],
    local_rank: int,
    compile_model: bool,
) -> tuple[DDP, dict[str, Any]]:
    """Initialize model for DDP training.

    Args:
        model_cls_str: Model class name as string.
        model_cls_kwargs: Keyword arguments for the model class.
        datasets: List of datasets used for training (used for inferring model grid
            parameters).
        local_rank: Local rank of the current process (GPU index).
        compile_model: Whether to compile the model with torch.compile().

    Returns:
        model: Initialized DDP model.
        model_kwargs: Dictionary of model keyword arguments used for initialization
            (useful for logging).
    """
    model_kwargs = {
        "grid_size": datasets[0].grid_size,
        "grid_range": datasets[0].grid_range,
        "grid_dx": datasets[0].grid_dx,
        "grid_units": datasets[0].grid_units,
    }

    if isinstance(datasets[0], BasewConditionersDataset):
        model_kwargs["conditioners_size"] = datasets[0].conditioners_size
        if model_cls_kwargs.get("normalize_conditioners", False):
            c_values = []
            for i in range(len(datasets)):
                d_aux = datasets[i]
                assert isinstance(d_aux, BasewConditionersDataset)
                c_values.append(d_aux.conditioners_array)
            c_values = np.stack(c_values, axis=0)
            conditioners_min_values = np.min(c_values, axis=0)
            conditioners_max_values = np.max(c_values, axis=0)

            # Get min/max values accross all ranks
            conditioners_min_values = torch.from_numpy(conditioners_min_values).to(
                local_rank
            )
            conditioners_max_values = torch.from_numpy(conditioners_max_values).to(
                local_rank
            )
            dist.all_reduce(conditioners_min_values, op=dist.ReduceOp.MIN)
            dist.all_reduce(conditioners_max_values, op=dist.ReduceOp.MAX)

            model_kwargs["conditioners_min_values"] = (
                conditioners_min_values.cpu().numpy().flatten()  # type: ignore
            )
            model_kwargs["conditioners_max_values"] = (
                conditioners_max_values.cpu().numpy().flatten()  # type: ignore
            )

    model_cls = utils.class_from_str(model_cls_str, "ml_pic_collision_operators.models")
    model_kwargs = model_kwargs | model_cls_kwargs
    model = model_cls(**model_kwargs)
    if compile_model:
        model = torch.compile(model)
    model = model.to(local_rank)
    model = DDP(
        model,
        device_ids=[
            local_rank,
        ],
    )

    return model, model_kwargs


def _initialize_optimizer(
    optimizer_cls_str: str,
    optimizer_cls_kwargs: dict[str, Any],
    lr: float | None,
    model: nn.Module,
) -> torch.optim.Optimizer:
    """Initializes optimizer for training.

    Args:
        optimizer_cls_str: Torch optimizer class to use.
        optimizer_cls_kwargs: Keyword arguments for the optimizer class.
        lr: Learning rate can be passed separetely for convenience. If lr is specified
            in both lr argument and optimizer_cls_kwargs, a ValueError is raised.
        model: Model to optimize.

    Returns:
        Initialized optimizer.
    """
    optimizer_cls = utils.class_from_str(optimizer_cls_str)
    optimizer_cls_kwargs = optimizer_cls_kwargs

    if lr is None:
        return optimizer_cls(model.parameters(), **optimizer_cls_kwargs)
    if "lr" not in optimizer_cls_kwargs:
        return optimizer_cls(model.parameters(), lr, **optimizer_cls_kwargs)
    raise ValueError(
        "Learning rate specified in both lr argument and optimizer_cls_kwargs."
    )


def _generate_loss_fn(
    loss_name: str,
    loss_mode: str,
    unrolling_steps: int = 1,
    y_extra_cells: int = 0,
) -> Callable[
    [nn.Module, torch.Tensor, torch.Tensor, float, torch.Tensor | None], torch.Tensor
]:
    """Generates loss function for temporal unrolling training.

    Args:
        loss_name: Name of the loss function to use. Valid options are 'mae' and 'mse'.
        loss_mode: Mode of loss accumulation. Valid options are 'accumulated' and 'last'.
        unrolling_steps: Number of temporal unrolling steps.
        y_extra_cells: Number of extra cells in the target y tensor (used to exclude
            padding from loss computation).

    Returns:
        A loss function that can be used for temporal unrolling training. The loss
        function receives as inputs the model, input tensor x, target tensor y, time
        step dt, and optional conditioners tensor c as input and returns the computed
        loss as output.
    """

    def single_step_loss_fn(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        error = y - y_pred
        if y_extra_cells != 0:
            error = error[
                :,
                y_extra_cells:-y_extra_cells,
                y_extra_cells:-y_extra_cells,
            ]
        if loss_name == "mae":
            return torch.mean(torch.abs(error))
        elif loss_name == "mse":
            return torch.mean(torch.square(error))
        else:
            raise ValueError(
                f"Unknown loss function: {loss_name}. Valid options are 'mae' and 'mse'."
            )

    def loss_accumulated(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        dt: float,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.tensor([0.0], device=x.device)
        y_pred = x.clone()
        for step in range(unrolling_steps):
            if c is None:
                y_pred = model(y_pred, dt)
            else:
                y_pred = model(y_pred, dt, c)
            loss = loss + single_step_loss_fn(y[:, step], y_pred)
        loss = loss / unrolling_steps
        return loss

    def loss_last(
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        dt: float,
        c: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y_pred = x.clone()
        for step in range(unrolling_steps):
            if c is None:
                y_pred = model(y_pred, dt)
            else:
                y_pred = model(y_pred, dt, c)
        loss = single_step_loss_fn(y[:, step], y_pred)
        return loss

    if loss_mode == "accumulated":
        return loss_accumulated
    elif loss_mode == "last":
        return loss_last
    else:
        raise ValueError(
            f"Unknown loss mode: {loss_mode}. Valid options are 'accumulated' and 'last'."
        )


def _log_model_plot(
    model: FPModelType,
    model_img_path: str,
    datasets: list[BaseDataset] | list[BasewConditionersDataset],
):
    """Log model plot to MLflow.

    Args:
        model: The model to be plotted (must be non-DDP object).
        model_img_path: Path to save the model plot image.
        datasets: List of datasets used for training (used for plotting when
            conditioners are available).
    """
    with torch.no_grad():
        model.eval()
        if isinstance(model, FokkerPlanck2D_Base):
            model.plot(model_img_path)
            mlflow.log_artifact(model_img_path, artifact_path="model_img")
        elif isinstance(model, FokkerPlanck2DBaseConditioned):
            seen_conditioners = []
            for i in range(len(datasets)):
                d_aux = datasets[i]
                assert isinstance(d_aux, BasewConditionersDataset)
                c_str = str(d_aux.conditioners)
                if c_str not in seen_conditioners:
                    seen_conditioners.append(c_str)
                    c_img_path = model_img_path.replace(".png", f"-{c_str}.png")
                    model_device = next(model.parameters()).device
                    model.plot(
                        torch.Tensor(
                            d_aux.conditioners_array,
                        )
                        .to(model_device)
                        .unsqueeze(0),
                        save_to=c_img_path,
                    )
                    mlflow.log_artifact(c_img_path, artifact_path="model_img")
        model.train()


def _do_epoch_callbacks(
    callbacks: TrainCallbackConfig | None,
    epoch: int,
    tmp_dir: str,
    model: FPModelType,
    is_best_model: bool,
    datasets: list[BaseDataset] | list[BasewConditionersDataset],
    include_time: bool,
    compiled_model: bool = False,
):
    """Callbacks to be done at the end of a training epoch.

    Plotting callbacks are disabled if the dataset includes time dimension or if the
    model is a DDP model.

    Args:
        callbacks: TrainCallbackConfig object containing callback configuration.
        tmp_dir: Temporary directory for saving model checkpoints and plots.
        model: The trained model (must be non-DDP object).
        is_best_model: Whether the current model is the best model observed during
            training.
        datasets: List of datasets used for training (used for plotting when
            conditioners are available).
        include_time: Whether the dataset includes time dimension (used to disable
            plotting).
        compiled_model: Whether the model was compiled with torch.compile()
    """
    if callbacks is None:
        return

    if callbacks.log_model.enabled:
        assert callbacks.log_model.frequency is not None
        if epoch % callbacks.log_model.frequency == 0:
            logging.log_model(model, tmp_dir, "weights.pth", compiled_model)
    if (
        callbacks.log_best_model.enabled
        and callbacks.log_best_model.frequency == "always"
        and is_best_model
    ):
        logging.log_model(model, tmp_dir, "weights-best.pth", compiled_model)

    if callbacks.plot_model.enabled and not include_time and not isinstance(model, DDP):
        assert callbacks.plot_model.frequency is not None
        if epoch % callbacks.plot_model.frequency == 0:
            _log_model_plot(
                model,
                model_img_path=os.path.join(tmp_dir, f"model-{epoch:06d}.png"),
                datasets=datasets,
            )


def _do_stage_callbacks(
    callbacks: TrainCallbackConfig | None,
    stage_name: str,
    tmp_dir: str,
    model: FPModelType | DDP,
    best_model_dict: dict[str, Any],
    run_id: str,
    datasets: list[BaseDataset] | list[BasewConditionersDataset],
    include_time: bool,
    compiled_model: bool = False,
):
    """Callbacks to be done at the end of a training stage.

    Plotting callbacks are disabled if the dataset includes time dimension or if the
    model is a DDP model.

    Args:
        callbacks: TrainCallbackConfig object containing callback configuration.
        stage_name: Name of the current training stage (used for naming checkpoints and
            plots).
        tmp_dir: Temporary directory for saving model checkpoints and plots.
        model: The trained model (can be DDP or non-DDP).
        best_model_dict: Dictionary containing the state dict of the best model observed
            during training.
        run_id: MLflow run ID for logging.
        datasets: List of datasets used for training (used for plotting when
            conditioners are available).
        include_time: Whether the dataset includes time dimension (used to disable
            plotting).
        compiled_model: Whether the model was compiled with torch.compile()
    """
    if callbacks is None:
        return

    if (
        callbacks.log_best_model.enabled
        and callbacks.log_best_model.frequency == "stage_end"
    ):
        init_params_dict = logging.get_model_init_params_dict(model, compiled_model)
        logging.log_model_init_params_and_state_dict(
            init_params_dict, best_model_dict, tmp_dir, "weights-best.pth"
        )

    model_aux = None
    if callbacks.log_best_stage_model.enabled:
        model_aux = logging.load_model(run_id, "weights-best.pth")
        logging.log_model(model_aux, tmp_dir, f"weights-stage-{stage_name}.pth")
    if (
        callbacks.plot_best_stage_model.enabled
        and not include_time
        and not isinstance(model, DDP)
    ):
        if model_aux is None:
            model_aux = logging.load_model(run_id, "weights-best.pth")
        _log_model_plot(
            datasets=datasets,
            model=model_aux,
            model_img_path=os.path.join(tmp_dir, f"model-stage-{stage_name}.png"),
        )


def _do_end_callbacks(
    callbacks: TrainCallbackConfig | None,
    tmp_dir: str,
    model: FPModelType | DDP,
    best_model_dict: dict[str, Any],
    run_id: str,
    datasets: list[BaseDataset] | list[BasewConditionersDataset],
    include_time: bool,
    compiled_model: bool = False,
):
    """Callbacks to be done at the end of training after all stages are completed.

    Plotting callbacks are disabled if the dataset includes time dimension or if the
    model is a DDP model.

    Args:
        callbacks: TrainCallbackConfig object containing callback configuration.
        tmp_dir: Temporary directory for saving model checkpoints and plots.
        model: The trained model (can be DDP or non-DDP).
        best_model_dict: Dictionary containing the state dict of the best model observed
            during training.
        run_id: MLflow run ID for logging.
        datasets: List of datasets used for training (used for plotting when
            conditioners are available).
        include_time: Whether the dataset includes time dimension (used to disable
            plotting).
        compiled_model: Whether the model was compiled with torch.compile()
    """
    if callbacks is None:
        return

    if (
        callbacks.log_best_model.enabled
        and callbacks.log_best_model.frequency == "train_end"
    ):
        init_params_dict = logging.get_model_init_params_dict(model, compiled_model)
        logging.log_model_init_params_and_state_dict(
            init_params_dict, best_model_dict, tmp_dir, "weights-best.pth"
        )

    if (
        callbacks.plot_best_final_model.enabled
        and not include_time
        and not isinstance(model, DDP)
    ):
        if callbacks.log_best_model.enabled:
            model = logging.load_model(run_id, "weights-best.pth")
        _log_model_plot(
            datasets=datasets,
            model=model,
            model_img_path=os.path.join(tmp_dir, f"model-final.png"),
        )


def _train_temporal_unrolling(
    cfg: TrainConfig,
    run_id: str,
    tmp_dir: str,
    compile_model: bool = False,
):
    """Train model with temporal unrolling on a sigle device.

    Args:
        cfg: TrainConfig object containing training configuration.
        run_id: MLflow run ID for logging.
        tmp_dir: Temporary directory for saving model checkpoints and plots.
        compile_model: Whether to compile the model with torch.compile().
    """
    utils.set_random_seeds(cfg.random_seed)

    for i_stage, (stage_name, stage_cfg) in enumerate(
        cfg.temporal_unrolling_stages.items()
    ):
        print()
        print(f"Stage: {stage_name} (#{i_stage+1})")

        conditioners = cfg.data.conditioners
        datasets = _initialize_datasets(
            dataset_cls=cfg.dataset_cls,
            folders=cfg.data.folders,
            temporal_unroll_steps=stage_cfg.unrolling_steps,
            dataset_cls_kwargs=cfg.dataset_cls_kwargs,
            conditioners=conditioners,
        )
        train_datasets, valid_datasets = _do_train_valid_split(
            datasets, cfg.data.train_valid_ratio
        )
        train_dataset_size = np.sum([len(d) for d in train_datasets])
        valid_dataset_size = np.sum([len(d) for d in valid_datasets])
        print("Train Dataset Size:", int(train_dataset_size))
        print("Valid Dataset Size:", int(valid_dataset_size))

        train_dataloader = _initialze_dataloader(
            dataset=ConcatDataset(train_datasets),
            dataloader_cls=cfg.dataloader_cls,
            dataloader_cls_kwargs=cfg.dataloader_cls_kwargs,
            device=cfg.device,
        )
        # load valid data
        valid_dataloader: BaseDataLoader | list[list[torch.Tensor]] = []
        if valid_datasets:
            valid_dataloader = _initialze_dataloader(
                dataset=ConcatDataset(valid_datasets),
                dataloader_cls=cfg.dataloader_cls,
                dataloader_cls_kwargs=cfg.dataloader_cls_kwargs,
                device=cfg.device,
            )

        # Actions only done in first stage
        if i_stage == 0:
            # Initialize Model
            model, model_kwargs = _initialize_model(
                model_cls_str=cfg.model_cls,
                model_cls_kwargs=cfg.model_cls_kwargs,
                datasets=datasets,
                device=cfg.device,
                compile_model=compile_model,
            )
            print("Model Kwargs:", model_kwargs)
            mlflow.log_params({"model_kwargs": model_kwargs})
            print("Model:", model)

            if not cfg.dataset_cls_kwargs.get("include_time", False):
                _log_model_plot(
                    datasets=datasets,
                    model=model,
                    model_img_path=os.path.join(tmp_dir, f"model-start.png"),
                )
            # Initialize optimizer
            optimizer = _initialize_optimizer(
                optimizer_cls_str=cfg.optimizer_cls,
                optimizer_cls_kwargs=cfg.optimizer_cls_kwargs,
                lr=stage_cfg.lr,
                model=model,
            )
        # Actions done in other stages
        else:
            if stage_cfg.lr is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = stage_cfg.lr

        loss_fn = _generate_loss_fn(
            loss_name=cfg.loss.name,
            loss_mode=cfg.loss.mode,
            unrolling_steps=stage_cfg.unrolling_steps,
            y_extra_cells=datasets[0].extra_cells,
        )

        def train_step(model, optimizer, batch):
            loss_data = loss_fn(model, *batch)
            loss_reg = torch.tensor([0.0], device=model.device)
            if cfg.loss.reg_first_deriv > 0:
                loss_reg += cfg.loss.reg_first_deriv * model.get_first_deriv_norm()
            if cfg.loss.reg_second_deriv > 0:
                loss_reg += cfg.loss.reg_second_deriv * model.get_second_deriv_norm()
            loss = loss_data + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return model, optimizer, (loss, loss_data, loss_reg)

        def valid_step(model, batch):
            return loss_fn(model, *batch)

        # Start training
        min_train_loss = np.inf
        min_valid_loss = np.inf
        # Buffer to store best model parameters
        best_model_dict: dict[str, Any] = {}
        is_best_model = False

        if i_stage == 0:
            step = 0
            epoch = 0

        for _ in tqdm(range(stage_cfg.epochs), leave=True):
            # train epoch
            train_loss = 0
            train_loss_data = 0
            train_loss_reg = 0

            min_train_loss_flag = False
            min_valid_loss_flag = False

            for batch in tqdm(
                train_dataloader, leave=False, disable=len(train_dataloader) == 1
            ):
                batch = [b.to(cfg.device, non_blocking=True) for b in batch]
                model, optimizer, loss = train_step(model, optimizer, batch)
                # Split losses
                train_loss_step = loss[0].detach().cpu()
                train_loss_data_step = loss[1].detach().cpu()
                train_loss_reg_step = loss[2]
                # Accumulate for epoch
                train_loss += train_loss_step * len(batch[0]) / train_dataset_size
                train_loss_data += (
                    train_loss_data_step * len(batch[0]) / train_dataset_size
                )
                train_loss_reg += (
                    train_loss_reg_step * len(batch[0]) / train_dataset_size
                )
                # Log step loss
                mlflow.log_metrics(
                    {
                        "train_loss_step": train_loss_step,
                        "train_loss_data_step": train_loss_data_step,
                        "train_loss_reg_step": train_loss_reg_step,
                    },
                    step=step,
                    run_id=run_id,
                )
                step += 1

            # Validation epoch
            valid_loss = 0
            with torch.no_grad():
                for batch in valid_dataloader:
                    valid_loss += (
                        valid_step(model, batch) * len(batch[0]) / valid_dataset_size
                    )

            # Log epoch loss
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_loss_data": train_loss_data,
                    "train_loss_reg": train_loss_reg,
                    "valid_loss": valid_loss,
                },
                step=epoch,
            )

            # Check if we observed minimum loss values
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                mlflow.log_metric(f"min_train_loss-stage-{stage_name}", min_train_loss)
                min_train_loss_flag = True
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                mlflow.log_metric(f"min_valid_loss-stage-{stage_name}", min_valid_loss)
                min_valid_loss_flag = True
            if (min_valid_loss_flag and valid_dataset_size != 0) or (
                min_train_loss_flag and valid_dataset_size == 0
            ):
                is_best_model = True
                best_model_dict = logging.get_model_state_dict(
                    model,
                    compiled_model=compile_model,
                ).copy()
            else:
                is_best_model = False

            # Update epoch value
            epoch += 1

            # Do callbacks
            _do_epoch_callbacks(
                callbacks=cfg.callbacks,
                epoch=epoch,
                model=model,
                is_best_model=is_best_model,
                datasets=datasets,
                tmp_dir=tmp_dir,
                include_time=cfg.dataset_cls_kwargs.get("include_time", False),
                compiled_model=compile_model,
            )

        _do_stage_callbacks(
            callbacks=cfg.callbacks,
            stage_name=stage_name,
            tmp_dir=tmp_dir,
            model=model,
            best_model_dict=best_model_dict,
            run_id=run_id,
            datasets=datasets,
            include_time=cfg.dataset_cls_kwargs.get("include_time", False),
            compiled_model=compile_model,
        )

    _do_end_callbacks(
        callbacks=cfg.callbacks,
        model=model,
        tmp_dir=tmp_dir,
        best_model_dict=best_model_dict,
        run_id=run_id,
        datasets=datasets,
        include_time=cfg.dataset_cls_kwargs.get("include_time", False),
        compiled_model=compile_model,
    )


def _train_temporal_unrolling_ddp(
    cfg: TrainConfig,
    run_id: str,
    tmp_dir: str,
    rank: int,
    local_rank: int,
    world_size: int,
    compile_model: bool = False,
):
    """Train model with temporal unrolling using Distributed Data Parallel (DDP) mode.

    Each rank gets a portion of the dataset and trains the model in parallel.
    Only rank 0 logs metrics and model checkpoints to MLflow.

    Plotting callbacks are disabled in DDP mode.

    Args:
        cfg: TrainConfig object containing training configuration.
        run_id: MLflow run ID for logging.
        tmp_dir: Temporary directory for saving model checkpoints and plots.
        rank: Rank of the current process in DDP.
        local_rank: Local rank of the current process (GPU index).
        world_size: Total number of processes in DDP.
        compile_model: Whether to compile the model with torch.compile().
    """
    utils.set_random_seeds(cfg.random_seed)

    dist.barrier(
        device_ids=[
            local_rank,
        ]
    )

    utils.rank_print("Started")

    for i_stage, (stage_name, stage_cfg) in enumerate(
        cfg.temporal_unrolling_stages.items()
    ):
        utils.rank_print(f"Stage: {stage_name} (#{i_stage+1})")
        dist.barrier(
            device_ids=[
                local_rank,
            ]
        )

        # Each rank gets a portion of the dataset
        i_folders = np.linspace(0, len(cfg.data.folders), world_size + 1)
        i_folders = i_folders.astype(int)
        folders = cfg.data.folders[i_folders[rank] : i_folders[rank + 1]]
        utils.rank_print(f"Folder Range: [{i_folders[rank]}, {i_folders[rank+1]}]")
        conditioners = cfg.data.conditioners
        if conditioners is not None:
            conditioners = conditioners[i_folders[rank] : i_folders[rank + 1]]

        datasets = _initialize_datasets(
            dataset_cls=cfg.dataset_cls,
            folders=folders,
            temporal_unroll_steps=stage_cfg.unrolling_steps,
            dataset_cls_kwargs=cfg.dataset_cls_kwargs,
            conditioners=conditioners,
        )

        train_datasets, valid_datasets = _do_train_valid_split(
            datasets, cfg.data.train_valid_ratio
        )

        train_dataloader = _initialze_dataloader(
            dataset=ConcatDataset(train_datasets),
            dataloader_cls=cfg.dataloader_cls,
            dataloader_cls_kwargs=cfg.dataloader_cls_kwargs,
            device=local_rank,
        )

        train_dataset_size = np.sum([len(d) for d in train_datasets])
        total_train_dataset_size = torch.Tensor([train_dataset_size]).to(local_rank)
        dist.all_reduce(total_train_dataset_size, op=dist.ReduceOp.SUM)
        total_train_dataset_size = int(total_train_dataset_size.cpu().numpy()[0])
        utils.rank_print(
            f"Train Dataset Size: {train_dataset_size} (Global: {total_train_dataset_size})"
        )

        valid_dataloader: BaseDataLoader | list[list[torch.Tensor]] = []
        if valid_datasets:
            valid_dataloader = _initialze_dataloader(
                dataset=ConcatDataset(valid_datasets),
                dataloader_cls=cfg.dataloader_cls,
                dataloader_cls_kwargs=cfg.dataloader_cls_kwargs,
                device=local_rank,
            )
            valid_dataset_size = np.sum([len(d) for d in valid_datasets])
            total_valid_dataset_size = torch.tensor(
                [valid_dataset_size], device=local_rank
            )
            dist.all_reduce(total_valid_dataset_size, op=dist.ReduceOp.SUM)
            total_valid_dataset_size = int(total_valid_dataset_size.cpu().numpy()[0])
        else:
            valid_dataset_size = 0
            total_valid_dataset_size = 0
        utils.rank_print(
            f"Valid Dataset Size: {valid_dataset_size} (Global: {total_valid_dataset_size})"
        )

        # Actions only done in the first stage
        if i_stage == 0:
            # Initialize model
            model, model_kwargs = _initialize_model_ddp(
                model_cls_str=cfg.model_cls,
                model_cls_kwargs=cfg.model_cls_kwargs,
                datasets=datasets,
                local_rank=local_rank,
                compile_model=compile_model,
            )

            if rank == 0:
                print("Model Kwargs:", model_kwargs)
                mlflow.log_params({"model_kwargs": model_kwargs})
                print("Model:", model)

            # Initialize optimizer
            optimizer = _initialize_optimizer(
                optimizer_cls_str=cfg.optimizer_cls,
                optimizer_cls_kwargs=cfg.optimizer_cls_kwargs,
                lr=stage_cfg.lr,
                model=model,
            )
        # Actions done in other stages
        else:
            if stage_cfg.lr is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = stage_cfg.lr

        loss_fn = _generate_loss_fn(
            loss_name=cfg.loss.name,
            loss_mode=cfg.loss.mode,
            unrolling_steps=stage_cfg.unrolling_steps,
            y_extra_cells=datasets[0].extra_cells,
        )

        def train_step(model, optimizer, batch):
            loss = loss_fn(model, *batch)
            if cfg.loss.reg_first_deriv > 0 or cfg.loss.reg_second_deriv > 0:
                raise NotImplementedError("Regularization not implemented in DDP mode")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return model, optimizer, loss

        def valid_step(model, batch):
            return loss_fn(model, *batch)

        # Start training
        min_train_loss = np.inf
        min_valid_loss = np.inf
        # Buffer to store best model parameters
        best_model_dict: dict[str, Any] = {}
        is_best_model = False

        if i_stage == 0:
            step = 0
            epoch = 0

        dist.barrier()

        for _ in tqdm(range(stage_cfg.epochs), disable=rank != 0, leave=True):
            # Train epoch
            train_loss = 0

            min_train_loss_flag = False
            min_valid_loss_flag = False

            for batch in tqdm(
                train_dataloader,
                leave=False,
                disable=len(train_dataloader) == 1 or rank != 0,
            ):
                batch = [b.to(local_rank, non_blocking=True) for b in batch]
                model, optimizer, loss = train_step(model, optimizer, batch)

                train_loss_step = loss.clone().detach() * len(batch[0])
                total_batch_size = torch.Tensor([len(batch[0])]).to(local_rank)

                dist.all_reduce(train_loss_step, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_batch_size, op=dist.ReduceOp.SUM)

                if rank == 0:
                    # accumulate for epoch
                    train_loss += train_loss_step / total_train_dataset_size

                    # log step loss
                    mlflow.log_metric(
                        "train_loss_step",
                        train_loss_step / total_batch_size,
                        step=step,
                        run_id=run_id,
                    )
                # Update step
                step += 1

            # Valid epoch
            valid_loss = 0
            with torch.no_grad():
                for batch in valid_dataloader:
                    valid_loss += (
                        valid_step(model, batch)
                        * len(batch[0])
                        / total_valid_dataset_size
                    )
            dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)

            if rank == 0:
                # Log epoch loss
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                    },
                    step=epoch,
                )
                # Check if we observed minimum loss values
                if train_loss < min_train_loss:
                    min_train_loss = train_loss
                    mlflow.log_metric(
                        f"min_train_loss-stage-{stage_name}", min_train_loss
                    )
                    min_train_loss_flag = True
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    mlflow.log_metric(
                        f"min_valid_loss-stage-{stage_name}", min_valid_loss
                    )
                    min_valid_loss_flag = True
                if (min_valid_loss_flag and valid_dataset_size != 0) or (
                    min_train_loss_flag and valid_dataset_size == 0
                ):
                    is_best_model = True
                    best_model_dict = logging.get_model_state_dict(
                        model,
                        compiled_model=compile_model,
                    ).copy()
                else:
                    is_best_model = False

            # Update epoch value
            epoch += 1

            # Do callbacks
            if rank == 0:
                _do_epoch_callbacks(
                    callbacks=cfg.callbacks,
                    epoch=epoch,
                    tmp_dir=tmp_dir,
                    model=model.module,
                    is_best_model=is_best_model,
                    datasets=datasets,
                    include_time=cfg.dataset_cls_kwargs.get("include_time", False),
                    compiled_model=compile_model,
                )

        if rank == 0:
            _do_stage_callbacks(
                callbacks=cfg.callbacks,
                stage_name=stage_name,
                tmp_dir=tmp_dir,
                model=model,
                best_model_dict=best_model_dict,
                run_id=run_id,
                datasets=datasets,
                include_time=cfg.dataset_cls_kwargs.get("include_time", False),
                compiled_model=compile_model,
            )

    if rank == 0:
        _do_end_callbacks(
            callbacks=cfg.callbacks,
            tmp_dir=tmp_dir,
            model=model,
            best_model_dict=best_model_dict,
            run_id=run_id,
            datasets=datasets,
            include_time=cfg.dataset_cls_kwargs.get("include_time", False),
            compiled_model=compile_model,
        )


def train(
    cfg: TrainConfig,
    run_id: str,
    rank: int,
    local_rank: int,
    world_size: int,
    compile_model: bool,
):
    """Main training function.

    All code should be called from this function, which will select the
    appropriate training loop based on the configuration.

    Args:
        cfg: TrainConfig object containing all training configurations.
        run_id: MLflow run ID for logging.
        rank: Rank of the current process (for distributed training).
        local_rank: Local rank of the current process (for distributed training).
        world_size: Total number of processes (for distributed training).
        compile_model: Whether to compile the model with torch.compile.
    """

    if cfg.mode == "temporal_unrolling":
        with tempfile.TemporaryDirectory() as tmp_dir:
            if world_size == 1:
                _train_temporal_unrolling(cfg, run_id, tmp_dir, compile_model)
            else:
                _train_temporal_unrolling_ddp(
                    cfg,
                    run_id,
                    tmp_dir,
                    rank,
                    local_rank,
                    world_size,
                    compile_model,
                )
    else:
        raise ValueError(f"Invalid train mode selected: {cfg.mode}")

    if rank == 0:
        _plot_loss(run_id)
        # Loss with regularization is only enabled for non-DPP mode
        if (
            cfg.loss.reg_first_deriv > 0
            or cfg.loss.reg_second_deriv > 0
            and not utils.is_distributed()
        ):
            _plot_loss_with_regularization(run_id)
