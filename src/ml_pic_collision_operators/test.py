import os
import numpy as np
import mlflow
import tempfile
import matplotlib.pyplot as plt
import subprocess
import torch
import torch.nn as nn
import tqdm

from ml_pic_collision_operators.config.test import TestConfig
from ml_pic_collision_operators.logging_utils import (
    get_mlflow_run_id,
    load_model,
    load_model_from_AB_hdf,
)
from ml_pic_collision_operators.models import (
    FokkerPlanck2D_Base_Conditioned,
    FokkerPlanck2D_Tensor_Base_TimeDependent,
)
from ml_pic_collision_operators.datasets import BaseDataset, BasewConditionersDataset
from ml_pic_collision_operators.dataloaders import BaseDataLoader, BatchDatasetItem


def plot_comparison(
    f_true: np.ndarray,
    f_pred: np.ndarray,
    bin_range: list[float],
    bin_units: str,
    save_to: str | None = None,
):
    """Plot comparison between true and predicted distribution functions."""
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    f_max = np.max(np.abs(f_true))
    kwargs = {
        "extent": bin_range,
        "vmax": f_max,
        "vmin": -f_max,
        "origin": "lower",
        "cmap": "bwr",
    }

    ax[0].imshow(f_true.T, **kwargs)
    ax[1].imshow(f_pred.T, **kwargs)
    im = ax[2].imshow((f_true - f_pred).T, **kwargs)
    cbaxes = ax[2].inset_axes([1.05, 0, 0.05, 1])
    cbar = fig.colorbar(im, cax=cbaxes, orientation="vertical")
    ax[0].set_title("Target")
    ax[1].set_title("Predicted")
    ax[2].set_title("Difference")
    xlabel = f"$v_x{bin_units}$"
    ylabel = f"$v_y{bin_units}$"
    plt.setp(ax, xlabel=xlabel)
    ax[0].set_ylabel(ylabel)
    for a in ax[1:]:
        a.set_yticklabels([])
    if save_to is not None:
        plt.savefig(save_to, dpi=200)
    plt.show()
    plt.close()


def compute_all_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, metrics: list[str]
) -> dict[str, float]:
    """Compute all specified metrics between y_true and y_pred.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.
        metrics: List of metric names to compute.

    Returns:
        metric_values: Dictionary of computed metric values.
    """
    metric_values: dict[str, float] = {}
    l1 = None
    l2 = None
    if "mse" in metrics:
        mse = torch.mean(torch.square(y_pred - y_true))
        metric_values["mse"] = float(mse.numpy())
    if "l1" in metrics:
        l1 = torch.sum(torch.abs(y_pred - y_true))
        metric_values["l1"] = float(l1.numpy())
    if "l2" in metrics:
        l2 = torch.sqrt(torch.sum(torch.square(y_pred - y_true)))
        metric_values["l2"] = float(l2.numpy())
    if "l1_norm" in metrics:
        if l1 is None:
            l1 = torch.sum(torch.abs(y_pred - y_true))
        l1_norm = l1 / torch.sum(y_true)
        metric_values["l1_norm"] = float(l1_norm.numpy())
    if "l2_norm" in metrics:
        if l2 is None:
            l2 = torch.sqrt(torch.sum(torch.square(y_pred - y_true)))
        l2_norm = l2 / torch.linalg.norm(y_true)
        metric_values["l2_norm"] = float(l2_norm.numpy())
    return metric_values


def generate_video_from_frames(frame_dir: str, video_fname: str, fps: int):
    """Generate a video from a sequence of image frames using ffmpeg.

    Args:
        frame_dir: Directory containing the image frames named as 000001.png, 000002.png, etc.
        video_fname: Output video filename (e.g., output.mp4).
        fps: Frames per second for the output video.
    """
    command = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frame_dir, "%06d.png"),
        "-c:v",
        "libx264",
        "-r",
        str(fps),
        "-pix_fmt",
        "yuv420p",
        video_fname,
        "-y",
    ]

    print(" ".join(command))
    subprocess.run(command, check=True, capture_output=True)


def test_rollout(cfg: TestConfig, model: nn.Module, run_id: str, tmp_dir: str):
    """This function performs rollout testing of a model.

    It iterates over the test datasets specified in the configuration,
    performs rollouts, computes error metrics, and logs results to MLflow.
    Additionally, if video generation is enabled, it creates comparison videos
    of the model predictions versus ground truth.

    Args:
        cfg: Configuration for the test, including data and metrics.
        model: The trained model to be tested.
        run_id: MLflow run ID for logging.
        tmp_dir: Temporary directory for storing intermediate files.
    """

    if cfg.data.step_size >= 1:
        test_datasets = [
            BaseDataset(folder=folder, step_size=int(cfg.data.step_size))
            for folder in cfg.data.folders
        ]
        dt_undersample = 1
    else:
        test_datasets = [
            BaseDataset(folder=folder, step_size=1) for folder in cfg.data.folders
        ]
        dt_undersample = int(np.round(1 / cfg.data.step_size))

    metrics = [m.value for m in cfg.metrics]
    dataset_metrics: dict[str, list[float]] = {m: [] for m in metrics}

    # Loop over datasets
    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(
            dataset,
            batch_size=1,
        )

        # Load t = 0
        batch: BatchDatasetItem = next(iter(dataloader))
        y_true = batch.inputs
        y_pred = y_true.clone()
        if cfg.video:
            plot_comparison(
                y_true.numpy(),
                y_pred.numpy(),
                bin_range=dataset.grid_range,
                bin_units=dataset.grid_units,
                save_to=os.path.join(tmp_dir, f"{0:06d}.png"),
            )

        # Perform rollout
        all_steps_metrics: dict[str, list[float]] = {m: [] for m in metrics}
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataset)):
            y_true = batch.targets
            for _ in range(dt_undersample):
                y_pred = model(y_pred, batch.dt / dt_undersample)

            # Compute error metrics
            current_step_metrics = compute_all_metrics(y_true, y_pred, metrics)
            for m in metrics:
                all_steps_metrics[m].append(current_step_metrics[m])
            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in current_step_metrics.items()},
                step=i,
            )

            # Plot single frame comparison
            if cfg.video:
                plot_comparison(
                    y_true.numpy(),
                    y_pred.numpy(),
                    bin_range=dataset.grid_range,
                    bin_units=dataset.grid_units,
                    save_to=os.path.join(tmp_dir, f"{i+1:06d}.png"),
                )

        # Accumulate rollout metrics for the dataset
        rollout_metrics: dict[str, float] = {}
        for m in metrics:
            rollout_metrics[m] = float(np.mean(all_steps_metrics[m]))
            dataset_metrics[m].append(float(rollout_metrics[m]))

        mlflow.log_metrics(
            {f"{m}_rollout_{i_dataset}": v for m, v in rollout_metrics.items()}
        )

        # Generate rollout video from frames
        if cfg.video:
            video_fname = os.path.join(tmp_dir, f"rollout_{i_dataset}.mp4")
            generate_video_from_frames(
                frame_dir=tmp_dir, video_fname=video_fname, fps=cfg.video_fps
            )
            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)

    mlflow.log_metrics(
        {f"{m}_avg": float(np.mean(v)) for m, v in dataset_metrics.items()}
        | {f"{m}_std": float(np.std(v)) for m, v in dataset_metrics.items()}
    )


def test_rollout_conditioned(
    cfg: TestConfig, model: nn.Module, run_id: str, tmp_dir: str
):
    """This function performs rollout testing of a model with conditioners.

    Conditioners include any additional inputs to the model which are used
    alongside the current state, such as time, numerical parameters, etc.

    The function iterates over the test datasets specified in the configuration,
    performs rollouts, computes error metrics, and logs results to MLflow.
    Additionally, if video generation is enabled, it creates comparison videos
    of the model predictions versus ground truth.

    Args:
        cfg: Configuration for the test, including data and metrics.
        model: The trained model to be tested.
        run_id: MLflow run ID for logging.
        tmp_dir: Temporary directory for storing intermediate files.
    """
    if cfg.data.step_size >= 1:
        step_size = int(cfg.data.step_size)
        dt_undersample = 1
    else:
        step_size = 1
        dt_undersample = int(np.round(1 / cfg.data.step_size))

    if cfg.data.conditioners is None:
        test_datasets = [
            BasewConditionersDataset(
                folder=f,
                step_size=step_size,
                conditioners=None,
                include_time=cfg.data.include_time,
            )
            for f in cfg.data.folders
        ]
    else:
        test_datasets = [
            BasewConditionersDataset(
                folder=f,
                step_size=step_size,
                conditioners=c,
                include_time=cfg.data.include_time,
            )
            for f, c in zip(cfg.data.folders, cfg.data.conditioners)
        ]

    metrics = [m.value for m in cfg.metrics]
    dataset_metrics: dict[str, list[float]] = {m: [] for m in metrics}

    # Loop over datasets
    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(
            dataset,
            batch_size=1,
        )

        # Load t = 0
        batch: BatchDatasetItem = next(iter(dataloader))
        y_true = batch.inputs
        y_pred = y_true.clone()
        if cfg.video:
            plot_comparison(
                y_true.numpy(),
                y_pred.numpy(),
                bin_range=dataset.grid_range,
                bin_units=dataset.grid_units,
                save_to=os.path.join(tmp_dir, f"{0:06d}.png"),
            )

        all_steps_metrics: dict[str, list[float]] = {m: [] for m in metrics}
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataset)):
            y_true = batch.targets
            c = batch.conditioners
            for _ in range(dt_undersample):
                y_pred = model(y_pred, batch.dt / dt_undersample, c)
                if cfg.data.include_time:
                    c[0] += batch.dt / dt_undersample

            # Compute error metrics
            current_step_metrics = compute_all_metrics(y_true, y_pred, metrics)
            for m in metrics:
                all_steps_metrics[m].append(current_step_metrics[m])
            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in current_step_metrics.items()},
                step=i,
            )

            # Plot single frame comparison
            if cfg.video:
                plot_comparison(
                    y_true.numpy(),
                    y_pred.numpy(),
                    bin_range=dataset.grid_range,
                    bin_units=dataset.grid_units,
                    save_to=os.path.join(tmp_dir, f"{i+1:06d}.png"),
                )

        rollout_metrics: dict[str, float] = {}
        for m in metrics:
            rollout_metrics[m] = float(np.mean(all_steps_metrics[m]))
            dataset_metrics[m].append(rollout_metrics[m])

        mlflow.log_metrics(
            {f"{m}_rollout_{i_dataset}": v for m, v in rollout_metrics.items()}
        )

        # Generate rollout video from frames
        if cfg.video:
            video_fname = os.path.join(tmp_dir, f"rollout_{i_dataset}.mp4")
            generate_video_from_frames(
                frame_dir=tmp_dir, video_fname=video_fname, fps=cfg.video_fps
            )
            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)

    mlflow.log_metrics(
        {f"{m}_avg": float(np.mean(v)) for m, v in dataset_metrics.items()}
        | {f"{m}_std": float(np.std(v)) for m, v in dataset_metrics.items()}
    )


def test(cfg: TestConfig, run_id: str):

    if cfg.model.type == "mlflow":
        model_run_id = get_mlflow_run_id(
            experiment_name=cfg.model.experiment_name,
            run_name=cfg.model.run_name,
        )

        print("Pre-trained model run found.")
        print("experiment_name:", cfg.model.experiment_name)
        print("run_name:", cfg.model.run_name)
        print("run_id:", model_run_id)

        model = load_model(model_run_id, cfg.model.fname)

    elif cfg.model.type == "hdf":
        if cfg.model.params is None:
            model = load_model_from_AB_hdf(cfg.model.hdf_file)
        else:
            model = load_model_from_AB_hdf(cfg.model.hdf_file, **cfg.model.params)
        print("HDF model found.")
        print("hdf_file:", cfg.model.hdf_file)

    if cfg.model.change_params is not None:
        for key, value in cfg.model.change_params.items():
            model.change_attribute(key, value)

    model = model.eval()
    print("model:", model)

    with torch.no_grad():
        with tempfile.TemporaryDirectory() as tmp_dir:
            if cfg.mode == "rollout":
                if isinstance(model, FokkerPlanck2D_Base_Conditioned) or isinstance(
                    model, FokkerPlanck2D_Tensor_Base_TimeDependent
                ):
                    test_rollout_conditioned(cfg, model, run_id, tmp_dir)
                else:
                    model_img = os.path.join(tmp_dir, f"model-AB.png")
                    model.plot(model_img)
                    mlflow.log_artifact(model_img, artifact_path="model_img")
                    test_rollout(cfg, model, run_id, tmp_dir)
            else:
                raise NotImplementedError(f"Test mode {cfg.mode} not implemented.")
