import os
import numpy as np
import mlflow
import tempfile
import matplotlib.pyplot as plt
import subprocess
import torch
import torch.nn as nn
import tqdm

from ml_pic_collision_operators.config.test import (
    TestConfig,
    TestFunctionConfig,
    PlotSliceConfig,
    PlotTracksConfig,
)
from ml_pic_collision_operators.logging_utils import (
    get_mlflow_run_id,
    load_model,
    load_model_from_AD_hdf,
    load_model_from_AD_ParPerp_hdf,
)
from ml_pic_collision_operators.models import (
    FokkerPlanck2D_Base_Conditioned,
    FokkerPlanck2D_Tensor_Base_TimeDependent,
    FokkerPlanck2D_NN_Gridless_Base,
)
from ml_pic_collision_operators.datasets import (
    BaseDataset,
    BasewConditionersDataset,
    BaseTracksDataset,
)
from ml_pic_collision_operators.dataloaders import BaseDataLoader, BatchDatasetItem
from ml_pic_collision_operators.losses.test_functions import (
    TestFunction,
    ConcatTestFunctions,
)
from ml_pic_collision_operators.utils import class_from_str


def plot_hist_comparison(
    f_true: np.ndarray,
    f_pred: np.ndarray,
    bin_range: list[float],
    bin_units: str,
    save_to: str | None = None,
    plot_slice: PlotSliceConfig | None = None,
):
    """Plot comparison between true and predicted distribution functions."""
    _axis_names = ["x", "y", "z"]
    # In 2D we are plotting v_x and v_y by construction.
    x_axis_name, y_axis_name = "x", "y"
    # In 3D we only plot a slice.
    # Checks for ndim = 4 because the first dimension is the batch dimension.
    if f_true.ndim == 4:
        if plot_slice is None:
            raise ValueError("plot_slice values must be set for 3D fdist plots.")
        f_true = np.take(f_true, plot_slice.index, axis=plot_slice.axis + 1)
        f_pred = np.take(f_pred, plot_slice.index, axis=plot_slice.axis + 1)
        axes_2d = [i for i in range(3) if i != plot_slice.axis]
        bin_range = [v for i in axes_2d for v in bin_range[2 * i : 2 * i + 2]]
        x_axis_name, y_axis_name = _axis_names[axes_2d[0]], _axis_names[axes_2d[1]]
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
    fig.colorbar(im, cax=cbaxes, orientation="vertical")
    ax[0].set_title("Target")
    ax[1].set_title("Predicted")
    ax[2].set_title("Difference")
    xlabel = f"$v_{x_axis_name}{bin_units}$"
    ylabel = f"$v_{y_axis_name}{bin_units}$"
    plt.setp(ax, xlabel=xlabel)
    ax[0].set_ylabel(ylabel)
    for a in ax[1:]:
        a.set_yticklabels([])
    if save_to is not None:
        plt.savefig(save_to, dpi=200)
    plt.show()
    plt.close()


def plot_scatter_comparison(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    bin_range: tuple[float, ...],
    bin_units: str,
    save_to: str | None = None,
):
    """Scatter comparison of true vs predicted particle clouds (2D only)."""
    D = v_true.shape[-1]
    if D != 2:
        raise NotImplementedError(
            f"scatter plotting only implemented for 2D phase space; got {D}D"
        )
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    kwargs = {"s": 2, "alpha": 0.05}
    ax[0].scatter(v_true[:, 0], v_true[:, 1], color="blue", **kwargs)
    ax[1].scatter(v_pred[:, 0], v_pred[:, 1], color="red", **kwargs)
    ax[2].scatter(v_true[:, 0], v_true[:, 1], color="blue", **kwargs)
    ax[2].scatter(v_pred[:, 0], v_pred[:, 1], color="red", **kwargs)
    ax[0].set_title("Target")
    ax[1].set_title("Predicted")
    ax[2].set_title("Overlay")
    xmin, xmax, ymin, ymax = bin_range
    xlabel = f"$v_x{bin_units}$"
    ylabel = f"$v_y{bin_units}$"
    for a in ax:
        a.set_xlim(xmin, xmax)
        a.set_ylim(ymin, ymax)
        a.set_aspect("equal")
        a.set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    for a in ax[1:]:
        a.set_yticklabels([])
    if save_to is not None:
        plt.savefig(save_to, dpi=200)
    plt.show()
    plt.close()


def _compute_all_metrics(
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
        metric_values["mse"] = mse.item()
    if "l1" in metrics:
        l1 = torch.sum(torch.abs(y_pred - y_true))
        metric_values["l1"] = l1.item()
    if "l2" in metrics:
        l2 = torch.sqrt(torch.sum(torch.square(y_pred - y_true)))
        metric_values["l2"] = l2.item()
    if "l1_norm" in metrics:
        if l1 is None:
            l1 = torch.sum(torch.abs(y_pred - y_true))
        l1_norm = l1 / torch.sum(y_true)
        metric_values["l1_norm"] = l1_norm.item()
    if "l2_norm" in metrics:
        if l2 is None:
            l2 = torch.sqrt(torch.sum(torch.square(y_pred - y_true)))
        l2_norm = l2 / torch.linalg.norm(y_true)
        metric_values["l2_norm"] = l2_norm.item()
    return metric_values


def _generate_video_from_frames(frame_dir: str, video_fname: str, fps: int):
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


def _test_rollout(cfg: TestConfig, model: nn.Module, run_id: str, tmp_dir: str):
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

    test_datasets = [
        BaseDataset(folder=folder, step_size=cfg.data.step_size, mode="test")
        for folder in cfg.data.folders
    ]
    n_substeps = cfg.data.n_substeps

    metrics = [m.value for m in cfg.metrics]
    dataset_metrics: dict[str, list[float]] = {m: [] for m in metrics}

    # Loop over datasets
    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(
            dataset,
            batch_size=1,
        )

        frame_dir = os.path.join(tmp_dir, f"frames_{i_dataset}")
        if cfg.video:
            os.makedirs(frame_dir)

        # Load t = 0
        batch: BatchDatasetItem = next(iter(dataloader))
        y_true = batch.inputs
        y_pred = y_true.clone()
        if cfg.video:
            plot_hist_comparison(
                y_true.numpy(),
                y_pred.numpy(),
                bin_range=dataset.grid_range,
                bin_units=dataset.grid_units,
                save_to=os.path.join(frame_dir, "000000.png"),
                plot_slice=cfg.plot_slice,
            )

        # Perform rollout
        all_steps_metrics: dict[str, list[float]] = {m: [] for m in metrics}
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            y_true = batch.targets
            for _ in range(n_substeps):
                y_pred = model(y_pred, batch.dt / n_substeps)

            # Compute error metrics
            current_step_metrics = _compute_all_metrics(y_true, y_pred, metrics)
            for m in metrics:
                all_steps_metrics[m].append(current_step_metrics[m])
            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in current_step_metrics.items()},
                step=i,
            )

            # Plot single frame comparison
            if cfg.video:
                plot_hist_comparison(
                    y_true.numpy(),
                    y_pred.numpy(),
                    bin_range=dataset.grid_range,
                    bin_units=dataset.grid_units,
                    save_to=os.path.join(frame_dir, f"{i+1:06d}.png"),
                    plot_slice=cfg.plot_slice,
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
            _generate_video_from_frames(
                frame_dir=frame_dir, video_fname=video_fname, fps=cfg.video_fps
            )
            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)

    mlflow.log_metrics(
        {f"{m}_avg": float(np.mean(v)) for m, v in dataset_metrics.items()}
        | {f"{m}_std": float(np.std(v)) for m, v in dataset_metrics.items()}
    )


def _test_rollout_conditioned(
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
    step_size = cfg.data.step_size
    n_substeps = cfg.data.n_substeps

    if cfg.data.conditioners is None:
        test_datasets = [
            BasewConditionersDataset(
                folder=f,
                step_size=step_size,
                conditioners=None,
                include_time=cfg.data.include_time,
                mode="test",
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
                mode="test",
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

        frame_dir = os.path.join(tmp_dir, f"frames_{i_dataset}")
        if cfg.video:
            os.makedirs(frame_dir)

        # Load t = 0
        batch: BatchDatasetItem = next(iter(dataloader))
        y_true = batch.inputs
        y_pred = y_true.clone()
        if cfg.video:
            plot_hist_comparison(
                y_true.numpy(),
                y_pred.numpy(),
                bin_range=dataset.grid_range,
                bin_units=dataset.grid_units,
                save_to=os.path.join(frame_dir, "000000.png"),
                plot_slice=cfg.plot_slice,
            )

        all_steps_metrics: dict[str, list[float]] = {m: [] for m in metrics}
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            y_true = batch.targets
            c = batch.conditioners
            if c is None:
                raise RuntimeError(
                    "Unexpected empty conditioners array during test."
                    " This should not happen if conditioners were provided in the input file"
                    " for all data entries."
                )
            for _ in range(n_substeps):
                y_pred = model(y_pred, batch.dt / n_substeps, c)
                if cfg.data.include_time:
                    # time is always the last conditioner
                    c[:, -1] += batch.dt / n_substeps

            # Compute error metrics
            current_step_metrics = _compute_all_metrics(y_true, y_pred, metrics)
            for m in metrics:
                all_steps_metrics[m].append(current_step_metrics[m])
            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in current_step_metrics.items()},
                step=i,
            )

            # Plot single frame comparison
            if cfg.video:
                plot_hist_comparison(
                    y_true.numpy(),
                    y_pred.numpy(),
                    bin_range=dataset.grid_range,
                    bin_units=dataset.grid_units,
                    save_to=os.path.join(frame_dir, f"{i+1:06d}.png"),
                    plot_slice=cfg.plot_slice,
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
            _generate_video_from_frames(
                frame_dir=frame_dir, video_fname=video_fname, fps=cfg.video_fps
            )
            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)

    mlflow.log_metrics(
        {f"{m}_avg": float(np.mean(v)) for m, v in dataset_metrics.items()}
        | {f"{m}_std": float(np.std(v)) for m, v in dataset_metrics.items()}
    )


def _build_test_functions(specs: list[TestFunctionConfig]) -> TestFunction:
    """Instantiate test functions from config."""
    tf_list: list[TestFunction] = []
    for tf_spec in specs:
        tf_cls = class_from_str(tf_spec.cls_name, "ml_pic_collision_operators.losses")
        tf_list.append(tf_cls(**tf_spec.cls_kwargs))
    if len(tf_list) == 1:
        return tf_list[0]
    return ConcatTestFunctions(tf_list)


def _histogram_from_tracks(
    v: np.ndarray,
    bin_range: tuple[float, ...],
    grid_size: tuple[int, ...],
) -> np.ndarray:
    """Bin a single (N, D) particle velocity cloud into a D-dim distribution function.

    Counts are normalized by the particle count N so the result estimates f
    with sum(f) = 1. Particles outside bin_range contribute 0, so sum(f) drops
    below 1 by the fraction of mass that left the binned region.
    """
    N, D = v.shape
    if len(bin_range) != 2 * D:
        raise ValueError(
            f"bin_range must have 2 * D = {2 * D} entries; got {len(bin_range)}"
        )
    if len(grid_size) != D:
        raise ValueError(f"grid_size must have D = {D} entries; got {len(grid_size)}")
    range_d = [(bin_range[2 * i], bin_range[2 * i + 1]) for i in range(D)]
    h, _ = np.histogramdd(v, bins=tuple(grid_size), range=range_d)
    return h / N


def _plot_tracks_frame(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    h_true: np.ndarray,
    h_pred: np.ndarray,
    tracks_cfg: PlotTracksConfig,
    bin_units: str,
    plot_slice: PlotSliceConfig | None,
    frame_dir_hist: str,
    frame_dir_scatter: str,
    frame_idx: int,
):
    """Render hist and/or scatter frames according to the tracks plot mode."""
    fname = f"{frame_idx:06d}.png"
    if tracks_cfg.plot_mode in ("hist", "both"):
        plot_hist_comparison(
            h_true[np.newaxis],
            h_pred[np.newaxis],
            bin_range=list(tracks_cfg.grid_range),
            bin_units=bin_units,
            save_to=os.path.join(frame_dir_hist, fname),
            plot_slice=plot_slice,
        )
    if tracks_cfg.plot_mode in ("scatter", "both"):
        plot_scatter_comparison(
            v_true,
            v_pred,
            bin_range=tuple(tracks_cfg.grid_range),
            bin_units=bin_units,
            save_to=os.path.join(frame_dir_scatter, fname),
        )


def _test_rollout_tracks(
    cfg: TestConfig,
    model: FokkerPlanck2D_NN_Gridless_Base,
    run_id: str,
    tmp_dir: str,
):
    """Rollout testing for gridless FP NN models on particle-track datasets.

    Logs histogram-based pixel metrics on a user-specified grid and, when
    cfg.test_functions is provided, weak-form residuals (⟨φ_k⟩_pred - ⟨φ_k⟩_true)
    against that test-function family. Plotting supports histogrammed-distribution
    comparisons, particle scatter (2D only), or both.
    """
    if cfg.plot_tracks is None:
        raise ValueError(
            "cfg.plot_tracks must be set for particle-track testing "
            "(bin_range, grid_size, plot_mode)."
        )

    test_datasets = [
        BaseTracksDataset(folder=folder, step_size=cfg.data.step_size, mode="test")
        for folder in cfg.data.folders
    ]
    n_substeps = cfg.data.n_substeps

    # Optional weak-form residual metrics against config-provided test functions.
    if cfg.test_functions is not None:
        test_function = _build_test_functions(cfg.test_functions)
        n_phi = test_function.n_functions
    else:
        test_function = None
        n_phi = 0

    metrics = [m.value for m in cfg.metrics]
    dataset_metrics: dict[str, list[float]] = {m: [] for m in metrics}

    bin_range = tuple(cfg.plot_tracks.grid_range)
    grid_size = tuple(cfg.plot_tracks.grid_size)
    plot_mode = cfg.plot_tracks.plot_mode

    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(dataset, batch_size=1)

        frame_dir_hist = os.path.join(tmp_dir, f"frames_hist_{i_dataset}")
        frame_dir_scatter = os.path.join(tmp_dir, f"frames_scatter_{i_dataset}")
        if cfg.video and plot_mode in ("hist", "both"):
            os.makedirs(frame_dir_hist)
        if cfg.video and plot_mode in ("scatter", "both"):
            os.makedirs(frame_dir_scatter)

        # Load t = 0
        batch: BatchDatasetItem = next(iter(dataloader))
        v_pred_t = batch.inputs.clone()  # (1, N, D)
        if cfg.video:
            v_t_np = v_pred_t.squeeze(0).numpy()  # (N, D)
            h_t = _histogram_from_tracks(v_t_np, bin_range, grid_size)
            _plot_tracks_frame(
                v_t_np,
                v_t_np,
                h_t,
                h_t,
                cfg.plot_tracks,
                dataset.v_units,
                cfg.plot_slice,
                frame_dir_hist,
                frame_dir_scatter,
                frame_idx=0,
            )

        all_steps_metrics: dict[str, list[float]] = {m: [] for m in metrics}
        all_steps_phi: list[list[float]] = [[] for _ in range(n_phi)]

        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            v_true_t = batch.targets  # (1, N, D), model also returns (B, N, D)
            for _ in range(n_substeps):
                v_pred_t = model(v_pred_t, batch.dt / n_substeps)

            # Weak-form residuals: (⟨φ_k⟩_pred − ⟨φ_k⟩_true), shape (n_phi,).
            phi_log: dict[str, float] = {}
            if test_function is not None:
                phi_pred = test_function.evaluate_phi(v_pred_t)
                phi_true = test_function.evaluate_phi(v_true_t)
                phi_res = (phi_pred.mean(dim=1) - phi_true.mean(dim=1)).squeeze(0)
                for k in range(n_phi):
                    res = float(phi_res[k].item())
                    all_steps_phi[k].append(res)
                    phi_log[f"phi_{k}_step_{i_dataset}"] = res

            # Histogram-based metrics.
            v_true_np = v_true_t.squeeze(0).numpy()  # (N, D)
            v_pred_np = v_pred_t.squeeze(0).numpy()  # (N, D)
            h_true = _histogram_from_tracks(v_true_np, bin_range, grid_size)
            h_pred = _histogram_from_tracks(v_pred_np, bin_range, grid_size)
            current_step_metrics = _compute_all_metrics(
                torch.from_numpy(h_true).to(torch.get_default_dtype()),
                torch.from_numpy(h_pred).to(torch.get_default_dtype()),
                metrics,
            )
            for m in metrics:
                all_steps_metrics[m].append(current_step_metrics[m])
            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in current_step_metrics.items()}
                | phi_log,
                step=i,
            )

            if cfg.video:
                _plot_tracks_frame(
                    v_true_np,
                    v_pred_np,
                    h_true,
                    h_pred,
                    cfg.plot_tracks,
                    dataset.v_units,
                    cfg.plot_slice,
                    frame_dir_hist,
                    frame_dir_scatter,
                    frame_idx=i + 1,
                )

        rollout_metrics: dict[str, float] = {}
        for m in metrics:
            rollout_metrics[m] = float(np.mean(all_steps_metrics[m]))
            dataset_metrics[m].append(rollout_metrics[m])
        for k in range(n_phi):
            rollout_metrics[f"phi_{k}_rms"] = float(
                np.sqrt(np.mean(np.square(all_steps_phi[k])))
            )

        mlflow.log_metrics(
            {f"{m}_rollout_{i_dataset}": v for m, v in rollout_metrics.items()}
        )

        if cfg.video and plot_mode in ("hist", "both"):
            video_fname = os.path.join(tmp_dir, f"rollout_hist_{i_dataset}.mp4")
            _generate_video_from_frames(
                frame_dir=frame_dir_hist, video_fname=video_fname, fps=cfg.video_fps
            )
            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)
        if cfg.video and plot_mode in ("scatter", "both"):
            video_fname = os.path.join(tmp_dir, f"rollout_scatter_{i_dataset}.mp4")
            _generate_video_from_frames(
                frame_dir=frame_dir_scatter, video_fname=video_fname, fps=cfg.video_fps
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
            model = load_model_from_AD_hdf(cfg.model.hdf_file)
        else:
            model = load_model_from_AD_hdf(cfg.model.hdf_file, **cfg.model.params)
        print("HDF model found.")
        print("hdf_file:", cfg.model.hdf_file)

    elif cfg.model.type == "hdf_parperp":
        if cfg.model.params is None:
            model = load_model_from_AD_ParPerp_hdf(cfg.model.hdf_file)
        else:
            model = load_model_from_AD_ParPerp_hdf(
                cfg.model.hdf_file, **cfg.model.params
            )
        print("HDF_ParPerp model found.")
        print("hdf_file:", cfg.model.hdf_file)

    else:
        raise ValueError("Invalid model type. Available options are 'mlflow' or 'hdf'")

    if cfg.model.change_params is not None:
        # Verify the method exists and is callable
        if hasattr(model, "change_attribute"):
            change_attr = getattr(model, "change_attribute")
            if callable(change_attr):
                for key, value in cfg.model.change_params.items():
                    change_attr(key, value)
            else:
                raise ValueError(
                    f"{type(model).__name__}.change_attribute is not callable."
                )
        else:
            raise ValueError(
                f"{type(model).__name__} does not support change_attribute."
            )

    model = model.eval()
    print("model:", model)

    with torch.no_grad():
        with tempfile.TemporaryDirectory() as tmp_dir:
            if cfg.mode == "rollout":
                if isinstance(model, FokkerPlanck2D_Base_Conditioned) or isinstance(
                    model, FokkerPlanck2D_Tensor_Base_TimeDependent
                ):
                    _test_rollout_conditioned(cfg, model, run_id, tmp_dir)
                elif isinstance(model, FokkerPlanck2D_NN_Gridless_Base):
                    model_img = os.path.join(tmp_dir, "model.png")
                    model.plot(model_img, show=False)
                    mlflow.log_artifact(model_img, artifact_path="model_img")
                    _test_rollout_tracks(cfg, model, run_id, tmp_dir)
                else:
                    model_img = os.path.join(tmp_dir, "model.png")
                    model.plot(model_img)
                    mlflow.log_artifact(model_img, artifact_path="model_img")
                    _test_rollout(cfg, model, run_id, tmp_dir)
            else:
                raise NotImplementedError(f"Test mode {cfg.mode} not implemented.")
