import os
import numpy as np
import mlflow
import tempfile
import matplotlib.pyplot as plt
import subprocess
import torch

from tqdm import tqdm

from ml_pic_collision_operators.logging import (
    get_mlflow_run_id,
    load_model,
    load_model_from_AB_hdf,
)
from ml_pic_collision_operators.models import *
from ml_pic_collision_operators.datasets import BaseDataset, BasewConditionersDataset
from ml_pic_collision_operators.dataloaders import BaseDataLoader


def plot_comparison(f_true, f_pred, bin_range, bin_units, save_to=None):
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


def test_single_step(cfg, model, run_id, tmp_dir):
    raise NotImplementedError


def test_rollout(cfg, model, run_id, tmp_dir):

    model_img = os.path.join(tmp_dir, f"model.png")
    model.plot(model_img)
    mlflow.log_artifact(model_img, artifact_path="model_img")

    if cfg["data"]["step_size"] >= 1:
        # load train data
        test_datasets = [
            BaseDataset(folder=folder, step_size=cfg["data"]["step_size"])
            for folder in cfg["data"]["test"]["folders"]
        ]
        dt_undersample = 1
    else:
        # load train data
        test_datasets = [
            BaseDataset(folder=folder, step_size=1)
            for folder in cfg["data"]["test"]["folders"]
        ]
        dt_undersample = int(np.round(1 / cfg["data"]["step_size"]))

    metrics = ["mse", "l1", "l2", "l1_norm", "l2_norm"]
    dataset_metrics = {m: [] for m in metrics}

    # loop over datasets
    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(
            dataset,
            batch_size=1,
        )

        # load t = 0
        y_true, _, dt = next(iter(dataloader))
        y_pred = y_true.clone()
        plot_comparison(
            y_true.numpy(),
            y_pred.numpy(),
            bin_range=dataset.grid_range,
            bin_units=dataset.grid_units,
            save_to=os.path.join(tmp_dir, f"{0:06d}.png"),
        )

        # do rollout
        rollout_metrics = {m: [] for m in metrics}
        step_metrics = {m: None for m in metrics}
        for i, (_, y_true, _) in tqdm(enumerate(dataloader), total=len(dataset)):
            for _ in range(dt_undersample):
                y_pred = model(y_pred, dt / dt_undersample)
            # error metrics
            step_mse = torch.mean(torch.square(y_pred - y_true))
            step_l1 = torch.sum(torch.abs(y_pred - y_true))
            step_l2 = torch.sqrt(torch.sum(torch.square(y_pred - y_true)))
            step_l1_norm = step_l1 / torch.sum(y_true)
            step_l2_norm = step_l2 / torch.linalg.norm(y_true)

            step_metrics["mse"] = step_mse.numpy()
            step_metrics["l1"] = step_l1.numpy()
            step_metrics["l2"] = step_l2.numpy()
            step_metrics["l1_norm"] = step_l1_norm.numpy()
            step_metrics["l2_norm"] = step_l2_norm.numpy()

            for m in metrics:
                rollout_metrics[m].append(step_metrics[m])

            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in step_metrics.items()}, step=i
            )

            if cfg["video"]:
                # plot frame comparison
                plot_comparison(
                    y_true.numpy(),
                    y_pred.numpy(),
                    bin_range=dataset.grid_range,
                    bin_units=dataset.grid_units,
                    save_to=os.path.join(tmp_dir, f"{i+1:06d}.png"),
                )

        for m in metrics:
            rollout_metrics[m] = np.mean(rollout_metrics[m])
            dataset_metrics[m].append(rollout_metrics[m])

        mlflow.log_metrics(
            {f"{m}_rollout_{i_dataset}": v for m, v in rollout_metrics.items()}
        )

        if cfg["video"]:
            # generate video
            video_fname = os.path.join(tmp_dir, f"rollout_{i_dataset}.mp4")
            command = [
                "ffmpeg",
                "-framerate",
                str(cfg["fps"]),
                "-i",
                os.path.join(tmp_dir, "%06d.png"),
                "-c:v",
                "libx264",
                "-r",
                str(cfg["fps"]),
                "-pix_fmt",
                "yuv420p",
                video_fname,
                "-y",
            ]

            print(" ".join(command))
            subprocess.run(command, check=True, capture_output=True)

            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)

    mlflow.log_metrics(
        {f"{m}_avg": np.mean(v) for m, v in dataset_metrics.items()}
        | {f"{m}_std": np.std(v) for m, v in dataset_metrics.items()}
    )


def test_rollout_conditioned(cfg, model, run_id, tmp_dir):

    # load train data
    if cfg["data"]["step_size"] >= 1:
        step_size = cfg["data"]["step_size"]
        dt_undersample = 1
    else:
        step_size = 1
        dt_undersample = int(np.round(1 / cfg["data"]["step_size"]))

    if "conditioners" in cfg["data"]["test"]:
        test_datasets = [
            BasewConditionersDataset(
                folder=f,
                step_size=step_size,
                conditioners=c,
                include_time=cfg["data"].get("include_time", False),
            )
            for f, c in zip(
                cfg["data"]["test"]["folders"], cfg["data"]["test"]["conditioners"]
            )
        ]
    else:
        test_datasets = [
            BasewConditionersDataset(
                folder=f,
                step_size=step_size,
                conditioners=None,
                include_time=cfg["data"].get("include_time", False),
            )
            for f in cfg["data"]["test"]["folders"]
        ]
    metrics = ["mse", "l1", "l2", "l1_norm", "l2_norm"]
    dataset_metrics = {m: [] for m in metrics}

    # loop over datasets
    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(
            dataset,
            batch_size=1,
        )

        # load t = 0
        y_true, _, dt, c = next(iter(dataloader))
        y_pred = y_true.clone()
        plot_comparison(
            y_true.numpy(),
            y_pred.numpy(),
            bin_range=dataset.grid_range,
            bin_units=dataset.grid_units,
            save_to=os.path.join(tmp_dir, f"{0:06d}.png"),
        )

        # do rollout
        rollout_metrics = {m: [] for m in metrics}
        step_metrics = {m: None for m in metrics}
        for i, (_, y_true, _, c) in tqdm(enumerate(dataloader), total=len(dataset)):
            for _ in range(dt_undersample):
                y_pred = model(y_pred, dt / dt_undersample, c)
                if cfg["data"].get("include_time", False):
                    c[0] += dt / dt_undersample
            # error metrics
            step_mse = torch.mean(torch.square(y_pred - y_true))
            step_l1 = torch.sum(torch.abs(y_pred - y_true))
            step_l2 = torch.sqrt(torch.sum(torch.square(y_pred - y_true)))
            step_l1_norm = step_l1 / torch.sum(y_true)
            step_l2_norm = step_l2 / torch.linalg.norm(y_true)

            step_metrics["mse"] = step_mse.numpy()
            step_metrics["l1"] = step_l1.numpy()
            step_metrics["l2"] = step_l2.numpy()
            step_metrics["l1_norm"] = step_l1_norm.numpy()
            step_metrics["l2_norm"] = step_l2_norm.numpy()

            for m in metrics:
                rollout_metrics[m].append(step_metrics[m])

            mlflow.log_metrics(
                {f"{m}_step_{i_dataset}": v for m, v in step_metrics.items()}, step=i
            )

            if cfg["video"]:
                # plot frame comparison
                plot_comparison(
                    y_true.numpy(),
                    y_pred.numpy(),
                    bin_range=dataset.grid_range,
                    bin_units=dataset.grid_units,
                    save_to=os.path.join(tmp_dir, f"{i+1:06d}.png"),
                )

        for m in metrics:
            rollout_metrics[m] = np.mean(rollout_metrics[m])
            dataset_metrics[m].append(rollout_metrics[m])

        mlflow.log_metrics(
            {f"{m}_rollout_{i_dataset}": v for m, v in rollout_metrics.items()}
        )

        if cfg["video"]:
            # generate video
            video_fname = os.path.join(tmp_dir, f"rollout_{i_dataset}.mp4")
            command = [
                "ffmpeg",
                "-framerate",
                str(cfg["fps"]),
                "-i",
                os.path.join(tmp_dir, "%06d.png"),
                "-c:v",
                "libx264",
                "-r",
                str(cfg["fps"]),
                "-pix_fmt",
                "yuv420p",
                video_fname,
                "-y",
            ]

            print(" ".join(command))
            subprocess.run(command, check=True, capture_output=True)

            mlflow.log_artifact(video_fname, "rollout_videos", run_id=run_id)

    mlflow.log_metrics(
        {f"{m}_avg": np.mean(v) for m, v in dataset_metrics.items()}
        | {f"{m}_std": np.std(v) for m, v in dataset_metrics.items()}
    )


def test(cfg, run_id):

    mlflow.log_params(cfg)

    if cfg["model"]["type"] == "mlrun":
        model_run_id = get_mlflow_run_id(
            experiment_name=cfg["model"]["experiment_name"],
            run_name=cfg["model"]["run_name"],
        )

        if model_run_id is None:
            raise FileNotFoundError(
                "Pre-trained model run not found. Check provided experiment_name and run_name values."
            )
        else:
            print("Pre-trained model run found.")
            print("experiment_name:", cfg["model"]["experiment_name"])
            print("run_name:", cfg["model"]["run_name"])
            print("run_id:", model_run_id)

        model = load_model(model_run_id, cfg["model"]["fname"])

    elif cfg["model"]["type"] == "AB":
        if "params" in cfg["model"]:
            model = load_model_from_AB_hdf(
                cfg["model"]["hdf_file"], **cfg["model"]["params"]
            )
        else:
            model = load_model_from_AB_hdf(cfg["model"]["hdf_file"])
        print("AB model found.")
        print("location:", cfg["model"]["hdf_file"])
        if not isinstance(model, FokkerPlanck2DBaseTime):
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_img = os.path.join(tmp_dir, f"model-AB.png")
                model.plot(model_img)
                mlflow.log_artifact(model_img, artifact_path="model_img")

    if "change_params" in cfg["model"]:
        for key, value in cfg["model"]["change_params"].items():
            model.change_attribute(key, value)

    model = model.eval()
    print("model:", model)

    with torch.no_grad():
        with tempfile.TemporaryDirectory() as tmp_dir:
            if cfg["mode"] == "rollout":
                if issubclass(type(model), FokkerPlanck2DBaseConditioned) or issubclass(
                    type(model), FokkerPlanck2DBaseTime
                ):
                    test_rollout_conditioned(cfg, model, run_id, tmp_dir)
                else:
                    model_img = os.path.join(tmp_dir, f"model-AB.png")
                    model.plot(model_img)
                    mlflow.log_artifact(model_img, artifact_path="model_img")
                    test_rollout(cfg, model, run_id, tmp_dir)

            if cfg["mode"] == "single_step":
                test_single_step(cfg, model, run_id, tmp_dir)
