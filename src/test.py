import os
import jax
import jax.numpy as jnp
import numpy as np
import mlflow
import tempfile
import equinox as eqx
import matplotlib.pyplot as plt
import subprocess

from tqdm import tqdm

from src.logging import get_existing_run_id, load_equinox_model, load_AB_model
from src.models import *
from src.datasets import *
from src.dataloaders import *


def plot_comparison(f_true, f_pred, bin_range, save_to=None):
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
    xlabel = "$v1[c]$"
    ylabel = "$v2[c]$"
    plt.setp(ax, xlabel=xlabel)
    ax[0].set_ylabel(ylabel)
    for a in ax[1:]:
        a.set_yticklabels([])
    if save_to is not None:
        plt.savefig(save_to, dpi=200)
    plt.show()
    plt.close()


def test_single_step(cfg, model, run_id):
    raise NotImplementedError


def test_rollout(cfg, model, run_id):

    # load train data
    test_datasets = [
        BaseDataset(folder=folder, step_size=cfg["dataset"]["step_size"])
        for folder in cfg["dataset"]["test"]["folders"]
    ]

    # loop over datasets
    for i_dataset, dataset in enumerate(test_datasets):

        dataloader = BaseDataLoader(
            dataset,
            batch_size=1,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:

            # load t = 0
            y_true, _ = next(iter(dataloader))
            y_pred = y_true.copy()
            plot_comparison(
                y_true,
                y_pred,
                bin_range=dataset.grid_range,
                save_to=os.path.join(tmp_dir, f"{0:06d}.png"),
            )

            # do rollout
            rollout_mse = []
            for i, (_, y_true) in tqdm(enumerate(dataloader), total=len(dataset)):
                y_pred = eqx.filter_jit(model)(y_pred)
                # pass to numpy for plots and error metrics
                y_pred_np = np.array(y_pred)
                # error metrics
                step_mse = np.mean(np.square(y_pred_np - y_true))
                rollout_mse.append(step_mse)
                # log step mse
                mlflow.log_metric(f"mse_rollout_{i_dataset}_step", step_mse, step=i)

                if cfg["video"]:
                    # plot frame comparison
                    plot_comparison(
                        y_true,
                        y_pred,
                        bin_range=dataset.grid_range,
                        save_to=os.path.join(tmp_dir, f"{i+1:06d}.png"),
                    )

            # log average rollout mse
            mlflow.log_metric(f"mse_rollout_{i_dataset}", np.mean(rollout_mse))

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


def test(cfg, run_id):

    mlflow.log_params(cfg)

    if cfg["model"]["type"] == "mlrun":
        model_run_id = get_existing_run_id(
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

        model = load_equinox_model(model_run_id, FokkerPlanck2D, cfg["model"]["fname"])

    elif cfg["model"]["type"] == "AB":
        model = load_AB_model(cfg["model"]["hdf_file"])
        print("AB model found.")
        print("location:", cfg["model"]["hdf_file"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_img = os.path.join(tmp_dir, f"model-AB.png")
            model.plot(model_img)
            mlflow.log_artifact(model_img, artifact_path="model_img")

    print("model:", model)

    if cfg["mode"] == "rollout":
        test_rollout(cfg, model, run_id)

    if cfg["mode"] == "single_step":
        test_single_step(cfg, model, run_id)
