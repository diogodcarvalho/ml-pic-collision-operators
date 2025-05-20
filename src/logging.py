import os
import h5py  # type: ignore[import-untyped]
import mlflow
import torch
import torch.nn as nn
import numpy as np
from typing import Any
from scipy import ndimage

from mlflow.tracking import MlflowClient

from src.models import FokkerPlanck2D
from src.utils import class_from_name


def get_experiment_runs(experiment_name, client=MlflowClient()):
    experiment_id = client.get_experiment_by_name(experiment_name)
    if experiment_id is None:
        raise Exception(f"Experiment does not exist: {experiment_name}")
    else:
        print(f"Experiment found: {experiment_name}")
    runs = client.search_runs(
        experiment_ids=experiment_id.experiment_id,
        filter_string="",  # No filter, get all runs
        run_view_type=1,  # 1 = Active, 2 = Deleted, 3 = All
    )
    return runs


def get_existing_run_id(experiment_name: str, run_name: str) -> str:

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise Exception(f"Experiment does not exist: experiment_name={experiment_name}")

    existing_runs = mlflow.search_runs(
        experiment_ids=experiment.experiment_id, filter_string=f"run_name='{run_name}'"
    )

    if existing_runs.empty:
        raise KeyError(
            f"Experiment run not found: experiment_name='{experiment_name}'  run_name='{run_name}'"
        )
    elif len(existing_runs) == 1:
        run_id = existing_runs["run_id"][0]
    else:
        raise Exception(f"Multiple runs detected with the same name: {run_name}")

    return run_id


def get_existing_run_params(run_id: str) -> dict:
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def get_metric_history(metric_name, run_id):
    client = mlflow.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    steps = np.array([m.step for m in metric_history])
    values = np.array([m.value for m in metric_history])
    i_start = np.argwhere(steps == 0)[-1, 0]
    return steps[i_start:], values[i_start:]


def log_torch_model(model: nn.Module, tmp_dir: str, fname: str = "weights.pth"):
    checkpoint_path = os.path.join(tmp_dir, fname)
    checkpoint = {
        "state_dict": model.state_dict(),
        "init_params": model.init_params_dict,
    }
    torch.save(checkpoint, checkpoint_path)
    mlflow.log_artifact(checkpoint_path, artifact_path="model")


def log_torch_state_dict(
    model_init_params: dict[str, Any],
    model_state_dict: dict[str, Any],
    tmp_dir: str,
    fname: str = "weights.pth",
):
    checkpoint_path = os.path.join(tmp_dir, fname)
    checkpoint = {
        "state_dict": model_state_dict,
        "init_params": model_init_params,
    }
    torch.save(checkpoint, checkpoint_path)
    mlflow.log_artifact(checkpoint_path, artifact_path="model")


def load_torch_model(
    run_id: str, fname: str = "weights.pth", device: str = "cpu"
) -> nn.Module:

    run_params = get_existing_run_params(run_id)
    model_cls = class_from_name("src.models", run_params["model_cls"])
    checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"model/{fname}"
    )
    if checkpoint_path is None:
        return None

    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=None if torch.cuda.is_available() else "cpu",
    )
    model = model_cls(**checkpoint["init_params"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def load_AB_model(
    hdf_file: str,
    zero_A: bool = False,
    zero_B: bool = False,
    zero_B_cross: bool = False,
    zero_v_larger_than: float = 0.0,
    smooth_v_larger_than: float = 0.0,
    gaussian_filter_sigma: float = 0.0,
    median_filter_size: float = 0.0,
    ensure_non_negative_f: bool = True,
    ensure_non_negative_B: bool = False,
) -> nn.Module:
    data_dict = {}
    with h5py.File(hdf_file, "r") as f:
        for key, item in f.items():
            data_dict[key] = item[()]

    grid_size = data_dict["grid_size"]
    grid_dx = data_dict["grid_dx"]
    grid_range = data_dict["grid_range"]
    grid_units = data_dict["grid_range_units"].decode("ascii")
    v_th = data_dict["v_th"]

    A = data_dict["A"].copy()
    B = data_dict["B"].copy()

    # Match FokkerPLanck2D normalizations
    # divide by dx
    A /= np.array(grid_dx).reshape(2, 1, 1)
    B /= np.array([grid_dx[0] ** 2, grid_dx[1] ** 2, np.prod(grid_dx)]).reshape(3, 1, 1)

    if zero_A:
        A = np.zeros_like(A)
    if zero_B:
        B = np.zeros_like(B)
    if zero_B_cross:
        B[2] = np.zeros_like(B[2])

    # normalize grid range to vth (to match trained models)
    if grid_units == "[c]":
        grid_range = (np.array(grid_range) / v_th).tolist()
        grid_dx = (np.array(grid_dx) / v_th).tolist()
        grid_units = "[v_{{th}}]"
    elif data_dict["grid_range_units"] != "[v_th]":
        raise Exception(f"AB model was saved with non-accepted units: {grid_units}")

    if zero_v_larger_than > 0.0:
        vx = np.linspace(*grid_range[:2], grid_size[0], endpoint=False)
        vy = np.linspace(*grid_range[2:], grid_size[1], endpoint=False)
        vx += grid_dx[0] / 2
        vy += grid_dx[0] / 2
        VX, VY = np.meshgrid(vx, vy, indexing="ij")
        v_norm = np.sqrt(VX**2 + VY**2)
        mask = v_norm > zero_v_larger_than
        A[:, mask] = 0.0
        B[:, mask] = 0.0

    if smooth_v_larger_than > 0.0:
        vx = np.linspace(*grid_range[:2], grid_size[0], endpoint=False)
        vy = np.linspace(*grid_range[2:], grid_size[1], endpoint=False)
        vx += grid_dx[0] / 2
        vy += grid_dx[0] / 2
        VX, VY = np.meshgrid(vx, vy, indexing="ij")
        v_norm = np.sqrt(VX**2 + VY**2)
        mask = v_norm > smooth_v_larger_than

        A_smooth = A.copy()
        B_smooth = B.copy()

        if gaussian_filter_sigma > 0.0:
            A_smooth = ndimage.gaussian_filter(
                A, sigma=gaussian_filter_sigma, axes=(1, 2)
            )
            B_smooth = ndimage.gaussian_filter(
                B, sigma=gaussian_filter_sigma, axes=(1, 2)
            )
        elif median_filter_size > 0.0:
            A_smooth = ndimage.median_filter(A, size=median_filter_size, axes=(1, 2))
            B_smooth = ndimage.median_filter(B, size=median_filter_size, axes=(1, 2))

        A[:, mask] = A_smooth[:, mask]
        B[:, mask] = B_smooth[:, mask]

    model = FokkerPlanck2D(
        grid_size=grid_size,
        grid_dx=grid_dx,
        grid_range=grid_range,
        grid_units=grid_units,
        ensure_non_negative_f=ensure_non_negative_f,
        ensure_non_negative_B=ensure_non_negative_B,
    )

    return model.load_from_numpy(A, B)
