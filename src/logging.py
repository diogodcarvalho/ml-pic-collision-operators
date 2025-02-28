import os
import h5py  # type: ignore[import-untyped]
import mlflow
import torch
import torch.nn as nn
import equinox as eqx
import numpy as np
from typing import Type, Any

from mlflow.tracking import MlflowClient

from src.models import FokkerPlanck2D
from src.utils import class_from_name


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


def log_equinox_model(model: eqx.Module, tmp_dir: str, fname: str = "weights.eqx"):

    weights_path = os.path.join(tmp_dir, fname)
    eqx.tree_serialise_leaves(weights_path, model)
    mlflow.log_artifact(weights_path, artifact_path="model")


def load_equinox_model(run_id: str, fname: str = "weights.eqx") -> eqx.Module:

    run_params = get_existing_run_params(run_id)
    if "model_cls" in run_params:
        model_cls = class_from_name("src.models", run_params["model_cls"])
    else:
        model_cls = FokkerPlanck2D

    model_kwargs = eval(get_existing_run_params(run_id)["model_kwargs"])

    weights_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"model/{fname}"
    )

    if weights_path is None:
        return None

    return eqx.tree_deserialise_leaves(weights_path, model_cls(**(model_kwargs)))


def log_torch_model(model: nn.Module, tmp_dir: str, fname: str = "weights.pth"):
    weights_path = os.path.join(tmp_dir, fname)
    torch.save(model.state_dict(), weights_path)
    mlflow.log_artifact(weights_path, artifact_path="model")


def log_torch_state_dict(
    model_state_dict: dict[str, Any], tmp_dir: str, fname: str = "weights.pth"
):
    weights_path = os.path.join(tmp_dir, fname)
    torch.save(model_state_dict, weights_path)
    mlflow.log_artifact(weights_path, artifact_path="model")


def load_torch_model(
    run_id: str, fname: str = "weights.pth", device: str = "cpu"
) -> eqx.Module:

    run_params = get_existing_run_params(run_id)
    model_cls = class_from_name("src.models", run_params["model_cls"])
    model_kwargs = eval(get_existing_run_params(run_id)["model_kwargs"])

    weights_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"model/{fname}"
    )
    if weights_path is None:
        return None

    model = model_cls(**model_kwargs)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    return model.to(device)


def load_AB_model(
    hdf_file: str,
    zero_A: bool = False,
    zero_B: bool = False,
    zero_B_cross: bool = False,
) -> eqx.Module | nn.Module:
    data_dict = {}
    with h5py.File(hdf_file, "r") as f:
        for key, item in f.items():
            data_dict[key] = item[:]

    model = FokkerPlanck2D(
        grid_size=data_dict["grid_size"],
        grid_dx=data_dict["grid_dx"],
        grid_range=data_dict["grid_range"],
        ensure_non_negative_f=True,
    )

    A = data_dict["A"].copy()
    B = data_dict["B"].copy()

    # Match FokkerPLanck2D normalizations
    A /= np.array(model.dx).reshape(2, 1, 1)
    B /= np.array([model.dx[0] ** 2, model.dx[1] ** 2, np.prod(model.dx)]).reshape(
        3, 1, 1
    )

    if zero_A:
        A = np.zeros_like(A)
    if zero_B:
        B = np.zeros_like(B)
    if zero_B_cross:
        B[2] = np.zeros_like(B[2])

    return model.load_from_numpy(A, B)
