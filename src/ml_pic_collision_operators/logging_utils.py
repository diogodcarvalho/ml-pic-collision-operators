import os
import h5py  # type: ignore[import-untyped]
import mlflow
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Any
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_pic_collision_operators.models import (
    FokkerPlanck2D_Tensor_AD,
    FokkerPlanck2D_Tensor_TimeDependent_AD,
    ModelType,
)
from ml_pic_collision_operators.utils import class_from_str


def configure_mlflow_experiment(
    database_name: str,
    experiment_name: str,
    no_sql_db: bool = False,
) -> mlflow.entities.Experiment:
    """Configure MLflow tracking database for the experiment.

    Will create a folder with the name of `database_name` in the current working
    directory if it does not exist.

    Experiment metadata will be stored in a SQLite database file named `database_name.db`
    inside the `database_name` folder. Model artifacts will be stored in a subfolder
    named `experiment_name` inside the `database_name` folder. If `no_sql_db` is True,
    then all data will be stored in the `database_name` folder without using a SQLite
    database.

    Args:
        database_name: Name of folder to store MLflow database and artifacts.
        no_sql_db: If True, a SQLite database is not used, and all files are stored in
            the experiment folder. This is not recommended, as it is slower and MLflow
            will deprecate support for local file storage in the future.
    Returns:
        MLflow Experiment object corresponding to the experiment_name.
    """
    if no_sql_db:
        mlflow.set_tracking_uri(f"file://{os.path.abspath(database_name)}")
    else:
        mlflow.set_tracking_uri(
            f"sqlite:///{os.path.abspath(database_name)}/{database_name}.db"
        )
    if mlflow.get_experiment_by_name(experiment_name) is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location="file://"
            + os.path.abspath(database_name)
            + "/"
            + experiment_name,
        )
        return mlflow.get_experiment(experiment_id)
    else:
        return mlflow.set_experiment(experiment_name)


def get_mlflow_run_id(experiment_name: str, run_name: str) -> str:
    """Get MLflow run ID from experiment and run name"""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment does not exist: experiment_name={experiment_name}"
        )

    existing_runs = mlflow.search_runs(
        experiment_ids=experiment.experiment_id, filter_string=f"run_name='{run_name}'"
    )
    assert isinstance(existing_runs, pd.DataFrame)
    if existing_runs.empty:
        raise ValueError(
            f"Experiment run not found: experiment_name='{experiment_name}'  run_name='{run_name}'"
        )
    elif len(existing_runs) == 1:
        run_id = existing_runs["run_id"][0]
    else:
        raise ValueError(f"Multiple runs detected with the same name: {run_name}")

    return run_id


def get_mlflow_run_params(run_id: str) -> dict:
    """Get MLflow run parameters from run ID"""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def get_mlflow_metric_history(
    metric_name: str, run_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Get metric history from MLflow run ID"""
    client = mlflow.MlflowClient()
    metric_history = client.get_metric_history(run_id, metric_name)
    steps = np.array([m.step for m in metric_history])
    values = np.array([m.value for m in metric_history])
    i_start = np.argwhere(steps == 0)[-1, 0]
    return steps[i_start:], values[i_start:]


def get_model_state_dict(
    model: ModelType | DDP,
    compiled_model: bool = False,
) -> dict[str, Any]:
    if isinstance(model, DDP):
        if compiled_model:
            return model.module._orig_mod.state_dict()
        else:
            return model.module.state_dict()
    else:
        if compiled_model:
            # mypy complains about _orig_mod possible being a nn.Tensor
            # so we must do the type assertion here
            assert isinstance(model._orig_mod, nn.Module)
            return model._orig_mod.state_dict()
        else:
            return model.state_dict()


def get_model_init_params_dict(
    model: nn.Module,
    compiled_model: bool = False,
):
    if isinstance(model, DDP):
        return model.module.init_params_dict
    else:
        return model.init_params_dict


def log_model(
    model: ModelType | DDP,
    tmp_dir: str,
    fname: str = "weights.pth",
    compiled_model: bool = False,
):
    """Log PyTorch model to MLflow.

    For now, this is a simple wrapper around log_model_init_params_and_state_dict.

    Args:
        model: Model to log.
        tmp_dir (str): Temporary directory to save checkpoint before logging.
        fname (str, optional): Name of model weights file. Defaults to "weights.pth"
        compiled_model (bool, optional): Whether the model is compiled. Defaults to False.
    """
    init_params_dict = get_model_init_params_dict(model, compiled_model)
    state_dict = get_model_state_dict(model, compiled_model)
    log_model_init_params_and_state_dict(init_params_dict, state_dict, tmp_dir, fname)


def log_model_init_params_and_state_dict(
    model_init_params: dict[str, Any],
    model_state_dict: dict[str, Any],
    tmp_dir: str,
    fname: str = "weights.pth",
):
    """Log init_params and state_dict to MLflow.

    Args:
        model_init_params: Model initialization parameters.
        model_state_dict: Model state dict (weights).
        tmp_dir: Temporary directory to save checkpoint before logging.
        fname: Name of model weights file. Defaults to "weights.pth".
    """
    checkpoint_path = os.path.join(tmp_dir, fname)
    checkpoint = {
        "state_dict": model_state_dict,
        "init_params": model_init_params,
    }
    torch.save(checkpoint, checkpoint_path)
    mlflow.log_artifact(checkpoint_path, artifact_path="model")


def load_model(
    run_id: str, fname: str = "weights.pth", device: str = "cpu"
) -> ModelType:
    """Load torchmodel from MLflow run ID

    Args:
        run_id: MLflow run ID where model is logged
        fname: Name of model weights file. Defaults to "weights.pth".
        device: Device where to load the model. Defaults to "cpu".

    Returns:
        model: PyTorch model loaded from checkpoint with weights restored
    """
    run_params = get_mlflow_run_params(run_id)
    model_cls = class_from_str(
        run_params["model_cls"], "ml_pic_collision_operators.models"
    )
    checkpoint_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=f"model/{fname}"
    )
    if checkpoint_path is None:
        raise Exception(f"Could not find model checkpoint at run_id={run_id}")

    checkpoint = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location=None if torch.cuda.is_available() else "cpu",
    )
    model = model_cls(**checkpoint["init_params"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def load_model_from_AD_hdf(
    hdf_file: str,
    ensure_non_negative_f: bool = True,
    ensure_non_negative_D: bool = False,
    includes_time: bool = False,
) -> FokkerPlanck2D_Tensor_AD | FokkerPlanck2D_Tensor_TimeDependent_AD:
    """Load A and D coefficients from HDF file and create FokkerPlanck model

    This is useful for loading precomputed A and D coefficients from particle tracks.

    HDF File should contain the following datasets:
        - grid_size: tuple of 2 ints, number of grid points in each dimension
        - grid_dx: tuple of 2 floats, grid spacing in each dimension
        - grid_range: tuple of 4 floats, min and max values in each dimension
        - grid_range_units: str, units of grid range (should be "[v_th]" or "[c]")
        - v_th: float, thermal velocity used for normalization
        - A: np.ndarray, A coefficients
        - D: np.ndarray, D coefficients
        - dt: float, time step size (only used if includes_time=True)

    Args:
        hdf_file: path to HDF5 file containing A and D coefficients
        ensure_non_negative_f: if True, ensure distribution function remains non-negative
        ensure_non_negative_D: if True, ensure D coefficients remain non-negative
        includes_time: if True, load time-dependent A and D coefficients

    Returns:
        fp_model: `FokkerPlanck2D_Tensor_AD` model if includes_time=False or
            `FokkerPlanck2D_Tensor_Base_TimeDependent` model if includues_time=True.
    """
    data_dict = {}
    with h5py.File(hdf_file, "r") as f:
        for key, item in f.items():
            data_dict[key] = item[()]

    grid_size: tuple[int, int] = data_dict["grid_size"]
    grid_dx: tuple[int, int] = data_dict["grid_dx"]
    grid_range: tuple[int, int, int, int] = data_dict["grid_range"]
    grid_units: str = data_dict["grid_range_units"].decode("ascii")
    v_th: float = data_dict["v_th"]

    A: np.ndarray = data_dict["A"].copy()
    D: np.ndarray = data_dict["D"].copy()

    # Normalize A/D to match trained FokkerPlanck models
    # Must divide A by dx and D by dx^2
    if includes_time:
        A /= np.array(grid_dx).reshape(1, 2, 1, 1)
        D /= np.array([grid_dx[0] ** 2, grid_dx[1] ** 2, np.prod(grid_dx)]).reshape(
            1, 3, 1, 1
        )
        A[np.isnan(A)] = 0
        D[np.isnan(D)] = 0
    else:
        A /= np.array(grid_dx).reshape(2, 1, 1)
        D /= np.array([grid_dx[0] ** 2, grid_dx[1] ** 2, np.prod(grid_dx)]).reshape(
            3, 1, 1
        )

    # Normalize grid range to vth (to match trained models)
    if grid_units == "[c]":
        grid_range = (np.array(grid_range) / v_th).tolist()
        grid_dx = (np.array(grid_dx) / v_th).tolist()
        grid_units = "[v_{{th}}]"
    elif data_dict["grid_range_units"] != "[v_th]":
        raise Exception(f"AB model was saved with non-accepted units: {grid_units}")

    model: FokkerPlanck2D_Tensor_AD | FokkerPlanck2D_Tensor_TimeDependent_AD
    if includes_time:
        model = FokkerPlanck2D_Tensor_TimeDependent_AD(
            grid_size=grid_size,
            grid_dx=grid_dx,
            grid_range=grid_range,
            grid_units=grid_units,
            grid_size_t=A.shape[0],
            grid_dt=data_dict["dt"],
            n_t=A.shape[0],
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
        )

    else:
        model = FokkerPlanck2D_Tensor_AD(
            grid_size=grid_size,
            grid_dx=grid_dx,
            grid_range=grid_range,
            grid_units=grid_units,
            ensure_non_negative_f=ensure_non_negative_f,
            ensure_non_negative_D=ensure_non_negative_D,
        )

    return model.load_from_numpy(A, D)
