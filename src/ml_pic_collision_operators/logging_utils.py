import os
import h5py  # type: ignore[import-untyped]
import mlflow
import torch
import numpy as np
import pandas as pd
from typing import Any
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_pic_collision_operators.models import (
    FokkerPlanck2D_Tensor_AD,
    FokkerPlanck2D_Tensor_TimeDependent_AD,
    FokkerPlanck3D_Tensor_AD_ParPerp,
    ModelType,
)
from ml_pic_collision_operators.models.utils import torch_interpolate
from ml_pic_collision_operators.utils import class_from_str


def configure_mlflow_experiment(
    database_name: str,
    experiment_name: str,
) -> mlflow.entities.Experiment:
    """Configure MLflow tracking database for the experiment.

    Will create a folder with the name of `database_name` in the current working
    directory if it does not exist.

    Experiment metadata will be stored in a SQLite database file named `database_name.db`
    inside the `database_name` folder. Model artifacts will be stored in a subfolder
    named `experiment_name` inside the `database_name` folder

    Args:
        database_name: Name of folder to store MLflow database and artifacts.
    Returns:
        MLflow Experiment object corresponding to the experiment_name.
    """
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
    if steps.size == 0:
        return steps, values
    i_start = np.argwhere(steps == 0)[-1, 0]
    return steps[i_start:], values[i_start:]


def get_model_state_dict(
    model: ModelType | DDP,
    compiled_model: bool = False,
) -> dict[str, Any]:
    m_ = model.module if isinstance(model, DDP) else model
    if compiled_model:
        assert isinstance(m_._orig_mod, ModelType)
        return m_._orig_mod.state_dict()
    else:
        return m_.state_dict()


def get_model_init_params_dict(
    model: ModelType | DDP,
    compiled_model: bool = False,
) -> dict[str, Any]:
    m_ = model.module if isinstance(model, DDP) else model
    if compiled_model:
        assert isinstance(m_._orig_mod, ModelType)
        return m_._orig_mod.init_params_dict
    else:
        return m_.init_params_dict


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
    elif grid_units != "[v_th]":
        raise Exception(f"AD model was saved with non-accepted units: {grid_units}")

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


def load_model_from_AD_ParPerp_hdf(
    hdf_file: str,
    grid_size: int | None = None,
    ensure_non_negative_f: bool = True,
    ensure_non_negative_D: bool = False,
) -> FokkerPlanck3D_Tensor_AD_ParPerp:
    """Load radial A_par, D_par, D_perp profiles from HDF file and create a 3D model.

    This is the parallel-perpendicular analogue of `load_model_from_AD_hdf`. The HDF
    file stores 1D *radial* profiles (one value per velocity-magnitude bin), as produced
    from particle tracks. The drift is assumed purely radial (A_par) and the diffusion
    isotropic in the plane perpendicular to v̂ (D_par, D_perp), matching
    `FokkerPlanck3D_Tensor_AD_ParPerp`.

    HDF File should contain the following datasets:
        - grid_size: int, number of radial velocity bins (== n_radial)
        - grid_dx: float, radial velocity bin width (unused, kept for symmetry with the
            AD loader. the model grid spacing is derived from grid_range/grid_size)
        - grid_range: tuple of 2 floats, (0, v_max) velocity-magnitude range
        - grid_range_units: str, units of grid range (should be "[v_th]" or "[c]")
        - v_th: float, thermal velocity used for normalization
        - A_par: np.ndarray, radial drift <dv_par>/dt
        - D_par: np.ndarray, parallel (per-direction) diffusion <dv_par^2>/dt
        - D_perp: np.ndarray, perpendicular (per-direction) diffusion. note in 3D the
            perpendicular subspace spans 2 directions, so the raw <dv_perp^2>/dt summed
            over both must be halved before being stored in the file.

    The model enforces the v=0 boundary conditions A_par(0)=0 and D_par(0)=D_perp(0) by
    construction. A warning is raised if the loaded profiles violate them (see
    `FokkerPlanck3D_Tensor_AD_ParPerp.load_from_numpy`).

    Args:
        hdf_file: path to HDF5 file containing the radial A/D profiles.
        grid_size: per-dimension resolution of the symmetric 3D velocity grid the
            profiles are interpolated onto. Defaults to 2 * n_radial - 1 (a grid that
            spans [-v_max, v_max] in each dimension and includes a cell at v=0).
        ensure_non_negative_f: if True, ensure distribution function remains non-negative.
        ensure_non_negative_D: if True, ensure D coefficients remain non-negative.

    Returns:
        fp_model: `FokkerPlanck3D_Tensor_AD_ParPerp` model with the loaded profiles.
    """
    data_dict = {}
    with h5py.File(hdf_file, "r") as f:
        for key, item in f.items():
            data_dict[key] = item[()]

    n_radial = int(data_dict["grid_size"])
    v_th = float(data_dict["v_th"])
    grid_units = data_dict["grid_range_units"].decode("ascii")

    # radial range (0, v_max). the profiles live on the centers of n_radial uniform
    # bins over (0, v_max), reconstructed here rather than read from the file
    v_max = float(np.array(data_dict["grid_range"])[1])
    v_edges = np.linspace(0.0, v_max, n_radial + 1)
    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])

    A_par = np.asarray(data_dict["A_par"])
    D_par = np.asarray(data_dict["D_par"])
    D_perp = np.asarray(data_dict["D_perp"])
    # empty velocity bins come back as NaN. zero them so interpolation stays finite
    A_par[np.isnan(A_par)] = 0
    D_par[np.isnan(D_par)] = 0
    D_perp[np.isnan(D_perp)] = 0

    # symmetric 3D velocity grid spanning [-v_max, v_max] in each dimension
    if grid_size is None:
        grid_size = 2 * n_radial - 1
    dx = 2 * v_max / grid_size
    grid_dx = (dx, dx, dx)
    grid_range = (-v_max, v_max, -v_max, v_max, -v_max, v_max)

    # Normalize A/D to match trained FokkerPlanck models (in the file's native units)
    # Must divide A by dx and D by dx^2
    A_par /= grid_dx[0]
    D_par /= grid_dx[0] ** 2
    D_perp /= grid_dx[0] ** 2

    # Normalize grid range and profile axis to vth (to match trained models)
    if grid_units == "[c]":
        grid_range = (np.array(grid_range) / v_th).tolist()
        grid_dx = (np.array(grid_dx) / v_th).tolist()
        v_centers = v_centers / v_th
        grid_units = "[v_{{th}}]"
    elif grid_units != "[v_th]":
        raise Exception(f"AD model was saved with non-accepted units: {grid_units}")

    model = FokkerPlanck3D_Tensor_AD_ParPerp(
        grid_size=(grid_size, grid_size, grid_size),
        grid_range=grid_range,
        grid_dx=grid_dx,
        grid_units=grid_units,
        n_radial=n_radial,
        ensure_non_negative_f=ensure_non_negative_f,
        ensure_non_negative_D=ensure_non_negative_D,
    )

    # The profiles are defined on `v_centers` (0, v_max), but the model stores them
    # on `vr_axis` (0, box diagonal). Re-interpolate onto the model axis, holding the
    # boundary values where vr_axis extends past the data (e.g. the box corners)
    v_centers_t = torch.as_tensor(v_centers, dtype=model.vr_axis.dtype)
    Apar = torch_interpolate(
        model.vr_axis,
        v_centers_t,
        torch.as_tensor(A_par, dtype=model.vr_axis.dtype),
    ).numpy()
    Dpar = torch_interpolate(
        model.vr_axis,
        v_centers_t,
        torch.as_tensor(D_par, dtype=model.vr_axis.dtype),
    ).numpy()
    Dperp = torch_interpolate(
        model.vr_axis,
        v_centers_t,
        torch.as_tensor(D_perp, dtype=model.vr_axis.dtype),
    ).numpy()

    return model.load_from_numpy(Apar, Dpar, Dperp)
