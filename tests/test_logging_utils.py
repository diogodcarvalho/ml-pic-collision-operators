import os
import pytest
import socket
import numpy as np
import h5py  # type: ignore[import-untyped]
import mlflow
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ml_pic_collision_operators.logging_utils import (
    configure_mlflow_experiment,
    get_mlflow_run_id,
    get_mlflow_metric_history,
    get_model_state_dict,
    get_model_init_params_dict,
    log_model,
    load_model,
    load_model_from_AD_hdf,
)
from ml_pic_collision_operators.models import (
    FokkerPlanck2D_Tensor_AD,
    FokkerPlanck2D_Tensor_TimeDependent_AD,
)


@pytest.fixture
def mlflow_experiment(tmp_path):
    """Real sqlite-backed MLflow tracking in a temp dir, so logging code runs unmocked."""
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlruns.db")
    experiment_name = "test-exp"
    mlflow.set_experiment(experiment_name)
    return experiment_name


@pytest.fixture
def ddp_group():
    """Single-rank gloo process group so DDP-wrapped models build without mp.spawn."""

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
    sock.close()
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    yield
    dist.destroy_process_group()


def _tiny_AD_model():
    """Small valid tensor model for state-dict and logging round-trips."""
    return FokkerPlanck2D_Tensor_AD(
        grid_size=(2, 2),
        grid_range=(-1.0, 1.0, -1.0, 1.0),
        grid_dx=(1.0, 1.0),
        grid_units="[v_{th}]",
    )


def _write_AD_hdf(path, A, D, units="[v_th]", v_th=1.0, dt=0.1):
    """Write a minimal A/D HDF5 file matching load_model_from_AD_hdf's expected schema."""
    with h5py.File(path, "w") as f:
        f.create_dataset("grid_size", data=[2, 2])
        f.create_dataset("grid_dx", data=[0.5, 0.5])
        f.create_dataset("grid_range", data=[-1.0, 1.0, -1.0, 1.0])
        f.create_dataset("grid_range_units", data=units)
        f.create_dataset("v_th", data=v_th)
        f.create_dataset("A", data=A)
        f.create_dataset("D", data=D)
        f.create_dataset("dt", data=dt)


class TestConfigureMlflowExperiment:
    def test_sqlite(self, tmp_path, monkeypatch):
        # sqlite backend should create the <db>.db file and register the requested experiment
        monkeypatch.chdir(tmp_path)
        experiment = configure_mlflow_experiment("mydb", "exp-a")
        assert experiment.name == "exp-a"
        assert (tmp_path / "mydb" / "mydb.db").exists()

    def test_file_store_and_reuse(self, tmp_path, monkeypatch):
        # no_sql_db uses a file store, and a repeat call must return the existing experiment
        monkeypatch.chdir(tmp_path)
        first = configure_mlflow_experiment("mydb", "exp-b", no_sql_db=True)
        second = configure_mlflow_experiment("mydb", "exp-b", no_sql_db=True)
        assert first.experiment_id == second.experiment_id


class TestGetMlflowRunId:
    def test_found(self, mlflow_experiment):
        # a logged run is retrievable by its run_name within the experiment
        with mlflow.start_run(run_name="findme") as run:
            expected = run.info.run_id
        assert get_mlflow_run_id(mlflow_experiment, "findme") == expected

    def test_experiment_not_found(self, mlflow_experiment):
        # an unknown experiment name is an explicit ValueError, not a silent None
        with pytest.raises(ValueError, match="Experiment does not exist"):
            get_mlflow_run_id("no-such-exp", "findme")

    def test_run_not_found(self, mlflow_experiment):
        # an existing experiment with no matching run name raises a descriptive ValueError
        with pytest.raises(ValueError, match="Experiment run not found"):
            get_mlflow_run_id(mlflow_experiment, "ghost-run")

    def test_duplicate_runs(self, mlflow_experiment):
        # two runs sharing a name is ambiguous and must raise rather than silently pick one
        for _ in range(2):
            with mlflow.start_run(run_name="dup"):
                pass
        with pytest.raises(ValueError, match="Multiple runs detected"):
            get_mlflow_run_id(mlflow_experiment, "dup")


class TestGetMlflowMetricHistory:
    def test_restarts_at_last_step_zero(self, mlflow_experiment):
        # history is trimmed to the final step-0 onward, so a resumed run drops the stale prefix
        with mlflow.start_run() as run:
            for step, value in [(0, 1.0), (1, 2.0), (0, 9.0), (1, 8.0)]:
                mlflow.log_metric("loss", value, step=step)
            run_id = run.info.run_id
        steps, values = get_mlflow_metric_history("loss", run_id)
        assert steps.tolist() == [0, 1]
        assert values.tolist() == [9.0, 8.0]

    def test_empty(self, mlflow_experiment):
        # an unlogged metric returns empty arrays instead of indexing into nothing
        with mlflow.start_run() as run:
            run_id = run.info.run_id
        steps, values = get_mlflow_metric_history("never-logged", run_id)
        assert steps.size == 0 and values.size == 0


class TestGetModelStateDict:
    def test_plain(self):
        # the non-DDP path returns the plain model's own weights
        assert set(get_model_state_dict(_tiny_AD_model())) == {"A", "D"}

    def test_compiled(self):
        # a torch.compile-wrapped model exposes its weights through _orig_mod, not the wrapper
        compiled = torch.compile(_tiny_AD_model(), backend="eager")
        assert set(get_model_state_dict(compiled, compiled_model=True)) == {"A", "D"}

    def test_ddp(self, ddp_group):
        # a DDP-wrapped model unwraps to model.module for its state dict
        ddp_model = DDP(_tiny_AD_model())
        assert set(get_model_state_dict(ddp_model)) == {"A", "D"}

    def test_ddp_compiled(self, ddp_group):
        # DDP wrapping a compiled model must unwrap both layers (module._orig_mod)
        ddp_model = DDP(torch.compile(_tiny_AD_model(), backend="eager"))
        assert set(get_model_state_dict(ddp_model, compiled_model=True)) == {"A", "D"}


class TestGetModelInitParamsDict:
    def test_plain(self):
        # the non-DDP path reads init params straight off the model
        assert get_model_init_params_dict(_tiny_AD_model())["grid_size"] == (2, 2)

    def test_compiled(self):
        # a torch.compile-wrapped model exposes init params through _orig_mod, not the wrapper
        compiled = torch.compile(_tiny_AD_model(), backend="eager")
        assert get_model_init_params_dict(compiled, compiled_model=True)["grid_size"] == (
            2,
            2,
        )

    def test_ddp(self, ddp_group):
        # init params are read from model.module when the model is DDP-wrapped
        ddp_model = DDP(_tiny_AD_model())
        assert get_model_init_params_dict(ddp_model)["grid_size"] == (2, 2)

    def test_ddp_compiled(self, ddp_group):
        # DDP wrapping a compiled model must unwrap both layers (module._orig_mod)
        ddp_model = DDP(torch.compile(_tiny_AD_model(), backend="eager"))
        params = get_model_init_params_dict(ddp_model, compiled_model=True)
        assert params["grid_size"] == (2, 2)


class TestLogAndLoadModel:
    def test_round_trip(self, mlflow_experiment, tmp_path):
        # log_model then load_model must restore both the model class and its trained weights
        model = _tiny_AD_model()
        with torch.no_grad():
            model.A.add_(3.0)
        with mlflow.start_run(run_name="round-trip") as run:
            mlflow.log_param("model_cls", "FokkerPlanck2D_Tensor_AD")
            log_model(model, str(tmp_path))
            run_id = run.info.run_id
        loaded = load_model(run_id)
        assert isinstance(loaded, FokkerPlanck2D_Tensor_AD)
        assert torch.allclose(loaded.A, torch.full_like(loaded.A, 3.0))


class TestLoadModelFromADHdf:

    def test_includes_time(self, tmp_path):
        # includes_time=True builds the time-dependent model with n_t inferred from the A axis
        file_path = tmp_path / "time.h5"
        n_t = 3
        _write_AD_hdf(
            file_path, np.ones((n_t, 2, 2, 2)), np.ones((n_t, 3, 2, 2)), dt=0.1
        )
        model = load_model_from_AD_hdf(str(file_path), includes_time=True)
        assert isinstance(model, FokkerPlanck2D_Tensor_TimeDependent_AD)
        assert model.n_t == n_t
        assert model.grid_dt == 0.1

    def test_v_th_units(self, tmp_path):
        # "[v_th]" is a documented accepted unit and must load without the units check rejecting it
        file_path = tmp_path / "vth.h5"
        _write_AD_hdf(file_path, np.ones((2, 2, 2)), np.ones((3, 2, 2)))
        model = load_model_from_AD_hdf(str(file_path))
        assert isinstance(model, FokkerPlanck2D_Tensor_AD)
        # A is normalized by grid_dx=0.5 (1.0 / 0.5 = 2.0), with no grid rescaling for [v_th]
        assert np.allclose(model.A.detach().numpy(), 2.0)

    def test_c_units(self, tmp_path):
        # "[c]" units are converted to thermal-velocity units by dividing grid_range by v_th
        file_path = tmp_path / "test_data.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("grid_size", data=[4, 4])
            f.create_dataset("grid_dx", data=[0.5, 0.5])
            f.create_dataset("grid_range", data=[-1.0, 1.0, -1.0, 1.0])
            # Using units of [c] to test conversion to [v_th]
            f.create_dataset("grid_range_units", data="[c]")
            # v_th != 1.0 to test unit conversion
            f.create_dataset("v_th", data=2.0)
            # A, D all ones for simplicity
            f.create_dataset("A", data=np.ones((2, 4, 4)))
            f.create_dataset("D", data=np.ones((3, 4, 4)))

        model = load_model_from_AD_hdf(str(file_path), includes_time=False)
        assert model.grid_range == [-0.5, 0.5, -0.5, 0.5]
        A_result = model.A.detach().numpy()
        D_result = model.D.detach().numpy()
        assert np.allclose(
            A_result, 2.0
        ), f"A normalization failed: expected 2.0, got {A_result.max()}"
        assert np.allclose(
            D_result, 4.0
        ), f"D normalization failed: expected 4.0, got {D_result.max()}"

    def test_invalid_units(self, tmp_path):
        # units other than "[v_th]"/"[c]" are rejected with a clear error
        file_path = tmp_path / "bad_units.h5"
        with h5py.File(file_path, "w") as f:
            f.create_dataset("grid_range_units", data=b"[meters]")
            # ... (add other required keys with dummy data)
            f.create_dataset("grid_size", data=[1, 1])
            f.create_dataset("grid_dx", data=[1, 1])
            f.create_dataset("grid_range", data=[0, 1, 0, 1])
            f.create_dataset("v_th", data=1.0)
            f.create_dataset("A", data=np.zeros((2, 1, 1)))
            f.create_dataset("D", data=np.zeros((3, 1, 1)))

        with pytest.raises(Exception, match="non-accepted units"):
            load_model_from_AD_hdf(str(file_path))
