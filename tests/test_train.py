import os
import queue
import socket
import pytest
import tempfile
import mlflow
import shutil
from pathlib import Path
import torch
import torch.multiprocessing as mp
from types import MappingProxyType

from ml_pic_collision_operators.train import (
    _train_temporal_unrolling,
    _train_temporal_unrolling_ddp,
)
from ml_pic_collision_operators.config.schema import MainConfig
import ml_pic_collision_operators.utils as utils

# Needed to debug potential DDP issues
torch.autograd.set_detect_anomaly(True)


def _freeze(d: dict) -> MappingProxyType:
    """Recursively freeeze dictionary."""
    return MappingProxyType(
        {k: _freeze(v) if isinstance(v, dict) else v for k, v in d.items()}
    )


def _thaw(obj) -> dict:
    """Recursively convert MappingProxyType (and plain dicts) to mutable dicts."""
    if isinstance(obj, (MappingProxyType, dict)):
        return {k: _thaw(v) for k, v in obj.items()}
    return obj


# ============================================================================
# Helper Test Configurations
# ============================================================================

_BASE_DIR = Path(__file__).resolve().parent
_PHASESPACE_DATA_DIR = _BASE_DIR.parent / "examples" / "dataset"
_TRACKS_DATA_DIR = _BASE_DIR.parent / "examples" / "dataset_tracks"

_BASE_DATASET_CONFIG = _freeze(
    {
        "data": {
            "folders": [
                str(_PHASESPACE_DATA_DIR / "normal_-2_0" / "f"),
                str(_PHASESPACE_DATA_DIR / "ring_normal_2_0.2" / "f"),
            ],
            "train_valid_ratio": 0.5,
        },
        "dataset_cls": "TemporalUnrolledDataset",
        "dataset_cls_kwargs": {"step_size": 1, "i_start": 5, "i_end": 10},
    }
)

_CONDITIONED_DATASET_CONFIG = _freeze(
    {
        "data": {
            "folders": [
                str(_PHASESPACE_DATA_DIR / "normal_-2_0" / "f"),
                str(_PHASESPACE_DATA_DIR / "ring_normal_2_0.2" / "f"),
                str(_PHASESPACE_DATA_DIR / "normal_-2_0_sim2" / "f"),
            ],
            "conditioners": [
                {"ppc": 4, "v_th": 0.01, "shape": 1, "dx_lD": 1.0},
                {"ppc": 4, "v_th": 0.01, "shape": 1, "dx_lD": 1.0},
                {"ppc": 25, "v_th": 0.1, "shape": 4, "dx_lD": 2.0},
            ],
            "train_valid_ratio": 0.5,
        },
        "dataset_cls": "TemporalUnrolledwConditionersDataset",
        "dataset_cls_kwargs": {"step_size": 1, "i_start": 5, "i_end": 10},
    }
)

_3D_DATASET_CONFIG = _freeze(
    {
        "data": {
            "folders": [
                str(_PHASESPACE_DATA_DIR / "normal_-2_0_0_3D" / "f"),
                str(_PHASESPACE_DATA_DIR / "ring_normal_2_0.2_3D" / "f"),
            ],
            "train_valid_ratio": 0.50,
        },
        "dataset_cls": "TemporalUnrolledDataset",
        # use less data for 3D models to speed up tests
        "dataset_cls_kwargs": {"step_size": 1, "i_start": 5, "i_end": 10},
        # use small batch size to avoid OOM with 3D data
        "dataloader_cls": "BaseDataLoader",
        "dataloader_cls_kwargs": {"batch_size": 1},
    }
)

_TIME_DEPENDENT_DATASET_CONFIG = _freeze(
    {
        "data": {
            "folders": [
                str(_PHASESPACE_DATA_DIR / "normal_-2_0" / "f"),
                str(_PHASESPACE_DATA_DIR / "ring_normal_2_0.2" / "f"),
            ],
            "train_valid_ratio": 0.50,
        },
        "dataset_cls": "TemporalUnrolledwConditionersDataset",
        "dataset_cls_kwargs": {
            "step_size": 1,
            "i_start": 5,
            "i_end": 10,
            "include_time": True,
        },
    }
)

_TRACKS_DATASET_CONFIG = _freeze(
    {
        "data": {
            "folders": [
                str(_TRACKS_DATA_DIR / "2D_generated" / "1_1k"),
                str(_TRACKS_DATA_DIR / "2D_generated" / "2_1k"),
            ],
            "train_valid_ratio": 0.50,
        },
        "dataset_cls": "TemporalUnrolledTracksDataset",
        "dataset_cls_kwargs": {"step_size": 1, "i_start": 5, "i_end": 10},
    }
)

_BASE_CONFIG = _freeze(
    {
        "random_seed": 42,
        "mode": "temporal_unrolling",
        "dataloader_cls": None,
        "temporal_unrolling_stages": {
            "stage-1": {"unrolling_steps": 1, "epochs": 2, "lr": 0.0001},
            "stage-2": {"unrolling_steps": 2, "epochs": 2, "lr": 0.0001},
        },
        "callbacks": {
            "log_best_model": {"enabled": True, "frequency": "stage_end"},
            "log_best_stage_model": {"enabled": True},
            # Plot each model once only (at end) to speed up tests.
            "plot_model_start": {"enabled": False},
            "plot_best_stage_model": {"enabled": False},
            "plot_best_final_model": {"enabled": True},
        },
        "optimizer_cls": "torch.optim.Adam",
        "optimizer_cls_kwargs": {},
        "loss": {"name": "mae", "mode": "accumulated"},
    }
)

_WEAK_SDE_LOSS_CONFIG = _freeze(
    {
        "kind": "weak_sde",
        "test_functions": [
            {
                "cls_name": "MonomialTestFunctions",
                "cls_kwargs": {"n_dims": 2, "degree": 2},
            },
            {
                "cls_name": "GaussianTestFunctions",
                "cls_kwargs": {
                    "centers": [[-0.2, 0.0], [0.0, 0.0], [0.2, 0.2], [0.0, 0.2]],
                    "sigma": 0.15,
                },
            },
        ],
    }
)

_BASE_NN_PARAMS = _freeze(
    {
        "model_cls_kwargs": {
            "ensure_non_negative_f": True,
            "guard_cells": True,
            "width_size": 16,
            "depth": 2,
            "activation": "torch.nn.LeakyReLU",
            "use_bias": True,
            "use_final_bias": True,
        },
    }
)

_BASE_NN_GRIDLESS_PARAMS = _freeze(
    {
        "model_cls_kwargs": {
            "v_range_norm": [-0.5, 0.5, -0.5, 0.5],
            "width_size": 16,
            "depth": 2,
            "activation": "torch.nn.LeakyReLU",
            "use_bias": True,
            "use_final_bias": True,
        },
    }
)


_BASE_TENSOR_PARAMS = _freeze(
    {
        "model_cls_kwargs": {
            "ensure_non_negative_f": True,
            "guard_cells": True,
        },
    }
)

_BASE_K_TENSOR_PARAMS = _freeze(
    {
        "model_cls_kwargs": {
            "kernel_size": 2,
            "padding_mode": "zeros",
            "ensure_non_negative_f": True,
            "gradient_scheme": "forward",
        },
    }
)

_BASE_K_NN_PARAMS = _freeze(
    {
        "model_cls_kwargs": {
            **_BASE_K_TENSOR_PARAMS["model_cls_kwargs"],
            **{
                k: v
                for k, v in _BASE_NN_PARAMS["model_cls_kwargs"].items()
                if k != "guard_cells"
            },
        }
    }
)

# ============================================================================
# Model Classes to Test
# ============================================================================

_FP_NN_MODEL_CLASSES = [
    "FokkerPlanck2D_NN_AD",
    "FokkerPlanck2D_NN_AD_T",
    "FokkerPlanck2D_NN_AD_Sym",
    "FokkerPlanck2D_NN_AD_ParPerp",
]

_FP_NN_CONDITIONED_MODEL_CLASSES = [
    "FokkerPlanck2D_NNConditioned_AD",
    "FokkerPlanck2D_NNConditioned_AD_T",
    "FokkerPlanck2D_NNConditioned_AD_Sym",
    "FokkerPlanck2D_NNConditioned_AD_ParPerp",
]

_FP_NN_GRIDLESS_MODEL_CLASSES = [
    "FokkerPlanck2D_NN_Gridless_AD",
    "FokkerPlanck2D_NN_Gridless_AD_T",
    "FokkerPlanck2D_NN_Gridless_AD_ParPerp",
]

_FP_TENSOR_MODEL_CLASSES = [
    "FokkerPlanck2D_Tensor_AD",
    "FokkerPlanck2D_Tensor_AD_T",
    "FokkerPlanck2D_Tensor_AD_Sym",
    "FokkerPlanck2D_Tensor_AD_ParPerp",
]

_FP_TENSOR_TIME_DEPENDENT_MODEL_CLASSES = [
    "FokkerPlanck2D_Tensor_TimeDependent_AD",
    "FokkerPlanck2D_Tensor_TimeDependent_AD_ParPerp",
]

_K_TENSOR_MODEL_CLASSES = [
    "K2D_Tensor",
    "K2D_Tensor_T",
]

_K_NN_MODEL_CLASSES = [
    "K2D_NN",
    "K2D_NN_T",
]

_FP_3D_TENSOR_MODEL_CLASSES = [
    "FokkerPlanck3D_Tensor_AD",
    "FokkerPlanck3D_Tensor_AD_ParPerp",
]

_FP_3D_NN_MODEL_CLASSES = [
    "FokkerPlanck3D_NN_AD",
    "FokkerPlanck3D_NN_AD_ParPerp",
]

# ============================================================================
# Utility Functions
# ============================================================================


def _get_base_nn_config(model_cls: str, is_conditioned: bool):
    if is_conditioned:
        print("Using conditioned dataset config")
        aux = _thaw({**_BASE_CONFIG, **_BASE_NN_PARAMS, **_CONDITIONED_DATASET_CONFIG})
    else:
        print("Using unconditioned dataset config")
        aux = _thaw({**_BASE_CONFIG, **_BASE_NN_PARAMS, **_BASE_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _get_base_tensor_config(model_cls: str, is_time_dependent: bool = False):
    if is_time_dependent:
        aux = _thaw(
            {**_BASE_CONFIG, **_BASE_TENSOR_PARAMS, **_TIME_DEPENDENT_DATASET_CONFIG}
        )
        aux["model_cls_kwargs"]["n_t"] = 5  # type: ignore
    else:
        aux = _thaw({**_BASE_CONFIG, **_BASE_TENSOR_PARAMS, **_BASE_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _get_base_gridless_nn_config(model_cls: str):
    aux = _thaw({**_BASE_CONFIG, **_BASE_NN_GRIDLESS_PARAMS, **_TRACKS_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    # The weak-SDE loss only supports single-step rollout, so every stage uses
    # unrolling_steps=1.
    # TODO change this once temporal unrolling is supported
    aux["temporal_unrolling_stages"] = {
        "stage-1": {"unrolling_steps": 1, "epochs": 2, "lr": 0.0001},
        "stage-2": {"unrolling_steps": 1, "epochs": 2, "lr": 0.0001},
    }
    aux["loss"] = {**aux["loss"], **_thaw(_WEAK_SDE_LOSS_CONFIG)}
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _get_base_k_tensor_config(model_cls: str):
    aux = _thaw({**_BASE_CONFIG, **_BASE_K_TENSOR_PARAMS, **_BASE_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _get_base_k_nn_config(model_cls: str):
    aux = _thaw({**_BASE_CONFIG, **_BASE_K_NN_PARAMS, **_BASE_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _get_base_3d_tensor_config(model_cls: str):
    aux = _thaw({**_BASE_CONFIG, **_BASE_TENSOR_PARAMS, **_3D_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _get_base_3d_nn_config(model_cls: str):
    aux = _thaw({**_BASE_CONFIG, **_BASE_NN_PARAMS, **_3D_DATASET_CONFIG})
    aux["model_cls"] = model_cls
    return MainConfig.model_validate({"mode": "train", "train": aux})


def _build_config(model_cls, model_type, is_conditioned=False, is_time_dependent=False):
    """Build the MainConfig for a model, dispatching on model_type."""
    if model_type == "nn":
        return _get_base_nn_config(model_cls, is_conditioned)
    if model_type == "tensor":
        return _get_base_tensor_config(model_cls, is_time_dependent)
    if model_type == "gridless-nn":
        return _get_base_gridless_nn_config(model_cls)
    if model_type == "k-tensor":
        return _get_base_k_tensor_config(model_cls)
    if model_type == "k-nn":
        return _get_base_k_nn_config(model_cls)
    if model_type == "3d-tensor":
        return _get_base_3d_tensor_config(model_cls)
    if model_type == "3d-nn":
        return _get_base_3d_nn_config(model_cls)
    raise ValueError(f"Unknown model_type: {model_type}")


def _set_mlflow_db(tmp_dir=None):
    tmp_dir = tmp_dir or tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_dir}/mlruns.db")
    return tmp_dir


def _start_mlflow_run(experiment_name, run_name):
    if mlflow.active_run() is not None:
        # end possible leftover active run from a prior failure
        mlflow.end_run()
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    run = mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id)
    return experiment, run


class _MlflowDatabaseClass:
    """Base for test classes needing MLflow (one sqlite database per class)."""

    @pytest.fixture(scope="class", autouse=True)
    def _mlflow_store(self):
        tmp_dir = _set_mlflow_db()
        yield
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# Serial Wrapper Functions
# ============================================================================


def _run_serial_train(
    model_cls: str,
    model_type: str,
    is_conditioned: bool = False,
    is_time_dependent: bool = False,
):
    """Run serial training test for a given model class."""
    config = _build_config(model_cls, model_type, is_conditioned, is_time_dependent)
    _, run = _start_mlflow_run(f"test-{model_type}", f"serial-{model_cls}")
    mlflow.log_params(config.train.model_dump())

    with tempfile.TemporaryDirectory() as tmp_run:
        _train_temporal_unrolling(
            cfg=config.train,
            run_id=run.info.run_id,
            tmp_dir=tmp_run,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compile_model=False,
        )

    mlflow.end_run()


# ============================================================================
# Serial Tests
# ============================================================================


class TestSerial(_MlflowDatabaseClass):

    @pytest.mark.parametrize("model_cls", _FP_NN_MODEL_CLASSES)
    def test_nn(self, model_cls):
        _run_serial_train(model_cls, "nn")

    @pytest.mark.parametrize("model_cls", _FP_NN_CONDITIONED_MODEL_CLASSES)
    def test_nn_conditioned(self, model_cls):
        _run_serial_train(model_cls, "nn", is_conditioned=True)

    @pytest.mark.parametrize("model_cls", _FP_NN_GRIDLESS_MODEL_CLASSES)
    def test_gridless(self, model_cls):
        _run_serial_train(model_cls, "gridless-nn")

    @pytest.mark.parametrize("model_cls", _FP_TENSOR_MODEL_CLASSES)
    def test_tensor(self, model_cls):
        _run_serial_train(model_cls, "tensor")

    @pytest.mark.parametrize("model_cls", _FP_TENSOR_TIME_DEPENDENT_MODEL_CLASSES)
    def test_tensor_time_dependent(self, model_cls):
        _run_serial_train(model_cls, "tensor", is_time_dependent=True)

    @pytest.mark.parametrize("model_cls", _K_TENSOR_MODEL_CLASSES)
    def test_k_tensor(self, model_cls):
        _run_serial_train(model_cls, "k-tensor")

    @pytest.mark.parametrize("model_cls", _K_NN_MODEL_CLASSES)
    def test_k_nn(self, model_cls):
        _run_serial_train(model_cls, "k-nn")

    @pytest.mark.parametrize("model_cls", _FP_3D_TENSOR_MODEL_CLASSES)
    def test_3d_tensor(self, model_cls):
        _run_serial_train(model_cls, "3d-tensor")

    @pytest.mark.parametrize("model_cls", _FP_3D_NN_MODEL_CLASSES)
    def test_3d_nn(self, model_cls):
        _run_serial_train(model_cls, "3d-nn")


# ============================================================================
# DDP Worker Pool
# ============================================================================

# Timeout for hanging jobs
_DDP_JOB_TIMEOUT_S = 120


def _ddp_pool_worker(
    rank: int,
    world_size: int,
    port: int,
    store_dir: str,
    task_queue: mp.Queue,
    done_queue: mp.Queue,
):
    """Single worker process for the DDP pool. Only rank 0 interacts with MLflow."""
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
    )
    _, _, _, device = utils.setup_distributed()
    if rank == 0:
        _set_mlflow_db(store_dir)
    try:
        for job in iter(task_queue.get, None):
            model_cls, model_type, is_conditioned, is_time_dependent = job
            try:
                config = _build_config(
                    model_cls, model_type, is_conditioned, is_time_dependent
                )
                if rank == 0:
                    _, run = _start_mlflow_run(
                        f"test-ddp-{model_type}", f"ddp-{model_cls}"
                    )
                    mlflow.log_params(config.train.model_dump())
                    run_id = run.info.run_id
                else:
                    run_id = None
                with tempfile.TemporaryDirectory() as tmp_dir:
                    _train_temporal_unrolling_ddp(
                        cfg=config.train,
                        run_id=run_id,
                        tmp_dir=tmp_dir,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                        compile_model=False,
                    )
                if rank == 0:
                    mlflow.end_run()
                done_queue.put(("ok", rank, model_cls, ""))
            except Exception as e:  # report so the pool survives a model failure
                done_queue.put(("err", rank, model_cls, repr(e)))
    finally:
        utils.cleanup_ddp()


class _DDPWorkerPool:
    """Persistent DDP pool reused across model tests.

    Unexpected job failures (or deadlocks) restart the pool.
    """

    def __init__(self, world_size: int = 2):
        self.world_size = world_size
        self._ctx = mp.get_context("spawn")  # CUDA-safe, unlike fork
        self._start()

    def _start(self):
        self._task_queues = [self._ctx.Queue() for _ in range(self.world_size)]
        self._done_queue = self._ctx.Queue()
        self._mlflow_dir = tempfile.mkdtemp()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self._workers = [
            self._ctx.Process(
                target=_ddp_pool_worker,
                args=(
                    rank,
                    self.world_size,
                    port,
                    self._mlflow_dir,
                    self._task_queues[rank],
                    self._done_queue,
                ),
            )
            for rank in range(self.world_size)
        ]
        for w in self._workers:
            w.start()

    def run(
        self,
        model_cls: str,
        model_type: str,
        is_conditioned: bool = False,
        is_time_dependent: bool = False,
        expected_error: str | None = None,
    ):
        """Runs a single test."""
        for q in self._task_queues:
            q.put((model_cls, model_type, is_conditioned, is_time_dependent))
        try:
            results = [
                self._done_queue.get(timeout=_DDP_JOB_TIMEOUT_S)
                for _ in range(self.world_size)
            ]
        except queue.Empty:
            self._restart()
            raise AssertionError(f"{model_cls}: DDP job timed out (rank deadlock)")

        statuses = {r[0] for r in results}
        if expected_error is not None:
            assert statuses == {"err"} and all(
                expected_error in r[3] for r in results
            ), f"{model_cls}: expected failure '{expected_error}', got {results}"
        elif statuses != {"ok"}:
            self._restart()
            raise AssertionError(f"{model_cls}: DDP train failed: {results}")

    def _stop_workers(self, grace: int = 0):
        for w in self._workers:
            w.join(timeout=grace)  # Chance to exit on its own
            if w.is_alive():
                w.terminate()  # SIGTERM
                w.join(timeout=5)
            if w.is_alive():
                w.kill()  # SIGKILL
                w.join(timeout=5)

    def _restart(self):
        self._stop_workers()
        shutil.rmtree(self._mlflow_dir, ignore_errors=True)
        self._start()

    def close(self):
        for q in self._task_queues:
            q.put(None)
        self._stop_workers(grace=30)
        shutil.rmtree(self._mlflow_dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ============================================================================
# DDP Tests
# ============================================================================


class TestDDP:

    @pytest.fixture(scope="class")
    def ddp_pool(self):
        with _DDPWorkerPool(world_size=2) as pool:
            yield pool

    @pytest.mark.parametrize("model_cls", _FP_NN_MODEL_CLASSES)
    def test_nn(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "nn")

    @pytest.mark.parametrize("model_cls", _FP_NN_CONDITIONED_MODEL_CLASSES)
    def test_nn_conditioned(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "nn", is_conditioned=True)

    @pytest.mark.parametrize("model_cls", _FP_NN_GRIDLESS_MODEL_CLASSES)
    def test_gridless_unsupported(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "gridless-nn", expected_error="not supported in DDP")

    @pytest.mark.parametrize("model_cls", _FP_TENSOR_MODEL_CLASSES)
    def test_tensor(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "tensor")

    @pytest.mark.parametrize("model_cls", _FP_TENSOR_TIME_DEPENDENT_MODEL_CLASSES)
    def test_tensor_time_dependent_unsupported(self, ddp_pool, model_cls):
        ddp_pool.run(
            model_cls,
            "tensor",
            is_time_dependent=True,
            expected_error="not supported in DDP",
        )

    @pytest.mark.parametrize("model_cls", _K_TENSOR_MODEL_CLASSES)
    def test_k_tensor(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "k-tensor")

    @pytest.mark.parametrize("model_cls", _K_NN_MODEL_CLASSES)
    def test_k_nn(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "k-nn")

    @pytest.mark.parametrize("model_cls", _FP_3D_TENSOR_MODEL_CLASSES)
    def test_3d_tensor(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "3d-tensor")

    @pytest.mark.parametrize("model_cls", _FP_3D_NN_MODEL_CLASSES)
    def test_3d_nn(self, ddp_pool, model_cls):
        ddp_pool.run(model_cls, "3d-nn")
