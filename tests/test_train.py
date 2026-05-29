import os
import pytest
import tempfile
import mlflow
import shutil
from pathlib import Path
import torch
import torch.multiprocessing as mp
from types import MappingProxyType

from ml_pic_collision_operators.train import (
    train,
    _train_temporal_unrolling,
    _train_temporal_unrolling_ddp,
)
from ml_pic_collision_operators.config.schema import MainConfig, TrainConfig
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


def _start_mlflow_run(experiment_name, run_name):
    # Use a temporary directory for MLflow to avoid conflicts with existing runs
    tmp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_dir}/mlruns.db")
    # Initialize experiment + run
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    run = mlflow.start_run(
        run_id=None,
        run_name=run_name,
        experiment_id=experiment.experiment_id,
        nested=True,
    )
    return experiment, run


def _close_mlflow_run(experiment):
    mlflow.end_run()
    # Soft delete (marks experiment as deleted in MLflow but files remain on disk)
    mlflow.delete_experiment(experiment.experiment_id)
    # Hard delete (deletes files on disk)
    shutil.rmtree(
        mlflow.get_tracking_uri().replace("sqlite:///", "").replace("mlruns.db", "")
    )


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
    if model_type == "nn":
        config = _get_base_nn_config(model_cls, is_conditioned)
    elif model_type == "tensor":
        config = _get_base_tensor_config(model_cls, is_time_dependent)
    elif model_type == "gridless-nn":
        config = _get_base_gridless_nn_config(model_cls)
    elif model_type == "k-tensor":
        config = _get_base_k_tensor_config(model_cls)
    elif model_type == "k-nn":
        config = _get_base_k_nn_config(model_cls)
    elif model_type == "3d-tensor":
        config = _get_base_3d_tensor_config(model_cls)
    elif model_type == "3d-nn":
        config = _get_base_3d_nn_config(model_cls)
    experiment_name = f"test-{model_type}"

    run_name = f"serial-{model_cls}"
    experiment, run = _start_mlflow_run(experiment_name, run_name)
    mlflow.log_params(config.train.model_dump())

    with tempfile.TemporaryDirectory() as tmp_run:
        _train_temporal_unrolling(
            cfg=config.train,
            run_id=run.info.run_id,
            tmp_dir=tmp_run,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compile_model=False,
        )

    _close_mlflow_run(experiment)


# ============================================================================
# Serial Tests
# ============================================================================


@pytest.mark.parametrize("model_cls", _FP_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_nn(model_cls):
    _run_serial_train(model_cls, model_type="nn")


@pytest.mark.parametrize("model_cls", _FP_NN_CONDITIONED_MODEL_CLASSES)
def test_train_temporal_unrolling_nn_conditioned(model_cls):
    _run_serial_train(model_cls, model_type="nn", is_conditioned=True)


@pytest.mark.parametrize("model_cls", _FP_NN_GRIDLESS_MODEL_CLASSES)
def test_train_temporal_unrolling_gridless(model_cls):
    _run_serial_train(model_cls, model_type="gridless-nn")


@pytest.mark.parametrize("model_cls", _FP_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_tensor(model_cls):
    _run_serial_train(model_cls, model_type="tensor")


@pytest.mark.parametrize("model_cls", _FP_TENSOR_TIME_DEPENDENT_MODEL_CLASSES)
def test_train_temporal_unrolling_tensor_time_dependent(model_cls):
    _run_serial_train(model_cls, model_type="tensor", is_time_dependent=True)


@pytest.mark.parametrize("model_cls", _K_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_k_tensor(model_cls):
    _run_serial_train(model_cls, model_type="k-tensor")


@pytest.mark.parametrize("model_cls", _K_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_k_nn(model_cls):
    _run_serial_train(model_cls, model_type="k-nn")


@pytest.mark.parametrize("model_cls", _FP_3D_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_3d_tensor(model_cls):
    _run_serial_train(model_cls, model_type="3d-tensor")


@pytest.mark.parametrize("model_cls", _FP_3D_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_3d_nn(model_cls):
    _run_serial_train(model_cls, model_type="3d-nn")


# ============================================================================
# DDP Wrapper Functions
# ============================================================================


def _train_ddp_worker(
    rank: int,
    world_size: int,
    model_cls: str,
    model_type: str,
    tmp_dir: str,
    is_conditioned: bool = False,
    is_time_dependent: bool = False,
):
    """Worker function that runs on each train process in DDP setup."""
    # Set environment variables for DDP
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["LOCAL_RANK"] = str(rank)

    _, _, _, device = utils.setup_distributed()

    try:
        if model_type == "nn":
            config = _get_base_nn_config(model_cls, is_conditioned)
        elif model_type == "tensor":
            config = _get_base_tensor_config(model_cls, is_time_dependent)
        elif model_type == "gridless-nn":
            config = _get_base_gridless_nn_config(model_cls)
        elif model_type == "k-tensor":
            config = _get_base_k_tensor_config(model_cls)
        elif model_type == "k-nn":
            config = _get_base_k_nn_config(model_cls)
        elif model_type == "3d-tensor":
            config = _get_base_3d_tensor_config(model_cls)
        elif model_type == "3d-nn":
            config = _get_base_3d_nn_config(model_cls)

        assert isinstance(config.train, TrainConfig)
        experiment_name = f"test-ddp-{model_type}"

        # Only rank 0 should log to MLflow
        if rank == 0:
            run_name = f"ddp-{model_cls}-rank{rank}"
            experiment, run = _start_mlflow_run(experiment_name, run_name)
            mlflow.log_params(config.train.model_dump())
            run_id = run.info.run_id
        else:
            run_id = None

        # Train
        _train_temporal_unrolling_ddp(
            cfg=config.train,
            run_id=run_id,
            tmp_dir=tmp_dir,
            rank=rank,
            world_size=world_size,
            device=device,
            compile_model=False,
        )

        # Cleanup MLflow (only on rank 0)
        if rank == 0:
            _close_mlflow_run(experiment)

    finally:
        utils.cleanup_ddp()


def _run_ddp_test(
    model_cls: str,
    model_type: str,
    world_size: int = 2,
    is_conditioned: bool = False,
    is_time_dependent: bool = False,
):
    """Spawn multiple processes for DDP training test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mp.spawn(
            _train_ddp_worker,
            args=(
                world_size,
                model_cls,
                model_type,
                tmp_dir,
                is_conditioned,
                is_time_dependent,
            ),
            nprocs=world_size,
        )


# ============================================================================
# DDP Tests
# ============================================================================


@pytest.mark.parametrize("model_cls", _FP_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_nn_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="nn")


@pytest.mark.parametrize("model_cls", _FP_NN_CONDITIONED_MODEL_CLASSES)
def test_train_temporal_unrolling_nn_conditioned_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="nn", is_conditioned=True)


@pytest.mark.parametrize("model_cls", _FP_NN_GRIDLESS_MODEL_CLASSES)
def test_train_temporal_unrolling_gridless_ddp_unsupported(model_cls):
    with pytest.raises(Exception, match="not supported in DDP"):
        _run_ddp_test(model_cls, model_type="gridless-nn")


@pytest.mark.parametrize("model_cls", _FP_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_tensor_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="tensor")


@pytest.mark.parametrize("model_cls", _FP_TENSOR_TIME_DEPENDENT_MODEL_CLASSES)
def test_train_temporal_unrolling_tensor_time_dependent_ddp_unsupported(model_cls):
    with pytest.raises(Exception, match="not supported in DDP"):
        _run_ddp_test(model_cls, model_type="tensor", is_time_dependent=True)


@pytest.mark.parametrize("model_cls", _K_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_k_tensor_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="k-tensor")


@pytest.mark.parametrize("model_cls", _K_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_k_nn_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="k-nn")


@pytest.mark.parametrize("model_cls", _FP_3D_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_3d_tensor_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="3d-tensor")


@pytest.mark.parametrize("model_cls", _FP_3D_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_3d_nn_ddp(model_cls):
    _run_ddp_test(model_cls, model_type="3d-nn")


# ============================================================================
# Public Entrypoint Tests
# ============================================================================


def _run_train_entrypoint(config: MainConfig, experiment_name: str, run_name: str):
    """Drive the public ``train()`` entrypoint in single-process mode.

    Unlike :func:`_run_serial_train`, which calls ``_train_temporal_unrolling``
    directly, this exercises the top-level dispatcher and the post-training loss
    plotting tail. ``train()`` manages its own temporary directory internally.
    """
    experiment, run = _start_mlflow_run(experiment_name, run_name)
    mlflow.log_params(config.train.model_dump())

    train(
        cfg=config.train,
        run_id=run.info.run_id,
        rank=0,
        world_size=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compile_model=False,
    )

    _close_mlflow_run(experiment)


class TestTrainEntrypoint:
    """Tests for the public ``train()`` dispatcher and its plotting tail."""

    def test_train_serial_dispatch(self):
        """world_size=1 routes to the serial loop and plots loss / loss_step."""
        config = _get_base_tensor_config("FokkerPlanck2D_Tensor_AD")
        _run_train_entrypoint(
            config, "test-entrypoint", "serial-FokkerPlanck2D_Tensor_AD"
        )

    def test_train_plots_with_regularization(self):
        """Non-zero regularization triggers the regularization loss plot."""
        # Only this model implements get_first_deriv_norm, so it is the only
        # serial path that can exercise the reg plotting branch in train().
        model_cls = "FokkerPlanck2D_Tensor_TimeDependent_AD_ParPerp"
        aux = _thaw(
            {**_BASE_CONFIG, **_BASE_TENSOR_PARAMS, **_TIME_DEPENDENT_DATASET_CONFIG}
        )
        aux["model_cls"] = model_cls
        aux["model_cls_kwargs"]["n_t"] = 5
        # The config is frozen, so enable regularization before validation.
        aux["loss"] = {**aux["loss"], "reg_first_deriv": 0.1}
        config = MainConfig.model_validate({"mode": "train", "train": aux})
        _run_train_entrypoint(config, "test-entrypoint-reg", f"serial-{model_cls}")

    def test_train_plots_all_callbacks(self):
        """All plot callbacks on, covering the start/stage plotting branches.

        The model matrix plots once per model (final only) for speed. This single
        run re-enables the start and per-stage plotting that _BASE_CONFIG disables.
        """
        model_cls = "FokkerPlanck2D_Tensor_AD"
        aux = _thaw({**_BASE_CONFIG, **_BASE_TENSOR_PARAMS, **_BASE_DATASET_CONFIG})
        aux["model_cls"] = model_cls
        aux["callbacks"] = {
            **aux["callbacks"],
            "plot_model_start": {"enabled": True},
            "plot_best_stage_model": {"enabled": True},
            "plot_best_final_model": {"enabled": True},
        }
        config = MainConfig.model_validate({"mode": "train", "train": aux})
        _run_train_entrypoint(config, "test-entrypoint-plots", f"serial-{model_cls}")
