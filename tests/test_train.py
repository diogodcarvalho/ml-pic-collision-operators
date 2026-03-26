import os
import pytest
import tempfile
import mlflow
import shutil
from pathlib import Path
import torch
import torch.multiprocessing as mp

from ml_pic_collision_operators.train import (
    _train_temporal_unrolling,
    _train_temporal_unrolling_ddp,
)
from ml_pic_collision_operators.config.schema import MainConfig, TrainConfig
import ml_pic_collision_operators.logging as logging
import ml_pic_collision_operators.utils as utils

# Needed to debug potential DDP issues
torch.autograd.set_detect_anomaly(True)

# ============================================================================
# Helper Variables
# ============================================================================

_BASE_DIR = Path(__file__).resolve().parent
_DATA_DIR = _BASE_DIR.parent / "examples" / "dataset"

_BASE_CONFIG = {
    "mode": "train",
    "train": {
        "random_seed": 42,
        "device": "cuda",
        "mode": "temporal_unrolling",
        "data": {
            "folders": [
                str(_DATA_DIR / "normal_-2_0" / "f"),
                str(_DATA_DIR / "ring_normal_2_0.2" / "f"),
            ],
            "train_valid_ratio": 0.95,
        },
        "dataset_cls": "TemporalUnrolledDataset",
        "dataset_cls_kwargs": {"step_size": 1, "i_start": 5},
        "dataloader_cls": None,
        "temporal_unrolling_stages": {
            "stage-1-0": {"unrolling_steps": 1, "epochs": 5, "lr": 0.001},
            "stage-1": {"unrolling_steps": 1, "epochs": 5, "lr": 0.0001},
            "stage-2": {"unrolling_steps": 2, "epochs": 5, "lr": 0.0001},
            "stage-5": {"unrolling_steps": 5, "epochs": 5, "lr": 0.0001},
            "stage-10": {"unrolling_steps": 10, "epochs": 5, "lr": 0.0001},
        },
        "callbacks": {
            "log_best_model": {"enabled": True, "frequency": "stage_end"},
            "log_best_stage_model": {"enabled": True},
            "plot_best_stage_model": {"enabled": True},
            "plot_best_final_model": {"enabled": True},
        },
        "optimizer_cls": "torch.optim.Adam",
        "optimizer_cls_kwargs": {},
        "loss": {"name": "mae", "mode": "accumulated"},
    },
}

_BASE_NN_PARAMS = {
    "model_cls": "FokkerPlanck2D_NN_AD",
    "model_cls_kwargs": {
        "ensure_non_negative_f": True,
        "guard_cells": True,
        "width_size": 64,
        "depth": 2,
        "activation": "torch.nn.LeakyReLU",
        "use_bias": True,
        "use_final_bias": True,
    },
}

_BASE_TENSOR_PARAMS = {
    "model_cls": "FokkerPlanck2D_Tensor_AD",
    "model_cls_kwargs": {
        "ensure_non_negative_f": True,
        "guard_cells": True,
    },
}


_FP_NN_MODEL_CLASSES = [
    "FokkerPlanck2D_NN_AD",
    "FokkerPlanck2D_NN_AD_T",
    "FokkerPlanck2D_NN_AD_Sym",
    "FokkerPlanck2D_NN_AD_ParPerp",
]

_FP_TENSOR_MODEL_CLASSES = [
    "FokkerPlanck2D_Tensor_AD",
    "FokkerPlanck2D_Tensor_AD_T",
    "FokkerPlanck2D_Tensor_AD_Sym",
    "FokkerPlanck2D_Tensor_AD_ParPerp",
]

# ============================================================================
# Utility Functions
# ============================================================================


def _get_base_nn_config(model_cls="FokkerPlanck2D_NN_AD"):
    aux = _BASE_CONFIG.copy()
    aux["train"].update(_BASE_NN_PARAMS)
    aux["train"]["model_cls"] = model_cls
    return MainConfig.model_validate(aux)


def _get_base_tensor_config(model_cls="FokkerPlanck2D_Tensor_AD"):
    aux = _BASE_CONFIG.copy()
    aux["train"].update(_BASE_TENSOR_PARAMS)
    aux["train"]["model_cls"] = model_cls
    return MainConfig.model_validate(aux)


def _start_mlflow_run(experiment_name, run_name):
    # Use a temporary directory for MLflow to avoid conflicts with existing runs
    mlflow.set_tracking_uri("file://" + tempfile.mkdtemp())
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
    shutil.rmtree(mlflow.get_tracking_uri().replace("file://", ""))


# ============================================================================
# Serial WRAPPER FUNCTIONS
# ============================================================================


def _run_serial_train(
    model_cls: str,
    is_nn: bool = True,
):
    if is_nn:
        config = _get_base_nn_config(model_cls)
        experiment_name = "test-nn"
    else:
        config = _get_base_tensor_config(model_cls)
        experiment_name = "test-tensor"

    run_name = f"serial-{model_cls}"
    experiment, run = _start_mlflow_run(experiment_name, run_name)
    mlflow.log_params(config.train.model_dump())

    with tempfile.TemporaryDirectory() as tmp_run:
        _train_temporal_unrolling(
            cfg=config.train,
            run_id=run.info.run_id,
            tmp_dir=tmp_run,
            compile_model=False,
        )

    _close_mlflow_run(experiment)


# ============================================================================
# Serial TESTS
# ============================================================================


@pytest.mark.parametrize("model_cls", _FP_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_nn(model_cls):
    _run_serial_train(model_cls, is_nn=True)


@pytest.mark.parametrize("model_cls", _FP_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_tensor(model_cls):
    _run_serial_train(model_cls, is_nn=False)


# ============================================================================
# DDP WRAPPER FUNCTIONS
# ============================================================================


def _train_ddp_worker(
    rank: int,
    world_size: int,
    model_cls: str,
    is_nn: bool,
    tmp_dir: str,
):
    """Worker function that runs on each train process in DDP setup."""
    # Set environment variables for DDP
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["LOCAL_RANK"] = str(rank)

    # Use gloo backend for CPU testing
    utils.setup_distributed(backend="gloo")

    try:
        if is_nn:
            config = _get_base_nn_config(model_cls)
            experiment_name = "test-ddp-nn"
        else:
            config = _get_base_tensor_config(model_cls)
            experiment_name = "test-ddp-tensor"

        assert isinstance(config.train, TrainConfig)

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
            local_rank=rank,
            world_size=world_size,
            compile_model=False,
        )

        # Cleanup MLflow (only on rank 0)
        if rank == 0:
            _close_mlflow_run(experiment)

    finally:
        utils.cleanup_ddp()


def _run_ddp_test(
    model_cls: str,
    is_nn: bool = True,
    world_size: int = 2,
):
    """Spawn multiple processes for DDP training test."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mp.spawn(
            _train_ddp_worker,
            args=(world_size, model_cls, is_nn, tmp_dir),
            nprocs=world_size,
        )


# ============================================================================
# DDP TESTS
# ============================================================================


@pytest.mark.parametrize("model_cls", _FP_NN_MODEL_CLASSES)
def test_train_temporal_unrolling_nn_ddp(model_cls):
    """Test DDP training with NN models."""
    _run_ddp_test(model_cls, is_nn=True, world_size=2)


@pytest.mark.parametrize("model_cls", _FP_TENSOR_MODEL_CLASSES)
def test_train_temporal_unrolling_tensor_ddp(model_cls):
    """Test DDP training with Tensor models."""
    _run_ddp_test(model_cls, is_nn=False, world_size=2)
