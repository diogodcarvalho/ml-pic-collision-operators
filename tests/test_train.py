import pytest
import tempfile
import mlflow
import shutil
from pathlib import Path

from ml_pic_collision_operators.train import _train_temporal_unrolling
from ml_pic_collision_operators.config.schema import MainConfig
import ml_pic_collision_operators.logging as logging

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
    "model_cls": "FokkerPlanck2D_NN",
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


def _get_base_nn_config(model_cls="FokkerPlanck2D_NN"):
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
    mlflow.start_run(
        run_id=None,
        run_name=run_name,
        experiment_id=experiment.experiment_id,
        nested=True,
    )
    return experiment


def _close_mlflow_run(experiment):
    mlflow.end_run()
    # Soft delete (marks experiment as deleted in MLflow but files remain on disk)
    mlflow.delete_experiment(experiment.experiment_id)
    # Hard delete (deletes files on disk)
    shutil.rmtree(mlflow.get_tracking_uri().replace("file://", ""))


@pytest.mark.parametrize(
    "model_cls",
    [
        "FokkerPlanck2D_NN_AD",
        "FokkerPlanck2D_NN_AD_T",
        "FokkerPlanck2D_NN_AD_Sym",
        "FokkerPlanck2D_NN_AD_ParPerp",
    ],
)
def test_train_temporal_unrolling_nn(model_cls):
    config = _get_base_nn_config(model_cls)

    experiment_name = "test-nn"
    run_name = "nn-base"
    experiment = _start_mlflow_run(experiment_name, run_name)
    mlflow.log_params(config.train.model_dump())

    with tempfile.TemporaryDirectory() as tmp_run:
        _train_temporal_unrolling(
            cfg=config.train,
            run_id=mlflow.active_run().info.run_id,
            tmp_dir=tmp_run,
            compile_model=False,
        )

    _close_mlflow_run(experiment)


@pytest.mark.parametrize(
    "model_cls",
    [
        "FokkerPlanck2D_Tensor_AD",
        "FokkerPlanck2D_Tensor_AD_T",
        "FokkerPlanck2D_Tensor_AD_Sym",
        "FokkerPlanck2D_Tensor_AD_ParPerp",
    ],
)
def test_train_temporal_unrolling_tensor(model_cls):
    config = _get_base_tensor_config(model_cls)

    experiment_name = "test-tensor"
    run_name = "tensor-base"
    experiment = _start_mlflow_run(experiment_name, run_name)
    mlflow.log_params(config.train.model_dump())

    with tempfile.TemporaryDirectory() as tmp_run:
        _train_temporal_unrolling(
            cfg=config.train,
            run_id=mlflow.active_run().info.run_id,
            tmp_dir=tmp_run,
            compile_model=False,
        )

    _close_mlflow_run(experiment)
