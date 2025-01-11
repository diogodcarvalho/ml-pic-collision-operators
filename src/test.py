import mlflow
import tempfile

from src.logging import get_existing_run_id, load_equinox_model, log_equinox_model
from src.models import *


def test(cfg):

    mlflow.log_params(cfg)

    model_run_id = get_existing_run_id(
        experiment_name=cfg["test"]["model"]["experiment_name"],
        run_name=cfg["test"]["model"]["run_name"],
    )

    if model_run_id is None:
        raise FileNotFoundError(
            "Pre-trained model run not found. Check provided experiment_name and run_name values."
        )
    else:
        print("Pre-trained model run found.")
        print("experiment_name:", cfg["test"]["model"]["experiment_name"])
        print("run_id:", cfg["test"]["model"]["run_name"])
        print("run_id:", model_run_id)

    model = load_equinox_model(model_run_id, FokkerPlanck2D)
    print("model:", model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_equinox_model(model, tmp_dir)
