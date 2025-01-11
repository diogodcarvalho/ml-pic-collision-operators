import os
import mlflow
import equinox as eqx
from typing import Type

from mlflow.tracking import MlflowClient


def get_existing_run_id(experiment_name: str, run_name: str) -> None | str:

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise Exception(f"Experiment does not exist: experiment_name={experiment_name}")

    existing_runs = mlflow.search_runs(
        experiment_ids=experiment.experiment_id, filter_string=f"run_name='{run_name}'"
    )

    if existing_runs.empty:
        run_id = None
    elif len(existing_runs) == 1:
        run_id = existing_runs["run_id"][0]
    else:
        raise Exception(f"Multiple runs detected with the same name: {run_name}")

    return run_id


def get_existing_run_params(run_id: str) -> dict:
    client = MlflowClient()
    run = client.get_run(run_id)
    return run.data.params


def log_equinox_model(model: eqx.Module, tmp_dir: str):

    weights_path = os.path.join(tmp_dir, "weights.eqx")
    eqx.tree_serialise_leaves(weights_path, model)
    mlflow.log_artifact(weights_path, artifact_path="models")


def load_equinox_model(run_id: str, model_cls: Type[eqx.Module]) -> eqx.Module:

    model_kwargs = eval(get_existing_run_params(run_id)["model_kwargs"])

    weights_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="models/weights.eqx"
    )

    if weights_path is None:
        return None

    return eqx.tree_deserialise_leaves(weights_path, model_cls(**(model_kwargs)))
