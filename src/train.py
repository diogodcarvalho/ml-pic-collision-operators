import os
import jax
import jax.numpy as jnp
import numpy as np
import mlflow
import equinox as eqx
import optax  # type: ignore[import-untyped]
import tempfile

from jaxtyping import PyTree
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src.datasets import *
from src.dataloaders import *
from src.models import *
from src.logging import log_equinox_model, load_equinox_model


def train_standard(cfg, run_id):
    # load train data
    train_datasets = [
        BaseDataset(folder=folder, step_size=cfg["dataset"]["step_size"])
        for folder in cfg["dataset"]["train"]["folders"]
    ]

    train_dataset = ConcatDataset(train_datasets)

    train_dataloader = BaseDataLoader(
        train_dataset,
        batch_size=len(train_dataset),
    )

    print("Train Dataset Size:", len(train_dataset))

    # load valid data
    valid_dataloader = []
    if cfg["dataset"]["valid"]["folders"] is not None:
        valid_datasets = [
            BaseDataset(folder=folder, step_size=cfg["dataset"]["step_size"])
            for folder in cfg["dataset"]["valid"]["folders"]
        ]
        valid_dataset = ConcatDataset(valid_datasets)
        valid_dataloader = BaseDataLoader(
            valid_dataset,
            batch_size=len(valid_dataset),
        )

        print("Valid Dataset Size:", len(valid_dataset))

    # initialize model
    model_kwargs = {
        "grid_size": train_datasets[0].grid_size,
        "grid_dx": train_datasets[0].grid_dx,
    }

    model = FokkerPlanck2D(**model_kwargs)
    mlflow.log_params({"model_kwargs": model_kwargs})

    print("Model:", model)

    # initialize optimizer
    optim = optax.adam(float(cfg["optimizer"]["learning_rate"]))

    # only model jax arrays are optimized
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # aux functions
    def loss_fn(model: eqx.Module, x: jax.Array, y: jax.Array):
        y_pred = jax.vmap(model)(x)
        return jnp.mean(jnp.square(y_pred - y))

    @eqx.filter_jit
    def train_step(model: eqx.Module, opt_state: PyTree, x: jax.Array, y: jax.Array):
        train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, train_loss

    @eqx.filter_jit
    def valid_step(model: eqx.Module, x: jax.Array, y: jax.Array):
        valid_loss = loss_fn(model, x, y)
        return valid_loss

    # start training
    min_loss = jnp.inf

    with tempfile.TemporaryDirectory() as tmp_dir:
        step = 0
        for epoch in tqdm(range(cfg["epochs"])):

            # train epoch
            train_loss_epoch = 0
            for x, y in train_dataloader:
                model, opt_state, train_loss_step = train_step(model, opt_state, x, y)
                train_loss_epoch += train_loss_step * len(x) / len(train_dataset)
                mlflow.log_metric("train_loss_step", train_loss_step, step=step)
                step += 1
            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)

            # validation epoch
            valid_loss_epoch = 0
            for x, y in valid_dataloader:
                valid_loss_epoch += (
                    valid_step(model, x, y) * len(x) / len(valid_dataset)
                )
            mlflow.log_metric("valid_loss", valid_loss_epoch, step=epoch)

            # do callbacks
            if cfg["callbacks"] is None:
                continue

            if "log_model" in cfg["callbacks"]:
                if epoch % cfg["callbacks"]["log_model"]["frequency"] == 0:
                    log_equinox_model(model, tmp_dir, "weights.eqx")

            if "log_model_best" in cfg["callbacks"]:
                if train_loss_epoch <= min_loss:
                    min_loss = train_loss_epoch
                    log_equinox_model(model, tmp_dir, "weights-best.eqx")

            if "plot_model" in cfg["callbacks"]:
                if epoch % cfg["callbacks"]["plot_model"]["frequency"] == 0:
                    model_img = os.path.join(tmp_dir, f"model-{epoch:06d}.png")
                    model.plot(model_img)
                    mlflow.log_artifact(model_img, artifact_path="model_img")

        if cfg["callbacks"] is None:
            return

        if "plot_model_end" in cfg["callbacks"]:
            if "log_model_best" in cfg["callbacks"]:
                model = load_equinox_model(run_id, type(model), "weights-best.eqx")
            model_img = os.path.join(tmp_dir, f"model-final.png")
            model.plot(model_img)
            mlflow.log_artifact(model_img, artifact_path="model_img")


def train_temporal_unrolling(cfg, run_id):

    for i_stage, (stage, stage_cfg) in enumerate(
        cfg["temporal_unrolling_stages"].items()
    ):
        # load train data
        train_datasets = [
            TemporalUnrolledDataset(
                folder=folder,
                step_size=cfg["dataset"]["step_size"],
                temporal_unroll_steps=stage_cfg["unrolling_steps"],
            )
            for folder in cfg["dataset"]["train"]["folders"]
        ]

        train_dataset = ConcatDataset(train_datasets)

        train_dataloader = BaseDataLoader(
            train_dataset,
            batch_size=len(train_dataset),
        )

        print("Train Dataset Size:", len(train_dataset))

        # load valid data
        valid_dataloader = []
        if cfg["dataset"]["valid"]["folders"] is not None:
            valid_datasets = [
                TemporalUnrolledDataset(
                    folder=folder,
                    step_size=cfg["dataset"]["step_size"],
                    temporal_unroll_steps=stage_cfg["unrolling_steps"],
                )
                for folder in cfg["dataset"]["valid"]["folders"]
            ]
            valid_dataset = ConcatDataset(valid_datasets)
            valid_dataloader = BaseDataLoader(
                valid_dataset,
                batch_size=len(valid_dataset),
            )

            print("Valid Dataset Size:", len(valid_dataset))

        # actions only done in first stage
        if i_stage == 0:
            model_kwargs = {
                "grid_size": train_datasets[0].grid_size,
                "grid_dx": train_datasets[0].grid_dx,
            }

            model = FokkerPlanck2D(**model_kwargs)
            mlflow.log_params({"model_kwargs": model_kwargs})

            print("Model:", model)

            # initialize optimizer
            optim = optax.adam(float(cfg["optimizer"]["learning_rate"]))

            # only model jax arrays are optimized
            opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # aux functions
        def loss_fn(model: eqx.Module, x: jax.Array, y: jax.Array):
            loss = 0
            y_pred = x.copy()
            for i in range(stage_cfg["unrolling_steps"]):
                y_pred = jax.vmap(model)(y_pred)
                loss += jnp.mean(jnp.square(y_pred - y[:, i]))
            return loss

        @eqx.filter_jit
        def train_step(
            model: eqx.Module, opt_state: PyTree, x: jax.Array, y: jax.Array
        ):
            train_loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
            updates, opt_state = optim.update(
                grads, opt_state, eqx.filter(model, eqx.is_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, train_loss

        @eqx.filter_jit
        def valid_step(model: eqx.Module, x: jax.Array, y: jax.Array):
            valid_loss = loss_fn(model, x, y)
            return valid_loss

        # start training
        min_loss = jnp.inf

        with tempfile.TemporaryDirectory() as tmp_dir:
            step = 0
            for epoch in tqdm(range(cfg["epochs"])):

                # train epoch
                train_loss_epoch = 0
                for x, y in train_dataloader:
                    model, opt_state, train_loss_step = train_step(
                        model, opt_state, x, y
                    )
                    train_loss_epoch += train_loss_step * len(x) / len(train_dataset)
                    mlflow.log_metric("train_loss_step", train_loss_step, step=step)
                    step += 1
                mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)

                # validation epoch
                valid_loss_epoch = 0
                for x, y in valid_dataloader:
                    valid_loss_epoch += (
                        valid_step(model, x, y) * len(x) / len(valid_dataset)
                    )
                mlflow.log_metric("valid_loss", valid_loss_epoch, step=epoch)

                # do callbacks
                if cfg["callbacks"] is None:
                    continue

                if "log_model" in cfg["callbacks"]:
                    if epoch % cfg["callbacks"]["log_model"]["frequency"] == 0:
                        log_equinox_model(model, tmp_dir, "weights.eqx")

                if "log_model_best" in cfg["callbacks"]:
                    if train_loss_epoch <= min_loss:
                        min_loss = train_loss_epoch
                        log_equinox_model(model, tmp_dir, "weights-best.eqx")

                if "plot_model" in cfg["callbacks"]:
                    if epoch % cfg["callbacks"]["plot_model"]["frequency"] == 0:
                        model_img = os.path.join(tmp_dir, f"model-{epoch:06d}.png")
                        model.plot(model_img)
                        mlflow.log_artifact(model_img, artifact_path="model_img")

            if cfg["callbacks"] is None:
                return

            if "plot_model_end" in cfg["callbacks"]:
                if "log_model_best" in cfg["callbacks"]:
                    model = load_equinox_model(run_id, type(model), "weights-best.eqx")
                model_img = os.path.join(tmp_dir, f"model-final.png")
                model.plot(model_img)
                mlflow.log_artifact(model_img, artifact_path="model_img")


def train(cfg, run_id):

    mlflow.log_params(cfg)

    if cfg["mode"] == "standard":
        train_standard(cfg, run_id)
    elif cfg["mode"] == "temporal_unrolling":
        train_temporal_unrolling(cfg, run_id)
