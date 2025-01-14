import os
import jax
import jax.numpy as jnp
import numpy as np
import mlflow
import equinox as eqx
import optax  # type: ignore[import-untyped]
import tempfile
import matplotlib.pyplot as plt

from jaxtyping import PyTree
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src.datasets import *
from src.dataloaders import *
from src.models import *
from src.logging import log_equinox_model, load_equinox_model, get_metric_history


def plot_loss(run_id):

    epochs, train_loss = get_metric_history("train_loss", run_id)
    _, valid_loss = get_metric_history("valid_loss", run_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plt.figure(figsize=(5, 4))
        plt.plot(epochs, train_loss, label="Train")
        if not np.all(valid_loss == 0):
            plt.plot(epochs, valid_loss, label="Valid")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0, epochs[-1])
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(tmp_dir, "loss.png")
        plt.savefig(fname, dpi=200)
        mlflow.log_artifact(fname, artifact_path="train_img")
        plt.show()
        plt.close()


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

    with tempfile.TemporaryDirectory() as tmp_dir:

        for i_stage, (stage, stage_cfg) in enumerate(
            cfg["temporal_unrolling_stages"].items()
        ):
            print()
            print(f"Stage: {stage} (#{i_stage+1})")
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

                # initialize model
                model_kwargs = {
                    "grid_size": train_datasets[0].grid_size,
                    "grid_dx": train_datasets[0].grid_dx,
                }

                model = FokkerPlanck2D(**model_kwargs)
                mlflow.log_params({"model_kwargs": model_kwargs})

                print("Model:", model)

                # initialize optimizer
                if "optimizer" in cfg:
                    optim = optax.adam(float(cfg["optimizer"]["learning_rate"]))
                else:
                    optim = optax.inject_hyperparams(optax.adam)(
                        learning_rate=stage_cfg["learning_rate"]
                    )
                # only model jax arrays are optimized
                opt_state = optim.init(eqx.filter(model, eqx.is_array))
            # actions done in other stages
            else:
                if not "optimizer" in cfg:
                    opt_state.hyperparams["learning_rate"] = stage_cfg["learning_rate"]
                    optim.update(eqx.filter(model, eqx.is_array), opt_state)

            # aux functions
            def loss_fn(model: eqx.Module, x: jax.Array, y: jax.Array):

                def single_step(i, state):
                    y_pred = jax.vmap(model)(state[0])
                    loss = state[1] + (
                        jnp.mean(jnp.square(y_pred - y[:, i]))
                        / stage_cfg["unrolling_steps"]
                    )
                    return (y_pred, loss)

                _, loss = jax.lax.fori_loop(
                    0, stage_cfg["unrolling_steps"], single_step, (x.copy(), 0)
                )
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
            if i_stage == 0:
                step = 0
                epoch = 0

            for _ in tqdm(range(stage_cfg["epochs"])):
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

                epoch += 1

            if cfg["callbacks"] is None:
                continue

            if "log_model_stage" in cfg["callbacks"]:
                if "log_model_best" in cfg["callbacks"]:
                    model = load_equinox_model(run_id, type(model), "weights-best.eqx")
                log_equinox_model(model, tmp_dir, f"weights-stage-{stage}.eqx")

            if "plot_model_stage" in cfg["callbacks"]:
                if "log_model_best" in cfg["callbacks"]:
                    model = load_equinox_model(run_id, type(model), "weights-best.eqx")
                model_img = os.path.join(tmp_dir, f"model-stage-{stage}.png")
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

    plot_loss(run_id)
