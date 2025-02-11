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
from src.utils import class_from_name, str_to_class


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


def plot_loss_with_regularization(run_id):

    epochs, train_loss = get_metric_history("train_loss", run_id)
    epochs, train_loss_data = get_metric_history("train_loss_data", run_id)
    epochs, train_loss_reg = get_metric_history("train_loss_reg", run_id)
    _, valid_loss = get_metric_history("valid_loss", run_id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        plt.figure(figsize=(5, 4))
        plt.plot(epochs, train_loss, label="Train")
        plt.plot(epochs, train_loss_data, label="Train-Data")
        plt.plot(epochs, train_loss_reg, label="Train-Reg")
        if not np.all(valid_loss == 0):
            plt.plot(epochs, valid_loss, label="Valid")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xlim(0, epochs[-1])
        plt.legend()
        plt.tight_layout()
        fname = os.path.join(tmp_dir, "loss_w_reg.png")
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
        if cfg["loss_fn"] == "mae":
            loss = jnp.mean(jnp.square(y_pred - y))
        elif cfg["loss_fn"] == "mse":
            loss = jnp.mean(jnp.square(y_pred - y))
        return loss

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


def train_temporal_unrolling(cfg, run_id, mode="accumulated"):

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
                    temporal_unroll_steps=stage_cfg["unrolling_steps"],
                    **cfg["dataset"]["cls_kwargs"],
                )
                for folder in cfg["dataset"]["train"]["folders"]
            ]

            train_dataset = ConcatDataset(train_datasets)

            train_dataloader = BaseDataLoader(
                train_dataset,
                batch_size=len(train_dataset),
            )
            # for now put all dataset into memory
            # this is silly as is, but MUUUUUCH faster
            train_dataloader = [next(iter(train_dataloader))]

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
                    "grid_range": train_datasets[0].grid_range,
                    "grid_dx": train_datasets[0].grid_dx,
                }

                if "model_cls_kwargs" in cfg:
                    model_kwargs = model_kwargs | cfg["model_cls_kwargs"]

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
            if mode == "accumulated":

                def loss_fn(model: eqx.Module, x: jax.Array, y: jax.Array):

                    def single_step(i, state):
                        y_pred = jax.vmap(model)(state[0])
                        if cfg["loss_fn"] == "mae":
                            loss = state[1] + (
                                jnp.mean(jnp.abs(y_pred - y[:, i]))
                                / stage_cfg["unrolling_steps"]
                            )
                        elif cfg["loss_fn"] == "mse":
                            loss = state[1] + (
                                jnp.mean(jnp.square(y_pred - y[:, i]))
                                / stage_cfg["unrolling_steps"]
                            )
                        return (y_pred, loss)

                    _, loss_data = jax.lax.fori_loop(
                        0, stage_cfg["unrolling_steps"], single_step, (x.copy(), 0)
                    )

                    if "reg_first_deriv" in cfg:
                        loss_reg = cfg["reg_first_deriv"] * model.get_first_deriv_norm()
                        loss = loss_data + loss_reg
                    else:
                        loss_reg = 0
                        loss = loss_data

                    return loss, (loss_data, loss_reg)

            elif mode == "last":

                def loss_fn(model: eqx.Module, x: jax.Array, y: jax.Array):

                    def single_step(i, state):
                        y_pred = jax.vmap(model)(state)
                        return y_pred

                    y_pred = jax.lax.fori_loop(
                        0, stage_cfg["unrolling_steps"], single_step, x.copy()
                    )

                    if cfg["loss_fn"] == "mae":
                        loss_data = jnp.mean(jnp.abs(y_pred - y[:, -1]))
                    elif cfg["loss_fn"] == "mse":
                        loss_data = jnp.mean(jnp.squared(y_pred - y[:, -1]))

                    if "reg_first_deriv" in cfg:
                        loss_reg = cfg["reg_first_deriv"] * model.get_first_deriv_norm()
                        loss = loss_data + loss_reg
                    else:
                        loss_reg = 0
                        loss = loss_data

                    return loss, (loss_data, loss_reg)

            @eqx.filter_jit
            def train_step(
                model: eqx.Module, opt_state: PyTree, x: jax.Array, y: jax.Array
            ):

                loss, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                    model, x, y
                )
                updates, opt_state = optim.update(
                    grads, opt_state, eqx.filter(model, eqx.is_array)
                )
                model = eqx.apply_updates(model, updates)
                return model, opt_state, loss

            @eqx.filter_jit
            def valid_step(model: eqx.Module, x: jax.Array, y: jax.Array):
                valid_loss, _ = loss_fn(model, x, y)
                return valid_loss

            # start training
            min_loss = jnp.inf
            if i_stage == 0:
                step = 0
                epoch = 0

            for _ in tqdm(range(stage_cfg["epochs"])):
                # train epoch
                train_loss = 0
                train_loss_data = 0
                train_loss_reg = 0
                for x, y in train_dataloader:
                    model, opt_state, loss = train_step(model, opt_state, x, y)
                    # split losses
                    train_loss_step = loss[0]
                    train_loss_data_step = loss[1][0]
                    train_loss_reg_step = loss[1][1]
                    # accumulate for epoch
                    train_loss += train_loss_step * len(x) / len(train_dataset)
                    train_loss_data += (
                        train_loss_data_step * len(x) / len(train_dataset)
                    )
                    train_loss_reg += train_loss_reg_step * len(x) / len(train_dataset)
                    # log step loss
                    mlflow.log_metric("train_loss_step", train_loss_step, step=step)
                    mlflow.log_metric(
                        "train_loss_data_step", train_loss_data_step, step=step
                    )
                    mlflow.log_metric(
                        "train_loss_reg_step", train_loss_reg_step, step=step
                    )
                    # update step
                    step += 1

                # log epoch loss
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_loss_data", train_loss_data, step=epoch)
                mlflow.log_metric("train_loss_reg", train_loss_reg, step=epoch)

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
                    if train_loss <= min_loss:
                        min_loss = train_loss
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
        train_temporal_unrolling(cfg, run_id, mode="accumulated")
    elif cfg["mode"] == "temporal_unrolling_last":
        train_temporal_unrolling(cfg, run_id, mode="last")

    plot_loss(run_id)
    try:
        plot_loss_with_regularization(run_id)
    except Exception as e:
        print(e)
