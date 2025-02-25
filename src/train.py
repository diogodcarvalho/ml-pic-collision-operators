import os
import mlflow.entities
import numpy as np
import mlflow
import tempfile
import matplotlib.pyplot as plt
import torch

from torch import optim
from tqdm import tqdm
from torch.utils.data import ConcatDataset

from src.datasets import *
from src.dataloaders import *
from src.models import *
from src.logging import (
    log_torch_model,
    load_torch_model,
    log_torch_state_dict,
    get_metric_history,
)
from src.utils import class_from_name, class_from_str


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


def train_temporal_unrolling(cfg, run_id, tmp_dir, mode="accumulated"):

    torch.manual_seed(cfg["random_seed"])
    np.random.seed(cfg["random_seed"])

    try:
        callbacks = cfg["callbacks"]
    except:
        callbacks = None

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

        if "dataloader_cls" not in cfg:
            train_dataloader = BaseDataLoader(
                train_dataset,
                batch_size=len(train_dataset),
            )
            train_dataloader = next(iter(train_dataloader))
            train_dataloader = [
                (
                    train_dataloader[0].to(cfg["device"]),
                    train_dataloader[1].to(cfg["device"]),
                )
            ]
        else:
            dataloader_cls = class_from_str(cfg["dataloader_cls"])
            train_dataloader = dataloader_cls(
                train_dataset, **eval(cfg["dataloader_kwargs"])
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
                "grid_range": train_datasets[0].grid_range,
                "grid_dx": train_datasets[0].grid_dx,
            }

            if "model_cls" in cfg:
                model_cls = class_from_name("src.models", cfg["model_cls"])
            else:
                model_cls = FokkerPlanck2D

            if "model_cls_kwargs" in cfg:
                model_kwargs = model_kwargs | cfg["model_cls_kwargs"]

            model = model_cls(**model_kwargs)
            model = model.to(cfg["device"])
            mlflow.log_params({"model_kwargs": model_kwargs})

            print("Model:", model)

            model_img = os.path.join(tmp_dir, f"model-start.png")
            model.plot(model_img)
            mlflow.log_artifact(model_img, artifact_path="model_img")

            # buffer to store best model
            best_model_dict = None

            # initialize optimizer
            if "optimizer_cls" in cfg:
                optimizer_cls = eval(cfg["optimizer_cls"])
            else:
                optimizer_cls = optim.Adam

            if "optimizer_cls_kwargs" in cfg:
                optimizer_cls_kwargs = cfg["optimizer_cls_kwargs"]
            else:
                optimizer_cls_kwargs = dict()

            if "lr" in stage_cfg:
                optimizer = optimizer_cls(
                    model.parameters(), stage_cfg["lr"], **optimizer_cls_kwargs
                )
            else:
                optimizer = optimizer_cls(model.parameters(), **optimizer_cls_kwargs)

        # actions done in other stages
        else:
            if "lr" in stage_cfg:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = stage_cfg["lr"]

        def loss_fn(y, y_pred):
            error = y - y_pred
            if cfg["loss_fn"] == "mae":
                loss = torch.mean(torch.abs(error))
            elif cfg["loss_fn"] == "mse":
                loss = torch.mean(torch.square(error))
            return loss

        def train_step_accumulated(model, x, y):
            loss = 0
            y_pred = x.clone()
            for step in range(stage_cfg["unrolling_steps"]):
                y_pred = model(y_pred)
                loss += loss_fn(y[:, step], y_pred) / stage_cfg["unrolling_steps"]
            return loss

        def train_step_last(model, x, y):
            y_pred = x.clone()
            for step in range(stage_cfg["unrolling_steps"]):
                y_pred = model(y_pred)
            loss = loss_fn(y, y_pred)
            return loss

        def train_step(model, optimizer, x, y):

            if mode == "accumulated":
                loss_data = train_step_accumulated(model, x, y)
            elif mode == "last":
                loss_data = train_step_last(model, x, y)

            loss_reg = 0
            if "reg_first_deriv" in cfg:
                loss_reg += cfg["reg_first_deriv"] * model.get_first_deriv_norm()
            if "reg_second_deriv" in cfg:
                loss_reg += cfg["reg_second_deriv"] * model.get_second_deriv_norm()
            loss = loss_data + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return model, optimizer, (loss, loss_data, loss_reg)

        def valid_step(model, x, y):
            with torch.no_grad():
                y_pred = model(x)
                loss = loss_fn(y, y_pred)
            return loss

        # start training
        min_train_loss = np.inf
        min_valid_loss = np.inf

        if i_stage == 0:
            step = 0
            epoch = 0

        for _ in tqdm(range(stage_cfg["epochs"]), leave=True):
            # train epoch
            train_loss = 0
            train_loss_data = 0
            train_loss_reg = 0
            metrics_step = []

            min_train_loss_flag = False
            min_valid_loss_flag = False

            for x, y in tqdm(
                train_dataloader, leave=False, disable=len(train_dataloader) == 1
            ):
                # x = x.to(cfg["device"], non_blocking=True)
                # y = y.to(cfg["device"], non_blocking=True)

                model, optimizer, loss = train_step(model, optimizer, x, y)
                # split losses
                train_loss_step = loss[0].detach().cpu()
                train_loss_data_step = loss[1].detach().cpu()
                train_loss_reg_step = loss[2]
                # accumulate for epoch
                train_loss += train_loss_step * len(x) / len(train_dataset)
                train_loss_data += train_loss_data_step * len(x) / len(train_dataset)
                train_loss_reg += train_loss_reg_step * len(x) / len(train_dataset)
                # log step loss
                mlflow.log_metrics(
                    {
                        "train_loss_step": train_loss_step,
                        "train_loss_data_step": train_loss_data_step,
                        "train_loss_reg_step": train_loss_reg_step,
                    },
                    step=step,
                    run_id=run_id,
                )
                # update step
                step += 1

            # validation epoch
            valid_loss = 0
            for x, y in valid_dataloader:
                valid_loss += valid_step(model, x, y) * len(x) / len(valid_dataset)

            # log epoch loss
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_loss_data": train_loss_data,
                    "train_loss_reg": train_loss_reg,
                    "valid_loss": valid_loss,
                },
                step=epoch,
            )

            # check if we observed minimum loss values
            if train_loss < min_train_loss:
                mlflow.log_metric(f"min_train_loss-stage-{stage}", min_train_loss)
                min_train_loss_flag = True
                min_train_loss = train_loss
            if valid_loss < min_valid_loss:
                mlflow.log_metric(f"min_valid_loss-stage-{stage}", min_valid_loss)
                min_valid_loss_flag = True
                min_valid_loss = valid_loss

            # update epoch value
            epoch += 1

            # do callbacks
            if callbacks is None:
                continue

            if "log_model" in callbacks:
                if epoch % callbacks["log_model"]["frequency"] == 0:
                    log_torch_model(model, tmp_dir, "weights.pth")

            if "log_model_best" in callbacks:
                if min_train_loss_flag:
                    if callbacks["log_model_best"]["frequency"] is None:
                        log_torch_model(model, tmp_dir, "weights-best.pth")
                    if callbacks["log_model_best"]["frequency"] == "stage":
                        best_model_dict = model.state_dict().copy()

            if "plot_model" in callbacks:
                if epoch % callbacks["plot_model"]["frequency"] == 0:
                    model_img = os.path.join(tmp_dir, f"model-{epoch:06d}.png")
                    model.plot(model_img)
                    mlflow.log_artifact(model_img, artifact_path="model_img")

        if callbacks is None:
            continue

        if "log_model_best" in callbacks:
            if callbacks["log_model_best"]["frequency"] == "stage":
                log_torch_state_dict(best_model_dict, tmp_dir, "weights-best.pth")

        if "log_model_stage" in callbacks:
            if "log_model_best" in callbacks:
                model_aux = load_torch_model(run_id, "weights-best.pth")
            log_torch_model(model_aux, tmp_dir, f"weights-stage-{stage}.pth")

        if "plot_model_stage" in callbacks:
            if "log_model_best" in callbacks:
                model_aux = load_torch_model(run_id, "weights-best.pth")
            model_img = os.path.join(tmp_dir, f"model-stage-{stage}.png")
            model_aux.plot(model_img)
            mlflow.log_artifact(model_img, artifact_path="model_img")

    if callbacks is None:
        return

    if "plot_model_end" in callbacks:
        if "log_model_best" in callbacks:
            model = load_torch_model(run_id, "weights-best.pth")
        model_img = os.path.join(tmp_dir, f"model-final.png")
        model.plot(model_img)
        mlflow.log_artifact(model_img, artifact_path="model_img")


def train(cfg, run_id):

    mlflow.log_params(cfg)

    with tempfile.TemporaryDirectory() as tmp_dir:
        if cfg["mode"] == "temporal_unrolling":
            train_temporal_unrolling(cfg, run_id, tmp_dir, mode="accumulated")
        elif cfg["mode"] == "temporal_unrolling_last":
            train_temporal_unrolling(cfg, run_id, tmp_dir, mode="last")

    plot_loss(run_id)
    try:
        plot_loss_with_regularization(run_id)
    except Exception as e:
        print(e)
