import os
import numpy as np
import mlflow
import tempfile
import matplotlib.pyplot as plt
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch import optim
from tqdm import tqdm
from torch.utils.data import ConcatDataset, random_split
from typing import Any

from ml_pic_collision_operators.datasets import *
from ml_pic_collision_operators.dataloaders import *
from ml_pic_collision_operators.models import *
from ml_pic_collision_operators.logging import (
    log_torch_model,
    load_torch_model,
    log_torch_state_dict,
    get_metric_history,
)
from ml_pic_collision_operators.utils import class_from_name, class_from_str, rank_print


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


def load_datasets(
    dataset_cls: str | BaseDataset,
    folders,
    temporal_unroll_steps,
    dataset_cls_kwargs: dict[str, Any] = {},
    conditioners: dict[str, Any] | None = None,
) -> list[BaseDataset]:
    dataset_cls = class_from_name("ml_pic_collision_operators.datasets", dataset_cls)

    if conditioners is None:
        datasets = [
            dataset_cls(
                folder=f,
                temporal_unroll_steps=temporal_unroll_steps,
                **dataset_cls_kwargs,
            )
            for f in folders
        ]
    else:
        datasets = [
            dataset_cls(
                folder=f,
                conditioners=c,
                temporal_unroll_steps=temporal_unroll_steps,
                **dataset_cls_kwargs,
            )
            for f, c in zip(folders, conditioners)
        ]

    for i in range(1, len(datasets)):
        # all datasets must share same spatial dimensions and units
        assert datasets[0].grid_size == datasets[i].grid_size
        assert np.equal(datasets[0].grid_range, datasets[i].grid_range).all()
        assert np.equal(datasets[0].grid_dx, datasets[i].grid_dx).all()
        assert datasets[0].grid_units == datasets[i].grid_units

    return datasets


def load_dataloader(
    dataset: BaseDataset,
    dataloader_cls: str | None = None,
    dataloader_cls_kwargs: dict[str, Any] = {},
    device: str | None = None,
) -> BaseDataLoader | list[list[torch.Tensor]]:

    if dataloader_cls is None:
        dataloader = BaseDataLoader(
            dataset,
            batch_size=len(dataset),
        )
        dataloader = next(iter(dataloader))
        dataloader = [[dataloader[i].to(device) for i in range(len(dataloader))]]
    else:
        dataloader_cls = class_from_name(
            "ml_pic_collision_operators.dataloaders", dataloader_cls
        )
        dataloader = dataloader_cls(dataset, **dataloader_cls_kwargs)

    return dataloader


def train_temporal_unrolling(
    cfg, run_id, tmp_dir, mode="accumulated", compile_model=False
):

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

        try:
            conditioners = cfg["data"]["train"]["conditioners"]
        except:
            conditioners = None

        datasets = load_datasets(
            dataset_cls=cfg["dataset_cls"],
            folders=cfg["data"]["train"]["folders"],
            temporal_unroll_steps=stage_cfg["unrolling_steps"],
            dataset_cls_kwargs=cfg["dataset_cls_kwargs"],
            conditioners=conditioners,
        )

        train_datasets = []
        valid_datasets = []
        train_valid_ratio = cfg["data"]["train_valid_ratio"]
        if train_valid_ratio == 1.0:
            train_datasets = datasets
        else:
            for i in range(len(datasets)):
                train_size = int(len(datasets[i]) * train_valid_ratio)
                valid_size = len(datasets[i]) - train_size
                train_portion, valid_portion = random_split(
                    datasets[i], [train_size, valid_size]
                )
                train_datasets.append(train_portion)
                valid_datasets.append(valid_portion)

        train_dataloader = load_dataloader(
            dataset=ConcatDataset(train_datasets),
            dataloader_cls=cfg["dataloader_cls"],
            dataloader_cls_kwargs=cfg["dataloader_cls_kwargs"],
            device=cfg["device"],
        )

        train_dataset_size = np.sum([len(d) for d in train_datasets])
        print("Train Dataset Size:", train_dataset_size)

        # load valid data
        valid_dataloader = []
        valid_dataset_size = 0
        if valid_datasets:
            valid_dataloader = load_dataloader(
                dataset=ConcatDataset(valid_datasets),
                dataloader_cls=cfg["dataloader_cls"],
                dataloader_cls_kwargs=cfg["dataloader_cls_kwargs"],
                device=cfg["device"],
            )

            valid_dataset_size = np.sum([len(d) for d in valid_datasets])
            print("Valid Dataset Size:", valid_dataset_size)

        # actions only done in first stage
        if i_stage == 0:

            # initialize model
            model_kwargs = {
                "grid_size": datasets[0].grid_size,
                "grid_range": datasets[0].grid_range,
                "grid_dx": datasets[0].grid_dx,
                "grid_units": datasets[0].grid_units,
            }

            if conditioners is not None:
                model_kwargs["conditioners_size"] = datasets[0].conditioners_size
                if "normalize_conditioners" in cfg["model_cls_kwargs"]:
                    if cfg["model_cls_kwargs"]["normalize_conditioners"]:
                        c_values = []
                        for i in range(len(datasets)):
                            c_values.append(datasets[i].conditioners_array)
                        c_values = np.stack(c_values, axis=0)
                        model_kwargs["conditioners_min_values"] = np.min(
                            c_values, axis=0
                        )
                        model_kwargs["conditioners_max_values"] = np.max(
                            c_values, axis=0
                        )
            if "include_time" in cfg["dataset_cls_kwargs"]:
                model_kwargs["conditioners_size"] = 1
                model_kwargs["conditioners_min_values"] = np.array([0.0])
                model_kwargs["conditioners_max_values"] = np.array(
                    [datasets[0].dt * len(datasets[0])]
                )
            if cfg["model_cls"] == "FokkerPlanck2DTime_ABparperp":
                model_kwargs["grid_size_dt"] = len(datasets[0])
                model_kwargs["grid_dt"] = datasets[0].dt
                del model_kwargs["conditioners_size"]
                del model_kwargs["conditioners_min_values"]
                del model_kwargs["conditioners_max_values"]

            model_cls = class_from_name(
                "ml_pic_collision_operators.models", cfg["model_cls"]
            )

            if "model_cls_kwargs" in cfg:
                model_kwargs = model_kwargs | cfg["model_cls_kwargs"]
            print("Model Kwargs:", model_kwargs)

            model = model_cls(**model_kwargs)
            model = model.to(cfg["device"])
            mlflow.log_params({"model_kwargs": model_kwargs})

            print("Model:", model)

            if not cfg["dataset_cls_kwargs"].get("include_time", False):
                if conditioners is None:
                    model_img = os.path.join(tmp_dir, f"model-start.png")
                    model.plot(model_img)
                    mlflow.log_artifact(model_img, artifact_path="model_img")
                else:
                    seen_conditioners = []
                    for i in range(len(datasets)):
                        c_str = str(datasets[i].conditioners)
                        if c_str not in seen_conditioners:
                            seen_conditioners.append(c_str)
                            model_img = os.path.join(
                                tmp_dir, f"model-start-dataset-{c_str}.png"
                            )
                            model.plot(
                                torch.Tensor(datasets[i].conditioners_array)
                                .to(cfg["device"])
                                .unsqueeze(0),
                                save_to=model_img,
                            )
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
            if datasets[0].extra_cells != 0:
                error = error[
                    :,
                    datasets[0].extra_cells : -datasets[0].extra_cells,
                    datasets[0].extra_cells : -datasets[0].extra_cells,
                ]
            if cfg["loss_fn"] == "mae":
                loss = torch.mean(torch.abs(error))
            elif cfg["loss_fn"] == "mse":
                loss = torch.mean(torch.square(error))
            return loss

        def loss_accumulated(model, x, y, dt, c=None):
            loss = 0
            y_pred = x.clone()
            for step in range(stage_cfg["unrolling_steps"]):
                if c is None:
                    y_pred = model(y_pred, dt)
                else:
                    y_pred = model(y_pred, dt, c)
                loss += loss_fn(y[:, step], y_pred) / stage_cfg["unrolling_steps"]
            return loss

        def loss_last(model, x, y, dt, c=None):
            y_pred = x.clone()
            for step in range(stage_cfg["unrolling_steps"]):
                if c is None:
                    y_pred = model(y_pred, dt)
                else:
                    y_pred = model(y_pred, dt, c)

            loss = loss_fn(y, y_pred)
            return loss

        def train_step(model, optimizer, batch):

            if mode == "accumulated":
                loss_data = loss_accumulated(model, *batch)
            elif mode == "last":
                loss_data = loss_last(model, *batch)

            loss_reg = 0
            if "reg_first_deriv" in cfg:
                loss_reg += float(cfg["reg_first_deriv"]) * model.get_first_deriv_norm()
            if "reg_second_deriv" in cfg:
                loss_reg += (
                    float(cfg["reg_second_deriv"]) * model.get_second_deriv_norm()
                )
            loss = loss_data + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return model, optimizer, (loss, loss_data, loss_reg)

        def valid_step(model, batch):
            with torch.no_grad():
                if mode == "accumulated":
                    return loss_accumulated(model, *batch)
                elif mode == "last":
                    return loss_last(model, *batch)

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

            min_train_loss_flag = False
            min_valid_loss_flag = False

            for batch in tqdm(
                train_dataloader, leave=False, disable=len(train_dataloader) == 1
            ):
                batch = [b.to(cfg["device"], non_blocking=True) for b in batch]
                # x = x.to(cfg["device"], non_blocking=True)
                # y = y.to(cfg["device"], non_blocking=True)
                model, optimizer, loss = train_step(model, optimizer, batch)
                # split losses
                train_loss_step = loss[0].detach().cpu()
                train_loss_data_step = loss[1].detach().cpu()
                train_loss_reg_step = loss[2]
                # accumulate for epoch
                train_loss += train_loss_step * len(batch[0]) / train_dataset_size
                train_loss_data += (
                    train_loss_data_step * len(batch[0]) / train_dataset_size
                )
                train_loss_reg += (
                    train_loss_reg_step * len(batch[0]) / train_dataset_size
                )
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
            for batch in valid_dataloader:
                valid_loss += (
                    valid_step(model, batch) * len(batch[0]) / valid_dataset_size
                )

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
                min_train_loss = train_loss
                mlflow.log_metric(f"min_train_loss-stage-{stage}", min_train_loss)
                min_train_loss_flag = True
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                mlflow.log_metric(f"min_valid_loss-stage-{stage}", min_valid_loss)
                min_valid_loss_flag = True

            # update epoch value
            epoch += 1

            # do callbacks
            if callbacks is None:
                continue

            if "log_model" in callbacks:
                if epoch % callbacks["log_model"]["frequency"] == 0:
                    log_torch_model(model, tmp_dir, "weights.pth")

            if "log_model_best" in callbacks:
                if (min_valid_loss_flag and valid_dataset_size != 0) or (
                    min_train_loss_flag and valid_dataset_size == 0
                ):
                    if callbacks["log_model_best"]["frequency"] is None:
                        log_torch_model(model, tmp_dir, "weights-best.pth")
                    if callbacks["log_model_best"]["frequency"] == "stage":
                        best_model_dict = model.state_dict().copy()

            if "plot_model" in callbacks and not cfg["dataset_cls_kwargs"].get(
                "include_time", False
            ):
                if epoch % callbacks["plot_model"]["frequency"] == 0:
                    model_img = os.path.join(tmp_dir, f"model-{epoch:06d}.png")
                    model.eval()
                    model.plot(model_img)
                    model.train()
                    mlflow.log_artifact(model_img, artifact_path="model_img")

        if callbacks is None:
            continue

        if "log_model_best" in callbacks:
            if callbacks["log_model_best"]["frequency"] == "stage":
                log_torch_state_dict(
                    model.init_params_dict, best_model_dict, tmp_dir, "weights-best.pth"
                )

        if "log_model_stage" in callbacks:
            if "log_model_best" in callbacks:
                model_aux = load_torch_model(run_id, "weights-best.pth")
            log_torch_model(model_aux, tmp_dir, f"weights-stage-{stage}.pth")

        if "plot_model_stage" in callbacks and not cfg["dataset_cls_kwargs"].get(
            "include_time", False
        ):
            if "log_model_best" in callbacks:
                model_aux = load_torch_model(run_id, "weights-best.pth")
                model_aux.eval()
            if conditioners is None:
                model_img = os.path.join(tmp_dir, f"model-stage-{stage}.png")
                model_aux.plot(model_img)
                mlflow.log_artifact(model_img, artifact_path="model_img")
            else:
                seen_conditioners = []
                for i in range(len(datasets)):
                    c_str = str(datasets[i].conditioners)
                    if c_str not in seen_conditioners:
                        seen_conditioners.append(c_str)
                        model_img = os.path.join(
                            tmp_dir, f"model-stage-{stage}-{c_str}.png"
                        )
                        model_aux.plot(
                            torch.Tensor(datasets[i].conditioners_array).unsqueeze(0),
                            save_to=model_img,
                        )
                        mlflow.log_artifact(model_img, artifact_path="model_img")

    if callbacks is None:
        return

    if "plot_model_end" in callbacks and not cfg["dataset_cls_kwargs"].get(
        "include_time", False
    ):
        if "log_model_best" in callbacks:
            model = load_torch_model(run_id, "weights-best.pth")
            model.eval()
        if conditioners is None:
            model_img = os.path.join(tmp_dir, f"model-final.png")
            model.plot(model_img)
            mlflow.log_artifact(model_img, artifact_path="model_img")
        else:
            seen_conditioners = []
            for i in range(len(datasets)):
                c_str = str(datasets[i].conditioners)
                if c_str not in seen_conditioners:
                    seen_conditioners.append(c_str)
                    model_img = os.path.join(tmp_dir, f"model-final-{c_str}.png")
                    model_aux.plot(
                        torch.Tensor(datasets[i].conditioners_array).unsqueeze(0),
                        save_to=model_img,
                    )
                    mlflow.log_artifact(model_img, artifact_path="model_img")


def train_ddp(
    cfg,
    run_id,
    tmp_dir,
    rank,
    local_rank,
    world_size,
    mode="accumulated",
    compile_model=False,
):

    torch.manual_seed(cfg["random_seed"])  # + rank)
    np.random.seed(cfg["random_seed"])  # + rank)

    dist.barrier(
        device_ids=[
            local_rank,
        ]
    )

    rank_print("Started")

    try:
        callbacks = cfg["callbacks"]
    except:
        callbacks = None

    for i_stage, (stage, stage_cfg) in enumerate(
        cfg["temporal_unrolling_stages"].items()
    ):
        rank_print(f"Stage: {stage} (#{i_stage+1})")
        dist.barrier(
            device_ids=[
                local_rank,
            ]
        )

        # each rank gets a portion of the dataset
        i_folders = np.linspace(0, len(cfg["data"]["train"]["folders"]), world_size + 1)
        i_folders = i_folders.astype(int)
        folders = cfg["data"]["train"]["folders"][i_folders[rank] : i_folders[rank + 1]]
        rank_print(f"Folder Range: [{i_folders[rank]}, {i_folders[rank+1]}]")

        try:
            conditioners = cfg["data"]["train"]["conditioners"][
                i_folders[rank] : i_folders[rank + 1]
            ]
        except:
            conditioners = None

        datasets = load_datasets(
            dataset_cls=cfg["dataset_cls"],
            folders=folders,
            temporal_unroll_steps=stage_cfg["unrolling_steps"],
            dataset_cls_kwargs=cfg["dataset_cls_kwargs"],
            conditioners=conditioners,
        )

        train_datasets = []
        valid_datasets = []
        train_valid_ratio = cfg["data"]["train_valid_ratio"]
        if train_valid_ratio == 1.0:
            train_datasets = datasets
        else:
            for i in range(len(datasets)):
                train_size = int(len(datasets[i]) * train_valid_ratio)
                valid_size = len(datasets[i]) - train_size
                train_portion, valid_portion = random_split(
                    datasets[i], [train_size, valid_size]
                )
                train_datasets.append(train_portion)
                valid_datasets.append(valid_portion)

        train_dataloader = load_dataloader(
            dataset=ConcatDataset(train_datasets),
            dataloader_cls=cfg["dataloader_cls"],
            dataloader_cls_kwargs=cfg["dataloader_cls_kwargs"],
            device=local_rank,
        )

        train_dataset_size = np.sum([len(d) for d in train_datasets])

        total_train_dataset_size = torch.Tensor([train_dataset_size]).to(local_rank)
        dist.all_reduce(total_train_dataset_size, op=dist.ReduceOp.SUM)
        total_train_dataset_size = int(total_train_dataset_size.cpu().numpy()[0])

        rank_print(
            f"Train Dataset Size: {train_dataset_size} (Global: {total_train_dataset_size})"
        )

        # load valid data
        valid_dataloader = []
        valid_dataset_size = 0
        if valid_datasets:
            valid_dataloader = load_dataloader(
                dataset=ConcatDataset(valid_datasets),
                dataloader_cls=cfg["dataloader_cls"],
                dataloader_cls_kwargs=cfg["dataloader_cls_kwargs"],
                device=local_rank,
            )

            valid_dataset_size = np.sum([len(d) for d in valid_datasets])

            total_valid_dataset_size = torch.Tensor([valid_dataset_size]).to(local_rank)
            dist.all_reduce(total_valid_dataset_size, op=dist.ReduceOp.SUM)
            total_valid_dataset_size = int(total_valid_dataset_size.cpu().numpy()[0])

            rank_print(
                f"Valid Dataset Size: {valid_dataset_size} (Global: {total_valid_dataset_size})"
            )

        # actions only done in first stage
        if i_stage == 0:

            # initialize model
            model_kwargs = {
                "grid_size": datasets[0].grid_size,
                "grid_range": datasets[0].grid_range,
                "grid_dx": datasets[0].grid_dx,
                "grid_units": datasets[0].grid_units,
            }

            if conditioners is not None:
                model_kwargs["conditioners_size"] = datasets[0].conditioners_size
                if "normalize_conditioners" in cfg["model_cls_kwargs"]:
                    if cfg["model_cls_kwargs"]["normalize_conditioners"]:
                        c_values = []
                        for i in range(len(datasets)):
                            c_values.append(datasets[i].conditioners_array)
                        c_values = np.stack(c_values, axis=0)
                        conditioners_min_values = np.min(c_values, axis=0)
                        conditioners_max_values = np.max(c_values, axis=0)

                    conditioners_min_values = torch.from_numpy(
                        conditioners_min_values
                    ).to(local_rank)
                    conditioners_max_values = torch.from_numpy(
                        conditioners_max_values
                    ).to(local_rank)

                    dist.all_reduce(conditioners_min_values, op=dist.ReduceOp.MIN)
                    dist.all_reduce(conditioners_max_values, op=dist.ReduceOp.MAX)

                    model_kwargs["conditioners_min_values"] = (
                        conditioners_min_values.cpu().numpy()
                    )
                    model_kwargs["conditioners_max_values"] = (
                        conditioners_max_values.cpu().numpy()
                    )

            model_cls = class_from_name(
                "ml_pic_collision_operators.models", cfg["model_cls"]
            )

            if "model_cls_kwargs" in cfg:
                model_kwargs = model_kwargs | cfg["model_cls_kwargs"]

            model = model_cls(**model_kwargs)
            if compile_model:
                model = torch.compile(model)
            model = model.to(local_rank)
            model = DDP(
                model,
                device_ids=[
                    local_rank,
                ],
            )

            if rank == 0:

                mlflow.log_params({"model_kwargs": model_kwargs})
                print("Model:", model)

                # with torch.no_grad():
                #     if conditioners is None:
                #         model_img = os.path.join(tmp_dir, f"model-start.png")
                #         model.module.plot(model_img)
                #         mlflow.log_artifact(model_img, artifact_path="model_img")
                #     else:
                #         seen_conditioners = []
                #         for i in range(len(datasets)):
                #             c_str = str(datasets[i].conditioners)
                #             if c_str not in seen_conditioners:
                #                 seen_conditioners.append(c_str)
                #                 model_img = os.path.join(
                #                     tmp_dir, f"model-start-dataset-{c_str}.png"
                #                 )
                #                 model.module.plot(
                #                     torch.Tensor(datasets[i].conditioners_array)
                #                     .to(local_rank)
                #                     .unsqueeze(0)
                #                     .detach(),
                #                     save_to=model_img,
                #                 )
                #                 mlflow.log_artifact(
                #                     model_img, artifact_path="model_img"
                #                 )

            # buffer to store best model
            best_model_dict = None

            # initialize optimizer
            if "optimizer_cls" in cfg:
                optimizer_cls = class_from_str(cfg["optimizer_cls"])
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
            if datasets[0].extra_cells != 0:
                error = error[
                    :,
                    datasets[0].extra_cells : -datasets[0].extra_cells,
                    datasets[0].extra_cells : -datasets[0].extra_cells,
                ]
            if cfg["loss_fn"] == "mae":
                loss = torch.mean(torch.abs(error))
            elif cfg["loss_fn"] == "mse":
                loss = torch.mean(torch.square(error))
            return loss

        def loss_accumulated(model, x, y, dt, c=None):
            loss = 0
            y_pred = x.clone()
            for step in range(stage_cfg["unrolling_steps"]):
                if c is None:
                    y_pred = model(y_pred, dt)
                else:
                    y_pred = model(y_pred, dt, c)
                loss += loss_fn(y[:, step], y_pred) / stage_cfg["unrolling_steps"]
            return loss

        def loss_last(model, x, y, dt, c=None):
            y_pred = x.clone()
            for step in range(stage_cfg["unrolling_steps"]):
                if c is None:
                    y_pred = model(y_pred, dt)
                else:
                    y_pred = model(y_pred, dt, c)

            loss = loss_fn(y, y_pred)
            return loss

        def train_step(model, optimizer, batch):

            if mode == "accumulated":
                loss = loss_accumulated(model, *batch)
            elif mode == "last":
                loss = loss_last(model, *batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return model, optimizer, loss

        def valid_step(model, batch):
            with torch.no_grad():
                if mode == "accumulated":
                    return loss_accumulated(model, *batch)
                elif mode == "last":
                    return loss_last(model, *batch)

        # start training
        min_train_loss = np.inf
        min_valid_loss = np.inf

        if i_stage == 0:
            step = 0
            epoch = 0

        dist.barrier(
            # device_ids=[
            #     local_rank,
            # ]
        )

        for _ in tqdm(range(stage_cfg["epochs"]), disable=rank != 0, leave=True):
            # train epoch
            train_loss = 0

            min_train_loss_flag = False
            min_valid_loss_flag = False

            for batch in tqdm(
                train_dataloader,
                leave=False,
                disable=len(train_dataloader) == 1 or rank != 0,
            ):
                batch = [b.to(local_rank, non_blocking=True) for b in batch]
                model, optimizer, loss = train_step(model, optimizer, batch)

                train_loss_step = loss.clone().detach() * len(batch[0])
                total_batch_size = torch.Tensor([len(batch[0])]).to(local_rank)

                dist.all_reduce(train_loss_step, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_batch_size, op=dist.ReduceOp.SUM)

                if rank == 0:
                    # accumulate for epoch
                    train_loss += train_loss_step / total_train_dataset_size

                    # log step loss
                    mlflow.log_metric(
                        "train_loss_step",
                        train_loss_step / total_batch_size,
                        step=step,
                        run_id=run_id,
                    )
                # update step
                step += 1

            # validation epoch
            valid_loss = 0
            for batch in valid_dataloader:
                valid_loss += (
                    valid_step(model, batch) * len(batch[0]) / total_valid_dataset_size
                )

            dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)

            if rank == 0:
                # log epoch loss
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                    },
                    step=epoch,
                )

                # check if we observed minimum loss values
                if train_loss < min_train_loss:
                    min_train_loss = train_loss
                    mlflow.log_metric(f"min_train_loss-stage-{stage}", min_train_loss)
                    min_train_loss_flag = True
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    mlflow.log_metric(f"min_valid_loss-stage-{stage}", min_valid_loss)
                    min_valid_loss_flag = True

            # update epoch value
            epoch += 1

            # do callbacks
            if callbacks is None or rank != 0:
                continue

            if "log_model" in callbacks:
                if epoch % callbacks["log_model"]["frequency"] == 0:
                    log_torch_model(model, tmp_dir, "weights.pth")

            if "log_model_best" in callbacks:
                if (min_valid_loss_flag and total_valid_dataset_size != 0) or (
                    min_train_loss_flag and total_valid_dataset_size == 0
                ):
                    if callbacks["log_model_best"]["frequency"] is None:
                        log_torch_model(model, tmp_dir, "weights-best.pth")
                    if callbacks["log_model_best"]["frequency"] == "stage":
                        if compile_model:
                            best_model_dict = model.module._orig_mod.state_dict().copy()
                        else:
                            best_model_dict = model.module.state_dict().copy()

        if callbacks is None or rank != 0:
            continue

        if "log_model_best" in callbacks:
            if callbacks["log_model_best"]["frequency"] == "stage":
                log_torch_state_dict(
                    model.module.init_params_dict,
                    best_model_dict,
                    tmp_dir,
                    "weights-best.pth",
                )

        if "log_model_stage" in callbacks:
            if "log_model_best" in callbacks:
                model_aux = load_torch_model(run_id, "weights-best.pth")
            log_torch_model(model_aux, tmp_dir, f"weights-stage-{stage}.pth")

        # if "plot_model_stage" in callbacks:
        #     if "log_model_best" in callbacks:
        #         model_aux = load_torch_model(run_id, "weights-best.pth")
        #         model_aux.eval()
        #     if conditioners is None:
        #         model_img = os.path.join(tmp_dir, f"model-stage-{stage}.png")
        #         model_aux.plot(model_img)
        #         mlflow.log_artifact(model_img, artifact_path="model_img")
        #     else:
        #         seen_conditioners = []
        #         for i in range(len(datasets)):
        #             c_str = str(datasets[i].conditioners)
        #             if c_str not in seen_conditioners:
        #                 seen_conditioners.append(c_str)
        #                 model_img = os.path.join(
        #                     tmp_dir, f"model-stage-{stage}-{c_str}.png"
        #                 )
        #                 model_aux.plot(
        #                     torch.Tensor(datasets[i].conditioners_array).unsqueeze(0),
        #                     save_to=model_img,
        #                 )
        #                 mlflow.log_artifact(model_img, artifact_path="model_img")

    if callbacks is None or rank != 0:
        return

    # if "plot_model_end" in callbacks:
    #     if "log_model_best" in callbacks:
    #         model = load_torch_model(run_id, "weights-best.pth")
    #         model.eval()
    #     if conditioners is None:
    #         model_img = os.path.join(tmp_dir, f"model-final.png")
    #         model.plot(model_img)
    #         mlflow.log_artifact(model_img, artifact_path="model_img")
    #     else:
    #         seen_conditioners = []
    #         for i in range(len(datasets)):
    #             c_str = str(datasets[i].conditioners)
    #             if c_str not in seen_conditioners:
    #                 seen_conditioners.append(c_str)
    #                 model_img = os.path.join(tmp_dir, f"model-final-{c_str}.png")
    #                 model_aux.plot(
    #                     torch.Tensor(datasets[i].conditioners_array).unsqueeze(0),
    #                     save_to=model_img,
    #                 )
    #                 mlflow.log_artifact(model_img, artifact_path="model_img")


def train(cfg, run_id, rank, local_rank, world_size, compile_model):

    if rank == 0:
        mlflow.log_params(cfg)

    mode = None
    if cfg["mode"] == "temporal_unrolling":
        mode = "accumulated"
    elif cfg["mode"] == "temporal_unrolling_last":
        mode = "last"
    else:
        raise ValueError(f"Invalid train mode selected: {cfg['mode']}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        if world_size == 1:
            train_temporal_unrolling(cfg, run_id, tmp_dir, mode, compile_model)
        else:
            train_ddp(
                cfg, run_id, tmp_dir, rank, local_rank, world_size, mode, compile_model
            )

    if rank == 0:
        plot_loss(run_id)
        try:
            plot_loss_with_regularization(run_id)
        except Exception as e:
            print(e)
