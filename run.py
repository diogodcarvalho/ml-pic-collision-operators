import argparse
import mlflow
import yaml
import torch
import os

from src.train import train
from src.test import test
from src.logging import get_existing_run_id
from src.utils import (
    setup_distributed,
    root_print,
    rank_print,
    cleanup_ddp,
)

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Machine Learned Collision Operators from Particle In Cell Simulations"
    )
    parser.add_argument("cfg", type=str, help="path to cfg")
    parser.add_argument("experiment_name", type=str, help="MLFlow experiment name")
    parser.add_argument("run_name", type=str, help="MLFlow run name")
    parser.add_argument(
        "--run_overwrite",
        action="store_true",
        help="overwrite existing MLFlow with same name",
    )
    parser.add_argument(
        "--single_precision", action="store_true", help="use single precision"
    )
    parser.add_argument(
        "--mlflow_dir",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/mlruns",
        help="folder where MLFlow database is stored",
    )
    parser.add_argument(
        "--force_not_ddp",
        action="store_true",
        help="turn off ddp initialization even if in distributed environment",
    )
    parser.add_argument(
        "--compile_model", action="store_true", help="if True, jit compiles torch model"
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.force_not_ddp:
        rank = 0
        local_rank = 0
        world_size = 1
    else:
        rank, local_rank, world_size = setup_distributed()

    if world_size != 1:
        rank_print(
            f"Rank: {rank} \t Local Rank: {local_rank} \t World Size: {world_size}"
        )

    root_print("-" * 40)
    root_print("Input Args")
    root_print("-" * 40)
    for var in vars(args):
        root_print(f"{var}: {getattr(args, var)}")
    root_print()

    mlflow.set_tracking_uri(f"file:{args.mlflow_dir}")

    if args.single_precision:
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)

    root_print("-" * 40)
    root_print("Config")
    root_print("-" * 40)
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise Exception(f"Empty config file provided: {args.cfg}")
    root_print(yaml.dump(cfg, indent=2, sort_keys=False, default_flow_style=False))

    root_print("-" * 40)
    root_print("MLFlow")
    root_print("-" * 40)

    if rank == 0:
        # create new experiment or load existing
        mlflow.set_experiment(args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)

        # check if run with same name already exists
        # if it exists, raises an exception if run_overwrite is not True
        try:
            run_id = get_existing_run_id(args.experiment_name, args.run_name)
            if args.run_overwrite:
                root_print("Overwriting existing run")
                root_print(f"experiment_name: {args.experiment_name}")
                root_print(f"experiment_id: {experiment.experiment_id}")
                root_print(f"run_name: {args.run_name}")
                root_print(f"run_id: {run_id}")
            else:
                raise Exception(
                    "Previous run with the same name already exists in this experiment. "
                    + "Change run_name or set --run_overwrite to overwrite."
                )
        except:
            run_id = None
            root_print("New run")
            root_print(f"experiment_name: {args.experiment_name}")
            root_print(f"experiment_id: {experiment.experiment_id}")
            root_print(f"run_name: {args.run_name}")

        mlflow.start_run(
            run_id=run_id,
            run_name=args.run_name,
            experiment_id=experiment.experiment_id,
            nested=True,
        )
        root_print("Run started OK")
        root_print()

        run_id = mlflow.active_run().info.run_id
    else:
        run_id = None

    if cfg["mode"] == "train":
        root_print("-" * 40)
        root_print("Train")
        root_print("-" * 40)
        train(cfg["train"], run_id, rank, local_rank, world_size, args.compile_model)

    elif cfg["mode"] == "test":
        root_print("-" * 40)
        root_print("Test")
        root_print("-" * 40)
        test(cfg["test"], run_id)

    elif cfg["mode"] == "pred":
        raise NotImplementedError

    else:
        raise NotImplementedError

    if rank == 0:
        mlflow.end_run()
        root_print("Finished OK")

    if not args.force_not_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
