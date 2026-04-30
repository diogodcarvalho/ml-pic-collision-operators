import argparse
import mlflow
import yaml
import torch

from ml_pic_collision_operators.config.schema import load_config
from ml_pic_collision_operators.train import train
from ml_pic_collision_operators.test import test
from ml_pic_collision_operators.logging_utils import (
    configure_mlflow_experiment,
    get_mlflow_run_id,
)
from ml_pic_collision_operators.utils import (
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
    parser.add_argument("cfg", type=str, help="Path to YAML configuration file.")
    parser.add_argument("experiment_name", type=str, help="MLflow experiment name")
    parser.add_argument("run_name", type=str, help="MLflow run name")
    parser.add_argument(
        "mlflow_dir",
        type=str,
        help="Folder where MLflow database is stored.",
    )
    parser.add_argument(
        "--run_overwrite",
        action="store_true",
        help="Overwrite existing MLflow with same name",
    )
    parser.add_argument(
        "--single_precision", action="store_true", help="Use single precision"
    )
    parser.add_argument(
        "--force_not_ddp",
        action="store_true",
        help="Turn off ddp initialization even if in distributed environment",
    )
    parser.add_argument(
        "--compile_model", action="store_true", help="Use PyTorch JIT compilation"
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

    if args.single_precision:
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)

    root_print("-" * 40)
    root_print("Config")
    root_print("-" * 40)
    cfg, cfg_yaml = load_config(args.cfg)
    root_print(yaml.dump(cfg_yaml, indent=2, sort_keys=False, default_flow_style=False))

    root_print("-" * 40)
    root_print("MLflow")
    root_print("-" * 40)

    if rank == 0:
        # Create new experiment or load existing
        experiment = configure_mlflow_experiment(args.mlflow_dir, args.experiment_name)

        # Check if run with same name already exists
        # If it exists, raises an exception if run_overwrite is not True
        try:
            run_id = get_mlflow_run_id(args.experiment_name, args.run_name)
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
        except ValueError:
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
        root_print("MLFlow run started OK")
        root_print()

        run_id = mlflow.active_run().info.run_id
    else:
        run_id = None

    if cfg.mode == "train":
        root_print("-" * 40)
        root_print("Train")
        root_print("-" * 40)
        if rank == 0:
            mlflow.log_params(cfg_yaml["train"])
        train(cfg.train, run_id, rank, local_rank, world_size, args.compile_model)

    elif cfg.mode == "test":
        root_print("-" * 40)
        root_print("Test")
        root_print("-" * 40)
        mlflow.log_params(cfg_yaml["test"])
        test(cfg.test, run_id)
    else:
        raise NotImplementedError(
            f"Mode not implemented: {cfg.mode} (valid modes: train, test)"
        )

    if rank == 0:
        mlflow.end_run()
        root_print("-" * 40)
        root_print("MLflow run finished OK")

    if not args.force_not_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
