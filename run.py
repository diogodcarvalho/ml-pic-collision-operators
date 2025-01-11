import argparse
import mlflow
import yaml

from src.train import train
from src.test import test
from src.logging import get_existing_run_id


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
    args = parser.parse_args()
    print("-" * 40)
    print("Input Args")
    print("-" * 40)
    for var in vars(args):
        print(f"{var}: {getattr(args, var)}")
    print()
    return args


def main():
    args = parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    print("-" * 40)
    print("Config")
    print("-" * 40)
    print(yaml.dump(cfg, indent=2, sort_keys=False, default_flow_style=False))

    print("-" * 40)
    print("MLFlow")
    print("-" * 40)
    # create new experiment or load existing
    mlflow.set_experiment(args.experiment_name)
    experiment = mlflow.get_experiment_by_name(args.experiment_name)

    # check if run with same name already exists
    # if it exists, raises an exception if run_overwrite is not True
    run_id = get_existing_run_id(args.experiment_name, args.run_name)
    if run_id is None:
        print("New run")
        print(f"experiment_name: {args.experiment_name}")
        print(f"experiment_id: {experiment.experiment_id}")
        print(f"run_name: {args.run_name}")
    else:
        if args.run_overwrite:
            print("Overwriting existing run")
            print(f"experiment_name: {args.experiment_name}")
            print(f"experiment_id: {experiment.experiment_id}")
            print(f"run_name: {args.run_name}")
            print(f"run_id: {run_id}")
        else:
            raise Exception(
                "Previous run with the same name already exists in this experiment. "
                + "Change run_name or set --run_overwrite to overwrite."
            )

    with mlflow.start_run(
        run_id=run_id,
        run_name=args.run_name,
        experiment_id=experiment.experiment_id,
        nested=True,
    ):
        print("run started OK")
        print()

        if cfg["mode"] == "train":
            print("-" * 40)
            print("Train")
            print("-" * 40)
            train(cfg)

        elif cfg["mode"] == "test":
            print("-" * 40)
            print("Test")
            print("-" * 40)
            test(cfg)

        elif cfg["mode"] == "pred":
            raise NotImplementedError

        else:
            raise NotImplementedError

        print("Finished OK")


if __name__ == "__main__":
    main()
