import argparse
import yaml

from src.train import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Machine Learned Collision Operators from Particle In Cell Simulations"
    )
    parser.add_argument("cfg", type=str, help="enter path to cfg")
    parser.add_argument("run_id", type=str, help="enter run_id to continue")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    print(cfg)

    if cfg["mode"] == "train":
        train(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
