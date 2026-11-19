"""Export the procedural plate and three-box set without building a robot."""

import argparse
from pathlib import Path

import yaml

from demos.cherry_tomato_pick.tableware import Tableware


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cherry_tomato_pick_deposit.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tableware_assets"))
    args = parser.parse_args()
    props = Tableware(yaml.safe_load(args.config.read_text())["tableware"])
    props.export(args.output)
    print(props.asset_directory)


if __name__ == "__main__":
    main()
