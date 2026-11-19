#!/usr/bin/env python3
"""Convert a Sharpa LeRobot episode into the named Isaac joint replay format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True,
                        help="LeRobot dataset root containing meta/info.json")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    state_names = info["features"]["observation.state"]["names"]
    positions = [(index, name.removesuffix(".position"))
                 for index, name in enumerate(state_names) if name.endswith(".position")]
    if len(positions) != 58:
        raise ValueError(f"Expected 58 joint positions, found {len(positions)}")
    joint_names = [name.replace("_fr3_joint", "_fr3v2_joint") for _, name in positions]
    if len(set(joint_names)) != 58:
        raise ValueError("Mapped joint names are not unique")

    parquet_path = dataset / info["data_path"].format(
        episode_chunk=args.episode // info["chunks_size"], episode_index=args.episode)
    columns = pq.read_table(parquet_path, columns=["timestamp", "observation.state"]).to_pydict()
    timestamps = np.asarray(columns["timestamp"], dtype=np.float64)
    state = np.asarray(columns["observation.state"], dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != len(state_names):
        raise ValueError(f"Invalid episode state shape {state.shape}")
    if not (np.isfinite(state).all() and np.isfinite(timestamps).all()
            and np.all(np.diff(timestamps) > 0)):
        raise ValueError("Episode contains non-finite states or invalid timestamps")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as episode:
        metadata = episode.create_group("metadata")
        metadata.attrs["extra"] = json.dumps({
            "joint_names": [f"replay/{name}" for name in joint_names],
            "source_dataset": str(dataset), "source_episode": args.episode,
        })
        robot = episode.create_group("robot")
        robot.create_dataset("joint_pos", data=state[:, [i for i, _ in positions]])
        robot.create_dataset("timestamps", data=timestamps)
    print(f"{args.output}: {len(timestamps)} frames, {len(joint_names)} joints, "
          f"{timestamps[-1] - timestamps[0]:.3f} s")


if __name__ == "__main__":
    main()
