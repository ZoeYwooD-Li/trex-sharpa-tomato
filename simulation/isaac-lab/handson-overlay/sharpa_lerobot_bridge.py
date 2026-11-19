"""Read real dual-FR3/Sharpa LeRobot episodes for an Isaac task.

This module does not retarget Wuji joints or invent tactile calibration.  A
simulator must publish the named dual-FR3/Sharpa state and action contracts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TACTILE_SHAPE = (2, 5, 240, 240)
WRENCH_SHAPE = (2, 5, 6)


@dataclass(frozen=True)
class SharpaDataset:
    root: Path
    info: dict
    episodes: tuple[dict, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]

    def episode_file(self, episode_index: int) -> Path:
        chunk = episode_index // int(self.info["chunks_size"])
        rel = self.info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
        return self.root / rel

    def video_file(self, episode_index: int, camera: str) -> Path:
        key = f"observation.images.{camera}"
        if key not in self.info["features"]:
            raise KeyError(f"Camera {camera!r} is absent from {self.root}")
        chunk = episode_index // int(self.info["chunks_size"])
        rel = self.info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=key)
        return self.root / rel


@dataclass(frozen=True)
class SharpaBatch:
    state: np.ndarray                 # [B,116] float32
    action: np.ndarray                # [B,58] float32
    deformation: np.ndarray           # [B,2,5,240,240] uint8
    wrench: np.ndarray                # [B,2,5,6] float32
    arm_engaged: np.ndarray           # [B,2] float32
    frame_index: np.ndarray           # [B] int64
    source_timestamp_ns: np.ndarray   # [B] int64

    def as_policy_observation(self) -> dict[str, np.ndarray]:
        """Use common policy keys without labeling real measurements as PhysX."""
        return {
            "state": self.state,
            "deformation": self.deformation,
            "wrench": self.wrench,
            "arm_engaged": self.arm_engaged,
        }


def discover_datasets(root: Path) -> list[SharpaDataset]:
    root = Path(root).expanduser().absolute()
    candidates = [root] if (root / "meta/info.json").is_file() else sorted(root.glob("[AB]/*"))
    datasets = [load_dataset(p) for p in candidates if (p / "meta/info.json").is_file()]
    if not datasets:
        raise FileNotFoundError(f"No LeRobot dataset under {root}")
    reference = datasets[0]
    for dataset in datasets[1:]:
        if dataset.state_names != reference.state_names or dataset.action_names != reference.action_names:
            raise ValueError(f"Incompatible state/action schema: {dataset.root}")
    return datasets


def load_dataset(root: Path) -> SharpaDataset:
    root = Path(root).expanduser().absolute()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("robot_type") != "dual_fr3_sharpa":
        raise ValueError(f"Expected dual_fr3_sharpa, got {info.get('robot_type')!r}: {root}")
    features = info["features"]
    state_names = tuple(features["observation.state"]["names"])
    action_names = tuple(features["action"]["names"])
    if len(state_names) != 116 or len(action_names) != 58:
        raise ValueError(f"Expected 116 state and 58 action names: {root}")
    if len(set(state_names)) != 116 or len(set(action_names)) != 58:
        raise ValueError(f"Duplicate state or action names: {root}")
    if tuple(features["observation.state"]["shape"]) != (116,) or tuple(features["action"]["shape"]) != (58,):
        raise ValueError(f"Incorrect state/action shape: {root}")
    for side in SIDES:
        for finger in FINGERS:
            prefix = f"observation.tactile.{side}.{finger}"
            if tuple(features[f"{prefix}.deformation"]["shape"]) != (240, 240):
                raise ValueError(f"Incorrect deformation shape for {prefix}: {root}")
            if tuple(features[f"{prefix}.wrench"]["shape"]) != (6,):
                raise ValueError(f"Incorrect wrench shape for {prefix}: {root}")
    episodes = tuple(json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line)
    if len(episodes) != info["total_episodes"] or sum(e["length"] for e in episodes) != info["total_frames"]:
        raise ValueError(f"Episode metadata count mismatch: {root}")
    dataset = SharpaDataset(root, info, episodes, state_names, action_names)
    for episode in episodes:
        if not dataset.episode_file(episode["episode_index"]).is_file():
            raise FileNotFoundError(dataset.episode_file(episode["episode_index"]))
    return dataset


def _fixed_array(column, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Flatten Arrow fixed-size lists without building Python lists of pixels."""
    if column.null_count:
        raise ValueError("Null values in required LeRobot column")
    values = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    for _ in shape:
        values = values.flatten()
        if values.null_count:
            raise ValueError("Null values inside LeRobot fixed-size list")
    result = np.asarray(values.to_numpy(zero_copy_only=False), dtype=dtype)
    return result.reshape((len(column), *shape))


def iter_episode_batches(dataset: SharpaDataset, episode_index: int, batch_size: int = 2) -> Iterator[SharpaBatch]:
    """Stream one Parquet episode in bounded Arrow batches, including all ten sensors."""
    import pyarrow.parquet as pq

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tactile_keys = [f"observation.tactile.{side}.{finger}" for side in SIDES for finger in FINGERS]
    columns = ["observation.state", "action", "observation.arm_engaged", "frame_index", "source_timestamp_ns"]
    columns += [f"{key}.{kind}" for key in tactile_keys for kind in ("deformation", "wrench")]
    parquet = pq.ParquetFile(dataset.episode_file(episode_index))
    for record in parquet.iter_batches(batch_size=batch_size, columns=columns):
        fields = {name: record.column(i) for i, name in enumerate(record.schema.names)}
        deformation = np.stack([_fixed_array(fields[f"{key}.deformation"], (240, 240), np.uint8) for key in tactile_keys], axis=1)
        wrench = np.stack([_fixed_array(fields[f"{key}.wrench"], (6,), np.float32) for key in tactile_keys], axis=1)
        yield SharpaBatch(
            state=_fixed_array(fields["observation.state"], (116,), np.float32),
            action=_fixed_array(fields["action"], (58,), np.float32),
            deformation=deformation.reshape(-1, *TACTILE_SHAPE),
            wrench=wrench.reshape(-1, *WRENCH_SHAPE),
            arm_engaged=_fixed_array(fields["observation.arm_engaged"], (2,), np.float32),
            frame_index=np.asarray(fields["frame_index"].to_numpy(), dtype=np.int64),
            source_timestamp_ns=np.asarray(fields["source_timestamp_ns"].to_numpy(), dtype=np.int64),
        )


def _reorder_named(values: np.ndarray, source_names: Sequence[str], target_names: Sequence[str], label: str) -> np.ndarray:
    source_names, target_names = tuple(source_names), tuple(target_names)
    if len(source_names) != len(target_names) or set(source_names) != set(target_names):
        missing = sorted(set(target_names) - set(source_names))
        extra = sorted(set(source_names) - set(target_names))
        raise ValueError(f"{label} joint contract mismatch; missing={missing[:8]}, extra={extra[:8]}")
    array = np.asarray(values)
    if array.shape[-1] != len(source_names):
        raise ValueError(f"{label} last dimension {array.shape[-1]} != {len(source_names)}")
    lookup = {name: i for i, name in enumerate(source_names)}
    return array[..., [lookup[name] for name in target_names]]


def action_to_sim(action: np.ndarray, dataset: SharpaDataset, sim_action_names: Sequence[str]) -> np.ndarray:
    """Reorder real commands for an exact Sharpa joint contract; reject Wuji."""
    return _reorder_named(action, dataset.action_names, sim_action_names, "action")


def mjcf_joint_to_real_action_name(joint_name: str) -> str:
    """Name the physical FR3/Sharpa joint in the LeRobot 58-action schema."""
    arm = re.fullmatch(r"(left|right)_fr3v2_joint([1-7])", joint_name)
    if arm:
        return f"{arm.group(1)}_fr3_joint{arm.group(2)}.position"
    if re.fullmatch(r"(left|right)_(thumb|index|middle|ring|pinky)_[A-Za-z0-9_]+", joint_name):
        return f"{joint_name}.command"
    raise ValueError(f"Unknown dual-FR3/Sharpa physical joint: {joint_name}")


def action_to_mjcf_joint_order(
    action: np.ndarray, dataset: SharpaDataset, mjcf_joint_names: Sequence[str]
) -> np.ndarray:
    """Reorder the 58 values; command-to-angle calibration is a separate step."""
    names = tuple(mjcf_joint_to_real_action_name(name) for name in mjcf_joint_names)
    return action_to_sim(action, dataset, names)


def sim_observation_to_policy(
    observation: Mapping[str, np.ndarray], dataset: SharpaDataset, sim_state_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Normalize an Isaac observation to the real-data key/order contract."""
    state = _reorder_named(np.asarray(observation["state"]), sim_state_names, dataset.state_names, "state")
    deformation = np.asarray(observation["physx_tactile_deformation"])
    wrench = np.asarray(observation["physx_tactile_wrench"])
    if deformation.shape[-4:] != TACTILE_SHAPE or deformation.dtype != np.uint8:
        raise ValueError(f"Expected uint8 tactile maps (...,{TACTILE_SHAPE}), got {deformation.shape} {deformation.dtype}")
    if wrench.shape[-3:] != WRENCH_SHAPE:
        raise ValueError(f"Expected tactile wrench (...,{WRENCH_SHAPE}), got {wrench.shape}")
    if state.shape[:-1] != deformation.shape[:-4] or state.shape[:-1] != wrench.shape[:-3]:
        raise ValueError("State and tactile batch dimensions differ")
    return {"state": state.astype(np.float32, copy=False), "deformation": deformation,
            "wrench": wrench.astype(np.float32, copy=False)}
