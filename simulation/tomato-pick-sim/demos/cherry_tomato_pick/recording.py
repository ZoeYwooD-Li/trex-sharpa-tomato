"""Multi-rate HDF5 episode recording for cherry-tomato harvesting."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from loguru import logger
import numpy as np
from scipy.spatial.transform import Rotation

from lycheesim.training import (
    CameraSpec,
    EpisodeFrame,
)
from .quality import assess_episode


DATASET_SCHEMA = "lycheesim.cherry_tomato_pick.multirate/v1"


def next_episode_path(path: str | Path) -> Path:
    """Return the next numbered path after existing matching episodes."""

    requested = Path(path).expanduser()
    match = re.fullmatch(r"(.*?)(\d+)", requested.stem)
    if match is None:
        if not requested.exists():
            return requested
        prefix = f"{requested.stem}_"
        first_index = 1
        width = 6
    else:
        prefix, digits = match.groups()
        first_index = int(digits)
        width = len(digits)
    filename_pattern = re.compile(
        rf"{re.escape(prefix)}(\d+){re.escape(requested.suffix)}"
    )
    highest_index = first_index - 1
    if requested.parent.exists():
        for existing in requested.parent.iterdir():
            existing_match = filename_pattern.fullmatch(existing.name)
            if existing_match is not None:
                highest_index = max(highest_index, int(existing_match.group(1)))
    index = max(first_index, highest_index + 1)
    return requested.with_name(f"{prefix}{index:0{width}d}{requested.suffix}")


def world_from_pose(
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Return a homogeneous world-from-local transform from an ``xyzw`` pose."""

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        np.asarray(quaternion_xyzw, dtype=np.float64)
    ).as_matrix()
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


@dataclass(frozen=True)
class _CameraFrame:
    time: float
    color: np.ndarray
    depth_m: np.ndarray
    world_from_camera: np.ndarray
    valid_depth_pixels: int


class CherryTomatoEpisodeRecorder:
    """Collect independently timestamped robot and RGB-D streams."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        seed: int,
        object_ids: tuple[str, ...],
        cameras: Mapping[str, CameraSpec],
        metadata: Mapping[str, Any],
        overwrite: bool = False,
        auto_increment: bool = True,
    ) -> None:
        requested_path = Path(output_path).expanduser()
        if requested_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("recording.output_path must end in .h5 or .hdf5")
        self.output_path = requested_path
        if not overwrite and auto_increment:
            self.output_path = next_episode_path(requested_path)
            if self.output_path != requested_path:
                logger.info(
                    "existing episode sequence found; recording next as {}",
                    self.output_path,
                )
        elif requested_path.exists() and not overwrite:
            raise FileExistsError(
                f"episode already exists: {requested_path}; enable "
                "recording.auto_increment, choose another "
                "recording.output_path, or set recording.overwrite=true"
            )
        self.seed = int(seed)
        self.object_ids = tuple(object_ids)
        self.cameras = dict(cameras)
        self.metadata = dict(metadata)
        self.quality_report: dict[str, Any] | None = None
        self._quality_enabled = bool(self.metadata.get("quality_filter", {}).get("enabled", False))
        if self._quality_enabled:
            import subprocess
            from hashlib import sha256
            root = Path(__file__).resolve().parents[2]
            try:
                revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
                status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
                diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=root)
                self.metadata["source"] = {"git_revision": revision, "working_tree_dirty": bool(status),
                                           "tracked_diff_sha256": sha256(diff).hexdigest()}
                self.metadata["source"]["demo_python_sha256"] = {
                    str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
                }
            except (OSError, subprocess.CalledProcessError):
                self.metadata["source"] = {"git_revision": None, "working_tree_dirty": None}
            # A standalone extraction need not have its own Git history. Record
            # the copied-source manifest and current code hashes in either case.
            snapshot = root / "provenance/source_snapshot.json"
            if snapshot.is_file():
                source_snapshot = json.loads(snapshot.read_text(encoding="utf-8"))
                self.metadata["source"]["extraction"] = {
                    "source_commit": source_snapshot["source_commit"],
                    "source_working_tree_status": source_snapshot["source_working_tree_status"],
                    "manifest_sha256": sha256(snapshot.read_bytes()).hexdigest(),
                }
            self.metadata["source"]["demo_python_sha256"] = {
                str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
            }
            threshold = float(self.metadata["quality_filter"].get("min_valid_depth_fraction", .001))
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("quality_filter.min_valid_depth_fraction must be in [0, 1]")
        self._robot_frames: list[EpisodeFrame] = []
        self._camera_frames: dict[str, list[_CameraFrame]] = {
            name: [] for name in self.cameras
        }
        self._actions: list[np.ndarray] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._capture_finished = False
        self._saved = False
        self._overwrite = bool(overwrite)

    @property
    def frame_count(self) -> int:
        """Return the number of robot samples for compatibility."""

        return self.robot_frame_count

    @property
    def robot_frame_count(self) -> int:
        return len(self._robot_frames)

    @property
    def camera_frame_counts(self) -> dict[str, int]:
        return {
            name: len(frames) for name, frames in self._camera_frames.items()
        }

    @property
    def capture_finished(self) -> bool:
        return self._capture_finished

    def append_robot(
        self,
        frame: EpisodeFrame,
        *,
        transition_action: np.ndarray | None = None,
        terminated: bool = False,
    ) -> bool:
        """Append one robot observation and the action that led to it.

        The first sample has no preceding action. Every later sample provides
        one transition action, preserving ``T`` states and ``T-1`` actions.
        """

        if self._saved:
            raise RuntimeError("cannot append after the episode was saved")
        if self._capture_finished:
            return False
        if frame.images:
            raise ValueError("camera images must use append_camera")
        if self._robot_frames and frame.time <= self._robot_frames[-1].time:
            raise ValueError("robot timestamps must be strictly increasing")
        if self._robot_frames:
            if transition_action is None:
                raise ValueError("non-initial robot samples require an action")
            self._actions.append(
                np.asarray(transition_action, dtype=np.float32).copy()
            )
            self._terminated.append(bool(terminated))
            self._truncated.append(False)
        elif transition_action is not None:
            raise ValueError("the initial robot sample cannot have an action")
        self._robot_frames.append(frame)
        self._capture_finished = bool(
            terminated and len(self._robot_frames) > 1
        )
        return True

    def append_camera(
        self,
        name: str,
        *,
        time: float,
        color: np.ndarray,
        depth_m: np.ndarray,
        world_from_camera: np.ndarray,
        valid_depth_pixels: int,
    ) -> bool:
        """Append one RGB-D frame on its camera's own timeline."""

        if self._saved:
            raise RuntimeError("cannot append after the episode was saved")
        if self._capture_finished:
            return False
        if name not in self._camera_frames:
            raise KeyError(f"unknown camera stream {name!r}")
        timestamp = float(time)
        if not np.isfinite(timestamp):
            raise ValueError("camera timestamp must be finite")
        frames = self._camera_frames[name]
        if frames and timestamp <= frames[-1].time:
            raise ValueError("camera timestamps must be strictly increasing")

        spec = self.cameras[name]
        color_array = np.asarray(color, dtype=np.uint8)
        depth_array = np.asarray(depth_m, dtype=np.float32)
        expected_size = spec.image_size
        if expected_size is not None:
            height, width = expected_size
            if color_array.shape != (height, width, 3):
                raise ValueError(
                    f"camera {name!r} color must have shape "
                    f"{(height, width, 3)}, got {color_array.shape}"
                )
            if depth_array.shape != (height, width):
                raise ValueError(
                    f"camera {name!r} depth must have shape "
                    f"{(height, width)}, got {depth_array.shape}"
                )
        transform = np.asarray(world_from_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("world_from_camera must be a finite 4x4 matrix")
        valid_pixels = int(valid_depth_pixels)
        if valid_pixels < 0 or valid_pixels > depth_array.size:
            raise ValueError("valid_depth_pixels is outside the image bounds")
        frames.append(
            _CameraFrame(
                time=timestamp,
                color=color_array.copy(),
                depth_m=depth_array.copy(),
                world_from_camera=transform.copy(),
                valid_depth_pixels=valid_pixels,
            )
        )
        return True

    @staticmethod
    def _stack_optional(
        frames: list[EpisodeFrame], name: str
    ) -> np.ndarray | None:
        values = [getattr(frame, name) for frame in frames]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"episode field {name!r} must be present consistently")
        return np.stack(values, axis=0)

    def _validate(self) -> tuple[set[str], dict[str, Any]]:
        if not self._robot_frames:
            raise RuntimeError("cannot save an episode without robot samples")
        extra_names = set(self._robot_frames[0].extra)
        for frame in self._robot_frames[1:]:
            if set(frame.extra) != extra_names:
                raise ValueError("robot extra fields changed during recording")
        missing_cameras = [
            name for name, frames in self._camera_frames.items() if not frames
        ]
        if missing_cameras and not self._quality_enabled:
            raise RuntimeError(
                "camera streams captured no frames: " + ", ".join(missing_cameras)
            )

        if self._truncated and not any(self._terminated):
            self._truncated[-1] = True
        metadata = dict(self.metadata)
        metadata.update(
            {
                "dataset_schema": DATASET_SCHEMA,
                "output_filename": self.output_path.name,
                "robot_frame_count": self.robot_frame_count,
                "transition_count": len(self._actions),
                "camera_frame_counts": self.camera_frame_counts,
                "terminated": bool(any(self._terminated)),
                "truncated": not bool(any(self._terminated)),
            }
        )
        if self._quality_enabled:
            self.quality_report = assess_episode(
                self._robot_frames, self._actions, self._camera_frames,
                metadata, bool(any(self._terminated)),
            )
            metadata["quality"] = self.quality_report
        return extra_names, metadata

    @staticmethod
    def _write_array(group: Any, name: str, value: np.ndarray) -> None:
        array = np.asarray(value)
        kwargs = (
            {"compression": "gzip", "shuffle": True}
            if array.ndim > 0 and array.size > 0
            else {}
        )
        group.create_dataset(name, data=array, **kwargs)

    def _write(self, path: Path) -> None:
        import h5py

        extra_names, metadata_extra = self._validate()
        with h5py.File(path, "w") as archive:
            archive.attrs["schema"] = DATASET_SCHEMA
            metadata = archive.create_group("metadata")
            metadata.attrs["task_name"] = "cherry_tomato_pick"
            metadata.attrs["seed"] = self.seed
            metadata.attrs["object_ids"] = json.dumps(list(self.object_ids))
            metadata.attrs["extra"] = json.dumps(
                metadata_extra, sort_keys=True, separators=(",", ":")
            )

            robot = archive.create_group("robot")
            robot_timestamps = np.asarray(
                [frame.time for frame in self._robot_frames],
                dtype=np.float64,
            )
            self._write_array(
                robot,
                "timestamps",
                robot_timestamps,
            )
            for field_name in ("joint_pos", "joint_vel", "ee_pose", "gripper"):
                value = self._stack_optional(self._robot_frames, field_name)
                if value is not None:
                    self._write_array(robot, field_name, value)
            if self._actions:
                self._write_array(robot, "action", np.stack(self._actions))
                self._write_array(
                    robot,
                    "action_timestamps",
                    robot_timestamps[:-1],
                )
                self._write_array(
                    robot,
                    "terminated",
                    np.asarray(self._terminated, dtype=bool),
                )
                self._write_array(
                    robot,
                    "truncated",
                    np.asarray(self._truncated, dtype=bool),
                )
            extra = robot.create_group("extra")
            for name in sorted(extra_names):
                self._write_array(
                    extra,
                    name,
                    np.stack(
                        [frame.extra[name] for frame in self._robot_frames]
                    ),
                )

            cameras = archive.create_group("cameras")
            for name, spec in self.cameras.items():
                frames = self._camera_frames[name]
                camera = cameras.create_group(name)
                camera.attrs["image_size"] = json.dumps(
                    list(spec.image_size) if spec.image_size is not None else None
                )
                camera.attrs["metadata"] = json.dumps(
                    dict(spec.metadata), sort_keys=True, separators=(",", ":")
                )
                calibration = camera.create_group("calibration")
                for field_name in (
                    "intrinsics",
                    "distortion",
                    "world_from_camera",
                ):
                    value = getattr(spec, field_name)
                    if value is not None:
                        self._write_array(calibration, field_name, value)
                if not frames:
                    height, width = spec.image_size or (0, 0)
                    for field_name, value in {
                        "timestamps": np.empty(0, dtype=np.float64),
                        "color": np.empty((0, height, width, 3), dtype=np.uint8),
                        "depth_m": np.empty((0, height, width), dtype=np.float32),
                        "world_from_camera": np.empty((0, 4, 4), dtype=np.float64),
                        "valid_depth_pixels": np.empty(0, dtype=np.int32),
                    }.items():
                        self._write_array(camera, field_name, value)
                    continue
                self._write_array(
                    camera,
                    "timestamps",
                    np.asarray([frame.time for frame in frames]),
                )
                self._write_array(
                    camera,
                    "color",
                    np.stack([frame.color for frame in frames]),
                )
                self._write_array(
                    camera,
                    "depth_m",
                    np.stack([frame.depth_m for frame in frames]),
                )
                self._write_array(
                    camera,
                    "world_from_camera",
                    np.stack([frame.world_from_camera for frame in frames]),
                )
                self._write_array(
                    camera,
                    "valid_depth_pixels",
                    np.asarray(
                        [frame.valid_depth_pixels for frame in frames],
                        dtype=np.int32,
                    ),
                )

            camera_index = robot.create_group("camera_index")
            camera_age = robot.create_group("camera_age_s")
            for name, frames in self._camera_frames.items():
                camera_timestamps = np.asarray(
                    [frame.time for frame in frames], dtype=np.float64
                )
                indices = np.searchsorted(
                    camera_timestamps, robot_timestamps, side="right"
                ).astype(np.int64) - 1
                ages = np.full(robot_timestamps.shape, np.nan, dtype=np.float64)
                valid = indices >= 0
                ages[valid] = (
                    robot_timestamps[valid] - camera_timestamps[indices[valid]]
                )
                self._write_array(
                    camera_index, name, indices.astype(np.int32)
                )
                self._write_array(camera_age, name, ages)

    def close(self) -> Path | None:
        """Write the episode once; partial runs are marked truncated."""

        if self._saved:
            return self.output_path
        if not self._robot_frames:
            logger.warning(
                "cherry-tomato recorder captured no robot samples; not writing {}",
                self.output_path,
            )
            return None
        if self.output_path.exists() and not self._overwrite:
            raise FileExistsError(
                f"episode appeared while recording: {self.output_path}"
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(f".{self.output_path.name}.tmp")
        try:
            self._write(temporary)
            temporary.replace(self.output_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        self._saved = True
        if self.quality_report is not None:
            # Per-episode sidecars avoid concurrent append to a shared manifest.
            # The HDF5 metadata remains authoritative for training selection.
            sidecar = self.output_path.with_suffix(".quality.json")
            report = {**self.quality_report, "episode": self.output_path.name,
                      "seed": self.seed}
            sidecar.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            logger.info("episode quality: {} reasons={}", report["status"], report["reasons"])
        logger.info(
            "wrote cherry-tomato episode: {} robot samples, cameras={} -> {}",
            self.robot_frame_count,
            self.camera_frame_counts,
            self.output_path,
        )
        return self.output_path


__all__ = [
    "CherryTomatoEpisodeRecorder",
    "DATASET_SCHEMA",
    "next_episode_path",
    "world_from_pose",
]
