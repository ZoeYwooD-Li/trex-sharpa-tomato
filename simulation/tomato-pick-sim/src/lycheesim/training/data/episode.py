"""Versioned multimodal episodes for RL, WAM, and VLA training."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from ..scene_flow import SceneFlowSequence


EPISODE_SCHEMA = "lycheesim.episode/v2"
LEGACY_EPISODE_SCHEMA = "lycheesim.episode/v1"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return default
    return json.loads(str(value))


def _validate_name(name: str, kind: str) -> str:
    value = str(name)
    if not value or "/" in value:
        raise ValueError(f"{kind} names must be non-empty and cannot contain '/'")
    return value


def _as_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise TypeError(f"{name} cannot use object dtype")
    return array


@dataclass(slots=True)
class EpisodeMetadata:
    """Episode-level task and provenance information."""

    task_name: str
    seed: int
    object_ids: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_name = str(self.task_name)
        self.seed = int(self.seed)
        self.object_ids = tuple(str(item) for item in self.object_ids)
        self.extra = dict(self.extra)
        if not self.task_name:
            raise ValueError("metadata.task_name must be non-empty")
        if any(not item for item in self.object_ids):
            raise ValueError("metadata.object_ids cannot contain empty IDs")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("metadata.object_ids must be unique")
        try:
            _json_dumps(self.extra)
        except (TypeError, ValueError) as exc:
            raise TypeError("metadata.extra must be JSON-serializable") from exc


@dataclass(slots=True)
class CameraSpec:
    """Static calibration and provenance for one camera stream."""

    image_size: tuple[int, int] | None = None
    intrinsics: np.ndarray | None = None
    distortion: np.ndarray | None = None
    world_from_camera: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_size is not None:
            self.image_size = tuple(int(item) for item in self.image_size)
            if len(self.image_size) != 2 or min(self.image_size) <= 0:
                raise ValueError("camera image_size must be (height, width)")
        if self.intrinsics is not None:
            self.intrinsics = _as_array(
                self.intrinsics, "camera intrinsics"
            ).astype(np.float64, copy=False)
            if self.intrinsics.shape != (3, 3):
                raise ValueError("camera intrinsics must have shape (3, 3)")
        if self.distortion is not None:
            self.distortion = _as_array(
                self.distortion, "camera distortion"
            ).astype(np.float64, copy=False)
            if self.distortion.ndim != 1:
                raise ValueError("camera distortion must be one-dimensional")
        if self.world_from_camera is not None:
            self.world_from_camera = _as_array(
                self.world_from_camera, "camera world_from_camera"
            ).astype(np.float64, copy=False)
            if self.world_from_camera.shape != (4, 4):
                raise ValueError(
                    "camera world_from_camera must have shape (4, 4)"
                )
        self.metadata = dict(self.metadata)
        try:
            _json_dumps(self.metadata)
        except (TypeError, ValueError) as exc:
            raise TypeError("camera metadata must be JSON-serializable") from exc


@dataclass(slots=True)
class EpisodeFrame:
    """One synchronized observation before batching into an episode."""

    time: float
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    ee_pose: np.ndarray | None = None
    gripper: np.ndarray | None = None
    extra: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = float(self.time)
        if not np.isfinite(self.time):
            raise ValueError("frame time must be finite")
        self.images = {
            _validate_name(name, "camera"): _as_array(
                value, f"frame.images[{name!r}]"
            )
            for name, value in self.images.items()
        }
        for name in ("joint_pos", "joint_vel", "ee_pose", "gripper"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _as_array(value, f"frame.{name}"))
        if self.joint_pos is not None and self.joint_vel is not None:
            if self.joint_pos.shape != self.joint_vel.shape:
                raise ValueError("frame joint_pos and joint_vel shapes must match")
        if self.ee_pose is not None and (
            self.ee_pose.ndim < 1 or self.ee_pose.shape[-1] != 7
        ):
            raise ValueError("frame ee_pose must end in xyz + quaternion xyzw")
        self.extra = {
            _validate_name(name, "extra frame field"): _as_array(
                value, f"frame.extra[{name!r}]"
            )
            for name, value in self.extra.items()
        }


class EpisodeBuilder:
    """Incrementally align simulator transitions into an :class:`Episode`.

    Capture starts with :meth:`append_initial`.  Each subsequent call to
    :meth:`append_transition` records the action and outcome associated with
    the preceding frame, then appends the resulting next frame.  This ordering
    prevents action/observation shifts during online data production.
    """

    def __init__(
        self,
        *,
        metadata: EpisodeMetadata,
        cameras: Mapping[str, CameraSpec],
    ) -> None:
        self.metadata = metadata
        self.cameras = dict(cameras)
        self._frames: list[EpisodeFrame] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[np.ndarray | None] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._closed = False

    def append_initial(self, frame: EpisodeFrame) -> None:
        if self._frames:
            raise RuntimeError("append_initial can be called only once")
        self._frames.append(frame)

    def append_transition(
        self,
        *,
        action: np.ndarray,
        next_frame: EpisodeFrame,
        reward: np.ndarray | float | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        if not self._frames:
            raise RuntimeError("append_initial must precede append_transition")
        if self._closed:
            raise RuntimeError("cannot append after a terminal or truncated step")
        if next_frame.time <= self._frames[-1].time:
            raise ValueError("next frame time must be strictly increasing")
        self._actions.append(_as_array(action, "transition action"))
        self._rewards.append(
            None if reward is None else _as_array(reward, "transition reward")
        )
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._frames.append(next_frame)
        self._closed = bool(terminated or truncated)

    @staticmethod
    def _stack_optional(frames: list[EpisodeFrame], name: str) -> np.ndarray | None:
        values = [getattr(frame, name) for frame in frames]
        present = [value is not None for value in values]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"frame field {name!r} must be present consistently")
        return np.stack(values, axis=0)

    def build(
        self, *, scene_flow: SceneFlowSequence | None = None
    ) -> "Episode":
        if not self._frames:
            raise RuntimeError("cannot build an episode without an initial frame")
        image_names = set(self._frames[0].images)
        extra_names = set(self._frames[0].extra)
        for frame in self._frames[1:]:
            if set(frame.images) != image_names:
                raise ValueError("camera image streams must be present consistently")
            if set(frame.extra) != extra_names:
                raise ValueError("extra frame fields must be present consistently")

        rewards_present = [value is not None for value in self._rewards]
        if any(rewards_present) and not all(rewards_present):
            raise ValueError("transition rewards must be present consistently")
        rewards = (
            np.stack(self._rewards, axis=0)
            if self._rewards and all(rewards_present)
            else None
        )
        frames = EpisodeFrames(
            times=np.asarray([frame.time for frame in self._frames]),
            images={
                name: np.stack(
                    [frame.images[name] for frame in self._frames], axis=0
                )
                for name in sorted(image_names)
            },
            joint_pos=self._stack_optional(self._frames, "joint_pos"),
            joint_vel=self._stack_optional(self._frames, "joint_vel"),
            ee_pose=self._stack_optional(self._frames, "ee_pose"),
            gripper=self._stack_optional(self._frames, "gripper"),
            action=(
                np.stack(self._actions, axis=0) if self._actions else None
            ),
            reward=rewards,
            terminated=(
                np.asarray(self._terminated, dtype=bool)
                if self._terminated
                else None
            ),
            truncated=(
                np.asarray(self._truncated, dtype=bool)
                if self._truncated
                else None
            ),
            scene_flow=scene_flow,
            extra={
                name: np.stack(
                    [frame.extra[name] for frame in self._frames], axis=0
                )
                for name in sorted(extra_names)
            },
        )
        return Episode(
            metadata=self.metadata,
            cameras=self.cameras,
            frames=frames,
        )


@dataclass(slots=True)
class EpisodeFrames:
    """Time-aligned visual, robot-state, action, and geometry sequences.

    State and image arrays have ``T`` rows.  ``action`` has ``T-1`` rows and
    describes the transition from frame ``t`` to frame ``t+1``.  Scene flow,
    when present, follows the same transition alignment.
    """

    times: np.ndarray
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    ee_pose: np.ndarray | None = None
    gripper: np.ndarray | None = None
    action: np.ndarray | None = None
    reward: np.ndarray | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    scene_flow: SceneFlowSequence | None = None
    extra: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.times = _as_array(self.times, "frames.times").astype(
            np.float64, copy=False
        )
        self.images = {
            _validate_name(name, "camera"): _as_array(
                value, f"frames.images[{name!r}]"
            )
            for name, value in self.images.items()
        }
        for name in (
            "joint_pos",
            "joint_vel",
            "ee_pose",
            "gripper",
            "action",
            "reward",
            "terminated",
            "truncated",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _as_array(value, f"frames.{name}"))
        for name in ("terminated", "truncated"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.astype(bool, copy=False))
        self.extra = {
            _validate_name(name, "extra frame field"): _as_array(
                value, f"frames.extra[{name!r}]"
            )
            for name, value in self.extra.items()
        }
        self.validate()

    @property
    def frame_count(self) -> int:
        return int(len(self.times))

    @property
    def transition_count(self) -> int:
        return max(self.frame_count - 1, 0)

    def validate(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("frames.times must have shape (T,)")
        if not np.isfinite(self.times).all():
            raise ValueError("frames.times must be finite")
        if len(self.times) > 1 and not np.all(np.diff(self.times) > 0.0):
            raise ValueError("frames.times must be strictly increasing")

        frame_count = self.frame_count
        for name, value in self.images.items():
            if value.ndim < 3 or value.shape[0] != frame_count:
                raise ValueError(
                    f"camera {name!r} images must begin with T={frame_count}, "
                    f"got {value.shape}"
                )
        for name in ("joint_pos", "joint_vel", "ee_pose", "gripper"):
            value = getattr(self, name)
            if value is not None and (
                value.ndim < 1 or value.shape[0] != frame_count
            ):
                raise ValueError(
                    f"frames.{name} must begin with T={frame_count}, "
                    f"got {value.shape}"
                )
        if self.joint_pos is not None and self.joint_vel is not None:
            if self.joint_pos.shape != self.joint_vel.shape:
                raise ValueError("joint_pos and joint_vel must have equal shapes")
        if self.ee_pose is not None and self.ee_pose.shape[-1] != 7:
            raise ValueError(
                "ee_pose must end in 7 values: position xyz and quaternion xyzw"
            )
        if self.action is not None and (
            self.action.ndim < 1
            or self.action.shape[0] != self.transition_count
        ):
            raise ValueError(
                "frames.action must begin with T-1="
                f"{self.transition_count}, got {self.action.shape}"
            )
        for name in ("reward", "terminated", "truncated"):
            value = getattr(self, name)
            if value is not None and (
                value.ndim < 1 or value.shape[0] != self.transition_count
            ):
                raise ValueError(
                    f"frames.{name} must begin with T-1="
                    f"{self.transition_count}, got {value.shape}"
                )
        for name, value in self.extra.items():
            if value.ndim < 1 or value.shape[0] != frame_count:
                raise ValueError(
                    f"extra frame field {name!r} must begin with "
                    f"T={frame_count}, got {value.shape}"
                )
        if self.scene_flow is not None:
            self.scene_flow.validate()
            if self.scene_flow.times.shape != self.times.shape or not np.allclose(
                self.scene_flow.times, self.times, rtol=0.0, atol=1e-9
            ):
                raise ValueError(
                    "scene-flow timestamps must match episode frame timestamps"
                )

    def frame(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.frame_count
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        result: dict[str, Any] = {
            "time": self.times[index],
            "images": {name: value[index] for name, value in self.images.items()},
        }
        for name in ("joint_pos", "joint_vel", "ee_pose", "gripper"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value[index]
        if self.scene_flow is not None:
            result["scene_points"] = self.scene_flow.points[index]
            result["scene_valid"] = self.scene_flow.valid[index]
        if self.extra:
            result["extra"] = {
                name: value[index] for name, value in self.extra.items()
            }
        return result

    def transition(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.transition_count
        if index < 0 or index >= self.transition_count:
            raise IndexError(index)
        result: dict[str, Any] = {
            "observation": self.frame(index),
            "next_observation": self.frame(index + 1),
            "time": self.times[index],
            "dt": self.times[index + 1] - self.times[index],
        }
        if self.action is not None:
            result["action"] = self.action[index]
        for name in ("reward", "terminated", "truncated"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value[index]
        if self.scene_flow is not None:
            result["scene_flow"] = self.scene_flow.transition(index)
        return result

    def transitions(self) -> Iterator[dict[str, Any]]:
        for index in range(self.transition_count):
            yield self.transition(index)


@dataclass(slots=True)
class Episode:
    """A model-neutral multimodal episode with deterministic persistence."""

    metadata: EpisodeMetadata
    cameras: Mapping[str, CameraSpec]
    frames: EpisodeFrames

    def __post_init__(self) -> None:
        self.cameras = {
            _validate_name(name, "camera"): value
            for name, value in self.cameras.items()
        }
        if not isinstance(self.metadata, EpisodeMetadata):
            raise TypeError("metadata must be EpisodeMetadata")
        if not isinstance(self.frames, EpisodeFrames):
            raise TypeError("frames must be EpisodeFrames")
        for name, camera in self.cameras.items():
            if not isinstance(camera, CameraSpec):
                raise TypeError(f"camera {name!r} must be CameraSpec")
        unknown_images = set(self.frames.images).difference(self.cameras)
        if unknown_images:
            raise ValueError(
                "image streams reference unregistered cameras: "
                + ", ".join(sorted(unknown_images))
            )
        for name, images in self.frames.images.items():
            image_size = self.cameras[name].image_size
            if image_size is not None and tuple(images.shape[1:3]) != image_size:
                raise ValueError(
                    f"camera {name!r} declares image_size={image_size} but "
                    f"frames contain {tuple(images.shape[1:3])}"
                )

    def transition(self, index: int) -> dict[str, Any]:
        result = self.frames.transition(index)
        result["metadata"] = self.metadata
        return result

    def transitions(self) -> Iterator[dict[str, Any]]:
        for index in range(self.frames.transition_count):
            yield self.transition(index)

    @staticmethod
    def _write_array(group: Any, name: str, value: np.ndarray) -> None:
        kwargs = (
            {"compression": "gzip", "shuffle": True}
            if value.ndim > 0 and value.size > 0
            else {}
        )
        group.create_dataset(name, data=value, **kwargs)

    @classmethod
    def _write_scene_flow(cls, group: Any, value: SceneFlowSequence) -> None:
        for name in (
            "times",
            "points",
            "valid",
            "flow",
            "flow_valid",
            "point_ids",
            "source_index",
        ):
            cls._write_array(group, name, np.asarray(getattr(value, name)))
        if value.colors is not None:
            cls._write_array(group, "colors", np.asarray(value.colors))
        if value.actions is not None:
            cls._write_array(group, "actions", np.asarray(value.actions))
        group.attrs["source_names"] = _json_dumps(list(value.source_names))
        group.attrs["source_kinds"] = _json_dumps(list(value.source_kinds))
        group.attrs["action_names"] = _json_dumps(list(value.action_names))
        group.attrs["action_slices"] = _json_dumps(
            {name: list(bounds) for name, bounds in value.action_slices.items()}
        )
        group.attrs["source_metadata"] = _json_dumps(
            [dict(item) for item in value.source_metadata]
        )
        group.attrs["metadata"] = _json_dumps(dict(value.metadata))

    @staticmethod
    def _read_scene_flow(group: Any) -> SceneFlowSequence:
        sequence = SceneFlowSequence(
            times=np.asarray(group["times"]),
            points=np.asarray(group["points"]),
            valid=np.asarray(group["valid"], dtype=bool),
            flow=np.asarray(group["flow"]),
            flow_valid=np.asarray(group["flow_valid"], dtype=bool),
            point_ids=np.asarray(group["point_ids"], dtype=np.int64),
            source_index=np.asarray(group["source_index"], dtype=np.int32),
            source_names=tuple(_json_loads(group.attrs.get("source_names"), [])),
            source_kinds=tuple(_json_loads(group.attrs.get("source_kinds"), [])),
            colors=(
                np.asarray(group["colors"], dtype=np.uint8)
                if "colors" in group
                else None
            ),
            actions=(np.asarray(group["actions"]) if "actions" in group else None),
            action_names=tuple(_json_loads(group.attrs.get("action_names"), [])),
            action_slices={
                name: tuple(bounds)
                for name, bounds in _json_loads(
                    group.attrs.get("action_slices"), {}
                ).items()
            },
            source_metadata=tuple(
                _json_loads(group.attrs.get("source_metadata"), [])
            ),
            metadata=_json_loads(group.attrs.get("metadata"), {}),
        )
        sequence.validate()
        return sequence

    def save(self, path: str | Path) -> Path:
        """Write an atomic HDF5 episode using ``lycheesim.episode/v2``."""

        import h5py

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            with h5py.File(temporary, "w") as archive:
                archive.attrs["schema"] = EPISODE_SCHEMA

                metadata = archive.create_group("metadata")
                metadata.attrs["task_name"] = self.metadata.task_name
                metadata.attrs["seed"] = self.metadata.seed
                metadata.attrs["object_ids"] = _json_dumps(
                    list(self.metadata.object_ids)
                )
                metadata.attrs["extra"] = _json_dumps(
                    dict(self.metadata.extra)
                )

                cameras = archive.create_group("cameras")
                for name, spec in self.cameras.items():
                    camera = cameras.create_group(name)
                    camera.attrs["image_size"] = _json_dumps(
                        list(spec.image_size) if spec.image_size is not None else None
                    )
                    camera.attrs["metadata"] = _json_dumps(dict(spec.metadata))
                    for field_name in (
                        "intrinsics",
                        "distortion",
                        "world_from_camera",
                    ):
                        value = getattr(spec, field_name)
                        if value is not None:
                            self._write_array(camera, field_name, value)

                frames = archive.create_group("frames")
                self._write_array(frames, "times", self.frames.times)
                for name in (
                    "joint_pos",
                    "joint_vel",
                    "ee_pose",
                    "gripper",
                    "action",
                    "reward",
                    "terminated",
                    "truncated",
                ):
                    value = getattr(self.frames, name)
                    if value is not None:
                        self._write_array(frames, name, value)
                images = frames.create_group("images")
                for name, value in self.frames.images.items():
                    self._write_array(images, name, value)
                extras = frames.create_group("extra")
                for name, value in self.frames.extra.items():
                    self._write_array(extras, name, value)
                if self.frames.scene_flow is not None:
                    scene_flow = frames.create_group("scene_flow")
                    self._write_scene_flow(scene_flow, self.frames.scene_flow)
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        """Load and validate a versioned LycheeSim HDF5 episode."""

        import h5py

        with h5py.File(path, "r") as archive:
            schema = archive.attrs.get("schema")
            if isinstance(schema, bytes):
                schema = schema.decode("utf-8")
            if schema not in {EPISODE_SCHEMA, LEGACY_EPISODE_SCHEMA}:
                raise ValueError(f"unsupported episode schema: {schema!r}")

            metadata_group = archive["metadata"]
            metadata = EpisodeMetadata(
                task_name=str(metadata_group.attrs["task_name"]),
                seed=int(metadata_group.attrs["seed"]),
                object_ids=tuple(
                    _json_loads(metadata_group.attrs.get("object_ids"), [])
                ),
                extra=_json_loads(metadata_group.attrs.get("extra"), {}),
            )

            cameras: dict[str, CameraSpec] = {}
            for name, camera in archive["cameras"].items():
                image_size = _json_loads(camera.attrs.get("image_size"), None)
                cameras[name] = CameraSpec(
                    image_size=(tuple(image_size) if image_size is not None else None),
                    intrinsics=(
                        np.asarray(camera["intrinsics"])
                        if "intrinsics" in camera
                        else None
                    ),
                    distortion=(
                        np.asarray(camera["distortion"])
                        if "distortion" in camera
                        else None
                    ),
                    world_from_camera=(
                        np.asarray(camera["world_from_camera"])
                        if "world_from_camera" in camera
                        else None
                    ),
                    metadata=_json_loads(camera.attrs.get("metadata"), {}),
                )

            frames_group = archive["frames"]
            optional = {
                name: (
                    np.asarray(frames_group[name])
                    if name in frames_group
                    else None
                )
                for name in (
                    "joint_pos",
                    "joint_vel",
                    "ee_pose",
                    "gripper",
                    "action",
                    "reward",
                    "terminated",
                    "truncated",
                )
            }
            if schema == LEGACY_EPISODE_SCHEMA and optional["ee_pose"] is not None:
                legacy_ee_pose = np.array(optional["ee_pose"], copy=True)
                legacy_ee_pose[..., 3:7] = legacy_ee_pose[..., [4, 5, 6, 3]]
                optional["ee_pose"] = legacy_ee_pose
            images = {
                name: np.asarray(value)
                for name, value in frames_group["images"].items()
            }
            extras = {
                name: np.asarray(value)
                for name, value in frames_group["extra"].items()
            }
            frames = EpisodeFrames(
                times=np.asarray(frames_group["times"]),
                images=images,
                scene_flow=(
                    cls._read_scene_flow(frames_group["scene_flow"])
                    if "scene_flow" in frames_group
                    else None
                ),
                extra=extras,
                **optional,
            )
        return cls(metadata=metadata, cameras=cameras, frames=frames)
