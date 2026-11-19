"""Versioned dense scene-flow sequences for world/action-model training."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


SCENE_FLOW_SCHEMA = "lycheesim.scene-flow/v1"


def _json_scalar(value: Any) -> np.ndarray:
    return np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _decode_json(value: np.ndarray | str | bytes, default: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return default
    return json.loads(str(value))


@dataclass(slots=True)
class SceneFlowSequence:
    """A point-corresponded full-scene trajectory.

    ``flow[t, i]`` is the forward displacement of persistent point ``i`` from
    frame ``t`` to ``t + 1``.  It is a displacement in metres, not a velocity.
    ``actions[t]`` is aligned with that same transition when actions are
    present.
    """

    times: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    flow: np.ndarray
    flow_valid: np.ndarray
    point_ids: np.ndarray
    source_index: np.ndarray
    source_names: tuple[str, ...]
    source_kinds: tuple[str, ...]
    colors: np.ndarray | None = None
    actions: np.ndarray | None = None
    action_names: tuple[str, ...] = ()
    action_slices: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    source_metadata: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dense(
        cls,
        *,
        times: np.ndarray,
        points: np.ndarray,
        point_ids: np.ndarray,
        source_index: np.ndarray,
        source_names: Sequence[str],
        source_kinds: Sequence[str],
        valid: np.ndarray | None = None,
        colors: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        action_names: Sequence[str] = (),
        action_slices: Mapping[str, tuple[int, int]] | None = None,
        source_metadata: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "SceneFlowSequence":
        pts = np.asarray(points, dtype=np.float32)
        if valid is None:
            mask = np.isfinite(pts).all(axis=-1)
        else:
            mask = np.asarray(valid, dtype=bool)

        n_frames = pts.shape[0] if pts.ndim >= 1 else 0
        n_points = pts.shape[1] if pts.ndim >= 2 else 0
        flow = np.zeros(
            (max(n_frames - 1, 0), n_points, 3), dtype=np.float32
        )
        flow_valid = (
            mask[:-1] & mask[1:]
            if n_frames > 1
            else np.zeros((0, n_points), dtype=bool)
        )
        if n_frames > 1:
            delta = pts[1:] - pts[:-1]
            flow[flow_valid] = delta[flow_valid]

        sequence = cls(
            times=np.asarray(times, dtype=np.float64),
            points=pts,
            valid=mask,
            flow=flow,
            flow_valid=flow_valid,
            point_ids=np.asarray(point_ids, dtype=np.int64),
            source_index=np.asarray(source_index, dtype=np.int32),
            source_names=tuple(str(item) for item in source_names),
            source_kinds=tuple(str(item) for item in source_kinds),
            colors=(
                None if colors is None else np.asarray(colors, dtype=np.uint8)
            ),
            actions=(
                None if actions is None else np.asarray(actions)
            ),
            action_names=tuple(str(item) for item in action_names),
            action_slices=dict(action_slices or {}),
            source_metadata=tuple(dict(item) for item in source_metadata),
            metadata=dict(metadata or {}),
        )
        sequence.validate()
        return sequence

    @property
    def velocities(self) -> np.ndarray:
        """Forward flow divided by the corresponding frame interval."""
        velocity = np.zeros_like(self.flow)
        if len(self.times) < 2:
            return velocity
        dt = np.diff(self.times)
        positive = dt > 0.0
        if positive.any():
            velocity[positive] = (
                self.flow[positive] / dt[positive, None, None]
            )
        velocity[~self.flow_valid] = 0.0
        return velocity

    def validate(self) -> None:
        if self.points.ndim != 3 or self.points.shape[-1] != 3:
            raise ValueError(
                f"points must have shape (T, N, 3), got {self.points.shape}"
            )
        t_count, point_count, _ = self.points.shape
        if self.times.shape != (t_count,):
            raise ValueError(
                f"times must have shape ({t_count},), got {self.times.shape}"
            )
        if not np.isfinite(self.times).all():
            raise ValueError("times must be finite")
        if t_count > 1 and not np.all(np.diff(self.times) > 0.0):
            raise ValueError("times must be strictly increasing")
        if self.valid.shape != (t_count, point_count):
            raise ValueError(
                "valid must have shape "
                f"({t_count}, {point_count}), got {self.valid.shape}"
            )
        if not np.isfinite(self.points[self.valid]).all():
            raise ValueError("points marked valid must be finite")
        transition_shape = (max(t_count - 1, 0), point_count)
        if self.flow.shape != (*transition_shape, 3):
            raise ValueError(
                "flow must have shape "
                f"{(*transition_shape, 3)}, got {self.flow.shape}"
            )
        if self.flow_valid.shape != transition_shape:
            raise ValueError(
                "flow_valid must have shape "
                f"{transition_shape}, got {self.flow_valid.shape}"
            )
        correspondence_possible = (
            self.valid[:-1] & self.valid[1:]
            if t_count > 1
            else np.zeros(transition_shape, dtype=bool)
        )
        if np.any(self.flow_valid & ~correspondence_possible):
            raise ValueError(
                "flow_valid cannot include a point absent from either frame"
            )
        if not np.isfinite(self.flow[self.flow_valid]).all():
            raise ValueError("flow marked valid must be finite")
        if self.point_ids.shape != (point_count,):
            raise ValueError(
                f"point_ids must have shape ({point_count},)"
            )
        if len(np.unique(self.point_ids)) != point_count:
            raise ValueError("point_ids must be unique")
        if self.source_index.shape != (point_count,):
            raise ValueError(
                f"source_index must have shape ({point_count},)"
            )
        if len(self.source_names) != len(self.source_kinds):
            raise ValueError("source_names and source_kinds must have equal length")
        if len(set(self.source_names)) != len(self.source_names):
            raise ValueError("source_names must be unique")
        if point_count and (
            self.source_index.min() < 0
            or self.source_index.max() >= len(self.source_names)
        ):
            raise ValueError("source_index contains an out-of-range source")
        if self.colors is not None and self.colors.shape != self.points.shape:
            raise ValueError(
                f"colors must match points shape, got {self.colors.shape}"
            )
        if self.actions is not None:
            if self.actions.ndim != 2:
                raise ValueError(
                    f"actions must have shape (T-1, A), got {self.actions.shape}"
                )
            if self.actions.shape[0] != max(t_count - 1, 0):
                raise ValueError(
                    "actions must align with flow transitions: expected "
                    f"{max(t_count - 1, 0)} rows, got {self.actions.shape[0]}"
                )
            used = np.zeros(self.actions.shape[1], dtype=bool)
            for name, bounds in self.action_slices.items():
                start, stop = bounds
                if not (0 <= start <= stop <= self.actions.shape[1]):
                    raise ValueError(
                        f"invalid action slice for {name!r}: {bounds}"
                    )
                if used[start:stop].any():
                    raise ValueError("action_slices must not overlap")
                used[start:stop] = True
            if self.action_names and (
                set(self.action_slices) != set(self.action_names)
            ):
                raise ValueError(
                    "action_names and action_slices must name the same fields"
                )
        if self.source_metadata and (
            len(self.source_metadata) != len(self.source_names)
        ):
            raise ValueError(
                "source_metadata must be empty or have one entry per source"
            )

    def transition(self, index: int) -> dict[str, Any]:
        """Return one model-ready ``state, action, next_state, flow`` sample."""
        if index < 0:
            index += len(self.flow)
        if index < 0 or index >= len(self.flow):
            raise IndexError(index)
        result: dict[str, Any] = {
            "time": self.times[index],
            "dt": self.times[index + 1] - self.times[index],
            "points": self.points[index],
            "next_points": self.points[index + 1],
            "valid": self.valid[index],
            "next_valid": self.valid[index + 1],
            "flow": self.flow[index],
            "flow_valid": self.flow_valid[index],
            "point_ids": self.point_ids,
            "source_index": self.source_index,
        }
        if self.actions is not None:
            result["action"] = self.actions[index]
        return result

    def transitions(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self.flow)):
            yield self.transition(index)

    def to_torch(
        self,
        *,
        device: str | None = None,
        dtype: Any | None = None,
    ) -> dict[str, Any]:
        """Create Torch views/copies for model input without making Torch required."""
        import torch

        float_dtype = dtype or torch.float32
        result: dict[str, Any] = {
            "times": torch.as_tensor(
                self.times, dtype=float_dtype, device=device
            ),
            "points": torch.as_tensor(
                self.points, dtype=float_dtype, device=device
            ),
            "valid": torch.as_tensor(self.valid, device=device),
            "flow": torch.as_tensor(
                self.flow, dtype=float_dtype, device=device
            ),
            "flow_valid": torch.as_tensor(self.flow_valid, device=device),
            "point_ids": torch.as_tensor(self.point_ids, device=device),
            "source_index": torch.as_tensor(
                self.source_index, device=device
            ),
        }
        if self.colors is not None:
            result["colors"] = torch.as_tensor(self.colors, device=device)
        if self.actions is not None:
            result["actions"] = torch.as_tensor(
                self.actions, dtype=float_dtype, device=device
            )
        return result

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "schema": np.asarray(SCENE_FLOW_SCHEMA),
            "times": self.times.astype(np.float64, copy=False),
            "points": self.points.astype(np.float32, copy=False),
            "valid": self.valid.astype(bool, copy=False),
            "flow": self.flow.astype(np.float32, copy=False),
            "flow_valid": self.flow_valid.astype(bool, copy=False),
            "point_ids": self.point_ids.astype(np.int64, copy=False),
            "source_index": self.source_index.astype(np.int32, copy=False),
            "source_names": _json_scalar(list(self.source_names)),
            "source_kinds": _json_scalar(list(self.source_kinds)),
            "source_metadata": _json_scalar(
                [dict(item) for item in self.source_metadata]
            ),
            "metadata": _json_scalar(dict(self.metadata)),
            "action_names": _json_scalar(list(self.action_names)),
            "action_slices": _json_scalar(
                {key: list(value) for key, value in self.action_slices.items()}
            ),
        }
        if self.colors is not None:
            payload["colors"] = self.colors.astype(np.uint8, copy=False)
        if self.actions is not None:
            payload["actions"] = self.actions
        return payload

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        try:
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, **self.to_npz_dict())
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "SceneFlowSequence":
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            schema = (
                str(archive["schema"].reshape(()).item())
                if "schema" in keys
                else None
            )
            if schema not in (None, SCENE_FLOW_SCHEMA):
                raise ValueError(f"unsupported scene-flow schema: {schema!r}")

            points = np.asarray(archive["points"], dtype=np.float32)
            times = np.asarray(archive["times"], dtype=np.float64)
            valid = (
                np.asarray(archive["valid"], dtype=bool)
                if "valid" in keys
                else np.isfinite(points).all(axis=-1)
            )
            if "source_names" in keys:
                names = _decode_json(archive["source_names"], [])
            else:
                names = _decode_json(archive["labels"], [])
            if "source_kinds" in keys:
                kinds = _decode_json(archive["source_kinds"], [])
            else:
                kinds = ["unknown"] * len(names)

            if "source_index" in keys:
                source_index = np.asarray(
                    archive["source_index"], dtype=np.int32
                )
            else:
                legacy_labels = np.asarray(archive["label_idx"])
                source_index = legacy_labels[0].astype(np.int32, copy=False)

            sequence = cls.from_dense(
                times=times,
                points=points,
                valid=valid,
                colors=(
                    np.asarray(archive["colors"], dtype=np.uint8)
                    if "colors" in keys
                    else None
                ),
                point_ids=(
                    np.asarray(archive["point_ids"], dtype=np.int64)
                    if "point_ids" in keys
                    else np.arange(points.shape[1], dtype=np.int64)
                ),
                source_index=source_index,
                source_names=names,
                source_kinds=kinds,
                actions=(
                    np.asarray(archive["actions"])
                    if "actions" in keys
                    else None
                ),
                action_names=(
                    _decode_json(archive["action_names"], [])
                    if "action_names" in keys
                    else ()
                ),
                action_slices=(
                    {
                        key: tuple(value)
                        for key, value in _decode_json(
                            archive["action_slices"], {}
                        ).items()
                    }
                    if "action_slices" in keys
                    else None
                ),
                source_metadata=(
                    _decode_json(archive["source_metadata"], [])
                    if "source_metadata" in keys
                    else ()
                ),
                metadata=(
                    _decode_json(archive["metadata"], {})
                    if "metadata" in keys
                    else {"migrated_from": "legacy-pointcloud-npz"}
                ),
            )
            if "flow" in keys:
                stored_flow = np.asarray(archive["flow"], dtype=np.float32)
                if stored_flow.shape != sequence.flow.shape:
                    raise ValueError(
                        "stored flow shape does not match the point sequence"
                    )
                sequence.flow = stored_flow
            if "flow_valid" in keys:
                sequence.flow_valid = np.asarray(
                    archive["flow_valid"], dtype=bool
                )
            sequence.validate()
            return sequence
