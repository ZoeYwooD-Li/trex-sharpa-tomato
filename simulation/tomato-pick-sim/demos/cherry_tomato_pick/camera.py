"""Fixed Gemini 435Le and dual Gemini 305 wrist rigs for the tomato scene.

The task photographs show a horizontally mounted camera on the centre pedestal
between the two FR3 arms.  The procedural housing uses Orbbec's published
138.5 x 40.5 x 70.0 mm envelope and 95 mm stereo baseline.  Its configurable
optical pose follows the Newton/OpenGL convention: +X right, +Y up, -Z forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


@dataclass(frozen=True)
class CameraVisualPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]
    position: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class WristCameraVisualPart:
    """One Gemini 305 render part expressed in its Wuji2 wrist frame."""

    name: str
    side: str
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]
    local_position: np.ndarray
    local_quaternion_xyzw: np.ndarray


def _xyz(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"camera.{name} must contain three finite values")
    return result


def _look_at_quaternion(
    position: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray,
) -> np.ndarray:
    forward = target - position
    length = float(np.linalg.norm(forward))
    if length <= 1.0e-8:
        raise ValueError("camera position and look_at_target_m must differ")
    forward /= length
    right = np.cross(forward, world_up)
    right_length = float(np.linalg.norm(right))
    if right_length <= 1.0e-8:
        raise ValueError("camera world_up must not be parallel to its view direction")
    right /= right_length
    camera_up = np.cross(right, forward)
    return Rotation.from_matrix(
        np.column_stack((right, camera_up, -forward))
    ).as_quat()


def _visual_part(
    name: str,
    mesh: trimesh.Trimesh,
    color: tuple[float, float, float],
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> CameraVisualPart:
    return CameraVisualPart(
        name=name,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        color=color,
        position=np.asarray(position, dtype=np.float32),
        quaternion_xyzw=np.asarray(quaternion_xyzw, dtype=np.float32),
    )


def _normalized_quaternion(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(result))
    if result.shape != (4,) or not np.isfinite(result).all() or norm <= 1.0e-8:
        raise ValueError(
            f"camera.{name} must contain a finite nonzero xyzw quaternion"
        )
    return result / norm


def _compose_pose(
    parent_position: np.ndarray,
    parent_quaternion_xyzw: np.ndarray,
    local_position: np.ndarray,
    local_quaternion_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rotation = Rotation.from_quat(parent_quaternion_xyzw)
    world_position = parent_position + parent_rotation.apply(local_position)
    world_rotation = parent_rotation * Rotation.from_quat(local_quaternion_xyzw)
    return world_position, world_rotation.as_quat()


def _intrinsic_matrix(
    width: int,
    height: int,
    horizontal_fov_rad: float,
    vertical_fov_rad: float,
) -> np.ndarray:
    return np.asarray(
        (
            (
                0.5 * width / math.tan(0.5 * horizontal_fov_rad),
                0.0,
                0.5 * width,
            ),
            (
                0.0,
                0.5 * height / math.tan(0.5 * vertical_fov_rad),
                0.5 * height,
            ),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


class Gemini435LeCamera:
    """Calibrated fixed RGB-D camera plus a dimensioned display model."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.model = str(self.cfg.get("model", "orbbec_gemini_435le"))
        if self.model != "orbbec_gemini_435le":
            raise ValueError(f"unsupported cherry-tomato camera model {self.model!r}")

        self.position = _xyz(
            self.cfg.get("optical_position_m", (0.10, 0.0, 0.48)),
            name="optical_position_m",
        )
        self.target = _xyz(
            self.cfg.get("look_at_target_m", (0.3983, -0.0031, 0.0842)),
            name="look_at_target_m",
        )
        self.world_up = _xyz(
            self.cfg.get("world_up", (0.0, 0.0, 1.0)),
            name="world_up",
        )
        self.quaternion_xyzw = _look_at_quaternion(
            self.position, self.target, self.world_up
        )

        self.dimensions = _xyz(
            self.cfg.get("housing_dimensions_m", (0.1385, 0.0405, 0.0700)),
            name="housing_dimensions_m",
        )
        if np.any(self.dimensions <= 0.0):
            raise ValueError("camera housing dimensions must be positive")
        self.baseline_m = float(self.cfg.get("stereo_baseline_m", 0.095))
        if not math.isfinite(self.baseline_m) or not 0.0 < self.baseline_m < self.dimensions[0]:
            raise ValueError("camera stereo_baseline_m must fit inside the housing")

        self.sensor_cfg = dict(self.cfg.get("sensor", {}) or {})
        self.width = int(self.sensor_cfg.get("width", 320))
        self.height = int(self.sensor_cfg.get("height", 200))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera sensor width and height must be positive")
        self.horizontal_fov_rad = math.radians(
            float(self.sensor_cfg.get("horizontal_fov_deg", 94.0))
        )
        self.vertical_fov_rad = math.radians(
            float(self.sensor_cfg.get("vertical_fov_deg", 68.0))
        )
        if not (
            0.0 < self.horizontal_fov_rad < math.pi
            and 0.0 < self.vertical_fov_rad < math.pi
        ):
            raise ValueError("camera fields of view must lie between 0 and 180 degrees")
        clipping = np.asarray(
            self.sensor_cfg.get("clipping_range_m", (0.31, 10.0)),
            dtype=np.float64,
        )
        if (
            clipping.shape != (2,)
            or not np.isfinite(clipping).all()
            or clipping[0] < 0.0
            or clipping[1] <= clipping[0]
        ):
            raise ValueError("camera sensor clipping_range_m must be [near, far]")
        self.clipping_range_m = (float(clipping[0]), float(clipping[1]))
        target_distance = float(np.linalg.norm(self.target - self.position))
        if not self.clipping_range_m[0] <= target_distance <= self.clipping_range_m[1]:
            raise ValueError("camera look-at target lies outside its clipping range")

        self.intrinsic_matrix = _intrinsic_matrix(
            self.width,
            self.height,
            self.horizontal_fov_rad,
            self.vertical_fov_rad,
        )
        self.visual_parts = self._make_visual_parts()

    def _make_visual_parts(self) -> tuple[CameraVisualPart, ...]:
        width, height, depth = self.dimensions
        # Keep the entire housing behind the optical origin so the camera does
        # not ray-hit its own display mesh when it is also registered in Newton.
        front_clearance = 0.004
        housing = trimesh.creation.box(extents=self.dimensions)
        housing.apply_translation((0.0, 0.0, front_clearance + 0.5 * depth))
        front = trimesh.creation.box(extents=(0.96 * width, 0.82 * height, 0.003))
        front.apply_translation((0.0, 0.0, front_clearance + 0.0015))

        parts = [
            _visual_part(
                "sensors/gemini_435le/housing",
                housing,
                (0.08, 0.09, 0.10),
                self.position,
                self.quaternion_xyzw,
            ),
            _visual_part(
                "sensors/gemini_435le/front_panel",
                front,
                (0.025, 0.028, 0.032),
                self.position,
                self.quaternion_xyzw,
            ),
        ]
        lens_positions = (
            (-0.5 * self.baseline_m, (0.05, 0.12, 0.20)),
            (0.0, (0.10, 0.20, 0.32)),
            (0.5 * self.baseline_m, (0.05, 0.12, 0.20)),
        )
        for index, (x, color) in enumerate(lens_positions):
            lens = trimesh.creation.cylinder(radius=0.007, height=0.004, sections=24)
            lens.apply_translation((x, 0.0, front_clearance))
            parts.append(
                _visual_part(
                    f"sensors/gemini_435le/lens_{index}",
                    lens,
                    color,
                    self.position,
                    self.quaternion_xyzw,
                )
            )

        mount_cfg = dict(self.cfg.get("mount", {}) or {})
        pedestal_size = _xyz(
            mount_cfg.get("pedestal_size_m", (0.045, 0.055, 0.43)),
            name="mount.pedestal_size_m",
        )
        pedestal_center = _xyz(
            mount_cfg.get("pedestal_center_m", (0.075, 0.0, 0.215)),
            name="mount.pedestal_center_m",
        )
        if np.any(pedestal_size <= 0.0):
            raise ValueError("camera mount pedestal dimensions must be positive")
        pedestal = trimesh.creation.box(extents=pedestal_size)
        bracket_size = _xyz(
            mount_cfg.get("bracket_size_m", (0.065, 0.050, 0.042)),
            name="mount.bracket_size_m",
        )
        if np.any(bracket_size <= 0.0):
            raise ValueError("camera mount bracket dimensions must be positive")
        bracket = trimesh.creation.box(extents=bracket_size)
        bracket_center = _xyz(
            mount_cfg.get("bracket_center_m", (0.055, 0.0, 0.451)),
            name="mount.bracket_center_m",
        )
        identity = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        parts.extend(
            (
                _visual_part(
                    "sensors/gemini_435le/pedestal",
                    pedestal,
                    (0.82, 0.82, 0.78),
                    pedestal_center,
                    identity,
                ),
                _visual_part(
                    "sensors/gemini_435le/bracket",
                    bracket,
                    (0.12, 0.12, 0.13),
                    bracket_center,
                    identity,
                ),
            )
        )
        return tuple(parts)

    def frustum_lines(self, length_m: float | None = None) -> np.ndarray:
        length = float(
            self.cfg.get("frustum_length_m", 0.18)
            if length_m is None
            else length_m
        )
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("camera frustum length must be positive")
        half_width = length * math.tan(0.5 * self.horizontal_fov_rad)
        half_height = length * math.tan(0.5 * self.vertical_fov_rad)
        origin = np.zeros(3, dtype=np.float64)
        corners = np.asarray(
            (
                (-half_width, -half_height, -length),
                (half_width, -half_height, -length),
                (half_width, half_height, -length),
                (-half_width, half_height, -length),
            ),
            dtype=np.float64,
        )
        strips = [[origin, corner] for corner in corners]
        strips.extend(
            [[corners[index], corners[(index + 1) % 4]] for index in range(4)]
        )
        local = np.asarray(strips, dtype=np.float64)
        rotation = Rotation.from_quat(self.quaternion_xyzw)
        return (
            rotation.apply(local.reshape(-1, 3)).reshape(8, 2, 3)
            + self.position
        ).astype(np.float32)


class Gemini305WristCameras:
    """Pair of photo-mounted Gemini 305 RGB-D cameras on the Wuji2 wrists."""

    sides = ("left", "right")

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.model = str(self.cfg.get("model", "orbbec_gemini_305"))
        if self.model != "orbbec_gemini_305":
            raise ValueError(f"unsupported wrist camera model {self.model!r}")

        self.dimensions = _xyz(
            self.cfg.get("housing_dimensions_m", (0.042, 0.042, 0.023)),
            name="wrist_cameras.housing_dimensions_m",
        )
        if np.any(self.dimensions <= 0.0):
            raise ValueError("wrist camera housing dimensions must be positive")
        self.baseline_m = float(self.cfg.get("stereo_baseline_m", 0.018))
        if (
            not math.isfinite(self.baseline_m)
            or not 0.0 < self.baseline_m < self.dimensions[0]
        ):
            raise ValueError("wrist camera stereo baseline must fit inside housing")

        self.sensor_cfg = dict(self.cfg.get("sensor", {}) or {})
        self.width = int(self.sensor_cfg.get("width", 320))
        self.height = int(self.sensor_cfg.get("height", 200))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("wrist camera sensor width and height must be positive")
        self.horizontal_fov_rad = math.radians(
            float(self.sensor_cfg.get("horizontal_fov_deg", 94.0))
        )
        self.vertical_fov_rad = math.radians(
            float(self.sensor_cfg.get("vertical_fov_deg", 68.0))
        )
        if not (
            0.0 < self.horizontal_fov_rad < math.pi
            and 0.0 < self.vertical_fov_rad < math.pi
        ):
            raise ValueError(
                "wrist camera fields of view must lie between 0 and 180 degrees"
            )
        clipping = np.asarray(
            self.sensor_cfg.get("clipping_range_m", (0.04, 1.0)),
            dtype=np.float64,
        )
        if (
            clipping.shape != (2,)
            or not np.isfinite(clipping).all()
            or clipping[0] < 0.0
            or clipping[1] <= clipping[0]
        ):
            raise ValueError("wrist camera clipping_range_m must be [near, far]")
        self.clipping_range_m = (float(clipping[0]), float(clipping[1]))
        self.intrinsic_matrix = _intrinsic_matrix(
            self.width,
            self.height,
            self.horizontal_fov_rad,
            self.vertical_fov_rad,
        )

        mounts = dict(self.cfg.get("mounts", {}) or {})
        self.local_positions: dict[str, np.ndarray] = {}
        self.local_quaternions_xyzw: dict[str, np.ndarray] = {}
        self.bracket_anchors: dict[str, np.ndarray] = {}
        default_positions = {
            "left": (0.0, -0.080, -0.025),
            "right": (0.080, 0.0, -0.025),
        }
        for side in self.sides:
            mount = dict(mounts.get(side, {}) or {})
            self.local_positions[side] = _xyz(
                mount.get("optical_position_in_wrist_m", default_positions[side]),
                name=f"wrist_cameras.mounts.{side}.optical_position_in_wrist_m",
            )
            self.local_quaternions_xyzw[side] = _normalized_quaternion(
                mount.get("optical_quat_in_wrist_xyzw", (0.0, 0.0, 0.0, 1.0)),
                name=f"wrist_cameras.mounts.{side}.optical_quat_in_wrist_xyzw",
            )
            self.bracket_anchors[side] = _xyz(
                mount.get("bracket_anchor_in_wrist_m", (0.0, 0.0, 0.0)),
                name=f"wrist_cameras.mounts.{side}.bracket_anchor_in_wrist_m",
            )
        self.visual_parts = self._make_visual_parts()

    @staticmethod
    def _wrist_pose(
        side: str,
        wrist_poses: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        return wrist_poses[0 if side == "left" else 1]

    def camera_poses(
        self,
        wrist_poses: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        positions = []
        quaternions = []
        for side in self.sides:
            wrist_pose = self._wrist_pose(side, wrist_poses)
            position, quaternion = _compose_pose(
                wrist_pose[:3],
                wrist_pose[3:],
                self.local_positions[side],
                self.local_quaternions_xyzw[side],
            )
            positions.append(position)
            quaternions.append(quaternion)
        return (
            np.asarray(positions, dtype=np.float32),
            np.asarray(quaternions, dtype=np.float32),
        )

    def visual_pose(
        self,
        part: WristCameraVisualPart,
        wrist_poses: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        wrist_pose = self._wrist_pose(part.side, wrist_poses)
        position, quaternion = _compose_pose(
            wrist_pose[:3],
            wrist_pose[3:],
            part.local_position,
            part.local_quaternion_xyzw,
        )
        return position.astype(np.float32), quaternion.astype(np.float32)

    def _part(
        self,
        side: str,
        name: str,
        mesh: trimesh.Trimesh,
        color: tuple[float, float, float],
        *,
        optical_frame: bool,
    ) -> WristCameraVisualPart:
        return WristCameraVisualPart(
            name=f"sensors/gemini_305_{side}/{name}",
            side=side,
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            color=color,
            local_position=(
                self.local_positions[side]
                if optical_frame
                else np.zeros(3, dtype=np.float64)
            ),
            local_quaternion_xyzw=(
                self.local_quaternions_xyzw[side]
                if optical_frame
                else np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
            ),
        )

    @staticmethod
    def _bracket_bar(start: np.ndarray, end: np.ndarray) -> trimesh.Trimesh:
        vector = np.asarray(end, dtype=np.float64) - np.asarray(
            start, dtype=np.float64
        )
        length = float(np.linalg.norm(vector))
        if length <= 1.0e-6:
            raise ValueError("wrist camera bracket endpoints must differ")
        bar = trimesh.creation.box(extents=(0.014, 0.006, length))
        rotation, _ = Rotation.align_vectors(
            (vector[None, :] / length),
            np.asarray(((0.0, 0.0, 1.0),)),
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation.as_matrix()
        transform[:3, 3] = 0.5 * (start + end)
        bar.apply_transform(transform)
        return bar

    def _make_visual_parts(self) -> tuple[WristCameraVisualPart, ...]:
        width, height, depth = self.dimensions
        front_clearance = 0.003
        parts: list[WristCameraVisualPart] = []
        for side in self.sides:
            housing = trimesh.creation.box(extents=self.dimensions)
            housing.apply_translation((0.0, 0.0, front_clearance + 0.5 * depth))
            front = trimesh.creation.box(
                extents=(0.94 * width, 0.94 * height, 0.002)
            )
            front.apply_translation((0.0, 0.0, front_clearance + 0.001))
            parts.extend(
                (
                    self._part(
                        side,
                        "housing",
                        housing,
                        (0.58, 0.60, 0.61),
                        optical_frame=True,
                    ),
                    self._part(
                        side,
                        "front_panel",
                        front,
                        (0.13, 0.14, 0.15),
                        optical_frame=True,
                    ),
                )
            )
            lens_x = (-0.5 * self.baseline_m, 0.5 * self.baseline_m)
            for index, x in enumerate(lens_x):
                lens = trimesh.creation.cylinder(
                    radius=0.0042,
                    height=0.0025,
                    sections=20,
                )
                lens.apply_translation((x, 0.0, front_clearance))
                parts.append(
                    self._part(
                        side,
                        f"lens_{index}",
                        lens,
                        (0.035, 0.07, 0.11),
                        optical_frame=True,
                    )
                )
            plate = trimesh.creation.box(extents=(0.048, 0.006, 0.048))
            plate.apply_translation((0.0, 0.0, front_clearance + depth + 0.003))
            parts.append(
                self._part(
                    side,
                    "mount_plate",
                    plate,
                    (0.88, 0.86, 0.78),
                    optical_frame=True,
                )
            )
            camera_rear = self.local_positions[side] + Rotation.from_quat(
                self.local_quaternions_xyzw[side]
            ).apply((0.0, 0.0, front_clearance + depth + 0.006))
            bracket = self._bracket_bar(self.bracket_anchors[side], camera_rear)
            parts.append(
                self._part(
                    side,
                    "bracket_arm",
                    bracket,
                    (0.88, 0.86, 0.78),
                    optical_frame=False,
                )
            )
        return tuple(parts)

    def frustum_lines(
        self,
        wrist_poses: tuple[np.ndarray, np.ndarray],
        length_m: float | None = None,
    ) -> dict[str, np.ndarray]:
        length = float(
            self.cfg.get("frustum_length_m", 0.15)
            if length_m is None
            else length_m
        )
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("wrist camera frustum length must be positive")
        half_width = length * math.tan(0.5 * self.horizontal_fov_rad)
        half_height = length * math.tan(0.5 * self.vertical_fov_rad)
        origin = np.zeros(3, dtype=np.float64)
        corners = np.asarray(
            (
                (-half_width, -half_height, -length),
                (half_width, -half_height, -length),
                (half_width, half_height, -length),
                (-half_width, half_height, -length),
            ),
            dtype=np.float64,
        )
        strips = [[origin, corner] for corner in corners]
        strips.extend(
            [[corners[index], corners[(index + 1) % 4]] for index in range(4)]
        )
        local = np.asarray(strips, dtype=np.float64)
        positions, quaternions = self.camera_poses(wrist_poses)
        return {
            side: (
                Rotation.from_quat(quaternions[index])
                .apply(local.reshape(-1, 3))
                .reshape(8, 2, 3)
                + positions[index]
            ).astype(np.float32)
            for index, side in enumerate(self.sides)
        }


def packed_rgba_to_rgb(packed: np.ndarray) -> np.ndarray:
    """Decode Newton's packed little-endian RGBA image to uint8 RGB."""

    values = np.asarray(packed, dtype=np.uint32)
    result = np.empty((*values.shape, 3), dtype=np.uint8)
    result[..., 0] = values & np.uint32(0xFF)
    result[..., 1] = (values >> np.uint32(8)) & np.uint32(0xFF)
    result[..., 2] = (values >> np.uint32(16)) & np.uint32(0xFF)
    return result


__all__ = [
    "CameraVisualPart",
    "Gemini305WristCameras",
    "Gemini435LeCamera",
    "WristCameraVisualPart",
    "packed_rgba_to_rgb",
]
