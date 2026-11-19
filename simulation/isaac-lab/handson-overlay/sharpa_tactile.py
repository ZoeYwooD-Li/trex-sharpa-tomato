"""PhysX contact-to-tactile adapter for HandsOn's Wuji hands.

Each terminal finger link is treated as a small tactile pad. PhysX contact
impulses produce a local six-axis wrench; contact points are projected to a
240x240 mono8 map. The map is a practical visual-tactile proxy, not an optical
rendering or a calibrated model of the physical Sharpa sensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


IMAGE_SIZE = 240
SIDES = ("left", "right")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_HAND_LINK_INDEX = {name: index for index, name in enumerate(FINGERS, start=1)}


@dataclass(frozen=True)
class TactileGeometry:
    width_m: float = 0.017
    height_m: float = 0.017
    max_normal_force_n: float = 8.0
    contact_sigma_m: float = 0.0015


@dataclass(frozen=True)
class TactileReading:
    wrench: np.ndarray  # float32, [Fx, Fy, Fz, Tx, Ty, Tz], sensor frame
    deformation: np.ndarray  # uint8 mono8, 240x240


def quaternion_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    """Return the world-from-local rotation for an Isaac scalar-first quat."""

    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return matrix


def _as_xyz_rows(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return array.reshape(-1, 3)


def _contact_arrays(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read Isaac raw contacts while supporting current-frame API variants."""

    contacts = frame.get("contacts", []) if frame else []
    if contacts:
        positions, impulses = [], []
        for contact in contacts:
            if contact.get("position") is None or contact.get("impulse") is None:
                continue
            positions.append(_as_xyz_rows(contact["position"])[0])
            impulses.append(_as_xyz_rows(contact["impulse"])[0])
        if positions:
            return np.stack(positions), np.stack(impulses)
    # Retain support for adapters that expose flattened contact arrays.
    positions = _as_xyz_rows(frame.get("position", []) if frame else [])
    impulses = _as_xyz_rows(frame.get("impulse", []) if frame else [])
    if positions.shape != impulses.shape:
        return np.empty((0, 3)), np.empty((0, 3))
    return positions, impulses


def synthesize_tactile_reading(
    *,
    positions_world_m: np.ndarray,
    impulses_world_ns: np.ndarray,
    sensor_position_world_m: np.ndarray,
    sensor_rotation_world_from_local: np.ndarray,
    physics_dt_s: float,
    geometry: TactileGeometry,
) -> TactileReading:
    """Convert contact points and impulses to a mono8 map and local-frame F6."""

    positions = _as_xyz_rows(positions_world_m)
    impulses = _as_xyz_rows(impulses_world_ns)
    if positions.shape != impulses.shape:
        raise ValueError("Contact positions and impulses must have matching (N, 3) shapes")
    if not np.isfinite(physics_dt_s) or physics_dt_s <= 0:
        raise ValueError("physics_dt_s must be finite and positive")

    origin = np.asarray(sensor_position_world_m, dtype=np.float64).reshape(3)
    rotation = np.asarray(sensor_rotation_world_from_local, dtype=np.float64).reshape(3, 3)
    force_local = (impulses / float(physics_dt_s)) @ rotation
    lever_local = (positions - origin) @ rotation
    torque_local = np.cross(lever_local, force_local)
    wrench = np.concatenate((force_local.sum(axis=0), torque_local.sum(axis=0))).astype(np.float32)

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    if len(positions):
        point_local = (positions - origin) @ rotation
        # A 1 micron tolerance absorbs USD/PhysX roundoff at the active-patch edge.
        tolerance = 1e-6
        valid = (
            (np.abs(point_local[:, 0]) <= geometry.width_m / 2 + tolerance)
            & (np.abs(point_local[:, 1]) <= geometry.height_m / 2 + tolerance)
        )
        px = (point_local[:, 0] / geometry.width_m + 0.5) * (IMAGE_SIZE - 1)
        py = (0.5 - point_local[:, 1] / geometry.height_m) * (IMAGE_SIZE - 1)
        sigma_x = max(geometry.contact_sigma_m / geometry.width_m * IMAGE_SIZE, 0.5)
        sigma_y = max(geometry.contact_sigma_m / geometry.height_m * IMAGE_SIZE, 0.5)
        radius_x, radius_y = int(np.ceil(3 * sigma_x)), int(np.ceil(3 * sigma_y))
        normal_force = np.abs(force_local[:, 2])
        for x, y, force_n in zip(px[valid], py[valid], normal_force[valid]):
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - radius_x), min(IMAGE_SIZE, ix + radius_x + 1)
            y0, y1 = max(0, iy - radius_y), min(IMAGE_SIZE, iy + radius_y + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            xx = np.arange(x0, x1, dtype=np.float64)[None, :]
            yy = np.arange(y0, y1, dtype=np.float64)[:, None]
            gaussian = np.exp(-0.5 * (((xx - x) / sigma_x) ** 2 + ((yy - y) / sigma_y) ** 2))
            brightness = 255.0 * np.clip(force_n / geometry.max_normal_force_n, 0.0, 1.0)
            image[y0:y1, x0:x1] = np.maximum(image[y0:y1, x0:x1], brightness * gaussian)

    return TactileReading(wrench=wrench, deformation=np.rint(image).astype(np.uint8))


def _finger_body_prim(stage: Any, side: str, finger_index: int) -> Any:
    from pxr import Usd, UsdPhysics

    expected = f"{side}_finger{finger_index}_link4"
    side_root = "RightRobot" if side == "right" else "LeftRobot"
    root_path = f"/World/HandsOnRobots/{side_root}"
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise RuntimeError(f"Wuji robot root is missing: {root_path}")
    # The assembled MJCF has a deterministic base_{side} link. Prefer direct
    # USD lookup; traversing all mesh descendants is slow because each hand
    # contains detailed visual imports.
    for candidate_path in (
        f"{root_path}/base_{side}/{expected}",
        f"{root_path}/{expected}",
    ):
        candidate = stage.GetPrimAtPath(candidate_path)
        if candidate and candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"[HandsOn tactile] Found rigid body {candidate_path}", flush=True)
            return candidate
    matches = []
    for prim in Usd.PrimRange(root):
        name = str(prim.GetName())
        path = str(prim.GetPath())
        if (
            name in {expected, f"{expected}_1"}
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            matches.append(prim)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {side} Wuji fingertip body {expected!r} below {side_root}; "
            f"found {len(matches)}: {[str(prim.GetPath()) for prim in matches]}"
        )
    return matches[0]


class IsaacSharpaTactileRuntime:
    """Attach ten IsaacLab PhysX contact views and capture tactile observations."""

    def __init__(self, native_env: Any):
        if str(getattr(native_env, "_physics_backend", "physx")).lower() != "physx":
            raise RuntimeError("Sharpa-style contact sensing currently requires the Isaac Sim PhysX backend")
        if not getattr(native_env, "_stage", None):
            raise RuntimeError("Isaac task stage is not available; create the native task environment first")

        from pxr import PhysxSchema, Usd, UsdPhysics
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        self.env = native_env
        self.dt = float(getattr(native_env, "_physics_dt", 1.0 / 120.0))
        self._sensors: list[list[Any]] = []
        self._geometry = TactileGeometry()
        self._wrench_integral = np.zeros((2, 5, 6), dtype=np.float64)
        self._wrench_peak = np.zeros((2, 5, 6), dtype=np.float32)
        self._peak_norm = np.zeros((2, 5), dtype=np.float64)
        self._deformation_peak = np.zeros((2, 5, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        self._window_duration_s = 0.0

        # Contact filters need exact actor paths.  IsaacLab's tensor contact
        # view reports net forces even for unfiltered static contacts; these
        # paths additionally provide contact positions/friction for moment and
        # tactile-map estimates where the scene exposes rigid actors.
        excluded_roots = ("/World/HandsOnRobots/",)
        self._contact_filter_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(native_env._stage.GetPseudoRoot())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and not str(prim.GetPath()).startswith(excluded_roots)
        ]

        for side_index, side in enumerate(SIDES):
            side_sensors = []
            for finger_index in range(1, 6):
                print(
                    f"[HandsOn tactile] Resolving {side}/{FINGERS[finger_index - 1]} fingertip",
                    flush=True,
                )
                body = _finger_body_prim(native_env._stage, side, finger_index)
                body_path = str(body.GetPath())
                sensor_path = f"{body_path}/SharpaTactileContact"
                print(f"[HandsOn tactile] Enabling PhysX contact report on {body_path}", flush=True)
                PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
                print(f"[HandsOn tactile] Creating tensor contact view {sensor_path}", flush=True)
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=body_path,
                    update_period=0.0,
                    track_pose=True,
                    track_contact_points=bool(self._contact_filter_paths),
                    track_friction_forces=bool(self._contact_filter_paths),
                    max_contact_data_count_per_prim=16,
                    filter_prim_paths_expr=self._contact_filter_paths,
                ))
                side_sensors.append(sensor)
                print(
                    f"[HandsOn tactile] Attached {side}/{FINGERS[finger_index - 1]} contact sensor "
                    f"({side_index * 5 + finger_index}/10)",
                    flush=True,
                )
            self._sensors.append(side_sensors)

        self._original_step = native_env._step_isaaclab

        def step_and_capture() -> None:
            self._original_step()
            self.capture_substep()

        native_env._step_isaaclab = step_and_capture

    def capture_substep(self) -> None:
        self._window_duration_s += self.dt
        for side_index, side_sensors in enumerate(self._sensors):
            for finger_index, sensor in enumerate(side_sensors):
                sensor.update(self.dt, force_recompute=True)
                data = sensor.data
                normal_force_world = _as_xyz_rows(data.net_forces_w)[0]
                friction_forces = (
                    _as_xyz_rows(data.friction_forces_w)
                    if data.friction_forces_w is not None else np.empty((0, 3), dtype=np.float64)
                )
                force_world = normal_force_world + (
                    np.nansum(friction_forces, axis=0) if len(friction_forces) else 0.0
                )
                sensor_position = _as_xyz_rows(data.pos_w)[0]
                sensor_quaternion = np.asarray(data.quat_w).reshape(-1, 4)[0]
                rotation = quaternion_wxyz_to_matrix(sensor_quaternion)
                # Match the distal-pad offset used by the previous sensor
                # placement. IsaacLab's contact reporter itself is on the
                # rigid body; the local pad center is used for projection.
                pad_position = sensor_position + rotation @ np.array([0.0, 0.0, 0.0267])
                positions, forces = np.empty((0, 3)), np.empty((0, 3))
                if data.contact_pos_w is not None and data.force_matrix_w is not None:
                    contact_positions = _as_xyz_rows(data.contact_pos_w)
                    normal_by_object = _as_xyz_rows(data.force_matrix_w)
                    friction_by_object = (
                        _as_xyz_rows(data.friction_forces_w)
                        if data.friction_forces_w is not None else np.zeros_like(normal_by_object)
                    )
                    forces_by_object = normal_by_object + friction_by_object
                    valid = np.isfinite(contact_positions).all(axis=1) & np.isfinite(forces_by_object).all(axis=1)
                    positions, forces = contact_positions[valid], forces_by_object[valid]
                if len(positions) == 0 and np.linalg.norm(force_world) > 1e-6:
                    positions = pad_position.reshape(1, 3)
                    forces = force_world.reshape(1, 3)
                reading = synthesize_tactile_reading(
                    positions_world_m=positions,
                    impulses_world_ns=forces * self.dt,
                    sensor_position_world_m=pad_position,
                    sensor_rotation_world_from_local=rotation,
                    physics_dt_s=self.dt,
                    geometry=self._geometry,
                )
                reading.wrench[:3] = force_world @ rotation
                if len(positions):
                    torque_local = np.cross(positions - pad_position, forces) @ rotation
                    reading.wrench[3:] = torque_local.sum(axis=0)
                self._wrench_integral[side_index, finger_index] += reading.wrench * self.dt
                norm = float(np.linalg.norm(reading.wrench[:3]))
                if norm > self._peak_norm[side_index, finger_index]:
                    self._peak_norm[side_index, finger_index] = norm
                    self._wrench_peak[side_index, finger_index] = reading.wrench
                self._deformation_peak[side_index, finger_index] = np.maximum(
                    self._deformation_peak[side_index, finger_index], reading.deformation
                )

    def clear_window(self) -> None:
        self._wrench_integral.fill(0.0)
        self._wrench_peak.fill(0.0)
        self._peak_norm.fill(0.0)
        self._deformation_peak.fill(0)
        self._window_duration_s = 0.0

    def consume_observation(self) -> dict[str, np.ndarray]:
        if self._window_duration_s > 0.0:
            wrench = (self._wrench_integral / self._window_duration_s).astype(np.float32)
        else:
            wrench = np.zeros((2, 5, 6), dtype=np.float32)
        deformation = self._deformation_peak.copy()
        peak_wrench = self._wrench_peak.copy()
        self.clear_window()
        return {
            "physx_tactile_wrench": wrench,
            "physx_tactile_wrench_peak": peak_wrench,
            "physx_tactile_deformation": deformation,
        }

    def restore_step_hook(self) -> None:
        self.env._step_isaaclab = self._original_step
