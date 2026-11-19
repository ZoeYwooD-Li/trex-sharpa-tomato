"""Dual-FR3 / Wuji Hand 2 Beta2 adapter for cherry-tomato picking.

The official left and right ``with_mount`` models are attached directly to the
two FR3 ``link8`` frames.  MuJoCo supplies deterministic FK/IK and authored
mesh poses; Newton owns rigid contact through five fingertip pads and five
distal-link hulls per hand.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import _Fr3WujiKinematics
from .kinematics import TACTILE_FINGERS, VisualPart

from .robot import RobotCollisionPart, interpolate_pose


@dataclass(frozen=True)
class _HandMeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    source_id: int
    normal_local: np.ndarray


def _distal_first_amount(joint_name: str, closure: float) -> float:
    """Curl the tip before bringing the finger base into the bunch.

    Continuous at both endpoints; retain the controller's small overclosure.
    This reshapes the contact-controlled motion, never overrides its stopping.
    """
    if joint_name.lower().endswith(("_pip", "_dip")) or joint_name.lower() in ("thumb_mcp", "thumb_mcp_fe", "thumb_ip"):
        return min(2.0 * closure, 1.0) + max(closure - 1.0, 0.0)
    return max((closure - 0.25) / 0.75, 0.0)


def _resolve_robot_paths(
    cfg: dict[str, Any], repository_root: Path
) -> dict[str, Any]:
    result = dict(cfg)
    for key in (
        "carrier_mjcf",
        f"{cfg.get('hand_model', 'wuji')}_left_mjcf",
        f"{cfg.get('hand_model', 'wuji')}_right_mjcf",
        "fr3_visual_mesh_dir",
    ):
        if key not in result:
            raise ValueError(f"robot.{key} is required for Wuji2")
        path = Path(str(result[key])).expanduser()
        result[key] = str(path if path.is_absolute() else repository_root / path)
    return result


def _finger_body_name(side: str, finger: str, suffix: str) -> str:
    prefix = side[0]
    stem = "thumb" if finger == "thumb" else (
        "pinky" if finger == "pinky" else f"{finger}_finger"
    )
    return f"{prefix}_{stem}_{suffix}"


class _Fr3Wuji2Kinematics(_Fr3WujiKinematics):
    """One Wuji2 Beta2 ``with_mount`` model on one source-scene FR3."""

    def __init__(self, cfg: dict[str, Any], side: str) -> None:
        mount_position = cfg.get(
            f"{side}_mount_offset_m", cfg.get("mount_offset_m", (0.0, 0.0, 0.0))
        )
        mount_quaternion = cfg.get(
            f"{side}_mount_quat_xyzw",
            cfg.get("mount_quat_xyzw", (0.0, 0.0, 1.0, 0.0)),
        )
        self._mount_position = np.asarray(mount_position, dtype=np.float64)
        if self._mount_position.shape != (3,):
            raise ValueError(f"robot.{side}_mount_offset_m must contain XYZ")
        quaternion = np.asarray(mount_quaternion, dtype=np.float64)
        if quaternion.shape != (4,) or np.linalg.norm(quaternion) <= 1.0e-12:
            raise ValueError(
                f"robot.{side}_mount_quat_xyzw must contain a nonzero xyzw quaternion"
            )
        self._mount_rotation = Rotation.from_quat(quaternion / np.linalg.norm(quaternion))
        super().__init__(cfg, side)
        prefix = f"{side[0]}_"
        self.canonical_qadr = {
            str(mujoco.mj_id2name(self.hand, mujoco.mjtObj.mjOBJ_JOINT, joint_id))[
                len(prefix) :
            ]: int(self.hand.jnt_qposadr[joint_id])
            for joint_id in range(self.hand.njnt)
        }
        if self.hand.nq != 20 or len(self.canonical_qadr) != 20:
            raise ValueError(
                f"official Wuji2 {side} model must expose 20 scalar joints"
            )

    def _mesh_part_for_body(
        self, body_name: str, *, group: int, name: str
    ) -> _HandMeshPart:
        body_id = self._body_id(self.hand, body_name)
        for geom_id in range(self.hand.ngeom):
            if (
                int(self.hand.geom_bodyid[geom_id]) == body_id
                and int(self.hand.geom_type[geom_id])
                == int(mujoco.mjtGeom.mjGEOM_MESH)
                and int(self.hand.geom_group[geom_id]) == group
            ):
                vertices, faces = self._compiled_mesh(
                    self.hand, int(self.hand.geom_dataid[geom_id])
                )
                return _HandMeshPart(
                    name=name,
                    vertices=vertices,
                    faces=faces,
                    source_id=int(geom_id),
                    normal_local=np.asarray((0.0, 0.0, -1.0), dtype=np.float32),
                )
        raise ValueError(
            f"official Wuji2 model has no group-{group} mesh on {body_name!r}"
        )

    def _make_skin_parts(self) -> dict[str, _HandMeshPart]:
        return {
            finger: self._mesh_part_for_body(
                _finger_body_name(self.side, finger, "tip_sensor_frame"),
                group=2,
                name=f"{finger}_tip_sensor_frame",
            )
            for finger in TACTILE_FINGERS
        }

    def distal_parts(self) -> dict[str, _HandMeshPart]:
        return {
            finger: self._mesh_part_for_body(
                _finger_body_name(self.side, finger, "distal"),
                group=2,
                name=f"{finger}_distal",
            )
            for finger in TACTILE_FINGERS
        }

    def tip_positions(self) -> dict[str, np.ndarray]:
        """Return the five authored fingertip query sites in world coordinates."""

        result: dict[str, np.ndarray] = {}
        prefix = self.side[0]
        for finger in TACTILE_FINGERS:
            stem = "thumb" if finger == "thumb" else (
                "pinky" if finger == "pinky" else f"{finger}_finger"
            )
            site_name = f"{prefix}_{stem}_tip"
            site_id = mujoco.mj_name2id(
                self.hand, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if site_id < 0:
                raise ValueError(f"official Wuji2 model is missing site {site_name!r}")
            result[finger] = self._root_position + self._root_rotation.apply(
                self.hand_data.site_xpos[site_id]
            )
        return result

    def _make_visual_parts(self, visual_dir: Path) -> list[VisualPart]:
        return [
            replace(
                part,
                name=part.name.replace(
                    f"robot/{self.side}_wuji/", f"robot/{self.side}_wuji2/"
                ),
            )
            for part in super()._make_visual_parts(visual_dir)
        ]

    def sync_hand_to_arm(self) -> None:
        link_id = self.left_link8_id if self.side == "left" else self.right_link8_id
        link_rotation = Rotation.from_matrix(
            self.carrier_data.xmat[link_id].reshape(3, 3)
        )
        self._root_position = self.carrier_data.xpos[link_id] + link_rotation.apply(
            self._mount_position
        )
        self._root_rotation = link_rotation * self._mount_rotation

    def root_pose(self) -> np.ndarray:
        return np.r_[self._root_position, self._root_rotation.as_quat()]

    def set_hand_root_from_tcp(self, pose: np.ndarray) -> None:
        target = np.asarray(pose, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError("Wuji2 TCP pose must have shape (7,)")
        self.set_hand_root_pose(target[:3], target[3:])

    def solve_root_pose(
        self,
        position: np.ndarray,
        quaternion_xyzw: np.ndarray,
        seed_q: np.ndarray,
        *,
        max_iterations: int = 900,
    ) -> np.ndarray:
        target_root_position = np.asarray(position, dtype=np.float64)
        target_root_rotation = Rotation.from_quat(
            np.asarray(quaternion_xyzw, dtype=np.float64)
        )
        target_link_rotation = target_root_rotation * self._mount_rotation.inv()
        target_link_position = target_root_position - target_link_rotation.apply(
            self._mount_position
        )
        if self.side == "left":
            body_id, qadrs, dadrs = self.left_link8_id, self._left_qadr, self._left_dadr
            lower, upper = self._left_lo, self._left_hi
        else:
            body_id, qadrs, dadrs = self.right_link8_id, self._right_qadr, self._right_dadr
            lower, upper = self._right_lo, self._right_hi
        command = np.asarray(seed_q, dtype=np.float64)
        if command.shape != (7,):
            raise ValueError(f"{self.side} FR3 seed must contain seven positions")
        self.carrier_data.qpos[qadrs] = np.clip(command, lower, upper)
        position_error = np.full(3, np.inf)
        rotation_error = np.full(3, np.inf)
        for _ in range(max_iterations):
            mujoco.mj_forward(self.carrier, self.carrier_data)
            current_rotation = Rotation.from_matrix(
                self.carrier_data.xmat[body_id].reshape(3, 3)
            )
            position_error = target_link_position - self.carrier_data.xpos[body_id]
            rotation_error = (
                target_link_rotation * current_rotation.inv()
            ).as_rotvec()
            if (
                np.linalg.norm(position_error) <= 2.0e-5
                and np.linalg.norm(rotation_error) <= 2.0e-4
            ):
                break
            jacp = np.zeros((3, self.carrier.nv), dtype=np.float64)
            jacr = np.zeros((3, self.carrier.nv), dtype=np.float64)
            mujoco.mj_jacBody(
                self.carrier, self.carrier_data, jacp, jacr, body_id
            )
            jacobian = np.concatenate((jacp[:, dadrs], jacr[:, dadrs]), axis=0)
            error = np.concatenate((position_error, rotation_error))
            damping = 0.006
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            maximum = float(np.max(np.abs(delta)))
            if maximum > 0.05:
                delta *= 0.05 / maximum
            self.carrier_data.qpos[qadrs] = np.clip(
                self.carrier_data.qpos[qadrs] + 0.55 * delta,
                lower + 0.002,
                upper - 0.002,
            )
        if np.linalg.norm(position_error) > 0.015 or np.linalg.norm(
            rotation_error
        ) > 0.08:
            raise RuntimeError(
                f"{self.side} FR3 IK failed for Wuji2 root: "
                f"position_error={np.linalg.norm(position_error):.5f}, "
                f"orientation_error={np.linalg.norm(rotation_error):.5f}"
            )
        return self.carrier_data.qpos[qadrs].copy()


class DualWuji2PickRobot:
    """Two FR3 arms carrying official left/right Wuji2 Beta2 hands."""

    model_tag = "wuji2"
    kinematics_type = _Fr3Wuji2Kinematics
    tactile_fingers = TACTILE_FINGERS

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        repository_root: Path,
        tomato_position: np.ndarray,
        support_position: np.ndarray,
        harvest_validation_offset: np.ndarray | None = None,
    ) -> None:
        harvest_offset = np.asarray(
            (0., 0., 0.) if harvest_validation_offset is None else harvest_validation_offset,
            dtype=float,
        )
        if harvest_offset.shape != (3,) or not np.isfinite(harvest_offset).all():
            raise ValueError('harvest_validation_offset must contain finite XYZ')
        self.cfg = _resolve_robot_paths(cfg, repository_root)
        self.left = self.kinematics_type(self.cfg, "left")
        self.right = self.kinematics_type(self.cfg, "right")
        self.left_open_q = np.zeros(self.left.hand.nq, dtype=np.float64)
        if self.cfg.get("left_pregrasp_hand"):
            self.left_open_q = self._configured_hand_q("left_pregrasp_hand", self.left)
        self.right_open_q = np.zeros(self.right.hand.nq, dtype=np.float64)
        self.left_closed_q = self._configured_hand_q("left_harvest_hand", self.left)
        self.left_retention_q = self._retention_hand_q()
        self.right_closed_q = self._configured_hand_q(
            "right_cluster_grasp_hand", self.right
        )
        if self.cfg.get("right_pregrasp_hand"):
            self.right_open_q = self._configured_hand_q("right_pregrasp_hand", self.right)
        elif self.cfg.get("right_pregrasp_joint_offsets_rad"):
            self.right_open_q = self.right_closed_q.copy()
            for name, offset in self.cfg["right_pregrasp_joint_offsets_rad"].items():
                self.right_open_q[self.right.canonical_qadr[name]] += float(offset)
        self.left_grip = 0.0
        self.right_grip = 0.0
        self.right_contact_preload = np.zeros(3)
        self.right_force_fingers = tuple(
            self.cfg.get("right_force_fingers", ("thumb", "index", "middle"))
        )
        if (
            len(self.right_force_fingers) < 2
            or self.right_force_fingers[0] != "thumb"
            or len(set(self.right_force_fingers)) != len(self.right_force_fingers)
            or any(f not in self.tactile_fingers for f in self.right_force_fingers)
        ):
            raise ValueError("right_force_fingers must start with thumb and list distinct fingers")
        self.right_force_closure = None
        # Startup stays open. The scripted controller may form a separate
        # collision-calibrated reach aperture while parked away from the bunch.
        self.right_reach_q = (self._configured_hand_q("right_reach_hand", self.right)
                              if self.cfg.get("right_reach_hand") else None)
        self.right_preshape_amount = 0.0
        self.left_force_closure = None
        self.right_distal_first_fingers = tuple(self.cfg.get("right_distal_first_fingers", ()))
        if (len(set(self.right_distal_first_fingers)) != len(self.right_distal_first_fingers)
                or any(f not in self.right_force_fingers for f in self.right_distal_first_fingers)):
            raise ValueError("right_distal_first_fingers must select distinct right_force_fingers")
        self.ik_failed = False
        self._right_joint_waypoint = None

        self._set_kinematic_state(
            self.left.default_left_q,
            self.right.default_right_q,
            self.left_open_q,
            self.right_open_q,
        )
        self.default_left_tcp = self.left.root_pose()
        self.default_right_tcp = self.right.root_pose()

        self.left_grasp_tcp = self._calibrate_left_harvest(
            np.asarray(tomato_position, dtype=np.float64)
        )
        self.left_grasp_center = np.asarray(tomato_position, dtype=np.float64).copy()
        self.right_grasp_tcp = self._calibrate_right_cluster_grasp(
            np.asarray(support_position, dtype=np.float64)
        )
        self.right_grasp_center = np.asarray(support_position, dtype=np.float64) + np.asarray(
            self.cfg.get(
                "right_cluster_grasp_center_bias_m", (0.0, 0.0, 0.0)
            ),
            dtype=np.float64,
        )
        self._left_grasp_center_local = Rotation.from_quat(
            self.left_grasp_tcp[3:]
        ).inv().apply(self.left_grasp_center - self.left_grasp_tcp[:3])
        self._right_grasp_center_local = Rotation.from_quat(
            self.right_grasp_tcp[3:]
        ).inv().apply(self.right_grasp_center - self.right_grasp_tcp[:3])
        self.left_pregrasp_tcp = self.left_grasp_tcp.copy()
        self.left_pregrasp_tcp[:3] += np.asarray(
            self.cfg.get("left_pregrasp_offset_m", (0.0, -0.02, 0.11)),
            dtype=np.float64,
        )
        self.right_pregrasp_tcp = self.right_grasp_tcp.copy()
        self.right_pregrasp_tcp[:3] += np.asarray(
            self.cfg.get("right_pregrasp_offset_m", (0.0, 0.0, 0.09)),
            dtype=np.float64,
        )
        self.right_insert_tcp = None
        if self.cfg.get('right_stem_insert_offset_m') is not None:
            self.right_insert_tcp = self.right_grasp_tcp.copy()
            offset = np.asarray(self.cfg['right_stem_insert_offset_m'], dtype=float)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError('right_stem_insert_offset_m must contain finite XYZ')
            self.right_insert_tcp[:3] += offset

        self._left_q = self.left.default_left_q.copy()
        self._right_q = self.right.default_right_q.copy()
        left_seed = self._left_q.copy()
        right_seed = self._right_q.copy()
        self.right_approach_q = None
        if self.cfg.get('right_approach_ik_seed_q') is not None:
            right_seed = np.asarray(self.cfg['right_approach_ik_seed_q'], dtype=float)
            if (right_seed.shape != (7,) or not np.isfinite(right_seed).all()
                    or np.any(right_seed < self.right._right_lo) or np.any(right_seed > self.right._right_hi)):
                raise ValueError('right_approach_ik_seed_q must be seven joints within limits')
        for pose in (self.left_pregrasp_tcp, self.left_grasp_tcp):
            # Main-stem harvesting occurs after the bunch lift. Check that
            # actual planned height without translating the stored calibration
            # (the script adds the lift later, exactly once).
            left_seed = self.left.solve_root_pose(pose[:3] + harvest_offset, pose[3:], left_seed)
        right_waypoints = [self.right_pregrasp_tcp]
        if self.right_insert_tcp is not None:
            right_waypoints.append(self.right_insert_tcp)
        previous_pose = self.default_right_tcp
        for waypoint_index, pose in enumerate((*right_waypoints, self.right_grasp_tcp)):
            # Follow the same Cartesian route as execution. A large one-shot
            # orientation jump can trap local IK in a different elbow branch.
            joint_approach = waypoint_index == 0 and self.cfg.get('right_approach_ik_seed_q') is not None
            samples = (np.linspace(0., 1., 25)[1:] if self.cfg.get('right_validate_approach_path', False) and not joint_approach
                       else (1.,))
            for amount in samples:
                waypoint = interpolate_pose(previous_pose, pose, float(amount))
                right_seed = self.right.solve_root_pose(waypoint[:3], waypoint[3:], right_seed)
            if joint_approach:
                self.right_approach_q = right_seed.copy()
            previous_pose = pose
        self._set_kinematic_state(
            self.left.default_left_q,
            self.right.default_right_q,
            self.left_open_q,
            self.right_open_q,
        )

        self.collision_parts = self._make_collision_parts()
        self._left_visual_parts = tuple(self.left.visual_parts)
        self._right_visual_parts = tuple(
            part for part in self.right.visual_parts if part.source == "hand_geom"
        )
        self.visual_parts = self._left_visual_parts + self._right_visual_parts
        self.minimum_hand_clearance = float(self.cfg.get("minimum_interhand_distance_m", 0.0))
        if not np.isfinite(self.minimum_hand_clearance) or self.minimum_hand_clearance < 0.0:
            raise ValueError("minimum_interhand_distance_m must be finite and non-negative")
        self.hand_collision_blocked = False
        self.left_tableware = None
        self.minimum_left_tableware_clearance = 0.
        self.left_tableware_blocked = False
        self.left_tableware_clearance = float('inf')
        self.interhand_distance = 0.05 if self.minimum_hand_clearance > 0.0 else float("nan")
        self.minimum_observed_interhand_distance = self.interhand_distance
        self.hand_clearance = None
        if self.minimum_hand_clearance > 0.0:
            from .hand_clearance import HandClearance
            self.hand_clearance = HandClearance(self.visual_parts)
        # Convex-hull vertices preserve the mesh extrema used by the table
        # envelope exactly, while avoiding full visual tessellation per step.
        meshes = (zip(self.hand_clearance.parts, self.hand_clearance.hull_vertices, strict=True)
                  if self.hand_clearance is not None
                  else ((p, p.vertices) for p in self._left_visual_parts))
        self.left_tableware_meshes = tuple((p, v) for p, v in meshes
                                          if p.source == 'hand_geom' and f'/left_{self.model_tag}/' in p.name)

    @classmethod
    def tactile_body_name(cls, side: str, finger: str) -> str:
        return f"{side}_{cls.model_tag}_{finger}_patch"

    def _configured_hand_q(
        self, key: str, kin: _Fr3Wuji2Kinematics
    ) -> np.ndarray:
        entry = self.cfg.get(key)
        if not entry:
            raise ValueError(f"robot.{key} is required for Wuji2")
        names = tuple(str(name) for name in entry.get("joint_order", ()))
        positions = np.asarray(entry.get("positions", ()), dtype=np.float64)
        expected = set(kin.canonical_qadr)
        if (
            len(names) != kin.hand.nq
            or len(set(names)) != kin.hand.nq
            or set(names) != expected
            or positions.shape != (kin.hand.nq,)
        ):
            missing = sorted(expected - set(names))
            unknown = sorted(set(names) - expected)
            raise ValueError(
                f"robot.{key} must name all {kin.hand.nq} {self.model_tag} joints exactly once; "
                f"missing={missing}, unknown={unknown}"
            )
        values = np.zeros(kin.hand.nq, dtype=np.float64)
        for name, value in zip(names, positions, strict=True):
            values[kin.canonical_qadr[name]] = value
        kin.set_hand_q(values)
        return kin.hand_data.qpos.copy()

    def _retention_hand_q(self) -> np.ndarray:
        values = self.left_closed_q.copy()
        offsets = self.cfg.get("left_retention_joint_offsets_rad", {}) or {}
        unknown = sorted(set(offsets) - set(self.left.canonical_qadr))
        if unknown:
            raise ValueError(
                "robot.left_retention_joint_offsets_rad contains unknown "
                f"joints: {unknown}"
            )
        for name, offset in offsets.items():
            value = float(offset)
            if not np.isfinite(value):
                raise ValueError(
                    "robot.left_retention_joint_offsets_rad values must be finite"
                )
            values[self.left.canonical_qadr[name]] += value
        self.left.set_hand_q(values)
        return self.left.hand_data.qpos.copy()

    def _calibrate_left_harvest(self, target: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(self.cfg["left_harvest_quat_xyzw"], dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        pose = np.r_[target, quaternion]
        self.left.set_hand_q(self.left_closed_q)
        self.left.set_hand_root_from_tcp(pose)
        centers = self.left.tip_positions()
        midpoint = 0.5 * (
            centers["thumb"]
            + np.mean([centers["index"], centers["middle"]], axis=0)
        )
        pose[:3] += target - midpoint
        depth = float(self.cfg.get("left_harvest_depth_m", 0.0))
        if depth < 0.0:
            raise ValueError("robot.left_harvest_depth_m must be non-negative")
        palm_direction = target - pose[:3]
        palm_distance = float(np.linalg.norm(palm_direction))
        if depth > 0.0:
            if palm_distance <= depth:
                raise ValueError(
                    "robot.left_harvest_depth_m must be shorter than the "
                    "wrist-to-fruit distance"
                )
            pose[:3] += depth * palm_direction / palm_distance
        return pose

    def _calibrate_right_cluster_grasp(self, target: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(
            self.cfg["right_cluster_grasp_quat_xyzw"], dtype=np.float64
        )
        quaternion /= np.linalg.norm(quaternion)
        pose = np.r_[target, quaternion]
        self.right.set_hand_q(self.right_closed_q)
        self.right.set_hand_root_from_tcp(pose)
        centers = self.right.tip_positions()
        opposed = self.cfg.get("right_grasp_opposing_fingers", ["index", "middle"])
        midpoint = 0.5 * (
            centers["thumb"]
            + np.mean([centers[finger] for finger in opposed], axis=0)
        )
        pose[:3] += target - midpoint
        pose[:3] += np.asarray(
            self.cfg.get(
                "right_cluster_grasp_center_bias_m", (0.0, 0.0, 0.0)
            ),
            dtype=np.float64,
        )
        return pose

    def right_approach_pose(self, amount: float) -> np.ndarray:
        """FK of a bounded joint-space approach, preserving live robot state."""
        self._right_joint_waypoint = None
        if self.right_approach_q is None:
            return interpolate_pose(self.default_right_tcp, self.right_pregrasp_tcp, amount)
        try:
            self.right_approach_q = self.right.solve_root_pose(
                self.right_pregrasp_tcp[:3], self.right_pregrasp_tcp[3:], self.right_approach_q)
            u = float(np.clip(amount, 0., 1.)); u = u*u*(3.-2.*u)
            q = self.right.default_right_q + u * (self.right_approach_q - self.right.default_right_q)
            self.right.set_arm_q(self._left_q, q)
            self.right.sync_hand_to_arm()
            pose = self.right.root_pose()
            self._right_joint_waypoint = (pose.copy(), q.copy())
            return pose
        finally:
            self.right.set_arm_q(self._left_q, self._right_q)
            self.right.sync_hand_to_arm()

    def _make_collision_parts(self) -> tuple[RobotCollisionPart, ...]:
        parts: list[RobotCollisionPart] = []
        for side, kin in (("left", self.left), ("right", self.right)):
            for finger, source in kin.tactile_parts.items():
                parts.append(
                    RobotCollisionPart(
                        name=self.tactile_body_name(side, finger),
                        vertices=np.asarray(source.vertices, dtype=np.float32),
                        faces=np.asarray(source.faces, dtype=np.int32),
                        side=side,
                        finger=finger,
                        source_id=source.source_id,
                    )
                )
            for finger, source in kin.distal_parts().items():
                parts.append(
                    RobotCollisionPart(
                        name=f"{side}_{self.model_tag}_{finger}_distal",
                        vertices=np.asarray(source.vertices, dtype=np.float32),
                        faces=np.asarray(source.faces, dtype=np.int32),
                        side=side,
                        finger=None,
                        source_id=source.source_id,
                    )
                )
        return tuple(parts)

    def _set_kinematic_state(
        self,
        left_q: np.ndarray,
        right_q: np.ndarray,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
    ) -> None:
        self._left_q = np.asarray(left_q, dtype=np.float64).copy()
        self._right_q = np.asarray(right_q, dtype=np.float64).copy()
        self.left.set_arm_q(self._left_q, self._right_q)
        self.left.set_hand_q(left_hand_q)
        self.left.sync_hand_to_arm()
        self.right.set_arm_q(self._left_q, self._right_q)
        self.right.set_hand_q(right_hand_q)
        self.right.sync_hand_to_arm()

    def set_targets(
        self,
        left_tcp: np.ndarray,
        right_tcp: np.ndarray,
        left_grip: float,
        right_grip: float,
        left_retention: float = 0.0,
    ) -> None:
        joint_waypoint = self._right_joint_waypoint
        self._right_joint_waypoint = None  # One-shot; never reuse for policy/other TCP commands.
        if self.hand_collision_blocked or self.left_tableware_blocked:
            return
        previous = (self._left_q.copy(), self._right_q.copy(),
                    self.left.hand_data.qpos.copy(), self.right.hand_data.qpos.copy())
        left_pose = np.asarray(left_tcp, dtype=np.float64)
        right_pose = np.asarray(right_tcp, dtype=np.float64)
        if left_pose.shape != (7,) or right_pose.shape != (7,):
            raise ValueError("dual FR3 TCP targets must each have shape (7,)")
        self.left_grip = float(np.clip(left_grip, 0.0, 1.0))
        self.right_grip = float(np.clip(right_grip, 0.0, 1.0))
        retention = float(np.clip(left_retention, 0.0, 1.0))
        try:
            left_q = self.left.solve_root_pose(
                left_pose[:3], left_pose[3:], self._left_q, max_iterations=180
            )
            if joint_waypoint is not None and np.allclose(right_pose, joint_waypoint[0], atol=1e-6, rtol=0.):
                # This TCP was produced by FK of an already limit-checked
                # joint waypoint. Re-solving IK can switch redundant elbow
                # branches or fail at a singularity on the intended path.
                right_q = joint_waypoint[1]
            else:
                right_q = self.right.solve_root_pose(
                    right_pose[:3], right_pose[3:], self._right_q, max_iterations=180
                )
            self.ik_failed = False
        except RuntimeError:
            left_q = self._left_q
            right_q = self._right_q
            self.ik_failed = True
        left_hand_q = self.left_open_q + self.left_grip * (
            self.left_closed_q - self.left_open_q
        )
        left_hand_q += self.left_grip * retention * (
            self.left_retention_q - self.left_closed_q
        )
        if self.left_force_closure is not None:
            for finger_index, finger in enumerate(("thumb", "index", "middle")):
                for name, address in self.left.canonical_qadr.items():
                    if name.startswith(finger):
                        left_hand_q[address] = (
                            self.left_open_q[address]
                            + self.left_force_closure[finger_index]
                            * (self.left_closed_q[address] - self.left_open_q[address])
                            + self.left_grip * retention
                            * (self.left_retention_q[address] - self.left_closed_q[address])
                        )
        right_aperture_q = self.right_open_q
        if self.right_reach_q is not None:
            right_aperture_q = self.right_open_q + self.right_preshape_amount * (
                self.right_reach_q - self.right_open_q)
        right_hand_q = right_aperture_q + self.right_grip * (
            self.right_closed_q - right_aperture_q
        )
        if self.right_force_closure is not None:
            for finger_index, finger in enumerate(self.right_force_fingers):
                for name, address in self.right.canonical_qadr.items():
                    if name.startswith(finger):
                        amount = self.right_force_closure[finger_index]
                        if finger in self.right_distal_first_fingers:
                            amount = _distal_first_amount(name, float(amount))
                        right_hand_q[address] = (
                            right_aperture_q[address]
                            + amount
                            * (self.right_closed_q[address] - right_aperture_q[address])
                        )
        for name, offset in self.cfg.get("right_stem_preload_joint_offsets_rad", {}).items():
            finger_index = 0 if name.startswith("thumb") else 1 if name.startswith("index") else 2
            right_hand_q[self.right.canonical_qadr[name]] += (
                self.right_grip * self.right_contact_preload[finger_index] * float(offset)
            )
        self._set_kinematic_state(left_q, right_q, left_hand_q, right_hand_q)
        if self.left_tableware is not None and (
                not np.isfinite(self.left_tableware_clearance)
                or not np.array_equal(previous[0], left_q)
                or not np.array_equal(previous[2], left_hand_q)):
            from .stem_pinch import plate_clearance
            candidate = (left_q, right_q, left_hand_q, right_hand_q)
            for amount in (.5, 1.):
                self._set_kinematic_state(*[
                    old + amount * (new - old)
                    for old, new in zip(previous, candidate, strict=True)])
                gap = float('inf')
                for part, vertices in self.left_tableware_meshes:
                    pos, quat = self.visual_pose(part)
                    gap = min(gap, plate_clearance(
                        Rotation.from_quat(quat).apply(vertices) + pos, self.left_tableware))
                if gap < self.minimum_left_tableware_clearance:
                    self._set_kinematic_state(*previous)
                    self.left_tableware_blocked = True
                    return
                self.left_tableware_clearance = min(self.left_tableware_clearance, gap)
        if self.hand_clearance is not None:
            candidate = (left_q, right_q, left_hand_q, right_hand_q)
            distances = []
            # Sample the joint-space midpoint as well as the commanded endpoint.
            # This is not a continuous-collision or full-arm safety certificate.
            for amount in (0.5, 1.0):
                self._set_kinematic_state(*[
                    old + amount * (new - old)
                    for old, new in zip(previous, candidate, strict=True)
                ])
                distance = self.hand_clearance.distance(self.visual_pose)
                if distance < self.minimum_hand_clearance:
                    self._set_kinematic_state(*previous)
                    self.hand_collision_blocked = True
                    return
                distances.append(distance)
            self.interhand_distance = distances[-1]
            self.minimum_observed_interhand_distance = min(
                self.minimum_observed_interhand_distance, *distances
            )

    def collision_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            part.name: (
                self.left if part.side == "left" else self.right
            ).hand_geom_pose(part.source_id)
            for part in self.collision_parts
        }

    def tactile_centers(self, side: str) -> dict[str, np.ndarray]:
        if side == "left":
            return self.left.tactile_centers()
        if side == "right":
            return self.right.tactile_centers()
        raise ValueError("side must be 'left' or 'right'")

    def grasp_center(self, side: str) -> np.ndarray:
        centers = self.tactile_centers(side)
        return 0.5 * (
            centers["thumb"]
            + np.mean([centers[finger] for finger in ("index", "middle")], axis=0)
        )

    def nominal_grasp_center(self, side: str) -> np.ndarray:
        left_tcp, right_tcp = self.tcp_poses()
        if side == "left":
            return left_tcp[:3] + Rotation.from_quat(left_tcp[3:]).apply(
                self._left_grasp_center_local
            )
        if side == "right":
            return right_tcp[:3] + Rotation.from_quat(right_tcp[3:]).apply(
                self._right_grasp_center_local
            )
        raise ValueError("side must be 'left' or 'right'")

    def tcp_poses(self) -> tuple[np.ndarray, np.ndarray]:
        return self.left.root_pose(), self.right.root_pose()

    def arm_q(self) -> tuple[np.ndarray, np.ndarray]:
        return self._left_q.copy(), self._right_q.copy()

    def joint_names(self) -> tuple[str, ...]:
        """Return the dataset order for both FR3 arms and Wuji2 hands."""

        left_hand_names = tuple(
            name
            for name, _qadr in sorted(
                self.left.canonical_qadr.items(), key=lambda item: item[1]
            )
        )
        right_hand_names = tuple(
            name
            for name, _qadr in sorted(
                self.right.canonical_qadr.items(), key=lambda item: item[1]
            )
        )
        return (
            *(f"left_fr3/{name}" for name in self.left.left_joint_names),
            *(f"left_{self.model_tag}/{name}" for name in left_hand_names),
            *(f"right_fr3/{name}" for name in self.right.right_joint_names),
            *(f"right_{self.model_tag}/{name}" for name in right_hand_names),
        )

    def joint_q(self) -> np.ndarray:
        """Return positions in :meth:`joint_names` order."""

        return np.concatenate(
            (
                self._left_q,
                np.asarray(self.left.hand_data.qpos, dtype=np.float64),
                self._right_q,
                np.asarray(self.right.hand_data.qpos, dtype=np.float64),
            )
        ).astype(np.float32)

    def visual_pose(self, part: VisualPart) -> tuple[np.ndarray, np.ndarray]:
        kin = self.right if part.name.startswith(f"robot/right_{self.model_tag}/") else self.left
        position, quaternion = kin.visual_pose(part)
        return position.astype(np.float32), quaternion.astype(np.float32)


__all__ = ["DualWuji2PickRobot"]
