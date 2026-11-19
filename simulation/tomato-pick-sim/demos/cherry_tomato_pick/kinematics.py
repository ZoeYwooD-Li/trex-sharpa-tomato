"""Task-local FR3 / Wuji2 kinematic support extracted from LycheeSim.

MuJoCo computes FK/IK only; Newton owns contact and dynamics.
Source files and hashes are recorded in provenance/source_snapshot.json.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

TACTILE_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _rotation_to_xyzw(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()

def _xyzw_to_rotation(quaternion: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(quaternion))

def _color3(value: Any) -> np.ndarray:
    color = np.asarray(value, dtype=np.float32).reshape(-1)
    if color.size < 3:
        color = np.asarray((0.72, 0.74, 0.78), dtype=np.float32)
    color = color[:3]
    if float(np.max(color)) > 1.0:
        color = color / 255.0
    return color

@dataclass(frozen=True)
class VisualPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray
    source: str
    source_id: int

def _dae_link_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene = trimesh.load(path, force="scene", process=False)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        local = (
            np.asarray(geometry.vertices, dtype=np.float64)
            @ np.asarray(transform[:3, :3], dtype=np.float64).T
            + np.asarray(transform[:3, 3], dtype=np.float64)
        ).astype(np.float32)
        face = np.asarray(geometry.faces, dtype=np.int32)
        material = getattr(geometry.visual, "material", None)
        color = _color3(getattr(material, "main_color", (184, 189, 198, 255)))
        vertices.append(local)
        faces.append(face + offset)
        colors.append(np.repeat(color[None, :], len(local), axis=0))
        offset += len(local)
    return np.concatenate(vertices), np.concatenate(faces), np.concatenate(colors)

def minimum_jerk(amount: float) -> float:
    """Fifth-order minimum-jerk curve used by the Isaac implementation."""

    value = float(np.clip(amount, 0.0, 1.0))
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))

class _Fr3WujiKinematics:
    """Shared FK and visual helpers; Beta2 supplies mounting, IK and pads."""

    def __init__(self, cfg: dict[str, Any], side: str) -> None:
        if side not in {"left", "right"}:
            raise ValueError("Wuji hand side must be left or right")
        self.side = side
        self.carrier = mujoco.MjModel.from_xml_path(str(Path(cfg["carrier_mjcf"])))
        self.carrier_data = mujoco.MjData(self.carrier)
        self.hand = mujoco.MjModel.from_xml_path(str(Path(cfg[f"{cfg.get('hand_model', 'wuji')}_{side}_mjcf"])))
        self.hand_data = mujoco.MjData(self.hand)
        self._configure_source_mounts(cfg)

        self.left_joint_names = tuple(f"left_fr3v2_joint{i}" for i in range(1, 8))
        self.right_joint_names = tuple(f"right_fr3v2_joint{i}" for i in range(1, 8))
        self._left_joints = self._joint_ids(self.left_joint_names)
        self._right_joints = self._joint_ids(self.right_joint_names)
        self._left_qadr = self.carrier.jnt_qposadr[self._left_joints].astype(int)
        self._right_qadr = self.carrier.jnt_qposadr[self._right_joints].astype(int)
        self._left_dadr = self.carrier.jnt_dofadr[self._left_joints].astype(int)
        self._right_dadr = self.carrier.jnt_dofadr[self._right_joints].astype(int)
        self._left_lo, self._left_hi = self.carrier.jnt_range[self._left_joints].T
        self._right_lo, self._right_hi = self.carrier.jnt_range[self._right_joints].T
        self.left_link8_id = self._body_id(self.carrier, "left_fr3v2_link8")
        self.right_link8_id = self._body_id(self.carrier, "right_fr3v2_link8")

        default = np.asarray(
            cfg.get("default_arm_q", (0.0, -0.48, 0.0, -2.55, 0.0, 2.75, 0.78)),
            dtype=np.float64,
        )
        if default.shape != (7,):
            raise ValueError("robot.default_arm_q must contain seven positions")
        self.default_left_q = np.clip(default, self._left_lo, self._left_hi)
        self.default_right_q = np.clip(default, self._right_lo, self._right_hi)

        self._root_position = np.zeros(3, dtype=np.float64)
        self._root_rotation = Rotation.identity()
        self.tactile_parts = self._make_skin_parts()
        visual_dir = Path(str(cfg["fr3_visual_mesh_dir"])).expanduser().resolve()
        self.visual_parts = self._make_visual_parts(visual_dir)
        self.set_arm_q(self.default_left_q, self.default_right_q)
        self.set_hand_q(np.zeros(self.hand.nq, dtype=np.float64))
        self.sync_hand_to_arm()

    @staticmethod
    def _body_id(model: mujoco.MjModel, name: str) -> int:
        value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if value < 0:
            raise ValueError(f"kinematic model is missing body {name!r}")
        return int(value)

    def _joint_ids(self, names: tuple[str, ...]) -> np.ndarray:
        ids = np.asarray(
            [mujoco.mj_name2id(self.carrier, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names],
            dtype=int,
        )
        missing = [name for name, value in zip(names, ids, strict=True) if value < 0]
        if missing:
            raise ValueError(f"kinematic model is missing joints: {missing}")
        return ids

    def _configure_source_mounts(self, cfg: dict[str, Any]) -> None:
        for name, entry in (cfg.get("source_mounts", {}) or {}).items():
            body_id = self._body_id(self.carrier, name)
            self.carrier.body_pos[body_id] = np.asarray(entry["pos"], dtype=np.float64)
            quat = np.asarray(
                entry.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0)), dtype=np.float64
            )
            quat /= np.linalg.norm(quat)
            self.carrier.body_quat[body_id] = quat[[3, 0, 1, 2]]

    def set_arm_q(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        left = np.asarray(left_q, dtype=np.float64)
        right = np.asarray(right_q, dtype=np.float64)
        if left.shape != (7,) or right.shape != (7,):
            raise ValueError("FR3 arm commands must each have seven positions")
        self.carrier_data.qpos[self._left_qadr] = np.clip(left, self._left_lo, self._left_hi)
        self.carrier_data.qpos[self._right_qadr] = np.clip(
            right, self._right_lo, self._right_hi
        )
        mujoco.mj_forward(self.carrier, self.carrier_data)

    def set_hand_q(self, values: np.ndarray) -> None:
        q = np.asarray(values, dtype=np.float64)
        if q.shape != (self.hand.nq,):
            raise ValueError(f"Wuji Hand q must have shape {(self.hand.nq,)}, got {q.shape}")
        clipped = q.copy()
        for joint_id in range(self.hand.njnt):
            qadr = int(self.hand.jnt_qposadr[joint_id])
            if self.hand.jnt_limited[joint_id]:
                clipped[qadr] = np.clip(
                    clipped[qadr],
                    self.hand.jnt_range[joint_id, 0],
                    self.hand.jnt_range[joint_id, 1],
                )
        self.hand_data.qpos[:] = clipped
        mujoco.mj_forward(self.hand, self.hand_data)

    def set_hand_root_pose(self, position: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
        self._root_position = np.asarray(position, dtype=np.float64).copy()
        self._root_rotation = _xyzw_to_rotation(np.asarray(quaternion_xyzw))

    @staticmethod
    def _compiled_mesh(model: mujoco.MjModel, mesh_id: int) -> tuple[np.ndarray, np.ndarray]:
        va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        return (
            np.asarray(model.mesh_vert[va : va + vn], dtype=np.float32).copy(),
            np.asarray(model.mesh_face[fa : fa + fn], dtype=np.int32).copy(),
        )

    def hand_geom_pose(self, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
        local_position = self.hand_data.geom_xpos[geom_id]
        local_rotation = Rotation.from_matrix(self.hand_data.geom_xmat[geom_id].reshape(3, 3))
        return (
            self._root_position + self._root_rotation.apply(local_position),
            _rotation_to_xyzw(self._root_rotation * local_rotation),
        )

    def tactile_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            zone: self.hand_geom_pose(part.source_id)
            for zone, part in self.tactile_parts.items()
        }

    def tactile_centers(self) -> dict[str, np.ndarray]:
        result = {}
        for zone, part in self.tactile_parts.items():
            position, quaternion = self.hand_geom_pose(part.source_id)
            result[zone] = position + _xyzw_to_rotation(quaternion).apply(
                np.asarray(part.vertices, dtype=np.float64).mean(axis=0)
            )
        return result

    def _make_visual_parts(self, visual_dir: Path) -> list[VisualPart]:
        parts: list[VisualPart] = []
        for geom_id in range(self.carrier.ngeom):
            body_id = int(self.carrier.geom_bodyid[geom_id])
            if (
                mujoco.mj_id2name(self.carrier, mujoco.mjtObj.mjOBJ_BODY, body_id)
                == "base_link"
                and int(self.carrier.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            ):
                vertices, faces = self._compiled_mesh(
                    self.carrier, int(self.carrier.geom_dataid[geom_id])
                )
                color = _color3(self.carrier.geom_rgba[geom_id, :3])
                parts.append(
                    VisualPart(
                        "robot/dual_arm_base", vertices, faces,
                        np.repeat(color[None, :], len(vertices), axis=0),
                        "carrier_geom", geom_id,
                    )
                )
                break
        for side in ("left", "right"):
            for link_index in range(8):
                body_id = self._body_id(self.carrier, f"{side}_fr3v2_link{link_index}")
                vertices, faces, colors = _dae_link_mesh(
                    visual_dir / f"link{link_index}.dae"
                )
                parts.append(
                    VisualPart(
                        f"robot/{side}_fr3/link{link_index}", vertices, faces,
                        colors, "carrier_body", body_id,
                    )
                )
        for geom_id in range(self.hand.ngeom):
            if (
                int(self.hand.geom_group[geom_id]) != 1
                or float(self.hand.geom_rgba[geom_id, 3]) <= 0.0
                or int(self.hand.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)
            ):
                continue
            vertices, faces = self._compiled_mesh(
                self.hand, int(self.hand.geom_dataid[geom_id])
            )
            color = _color3(self.hand.geom_rgba[geom_id, :3])
            parts.append(
                VisualPart(
                    f"robot/{self.side}_wuji/geom_{geom_id:03d}", vertices, faces,
                    np.repeat(color[None, :], len(vertices), axis=0),
                    "hand_geom", geom_id,
                )
            )
        return parts

    def visual_pose(self, part: VisualPart) -> tuple[np.ndarray, np.ndarray]:
        if part.source == "carrier_body":
            position = self.carrier_data.xpos[part.source_id]
            rotation = Rotation.from_matrix(self.carrier_data.xmat[part.source_id].reshape(3, 3))
        elif part.source == "carrier_geom":
            position = self.carrier_data.geom_xpos[part.source_id]
            rotation = Rotation.from_matrix(
                self.carrier_data.geom_xmat[part.source_id].reshape(3, 3)
            )
        elif part.source == "hand_geom":
            return self.hand_geom_pose(part.source_id)
        else:
            raise ValueError(f"unknown visual source {part.source!r}")
        return np.asarray(position), _rotation_to_xyzw(rotation)
