"""Task-local bilateral Sharpa Wave adapter using authored left/right assets."""
from dataclasses import replace
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import _Fr3WujiKinematics, TACTILE_FINGERS
from .wuji2 import _Fr3Wuji2Kinematics, _HandMeshPart, DualWuji2PickRobot


class Fr3SharpaKinematics(_Fr3Wuji2Kinematics):
    def __init__(self, cfg, side):
        self._mount_position = np.asarray(cfg.get(f"{side}_mount_offset_m", cfg["mount_offset_m"]), dtype=float)
        self._mount_rotation = Rotation.from_quat(cfg.get(f"{side}_mount_quat_xyzw", cfg["mount_quat_xyzw"]))
        _Fr3WujiKinematics.__init__(self, cfg, side)
        self.canonical_qadr = {
            mujoco.mj_id2name(self.hand, mujoco.mjtObj.mjOBJ_JOINT, j).removeprefix(f"{side}_"):
            int(self.hand.jnt_qposadr[j]) for j in range(self.hand.njnt)
        }
        if self.hand.nq != 22 or len(self.canonical_qadr) != 22:
            raise ValueError("Sharpa Wave requires 22 scalar joints per hand")

    def _part(self, finger, suffix):
        name = f"{self.side}_{finger}_{suffix}"
        gid = mujoco.mj_name2id(self.hand, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0 or self.hand.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError(f"Sharpa model is missing mesh {name}")
        vertices, faces = self._compiled_mesh(self.hand, int(self.hand.geom_dataid[gid]))
        return _HandMeshPart(name, vertices, faces, gid, np.array([0., 0., -1.], dtype=np.float32))

    def _make_skin_parts(self):
        return {f: self._part(f, "elastomer") for f in TACTILE_FINGERS}

    def distal_parts(self):
        return {f: self._part(f, "DP") for f in TACTILE_FINGERS}

    def tip_positions(self):
        # Mesh centroids and poses use the same compiled frame as Newton contact.
        return {f: p + Rotation.from_quat(q).apply(self.tactile_parts[f].vertices.mean(axis=0))
                for f, (p, q) in self.tactile_poses().items()}

    def _make_visual_parts(self, visual_dir):
        return [replace(p, name=p.name.replace(f"robot/{self.side}_wuji/", f"robot/{self.side}_sharpa/"))
                for p in _Fr3WujiKinematics._make_visual_parts(self, visual_dir)]


class DualSharpaPickRobot(DualWuji2PickRobot):
    model_tag = "sharpa"
    kinematics_type = Fr3SharpaKinematics
