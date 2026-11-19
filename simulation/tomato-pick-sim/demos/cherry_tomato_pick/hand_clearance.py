"""CPU convex-mesh clearance checks for the two kinematically driven hands.

This is a motion guard, not a dynamic hand-contact solver. It includes palms
and proximal links from the authored visuals, not just Newton's fingertip hulls.
"""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


class HandClearance:
    def __init__(self, parts) -> None:
        self.parts = tuple(p for p in parts if p.source == "hand_geom")
        corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        self.local_bounds = [np.where(corners, p.vertices.max(axis=0), p.vertices.min(axis=0))
                             for p in self.parts]
        hulls = [ConvexHull(p.vertices) for p in self.parts]
        self.hull_vertices = [p.vertices[h.vertices] for p, h in zip(self.parts, hulls, strict=True)]
        self.hull_axes = [np.unique(np.round(h.equations[:, :3], 10), axis=0) for h in hulls]
        spec = mujoco.MjSpec()
        for i, part in enumerate(self.parts):
            spec.add_mesh(name=f"mesh_{i}", uservert=part.vertices.ravel(),
                          userface=part.faces.ravel())
            spec.worldbody.add_geom(name=f"geom_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                                    meshname=f"mesh_{i}", contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.local_positions = self.model.geom_pos.copy()
        self.local_rotations = Rotation.from_quat(self.model.geom_quat[:, [1, 2, 3, 0]])
        self.left = np.array([i for i, p in enumerate(self.parts) if p.name.startswith("robot/left_")], dtype=int)
        self.right = np.array([i for i, p in enumerate(self.parts) if p.name.startswith("robot/right_")], dtype=int)
        if not self.left.size or not self.right.size:
            raise ValueError("hand clearance requires meshes for both hands")
        self.closest_pair = None

    def distance(self, pose_for_part, *, limit: float = 0.05) -> float:
        world_vertices = []
        source_poses = []
        for i, part in enumerate(self.parts):
            position, quaternion = pose_for_part(part)
            rotation = Rotation.from_quat(quaternion)
            source_poses.append((position, rotation))
            self.data.geom_xpos[i] = position + rotation.apply(self.local_positions[i])
            self.data.geom_xmat[i] = (rotation * self.local_rotations[i]).as_matrix().ravel()
            world_vertices.append(rotation.apply(self.local_bounds[i]) + position)
        positions = self.data.geom_xpos
        radii = self.model.geom_rbound
        delta = positions[self.left, None] - positions[self.right][None]
        lower = np.linalg.norm(delta, axis=2) - radii[self.left, None] - radii[self.right][None]
        best = limit
        self.closest_pair = None
        for a, b in np.argwhere(lower < limit):
            i, j = int(self.left[a]), int(self.right[b])
            distance = mujoco.mj_geomDistance(self.model, self.data, i, j, limit, None)
            if distance <= 0.0:
                # A narrow-phase zero is ambiguous near the query cutoff. A
                # positive AABB separation certifies a real lower bound; never
                # override an overlapping AABB with an unproven positive gap.
                a, b = world_vertices[i], world_vertices[j]
                gaps = np.maximum(np.maximum(a.min(axis=0) - b.max(axis=0),
                                             b.min(axis=0) - a.max(axis=0)), 0.0)
                distance = max(distance, float(np.linalg.norm(gaps)))
                if distance < limit:
                    # Rotated/tapered links can have overlapping or narrowly
                    # separated world AABBs while the meshes are far apart.
                    # A tiny positive AABB bound is NOT the actual mesh gap;
                    # still seek a stronger certificate. A positive
                    # projection gap along ANY unit axis is a certified lower
                    # bound on Euclidean separation. Face axes need not form a
                    # complete SAT test: no found gap means keep the rejection.
                    pi, ri = source_poses[i]
                    pj, rj = source_poses[j]
                    vi = ri.apply(self.hull_vertices[i]) + pi
                    vj = rj.apply(self.hull_vertices[j]) + pj
                    axes = np.concatenate((ri.apply(self.hull_axes[i]), rj.apply(self.hull_axes[j])))
                    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
                    ai, aj = vi @ axes.T, vj @ axes.T
                    gaps = np.maximum(aj.min(axis=0)-ai.max(axis=0),
                                      ai.min(axis=0)-aj.max(axis=0))
                    distance = max(distance, float(gaps.max()))
            if distance < best:
                best = distance
                self.closest_pair = (self.parts[i].name, self.parts[j].name)
        return float(best)
