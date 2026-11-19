"""Recorder frontend: write a versioned .npz cache for the offline render layer.

Schema (see ARCHITECTURE.md §4.1)::

    schema_version : int                       (currently 2)
    times          : (F,)         float32
    pos            : (F, E, 3)    float32      (E = effector count, may be 0)
    quat           : (F, E, 4)    float32      xyzw
    particles      : (F, N, 3)    float32      (N = particle count, may be 0)
    lines          : (F, S, P, 3) float32      (S strips, P points/strip, may be 0)
    meta           : json-encoded dict

A sidecar directory `<stem>_meshes/` holds rest meshes per registered solid mesh.

Optional additive arrays (introduced after schema v1, ignored by older
loaders):

    robot_q__<id>  : (F, n_dof) float32        per-robot joint trajectory
    deformable_vertices : (F, V, 3) float32     concatenated deformable meshes

Robot specs are echoed into ``meta["robots"]`` as a list of dicts with keys
``id``, ``mjcf``, ``ee_link``, ``base_pose``, ``n_dof``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .base import Frontend


SCHEMA_VERSION = 2


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# {len(verts)} verts, {len(faces)} tris\n")
        for v in verts:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for t in faces:
            f.write(f"f {int(t[0])+1} {int(t[1])+1} {int(t[2])+1}\n")


class RecorderFrontend(Frontend):
    """Append-only buffer; flushes a versioned .npz on close()."""

    def __init__(self, out_path: str | Path, meta: dict[str, Any] | None = None) -> None:
        self.out_path = Path(out_path)
        self.meta: dict[str, Any] = dict(meta or {})
        self._mesh_dir = self.out_path.with_suffix("").with_name(
            self.out_path.stem + "_meshes")
        self._mesh_names: list[str] = []
        self._mesh_visible: list[bool] = []
        self._mesh_colors: list[list[float]] = []
        self._particle_names: list[str] = []
        self._line_names: list[str] = []
        self._robot_specs: list[dict[str, Any]] = []     # add_robot order
        self._robot_q: dict[str, list[np.ndarray]] = {}  # id → list of (n_dof,)
        self._deformable_mesh_names: list[str] = []
        self._deformable_vertex_counts: list[int] = []

        # Per-frame buffers, parallel by registered name.
        self._times: list[float] = []
        self._pos: list[list[np.ndarray]] = []           # [frame][effector] = (3,)
        self._quat: list[list[np.ndarray]] = []          # [frame][effector] = (4,)
        self._particles: list[list[np.ndarray]] = []     # [frame][cloud] = (N, 3)
        self._lines: list[list[np.ndarray]] = []         # [frame][strip-group] = (S, P, 3)

        # Per-frame staging: filled by update_*, drained by frame().
        self._stage_pos: dict[str, np.ndarray] = {}
        self._stage_quat: dict[str, np.ndarray] = {}
        self._stage_particles: dict[str, np.ndarray] = {}
        self._stage_lines: dict[str, np.ndarray] = {}
        self._stage_robot_q: dict[str, np.ndarray] = {}
        self._stage_deformable_vertices: dict[str, np.ndarray] = {}
        self._deformable_vertices: list[list[np.ndarray]] = []

    def add_mesh(self, name, vertices, indices, color=(0.8, 0.8, 0.85), solid=False,
                 normals=None, vertex_colors=None, texcoords=None,
                 albedo_texture=None):
        self._mesh_dir.mkdir(parents=True, exist_ok=True)
        _write_obj(self._mesh_dir / f"{name}.obj",
                   np.asarray(vertices, dtype=np.float32),
                   np.asarray(indices, dtype=np.int32))
        self._mesh_names.append(name)
        self._mesh_visible.append(True)
        self._mesh_colors.append(
            [float(value) for value in np.asarray(color).reshape(-1)[:3]]
        )

    def add_deformable_mesh(
        self,
        name,
        vertices,
        indices,
        color=(0.8, 0.8, 0.85),
        solid=True,
        normals=None,
        vertex_colors=None,
        texcoords=None,
        albedo_texture=None,
    ):
        self.add_mesh(
            name,
            vertices,
            indices,
            color=color,
            solid=solid,
            normals=normals,
            vertex_colors=vertex_colors,
            texcoords=texcoords,
            albedo_texture=albedo_texture,
        )
        values = np.asarray(vertices, dtype=np.float32)
        self._deformable_mesh_names.append(str(name))
        self._deformable_vertex_counts.append(int(len(values)))
        self._stage_deformable_vertices[str(name)] = values.copy()

    def add_particles(self, name, count, radius, color=(0.2, 0.5, 1.0)):
        self._particle_names.append(name)
        self.meta.setdefault("particle_radii", {})[name] = float(radius)

    def add_lines(self, name, n_strips, points_per_strip, color=(0.85, 0.55, 0.20),
                  radius=0.0008):
        self._line_names.append(name)
        self.meta.setdefault("line_radii", {})[name] = float(radius)

    def update_pose(self, name, pos, quat):
        if name in self._mesh_names:
            self._stage_pos[name] = np.asarray(pos, dtype=np.float32)
            self._stage_quat[name] = np.asarray(quat, dtype=np.float32)

    def update_mesh(self, name, vertices, normals=None):
        del normals
        if name not in self._deformable_mesh_names:
            return
        values = np.asarray(vertices, dtype=np.float32)
        expected = self._deformable_vertex_counts[
            self._deformable_mesh_names.index(name)
        ]
        if values.shape != (expected, 3):
            raise ValueError(
                f"deformable mesh {name!r} expected ({expected}, 3) vertices, "
                f"got {values.shape}"
            )
        self._stage_deformable_vertices[name] = values.copy()

    def update_particles(self, name, positions, colors=None):
        # ``colors`` is informational here — the recorder NPZ schema v1 does not
        # store per-particle colors yet, so this argument is accepted but ignored.
        if name in self._particle_names:
            self._stage_particles[name] = np.asarray(positions, dtype=np.float32)

    def update_lines(self, name, positions):
        if name in self._line_names:
            self._stage_lines[name] = np.asarray(positions, dtype=np.float32)

    def add_robot(self, name, mjcf, ee_link=None, base_pose=(0.0, 0.0, 0.0)):
        if any(s["id"] == name for s in self._robot_specs):
            raise ValueError(f"recorder: robot {name!r} already registered")
        self._robot_specs.append({
            "id": str(name),
            "mjcf": str(mjcf),
            "ee_link": ee_link,
            "base_pose": [float(x) for x in base_pose],
            "n_dof": None,             # filled in on first update_joint_q
        })
        self._robot_q[str(name)] = []

    def update_joint_q(self, name, q):
        key = str(name)
        if key not in self._robot_q:
            return
        self._stage_robot_q[key] = np.asarray(q, dtype=np.float32).reshape(-1)

    def frame(self, t):
        self._times.append(float(t))
        self._pos.append([self._stage_pos.get(n, np.zeros(3, dtype=np.float32))
                          for n in self._mesh_names])
        self._quat.append([self._stage_quat.get(n, np.array([0, 0, 0, 1], dtype=np.float32))
                           for n in self._mesh_names])
        self._particles.append([self._stage_particles.get(n) for n in self._particle_names])
        self._lines.append([self._stage_lines.get(n) for n in self._line_names])
        self._deformable_vertices.append(
            [
                self._stage_deformable_vertices[name].copy()
                for name in self._deformable_mesh_names
            ]
        )

        for spec in self._robot_specs:
            rid = spec["id"]
            q = self._stage_robot_q.get(rid)
            if q is None:
                # Reuse last frame if missing; if no frame seen yet, drop a
                # zero of unknown width — n_dof gets fixed at the first real q.
                q = (self._robot_q[rid][-1].copy() if self._robot_q[rid]
                     else np.zeros(0, dtype=np.float32))
            else:
                if spec["n_dof"] is None:
                    spec["n_dof"] = int(q.shape[0])
            self._robot_q[rid].append(q)

    def close(self):
        if not self._times:
            logger.warning("recorder: no frames captured, not writing {}", self.out_path)
            return

        F = len(self._times)
        E = len(self._mesh_names)
        times = np.asarray(self._times, dtype=np.float32)
        pos = (np.stack([np.stack(p) for p in self._pos]) if E
               else np.zeros((F, 0, 3), dtype=np.float32))
        quat = (np.stack([np.stack(q) for q in self._quat]) if E
                else np.zeros((F, 0, 4), dtype=np.float32))

        # Concat particles/lines if registered (single group is the common case;
        # multiple groups concatenate along axis 1, with name boundaries in meta).
        particles = self._stack_clouds(self._particle_names, self._particles, F)
        lines = self._stack_lines(self._line_names, self._lines, F)

        full_meta = dict(self.meta)
        full_meta["schema_version"] = SCHEMA_VERSION
        full_meta["quaternion_order"] = "xyzw"
        full_meta["mesh_names"] = self._mesh_names
        full_meta["mesh_visible"] = self._mesh_visible
        full_meta["mesh_colors"] = self._mesh_colors
        full_meta["mesh_dir"] = self._mesh_dir.name
        full_meta["particle_names"] = self._particle_names
        full_meta["line_names"] = self._line_names
        full_meta["deformable_mesh_names"] = self._deformable_mesh_names
        offsets = np.cumsum([0, *self._deformable_vertex_counts], dtype=np.int64)
        full_meta["deformable_mesh_offsets"] = offsets.tolist()

        # Robot trajectories: stored as additive npz arrays under
        # `robot_q__<id>`. Key list echoed into meta for discovery; the
        # render layer's cache loader intentionally ignores unknown arrays.
        robot_arrays: dict[str, np.ndarray] = {}
        if self._robot_specs:
            full_meta["robots"] = self._robot_specs
            for spec in self._robot_specs:
                rid = spec["id"]
                frames = self._robot_q[rid]
                if not frames or spec["n_dof"] is None:
                    continue
                n_dof = int(spec["n_dof"])
                # Pad earlier zero-width frames (before the first real q).
                arr = np.stack([
                    f if f.shape[0] == n_dof else np.zeros(n_dof, dtype=np.float32)
                    for f in frames
                ]).astype(np.float32)
                robot_arrays[f"robot_q__{rid}"] = arr

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._deformable_mesh_names:
            deformable_vertices = np.stack(
                [np.concatenate(frame, axis=0) for frame in self._deformable_vertices]
            ).astype(np.float32)
        else:
            deformable_vertices = np.zeros((F, 0, 3), dtype=np.float32)
        np.savez_compressed(
            self.out_path,
            schema_version=np.int32(SCHEMA_VERSION),
            times=times,
            pos=pos,
            quat=quat,
            particles=particles,
            lines=lines,
            deformable_vertices=deformable_vertices,
            meta=np.asarray(json.dumps(full_meta)),
            **robot_arrays,
        )
        logger.info(
            "recorder: wrote {} frames (effectors={}, deformable_vertices={}, "
            "particles={}, lines={}, robots={}) → {} ({:.1f} MB)",
            F, E, deformable_vertices.shape[1],
            particles.shape[1] if particles.size else 0,
            lines.shape[1] if lines.size else 0,
            len(self._robot_specs),
            self.out_path, self.out_path.stat().st_size / 1024 / 1024,
        )

    @staticmethod
    def _stack_clouds(names, frames, F):
        if not names:
            return np.zeros((F, 0, 3), dtype=np.float32)
        per_frame = []
        for fr in frames:
            chunks = [c for c in fr if c is not None]
            per_frame.append(np.concatenate(chunks, axis=0) if chunks
                             else np.zeros((0, 3), dtype=np.float32))
        return np.stack(per_frame).astype(np.float32)

    @staticmethod
    def _stack_lines(names, frames, F):
        if not names:
            return np.zeros((F, 0, 0, 3), dtype=np.float32)
        per_frame = []
        for fr in frames:
            chunks = [c for c in fr if c is not None]
            if not chunks:
                per_frame.append(None)
                continue
            per_frame.append(np.concatenate(chunks, axis=0))
        if any(c is None for c in per_frame):
            return np.zeros((F, 0, 0, 3), dtype=np.float32)
        return np.stack(per_frame).astype(np.float32)
