"""Rerun frontend: live viewer or .rrd recording."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from .base import Frontend


class RerunFrontend(Frontend):
    """Renders meshes (wireframe or solid), particle clouds, and polylines via rerun.

    `live=True` spawns a viewer; otherwise an .rrd file is written to `out_path`.
    `view_up` selects the world up axis for the viewer's default orientation.
    """

    def __init__(self, out_path: str | None = None, fps: int = 60,
                 live: bool = False, view_up: str = "y",
                 eye_position: tuple[float, float, float] | list[float] | None = None,
                 look_target: tuple[float, float, float] | list[float] | None = None) -> None:
        if view_up not in {"y", "z"}:
            raise ValueError(f"view_up must be 'y' or 'z', got {view_up!r}")
        self.out_path = out_path
        self.fps = fps
        self.live = live
        self.view_up = view_up
        self.eye_position = eye_position
        self.look_target = look_target
        self.frame_idx = 0
        self._meshes: dict[str, dict[str, Any]] = {}
        self._particles: dict[str, dict[str, Any]] = {}
        self._lines: dict[str, dict[str, Any]] = {}
        self._scalars: dict[str, float] = {}
        self._scalar_visibility: dict[str, bool] = {}
        self._images: dict[str, dict[str, Any]] = {}
        self._sensor_blueprint_sent = False

        import rerun as rr
        rr.init("lycheesim", recording_id=None)
        if live:
            # Direct ``.venv/bin/python`` invocations do not necessarily put
            # the sibling Rerun executable on PATH.
            sibling_viewer = Path(sys.executable).with_name("rerun")
            rr.spawn(
                executable_path=(
                    str(sibling_viewer) if sibling_viewer.is_file() else None
                )
            )
            logger.info("rerun: streaming live to spawned viewer")
        else:
            if not out_path:
                raise ValueError("rerun offline mode requires out_path (.rrd file); "
                                 "pass live=True to stream to a viewer instead.")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            rr.save(out_path)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP if view_up == "z"
                    else rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        import rerun.blueprint as rrb
        eye_up = [0.0, 0.0, 1.0] if view_up == "z" else [0.0, 1.0, 0.0]
        rr.send_blueprint(rrb.Blueprint(
            rrb.Spatial3DView(
                origin="/",
                eye_controls=rrb.EyeControls3D(
                    position=eye_position,
                    look_target=look_target,
                    eye_up=eye_up,
                ),
            ),
            auto_views=False,
            collapse_panels=True,
        ))
        self._rr = rr

    def add_mesh(self, name, vertices, indices, color=(0.8, 0.8, 0.85), solid=False,
                 normals=None, vertex_colors=None, texcoords=None,
                 albedo_texture=None):
        self._add_mesh(
            name,
            vertices,
            indices,
            color=color,
            solid=solid,
            normals=normals,
            vertex_colors=vertex_colors,
            texcoords=texcoords,
            albedo_texture=albedo_texture,
            deformable=False,
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
        self._add_mesh(
            name,
            vertices,
            indices,
            color=color,
            solid=solid,
            normals=normals,
            vertex_colors=vertex_colors,
            texcoords=texcoords,
            albedo_texture=albedo_texture,
            deformable=True,
        )

    def _add_mesh(
        self,
        name,
        vertices,
        indices,
        *,
        color,
        solid,
        normals,
        vertex_colors,
        texcoords,
        albedo_texture,
        deformable,
    ):
        idx = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        local_verts = np.asarray(vertices, dtype=np.float32)
        local_colors = (
            np.asarray(vertex_colors)
            if vertex_colors is not None
            else None
        )
        local_texcoords = (
            np.asarray(texcoords, dtype=np.float32)
            if texcoords is not None
            else None
        )
        local_texture = (
            np.asarray(albedo_texture)
            if albedo_texture is not None
            else None
        )
        entry: dict[str, Any] = {
            "vertices": local_verts.copy(),
            "local_vertices": local_verts,
            "indices": idx,
            "color": color,
            "solid": solid,
            "deformable": bool(deformable),
            "normals": np.asarray(normals, dtype=np.float32) if normals is not None else None,
            "vertex_colors": local_colors,
            "texcoords": local_texcoords,
            "albedo_texture": local_texture,
        }
        color_components = np.asarray(color).reshape(-1).size
        entry["albedo_factor"] = (
            color
            if local_colors is None
            and (local_texture is None or color_components == 4)
            else None
        )
        if not solid:
            e = np.vstack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
            entry["edges"] = np.unique(np.sort(e, axis=1), axis=0)
        self._meshes[name] = entry
        if solid and not deformable:
            # Log mesh topology once as static; pose is updated each frame via Transform3D.
            self._rr.log(f"/{name}", self._rr.Mesh3D(
                vertex_positions=local_verts,
                triangle_indices=idx,
                vertex_normals=entry.get("normals"),
                vertex_colors=local_colors,
                vertex_texcoords=local_texcoords,
                albedo_texture=local_texture,
                albedo_factor=entry["albedo_factor"],
            ), static=True)

    def add_particles(self, name, count, radius, color=(0.2, 0.5, 1.0)):
        # ``color`` is the fallback if no per-particle colors are pushed;
        # ``colors`` is the optional (N, 3) array set by update_particles.
        self._particles[name] = {"count": count, "radius": radius, "color": color,
                                 "positions": np.zeros((count, 3), dtype=np.float32),
                                 "colors": None}

    def add_lines(self, name, n_strips, points_per_strip, color=(0.85, 0.55, 0.20),
                  radius=0.0008):
        self._lines[name] = {
            "n_strips": int(n_strips),
            "points_per_strip": int(points_per_strip),
            "positions": np.zeros((n_strips, points_per_strip, 3), dtype=np.float32),
            "color": color,
            "radius": float(radius),
        }

    def update_pose(self, name, pos, quat):
        m = self._meshes.get(name)
        if m is None:
            return
        if m.get("solid", False):
            # Store transform; will be logged as Transform3D in frame().
            m["_pending_transform"] = (
                np.asarray(pos, dtype=np.float32),
                np.asarray(quat, dtype=np.float32),
            )
        else:
            # Wireframe: recompute vertex positions for LineStrips3D.
            from ..transforms import quat_to_mat3
            rest = m.setdefault("_rest", m["vertices"].copy())
            R = quat_to_mat3(np.asarray(quat, dtype=np.float32))
            m["vertices"] = (rest @ R.T + np.asarray(pos, dtype=np.float32)).astype(np.float32)

    def update_mesh(self, name, vertices, normals=None):
        if name in self._meshes:
            self._meshes[name]["vertices"] = np.asarray(vertices, dtype=np.float32)
            if normals is not None:
                self._meshes[name]["normals"] = np.asarray(normals, dtype=np.float32)
            self._meshes[name]["_mesh_dirty"] = True

    def update_dynamic_mesh(
        self, name, vertices, indices, normals=None, vertex_colors=None
    ):
        if name not in self._meshes:
            return
        mesh = self._meshes[name]
        mesh["vertices"] = np.asarray(vertices, dtype=np.float32)
        mesh["indices"] = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        mesh["normals"] = (
            np.asarray(normals, dtype=np.float32) if normals is not None else None
        )
        mesh["vertex_colors"] = (
            np.asarray(vertex_colors, dtype=np.float32)
            if vertex_colors is not None
            else None
        )
        # Dynamic topology cannot retain authored UVs because vertex identity
        # changes after every Marching Cubes extraction.
        mesh["texcoords"] = None
        mesh["albedo_texture"] = None
        mesh["albedo_factor"] = None
        mesh["_mesh_dirty"] = True

    def update_particles(self, name, positions, colors=None):
        if name in self._particles:
            self._particles[name]["positions"] = np.asarray(positions, dtype=np.float32)
            if colors is not None:
                arr = np.asarray(colors, dtype=np.float32)
                # Rerun Points3D accepts RGB or RGBA colors in [0, 1].
                self._particles[name]["colors"] = arr
            else:
                self._particles[name]["colors"] = None

    def update_lines(self, name, positions):
        if name in self._lines:
            self._lines[name]["positions"] = np.asarray(positions, dtype=np.float32)

    def add_scalar(self, name, *, visible=False):
        self._scalars[name] = 0.0
        self._scalar_visibility[name] = bool(visible)

    def update_scalar(self, name, value):
        if name in self._scalars:
            self._scalars[name] = float(value)

    def add_image(self, name, kind="color", *, visible=False):
        if kind not in {"color", "depth"}:
            raise ValueError(f"unsupported Rerun image kind: {kind!r}")
        self._images[name] = {
            "kind": kind,
            "image": None,
            "dirty": False,
            "visible": bool(visible),
        }

    def update_image(self, name, image):
        if name in self._images:
            item = self._images[name]
            array = np.asarray(image)
            if (
                item["image"] is not None
                and item["image"].shape == array.shape
                and np.array_equal(item["image"], array)
            ):
                return
            item["image"] = array.copy()
            item["dirty"] = True

    def _sensor_blueprint_views(self, rrb):
        """Build sensor views while retaining hidden entities for opt-in use."""

        sensor_views = [
            rrb.Spatial2DView(
                origin=f"/{name}",
                name=name.rsplit("/", 1)[-1],
                visible=bool(item.get("visible", False)),
            )
            for name, item in self._images.items()
        ]
        scalar_groups: dict[str, bool] = {}
        for name in self._scalars:
            group = f"/{name.strip('/')}".rsplit("/", 1)[0] or "/"
            scalar_groups[group] = scalar_groups.get(group, False) or bool(
                self._scalar_visibility.get(name, False)
            )
        for group, visible in sorted(scalar_groups.items()):
            title = group.rsplit("/", 1)[-1].replace("_", " ").title()
            sensor_views.append(
                rrb.TimeSeriesView(
                    origin=group,
                    name=title or "Sensor signals",
                    visible=visible,
                )
            )
        return sensor_views

    def _send_sensor_blueprint(self) -> None:
        if self._sensor_blueprint_sent or not (self._images or self._scalars):
            return
        import rerun.blueprint as rrb

        eye_up = [0.0, 0.0, 1.0] if self.view_up == "z" else [0.0, 1.0, 0.0]
        sensor_views = self._sensor_blueprint_views(rrb)
        visible_sensor_views = [view for view in sensor_views if bool(view.visible)]
        hidden_sensor_views = [view for view in sensor_views if not bool(view.visible)]
        simulation_view = rrb.Spatial3DView(
            origin="/",
            name="Simulation",
            eye_controls=rrb.EyeControls3D(
                position=self.eye_position,
                look_target=self.look_target,
                eye_up=eye_up,
            ),
        )
        if visible_sensor_views:
            layout = rrb.Horizontal(
                simulation_view,
                rrb.Vertical(*visible_sensor_views),
                column_shares=[2.0, 1.0],
            )
        else:
            layout = simulation_view
        self._rr.send_blueprint(
            rrb.Blueprint(
                layout,
                *hidden_sensor_views,
                auto_views=False,
                collapse_panels=True,
            )
        )
        self._sensor_blueprint_sent = True

    def frame(self, t):
        rr = self._rr
        self._send_sensor_blueprint()
        rr.set_time("sim", duration=float(t))
        for name, m in self._meshes.items():
            if m.get("solid", False):
                if m.get("deformable", False):
                    if not m.pop("_mesh_dirty", False):
                        continue
                    local_colors = m.get("vertex_colors")
                    local_texture = m.get("albedo_texture")
                    rr.log(f"/{name}", rr.Mesh3D(
                        vertex_positions=m["vertices"],
                        triangle_indices=m["indices"],
                        vertex_normals=m.get("normals"),
                        vertex_colors=local_colors,
                        vertex_texcoords=m.get("texcoords"),
                        albedo_texture=local_texture,
                        albedo_factor=m.get("albedo_factor"),
                    ))
                elif "_pending_transform" in m:
                    # Rigid body: update pose via Transform3D; static mesh is logged once.
                    pos, quat_xyzw = m["_pending_transform"]
                    rr.log(f"/{name}", rr.Transform3D(
                        translation=pos,
                        quaternion=rr.Quaternion(xyzw=quat_xyzw),
                    ))
                elif m.pop("_mesh_dirty", False):
                    # Deformable body updated via update_mesh. Static solids keep
                    # the textured Mesh3D logged once by add_mesh().
                    local_colors = m.get("vertex_colors")
                    local_texture = m.get("albedo_texture")
                    rr.log(f"/{name}", rr.Mesh3D(
                        vertex_positions=m["vertices"],
                        triangle_indices=m["indices"],
                        vertex_normals=m.get("normals"),
                        vertex_colors=local_colors,
                        vertex_texcoords=m.get("texcoords"),
                        albedo_texture=local_texture,
                        albedo_factor=m.get("albedo_factor"),
                    ))
            else:
                strips = m["vertices"][m["edges"]]
                rr.log(f"/{name}", rr.LineStrips3D(strips, colors=m["color"], radii=0.0005))
        for name, p in self._particles.items():
            # Per-particle colors override the per-cloud fallback when supplied.
            cols = p.get("colors") if p.get("colors") is not None else p["color"]
            rr.log(f"/{name}",
                   rr.Points3D(p["positions"], radii=p["radius"], colors=cols))
        for name, ln in self._lines.items():
            rr.log(f"/{name}",
                   rr.LineStrips3D(ln["positions"], colors=ln["color"], radii=ln["radius"]))
        for name, value in self._scalars.items():
            rr.log(f"/{name}", rr.Scalars([value]))
        for name, item in self._images.items():
            image = item["image"]
            if image is None or not item["dirty"]:
                continue
            if item["kind"] == "depth":
                rr.log(f"/{name}", rr.DepthImage(image, meter=1.0))
            else:
                rr.log(f"/{name}", rr.Image(image))
            item["dirty"] = False
        self.frame_idx += 1

    def close(self):
        if self.live:
            logger.info("rerun live session finished; keep the viewer open to scrub")
        else:
            logger.info(f"rerun recording saved: uv run rerun {self.out_path}")
