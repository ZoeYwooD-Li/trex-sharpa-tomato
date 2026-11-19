"""Frontend ABC: the seam between physics backend and any runtime sink."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Frontend(ABC):
    """Lifecycle: add_* once → (update_*; frame)* per step → close() once.

    Multiple frontends can be active in the same run; the runner broadcasts
    every call to all of them. Subclasses override only what they care about
    (HeadlessFrontend overrides nothing, RecorderFrontend ignores live updates,
    etc.).
    """

    def add_mesh(self, name: str, vertices: np.ndarray, indices: np.ndarray,
                 color: tuple[float, float, float] = (0.8, 0.8, 0.85),
                 solid: bool = False, normals: np.ndarray | None = None,
                 vertex_colors: np.ndarray | None = None,
                 texcoords: np.ndarray | None = None,
                 albedo_texture: np.ndarray | None = None) -> None:
        """Register a mesh, optionally with per-vertex colors or an albedo map."""
        pass

    def add_deformable_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        indices: np.ndarray,
        color: tuple[float, float, float] = (0.8, 0.8, 0.85),
        solid: bool = True,
        normals: np.ndarray | None = None,
        vertex_colors: np.ndarray | None = None,
        texcoords: np.ndarray | None = None,
        albedo_texture: np.ndarray | None = None,
    ) -> None:
        """Register fixed-topology geometry whose vertices change over time.

        Frontends without a specialized deformable path fall back to their
        normal mesh registration. ``update_mesh`` remains the update API.
        """
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

    def add_particles(self, name: str, count: int, radius: float,
                      color: tuple[float, float, float] = (0.2, 0.5, 1.0)) -> None:
        pass

    def add_lines(self, name: str, n_strips: int, points_per_strip: int,
                  color: tuple[float, float, float] = (0.85, 0.55, 0.20),
                  radius: float = 0.0008) -> None:
        pass

    def add_plane(
        self,
        name: str,
        normal: np.ndarray,
        offset: float,
        extent: float,
        color: tuple[float, float, float] = (0.45, 0.45, 0.48),
    ) -> None:
        """Register a finite visualization/sample patch of an implicit plane."""
        pass

    def update_pose(self, name: str, pos: np.ndarray, quat: np.ndarray) -> None:
        """Update a world pose; ``quat`` is always ordered ``xyzw``."""
        pass

    def update_mesh(self, name: str, vertices: np.ndarray,
                    normals: np.ndarray | None = None) -> None:
        pass

    def update_dynamic_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        indices: np.ndarray,
        normals: np.ndarray | None = None,
        vertex_colors: np.ndarray | None = None,
    ) -> None:
        """Replace a mesh whose vertex count and topology may change.

        Fixed-topology frontends deliberately default to a no-op.  Dynamic
        implicit surfaces are display-only, so unsupported sinks should retain
        the rest mesh rather than interpreting new vertices with stale faces.
        """
        pass

    def update_particles(self, name: str, positions: np.ndarray,
                         colors: np.ndarray | None = None) -> None:
        """Push new particle positions for ``name``.

        ``colors``: optional (N, 3) float array in [0, 1] for per-particle RGB.
        If ``None``, the cloud keeps the single ``color`` from add_particles().
        Frontends that don't support per-particle color may ignore ``colors``.
        """
        pass

    def update_lines(self, name: str, positions: np.ndarray) -> None:
        pass

    def set_source_metadata(
        self, name: str, metadata: dict[str, Any]
    ) -> None:
        """Attach training/export semantics to registered scene geometry.

        Examples include ``{"role": "robot", "dynamic": True}`` and
        ``{"role": "environment", "dynamic": False}``. Display frontends may
        ignore this; dataset frontends preserve it.
        """
        pass

    def set_sequence_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach episode-level semantics to a dataset export.

        Geometry/display frontends may ignore this. Dataset frontends preserve
        it alongside coordinate, timing, and action-alignment metadata.
        """
        pass

    def set_next_action(self, name: str, action: np.ndarray) -> None:
        """Record an action aligned with the transition after this frame.

        A dataset frontend stores this value as ``actions[t]`` for the scene
        flow from frame ``t`` to frame ``t + 1``. Display frontends ignore it.
        """
        pass

    def add_scalar(self, name: str, *, visible: bool = False) -> None:
        """Register a scalar and whether its Rerun view is visible by default."""
        pass

    def update_scalar(self, name: str, value: float) -> None:
        pass

    def add_image(
        self,
        name: str,
        kind: str = "color",
        *,
        visible: bool = False,
    ) -> None:
        """Register an image and whether its Rerun view is visible by default."""
        pass

    def update_image(self, name: str, image: np.ndarray) -> None:
        pass

    def add_camera(
        self,
        name: str,
        *,
        width: int,
        height: int,
        intrinsic_matrix: np.ndarray,
        horizontal_aperture: float,
        vertical_aperture: float,
        focal_length: float,
        clipping_range_m: tuple[float, float],
        pose_convention: str = "opengl",
        source_path: str | None = None,
    ) -> None:
        """Register a calibrated pinhole camera for artifact renderers.

        Runtime display frontends may ignore this declaration. Persisted camera
        poses use the public ``xyzw`` quaternion order; ``pose_convention``
        declares the optical-frame axis convention.
        """

        pass

    def update_camera_pose(
        self,
        name: str,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        """Update a registered camera's world pose in ``xyzw`` order."""

        pass

    def add_robot(self, name: str, mjcf: str,
                  ee_link: str | None = None,
                  base_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        """Register an articulated robot loaded from MJCF/URDF.

        Default: no-op (e.g. Rerun, Headless ignore robots; only Genesis
        renders them today). Recorder may persist the spec for offline
        replay tooling.
        """
        pass

    def update_joint_q(self, name: str, q: np.ndarray) -> None:
        """Set the robot's joint positions for this frame.

        Default: no-op. GenesisFrontend writes via set_dofs_position;
        RecorderFrontend appends to its per-frame buffer.
        """
        pass

    @abstractmethod
    def frame(self, t: float) -> None:
        ...

    def read_inputs(self, t: float) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass
