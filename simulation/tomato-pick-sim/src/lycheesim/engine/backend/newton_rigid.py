"""Multi-body rigid contact solver for Newton meshes.

The backend supports Newton XPBD and semi-implicit solvers for reconstructed
Real2Sim scenes, plus Newton ``SolverMuJoCo`` hydroelastic contact.

It hosts small sets of mesh-collider rigid bodies. Each body is either free
and dynamic or fully kinematic, with its pose overwritten through
``set_kinematic_pose``. Reusable joint chains can implement
``RigidModelExtension``; task-specific mechanisms belong beside their demo
and plug into the same extension point.

The intention is that the solver is swappable. It implements the shared
``PhysicsSolver`` timing contract plus this rigid-family interface:

    class RigidPairSolver(Protocol):
        dt: float
        substeps: int
        @property
        def n_bodies(self) -> int: ...
        def body_label(self, idx: int) -> str: ...
        def mesh_local_vertices(self, idx: int) -> np.ndarray: ...
        def mesh_local_faces(self, idx: int) -> np.ndarray: ...
        def body_pose(self, idx: int) -> tuple[np.ndarray, np.ndarray]: ...
        def set_kinematic_pose(self, idx, pos, quat_xyzw) -> None: ...
        def step(self) -> None: ...

Another implementation can match the same scene-facing shape without changing
runner integration.

Quaternion storage and every runtime API use Newton/Warp-native ``xyzw``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import trimesh
import warp as wp


@wp.kernel
def _set_free_joint_states(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    q_starts: wp.array[int],
    qd_starts: wp.array[int],
    poses: wp.array2d[float],
    velocities: wp.array2d[float],
):
    body = wp.tid()
    q_start = q_starts[body]
    qd_start = qd_starts[body]
    for component in range(7):
        joint_q[q_start + component] = poses[body, component]
    for component in range(6):
        joint_qd[qd_start + component] = velocities[body, component]


@wp.kernel
def _set_render_shape_transforms(
    shape_transforms: wp.array(dtype=wp.transform),
    shape_indices: wp.array(dtype=wp.int32),
    poses: wp.array2d(dtype=wp.float32),
):
    visual = wp.tid()
    shape = shape_indices[visual]
    shape_transforms[shape] = wp.transform(
        wp.vec3(poses[visual, 0], poses[visual, 1], poses[visual, 2]),
        wp.quat(
            poses[visual, 3],
            poses[visual, 4],
            poses[visual, 5],
            poses[visual, 6],
        ),
    )


@wp.kernel
def _add_commanded_body_forces(
    body_f: wp.array(dtype=wp.spatial_vector),
    body_ids: wp.array(dtype=wp.int32),
    forces: wp.array(dtype=wp.vec3),
    torques: wp.array(dtype=wp.vec3),
):
    command = wp.tid()
    body = body_ids[command]
    body_f[body] = body_f[body] + wp.spatial_vector(
        forces[command], torques[command]
    )


# ---------------- specs ----------------


@dataclass
class RigidBodySpec:
    """One rigid body in the scene.

    `kinematic=True` ⇒ infinite mass; pose driven externally each substep
    via `set_kinematic_pose`. Otherwise the body is dynamic (mass derived
    from mesh volume × density).

    `init_pos` is the world position of the body's local origin. With
    `center_origin=True` the local origin is the mesh AABB center, so
    `init_pos` is where you want the centroid to land.
    """

    name: str
    mesh_path: str | Path
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    kinematic: bool = False
    density: float = 8000.0     # steel
    # OBJ-frame fixup applied to vertices BEFORE recentering. Use to flip
    # an OBJ that was authored upside-down.
    mesh_rotate_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center_origin: bool = True
    collision_mesh_paths: tuple[str | Path, ...] | None = None
    collision_mass_fractions: tuple[float, ...] | None = None
    collision_geometry_kinds: tuple[str, ...] | None = None
    mass_kg: float | None = None
    mesh_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    friction: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    render_collision: bool = True
    render_uvs: Any | None = None
    render_texture: str | Path | np.ndarray | None = None


@dataclass
class RigidVisualSpec:
    """One camera-visible, non-colliding mesh with an independently driven pose.

    Visuals are attached to the global world rather than represented as rigid
    bodies, so a detailed robot can be rendered without adding free joints to
    the physics system. ``set_visual_poses`` updates their shape transforms.
    """

    name: str
    vertices: Any
    indices: Any
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    normals: Any | None = None
    uvs: Any | None = None
    texture: str | Path | np.ndarray | None = None


@dataclass
class RigidContactConfig:
    """Hydroelastic SDF contact knobs for SolverMuJoCo.

    Defaults follow Newton's `example_nut_bolt_hydro.py`, the shipping
    reference for tight-tolerance threaded-fastener contact.
    """

    mu: float = 0.01
    ke: float = 1.0e7
    kd: float = 1.0e4
    kh: float = 1.0e10
    gap: float = 0.005
    margin: float = 0.0
    is_hydroelastic: bool = True
    sdf_max_resolution: int = 256
    sdf_narrow_band: tuple[float, float] = (-0.005, 0.005)


class RigidModelExtension(Protocol):
    """Composable joint/constraint contribution to a rigid model."""

    @property
    def linked_body_indices(self) -> tuple[int, ...]:
        """Bodies that must be added as articulation links, not free bodies."""

    def validate(self, bodies: list[RigidBodySpec]) -> None:
        """Validate extension-specific body requirements."""

    def add_to_builder(
        self,
        builder,
        newton,
        body_indices: list[int],
        bodies: list[RigidBodySpec],
    ) -> None:
        """Add joints/articulations before model finalization."""

    def bind(self, solver: "NewtonRigidSolver") -> None:
        """Bind finalized state/control handles and seed initial joint state."""


@dataclass
class RigidContactSensorSpec:
    """Declarative rigid contact/tactile sensor constructed with the model.

    Newton contact-force attributes must be requested before ``Contacts`` is
    allocated.  Keeping sensor construction in the solver makes that ordering
    invariant explicit and also keeps captured contact buffers compatible.
    """

    name: str
    sensing_bodies: str | list[str] | list[int] | None = None
    sensing_shapes: str | list[str] | list[int] | None = None
    counterpart_bodies: str | list[str] | list[int] | None = None
    counterpart_shapes: str | list[str] | list[int] | None = None
    measure_total: bool = True


# ---------------- helpers ----------------


def _euler_xyz_deg_to_mat3(deg):
    rx, ry, rz = (np.deg2rad(a) for a in deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return Rx @ Ry @ Rz


def _load_mesh_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, force="mesh", process=False)
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32).flatten(),
    )


def prepare_rigid_vertices(
    vertices: np.ndarray,
    spec: RigidBodySpec,
    *,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the asset-frame transform shared by visual and collision meshes."""
    verts = np.asarray(vertices, dtype=np.float32) * np.asarray(
        spec.mesh_scale, dtype=np.float32
    )[None, :]
    if any(a != 0.0 for a in spec.mesh_rotate_euler_deg):
        verts = verts @ _euler_xyz_deg_to_mat3(spec.mesh_rotate_euler_deg).T
    if center is not None:
        verts = verts - center
    return np.ascontiguousarray(verts, dtype=np.float32)


def _load_and_prep_mesh(
    spec: RigidBodySpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load OBJ, optionally rotate then recenter to AABB center.

    Returns (vertices, indices_flat). Vertices are in the BODY local
    frame; the body's world pose places this local origin at `init_pos`.
    """
    raw_vertices, indices = _load_mesh_arrays(spec.mesh_path)
    transformed = prepare_rigid_vertices(raw_vertices, spec)
    center = np.zeros(3, dtype=np.float32)
    if spec.center_origin:
        center = (
            (transformed.min(axis=0) + transformed.max(axis=0)) * 0.5
        ).astype(np.float32)
    return transformed - center, indices, center


def _mesh_volume(vertices: np.ndarray, indices_flat: np.ndarray) -> float:
    """Return a positive volume estimate for mass-prior distribution."""
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=indices_flat.reshape(-1, 3),
        process=False,
    )
    volume = abs(float(mesh.volume))
    if not np.isfinite(volume) or volume <= 1.0e-12:
        try:
            volume = abs(float(mesh.convex_hull.volume))
        except (ValueError, RuntimeError):
            volume = 0.0
    if not np.isfinite(volume) or volume <= 1.0e-12:
        extent = np.ptp(vertices, axis=0)
        volume = float(np.prod(np.maximum(extent, 1.0e-6)))
    return max(volume, 1.0e-12)


def _prepare_collision_geometry(
    vertices: np.ndarray,
    indices_flat: np.ndarray,
    geometry_kind: str,
    *,
    is_hydroelastic: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return geometry compatible with Newton's selected contact path.

    Newton's CUDA texture-SDF finalizer consumes attached SDF payloads only
    for ``GeoType.MESH``. A hydroelastic ``GeoType.CONVEX_MESH`` is accepted
    by the builder but receives an empty generated SDF, silently removing its
    contacts. Bake requested convex hulls into watertight triangle meshes on
    the hydroelastic path. Point-contact solvers retain native convex hulls.
    """

    if not is_hydroelastic or geometry_kind != "convex_hull":
        return vertices, indices_flat, geometry_kind

    hull = trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(indices_flat, dtype=np.int32).reshape(-1, 3),
        process=False,
    ).convex_hull
    hull_vertices = np.ascontiguousarray(hull.vertices, dtype=np.float32)
    hull_indices = np.ascontiguousarray(hull.faces, dtype=np.int32).reshape(-1)
    if len(hull_vertices) < 4 or len(hull_indices) < 12 or not hull.is_watertight:
        raise ValueError("failed to build a watertight hydroelastic convex hull")
    return hull_vertices, hull_indices, "triangle_mesh"


# ---------------- solver ----------------


class NewtonRigidSolver:
    """Rigid bodies in mutual contact with composable model extensions."""

    def __init__(
        self,
        bodies: list[RigidBodySpec],
        *,
        visuals: list[RigidVisualSpec] | None = None,
        dt: float = 1.0 / 120.0,
        substeps: int = 5,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        contact: RigidContactConfig | None = None,
        extensions: list[RigidModelExtension] | None = None,
        contact_sensors: list[RigidContactSensorSpec] | None = None,
        ground_z: float | None = None,
        ground_friction: tuple[float, float, float] | None = None,
        solver_kind: str = "newton_mujoco_hydro",
        xpbd_iterations: int = 8,
        mujoco_iterations: int = 15,
        mujoco_ls_iterations: int = 100,
        rigid_contact_max: int = 100_000,
        broad_phase: str = "sap",
        use_capture: bool = True,
    ):
        if not bodies:
            raise ValueError("need at least one body")

        import newton

        self._newton = newton
        self.dt = float(dt)
        self.substeps = int(substeps)
        self._sub_dt = float(dt) / int(substeps)
        self._specs = list(bodies)
        self._contact = contact if contact is not None else RigidContactConfig()
        self.solver_kind = str(solver_kind)
        self.extensions = tuple(extensions or ())
        for extension in self.extensions:
            extension.validate(self._specs)

        # The body that's wired into a hard joint chain must use add_link;
        # add_body would auto-add a 6-DOF free joint, defeating the chain.
        linked_indices = [
            int(index)
            for extension in self.extensions
            for index in extension.linked_body_indices
        ]
        if any(index < 0 or index >= len(self._specs) for index in linked_indices):
            raise IndexError("rigid model extension body index is out of range")
        if len(set(linked_indices)) != len(linked_indices):
            raise ValueError("rigid model extensions cannot share linked bodies")
        chain_indices = set(linked_indices)

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = self._contact.gap

        if ground_z is not None:
            friction = ground_friction or (
                self._contact.mu,
                0.0,
                0.0,
            )
            ground_cfg = newton.ModelBuilder.ShapeConfig(
                margin=self._contact.margin,
                mu=float(friction[0]),
                ke=self._contact.ke,
                kd=self._contact.kd,
                gap=self._contact.gap,
                mu_torsional=float(friction[1]),
                mu_rolling=float(friction[2]),
                is_hydroelastic=False,
            )
            builder.add_shape_plane(
                plane=(0.0, 0.0, 1.0, -float(ground_z)),
                width=0.0, length=0.0, label="ground_plane",
                cfg=ground_cfg,
            )

        body_indices, self._local_verts, self._local_faces = self._add_bodies(
            builder, newton, bodies, chain_indices,
        )
        self._body_indices = body_indices
        self._visual_shape_indices, self._visual_names = self._add_visuals(
            builder, newton, visuals or []
        )
        self._visual_name_to_slot = {
            name: index for index, name in enumerate(self._visual_names)
        }

        for extension in self.extensions:
            extension.add_to_builder(
                builder,
                newton,
                self._body_indices,
                self._specs,
            )

        previous_target_layout = newton.use_coord_layout_targets
        newton.use_coord_layout_targets = True
        try:
            self.model = builder.finalize()
        finally:
            newton.use_coord_layout_targets = previous_target_layout
        if self._contact.is_hydroelastic:
            shape_flags = self.model.shape_flags.numpy()
            shape_types = self.model.shape_type.numpy()
            hydro_mask = (
                shape_flags & int(newton.ShapeFlags.HYDROELASTIC)
            ) != 0
            unsupported = np.flatnonzero(
                hydro_mask & (shape_types != int(newton.GeoType.MESH))
            )
            if len(unsupported):
                raise RuntimeError(
                    "hydroelastic mesh SDFs must be registered as GeoType.MESH; "
                    f"invalid shape indices={unsupported.tolist()}"
                )
        self.model.set_gravity(wp.vec3(*gravity))

        self._collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            rigid_contact_max=int(rigid_contact_max),
            broad_phase=broad_phase,
        )
        if self.solver_kind == "newton_xpbd":
            self.solver = newton.solvers.SolverXPBD(
                self.model,
                iterations=int(xpbd_iterations),
                enable_restitution=False,
            )
        elif self.solver_kind == "newton_semi_implicit":
            self.solver = newton.solvers.SolverSemiImplicit(
                self.model,
                enable_tri_contact=False,
            )
        elif self.solver_kind == "newton_mujoco_hydro":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_contacts=False,
                solver="newton",
                integrator="implicitfast",
                cone="elliptic",
                njmax=int(rigid_contact_max),
                nconmax=int(rigid_contact_max),
                iterations=int(mujoco_iterations),
                ls_iterations=int(mujoco_ls_iterations),
                impratio=1.0,
            )
        else:
            raise ValueError(
                "solver_kind must be newton_xpbd, newton_semi_implicit, or "
                f"newton_mujoco_hydro; got {self.solver_kind!r}"
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self._device = self.model.device
        self._commanded_body_forces = wp.zeros(
            len(self._specs), dtype=wp.vec3, device=self._device
        )
        self._commanded_body_torques = wp.zeros(
            len(self._specs), dtype=wp.vec3, device=self._device
        )
        self._commanded_body_ids = wp.array(
            self._body_indices, dtype=wp.int32, device=self._device
        )

        # Contact sensors request their extended force attributes here, before
        # the first Contacts allocation below.
        self.contact_sensors: dict[str, Any] = {}
        if contact_sensors:
            from lycheesim.sensor import NewtonContactSensor

            for spec in contact_sensors:
                if spec.name in self.contact_sensors:
                    raise ValueError(f"duplicate contact sensor name: {spec.name!r}")
                self.contact_sensors[spec.name] = NewtonContactSensor(
                    self.model,
                    sensing_bodies=spec.sensing_bodies,
                    sensing_shapes=spec.sensing_shapes,
                    counterpart_bodies=spec.counterpart_bodies,
                    counterpart_shapes=spec.counterpart_shapes,
                    measure_total=spec.measure_total,
                )

        # Map body_idx -> joint_idx of its parent (free) joint, used by
        # set_kinematic_pose to find the right slice of joint_q.
        joint_child = self.model.joint_child.numpy()
        self._body_joint_idx: dict[int, int] = {
            int(c): int(j) for j, c in enumerate(joint_child) if c >= 0
        }
        self._joint_q_starts = self.model.joint_q_start.numpy()
        self._joint_qd_starts = self.model.joint_qd_start.numpy()
        self._kinematic_state_buffers: dict[
            tuple[int, ...], tuple[Any, Any, Any, Any]
        ] = {}
        self._visual_pose_buffers: dict[tuple[int, ...], tuple[Any, Any]] = {}
        for extension in self.extensions:
            extension.bind(self)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = self._collision_pipeline.contacts()
        self._collision_pipeline.collide(self.state_0, self.contacts)

        self._graph = None
        self._graphs = None
        self._graph_index = 0
        if use_capture and wp.get_device().is_cuda:
            # Solver states are double-buffered.  For an odd number of
            # substeps, one captured graph always advances A -> B; repeatedly
            # launching only that graph would therefore recompute the same
            # first step from stale A forever.  Capture the reverse B -> A
            # graph too and alternate them at runtime.
            state_a, state_b = self.state_0, self.state_1
            with wp.ScopedCapture() as first_capture:
                self._inner_substep_loop()
            first_graph = first_capture.graph
            if self.substeps % 2:
                if self.state_0 is not state_b or self.state_1 is not state_a:
                    raise RuntimeError(
                        "unexpected Newton state swap during graph capture"
                    )
                with wp.ScopedCapture() as second_capture:
                    self._inner_substep_loop()
                second_graph = second_capture.graph
            else:
                second_graph = first_graph
            self.state_0, self.state_1 = state_a, state_b
            self._graphs = (first_graph, second_graph)
            self._graph = first_graph

    # ---------------- builder helpers ----------------

    def _add_bodies(self, builder, newton, specs, chain_indices):
        """Add each body's mesh + shape.

        Bodies in ``chain_indices`` use ``add_link`` so they do not receive an
        automatic free joint; a drive-specific articulation attaches them.
        """
        body_indices: list[int] = []
        local_verts: list[np.ndarray] = []
        local_faces: list[np.ndarray] = []
        for i, spec in enumerate(specs):
            verts, indices_flat, center = _load_and_prep_mesh(spec)
            local_verts.append(verts.copy())
            local_faces.append(indices_flat.reshape(-1, 3).astype(np.int32))

            collision_paths = (
                tuple(spec.collision_mesh_paths)
                if spec.collision_mesh_paths
                else (spec.mesh_path,)
            )
            collision_kinds = (
                tuple(spec.collision_geometry_kinds)
                if spec.collision_geometry_kinds is not None
                else ("triangle_mesh",) * len(collision_paths)
            )
            if len(collision_kinds) != len(collision_paths):
                raise ValueError(
                    f"{spec.name}: collision_geometry_kinds must match "
                    "collision_mesh_paths"
                )
            invalid_kinds = sorted(
                set(collision_kinds) - {"convex_hull", "triangle_mesh"}
            )
            if invalid_kinds:
                raise ValueError(
                    f"{spec.name}: unsupported collision geometry kinds "
                    f"{invalid_kinds}"
                )
            collision_meshes = []
            collision_volumes = []
            resolved_collision_kinds = []
            for collision_path, geometry_kind in zip(
                collision_paths, collision_kinds, strict=True
            ):
                raw_collision_verts, collision_indices = _load_mesh_arrays(
                    collision_path
                )
                collision_verts = prepare_rigid_vertices(
                    raw_collision_verts,
                    spec,
                    center=center if spec.center_origin else None,
                )
                collision_verts, collision_indices, resolved_kind = (
                    _prepare_collision_geometry(
                        collision_verts,
                        collision_indices,
                        geometry_kind,
                        is_hydroelastic=self._contact.is_hydroelastic,
                    )
                )
                render_uvs = None
                render_texture = None
                if len(collision_paths) == 1:
                    render_uvs = spec.render_uvs
                    render_texture = spec.render_texture
                collision_mesh = newton.Mesh(
                    collision_verts,
                    collision_indices,
                    uvs=render_uvs,
                    texture=render_texture,
                )
                if self._contact.is_hydroelastic:
                    collision_mesh.build_sdf(
                        max_resolution=int(self._contact.sdf_max_resolution),
                        narrow_band_range=self._contact.sdf_narrow_band,
                        margin=(
                            self._contact.gap
                            if self._contact.gap is not None
                            else 0.005
                        ),
                    )
                collision_meshes.append(collision_mesh)
                collision_volumes.append(
                    _mesh_volume(collision_verts, collision_indices)
                )
                resolved_collision_kinds.append(resolved_kind)

            if spec.collision_mass_fractions is None:
                fractions = np.asarray(collision_volumes, dtype=np.float64)
            else:
                fractions = np.asarray(
                    spec.collision_mass_fractions, dtype=np.float64
                )
                if fractions.shape != (len(collision_meshes),):
                    raise ValueError(
                        f"{spec.name}: collision_mass_fractions must match "
                        "collision_mesh_paths"
                    )
                if np.any(fractions <= 0.0):
                    raise ValueError(
                        f"{spec.name}: collision mass fractions must be positive"
                    )
            fractions /= float(fractions.sum())

            xform = wp.transform(
                wp.vec3(*spec.init_pos),
                wp.quat(*spec.init_quat_xyzw),
            )
            add_fn = builder.add_link if i in chain_indices else builder.add_body
            body = add_fn(xform=xform, label=spec.name,
                          is_kinematic=bool(spec.kinematic))
            friction = spec.friction or (
                self._contact.mu,
                0.0,
                0.0,
            )
            for part_index, (mesh, volume, geometry_kind) in enumerate(
                zip(
                    collision_meshes,
                    collision_volumes,
                    resolved_collision_kinds,
                    strict=True,
                )
            ):
                density = float(spec.density)
                if spec.mass_kg is not None:
                    if spec.mass_kg <= 0.0:
                        raise ValueError(f"{spec.name}: mass_kg must be positive")
                    density = float(spec.mass_kg) * float(
                        fractions[part_index]
                    ) / volume
                shape_cfg = newton.ModelBuilder.ShapeConfig(
                    margin=self._contact.margin,
                    mu=float(friction[0]),
                    ke=self._contact.ke,
                    kd=self._contact.kd,
                    kh=self._contact.kh,
                    gap=self._contact.gap,
                    mu_torsional=float(friction[1]),
                    mu_rolling=float(friction[2]),
                    is_hydroelastic=self._contact.is_hydroelastic,
                    density=density,
                    # Newton's tiled raytracer cannot consume GeoType.CONVEX_MESH.
                    # A render-only triangle copy is added below for that case.
                    is_visible=spec.render_collision
                    and not (
                        geometry_kind == "convex_hull" and spec.color is not None
                    ),
                )
                add_collision_shape = (
                    builder.add_shape_convex_hull
                    if geometry_kind == "convex_hull"
                    else builder.add_shape_mesh
                )
                shape_kwargs = dict(
                    body=body,
                    mesh=mesh,
                    scale=(1.0, 1.0, 1.0),
                    cfg=shape_cfg,
                    label=f"{spec.name}_collision_{part_index:03d}",
                )
                if spec.color is not None:
                    shape_kwargs["color"] = spec.color
                add_collision_shape(**shape_kwargs)
                if (
                    spec.render_collision
                    and geometry_kind == "convex_hull"
                    and spec.color is not None
                ):
                    render_cfg = newton.ModelBuilder.ShapeConfig(
                        density=0.0,
                        collision_group=0,
                        has_shape_collision=False,
                        has_particle_collision=False,
                        is_visible=True,
                    )
                    render_kwargs = dict(
                        body=body,
                        mesh=mesh,
                        scale=(1.0, 1.0, 1.0),
                        cfg=render_cfg,
                        label=f"{spec.name}_render_{part_index:03d}",
                    )
                    if spec.color is not None:
                        render_kwargs["color"] = spec.color
                    builder.add_shape_mesh(**render_kwargs)
            body_indices.append(body)
        return body_indices, local_verts, local_faces

    @staticmethod
    def _add_visuals(builder, newton, specs: list[RigidVisualSpec]):
        shape_indices: list[int] = []
        names: list[str] = []
        for spec in specs:
            if spec.name in names:
                raise ValueError(f"duplicate render visual name: {spec.name!r}")
            vertices = np.ascontiguousarray(spec.vertices, dtype=np.float32)
            indices = np.ascontiguousarray(spec.indices, dtype=np.int32).reshape(-1)
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError(f"{spec.name}: visual vertices must have shape (N,3)")
            if len(indices) % 3:
                raise ValueError(f"{spec.name}: visual indices must contain triangles")
            mesh = newton.Mesh(
                vertices,
                indices,
                normals=spec.normals,
                uvs=spec.uvs,
                compute_inertia=False,
                is_solid=False,
                color=spec.color,
                texture=spec.texture,
            )
            cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                collision_group=0,
                has_shape_collision=False,
                has_particle_collision=False,
                is_visible=True,
                is_solid=False,
            )
            shape_indices.append(
                builder.add_shape_mesh(
                    body=-1,
                    xform=wp.transform(
                        wp.vec3(*spec.init_pos), wp.quat(*spec.init_quat_xyzw)
                    ),
                    mesh=mesh,
                    cfg=cfg,
                    color=spec.color,
                    label=spec.name,
                )
            )
            names.append(spec.name)
        return tuple(shape_indices), tuple(names)

    # ---------------- step ----------------

    def _inner_substep_loop(self) -> None:
        self._collision_pipeline.collide(self.state_0, self.contacts)
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            wp.launch(
                _add_commanded_body_forces,
                dim=len(self._specs),
                inputs=(
                    self.state_0.body_f,
                    self._commanded_body_ids,
                    self._commanded_body_forces,
                    self._commanded_body_torques,
                ),
                device=self._device,
            )
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self._sub_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        """Advance one ``dt`` using the configured internal rigid substeps."""
        if self._graphs is not None:
            wp.capture_launch(self._graphs[self._graph_index])
            if self.substeps % 2:
                self.state_0, self.state_1 = self.state_1, self.state_0
                self._graph_index = 1 - self._graph_index
        else:
            self._inner_substep_loop()

    def substep(self) -> None:
        """Compatibility alias for :meth:`step`; prefer ``step``."""
        self.step()

    def update_contact_sensors(self) -> None:
        """Refresh all registered rigid contact sensors from the latest state."""

        if not self.contact_sensors:
            return
        self.solver.update_contacts(self.contacts, self.state_0)
        for sensor in self.contact_sensors.values():
            sensor.update(self.state_0, self.contacts)

    def set_body_force(self, idx: int, force_xyz: tuple[float, float, float]) -> None:
        """Set a persistent world-frame force applied on every inner substep."""

        if not 0 <= int(idx) < len(self._specs):
            raise IndexError("body force index is out of range")
        torque = self._commanded_body_torques.numpy()[int(idx)]
        self.set_body_wrench(idx, force_xyz, tuple(float(value) for value in torque))

    def set_body_wrench(
        self,
        idx: int,
        force_xyz: tuple[float, float, float],
        torque_xyz: tuple[float, float, float],
    ) -> None:
        """Set a persistent world-frame wrench applied on every inner substep."""

        if not 0 <= int(idx) < len(self._specs):
            raise IndexError("body wrench index is out of range")
        force = np.asarray(force_xyz, dtype=np.float32)
        torque = np.asarray(torque_xyz, dtype=np.float32)
        if (
            force.shape != (3,)
            or torque.shape != (3,)
            or not np.all(np.isfinite(force))
            or not np.all(np.isfinite(torque))
        ):
            raise ValueError("body wrench must contain finite force and torque")
        commands = self._commanded_body_forces.numpy()
        commands[int(idx)] = force
        self._commanded_body_forces.assign(commands)
        commands = self._commanded_body_torques.numpy()
        commands[int(idx)] = torque
        self._commanded_body_torques.assign(commands)

    def set_body_collision_enabled(self, idx: int, enabled: bool) -> None:
        """Enable or disable rigid-shape collision for one body, preserving visuals."""

        if not 0 <= int(idx) < len(self._specs):
            raise IndexError("body collision index is out of range")
        body = int(self._body_indices[int(idx)])
        shape_bodies = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy()
        mask = shape_bodies == body
        collide_flag = int(self._newton.ShapeFlags.COLLIDE_SHAPES)
        if enabled:
            shape_flags[mask] |= collide_flag
        else:
            shape_flags[mask] &= ~collide_flag
        self.model.shape_flags.assign(shape_flags)

    # ---------------- accessors ----------------

    @property
    def n_bodies(self) -> int:
        return len(self._specs)

    def body_label(self, idx: int) -> str:
        return self._specs[idx].name

    def mesh_local_vertices(self, idx: int) -> np.ndarray:
        """Body-local vertices after rotation + recentering. World vertices
        = local @ R(quat).T + body_pose(idx)[0]."""
        return self._local_verts[idx]

    def mesh_local_faces(self, idx: int) -> np.ndarray:
        return self._local_faces[idx]

    def body_pose(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position[3], quaternion_xyzw[4])`` in the world frame."""
        body_q = self.state_0.body_q.numpy()
        row = np.asarray(body_q[self._body_indices[idx]], dtype=np.float32)
        return row[:3].copy(), row[3:7].copy()

    def body_velocities(
        self, indices=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return world-frame linear and angular velocity for selected bodies."""

        selected = (
            tuple(range(len(self._specs)))
            if indices is None
            else tuple(int(index) for index in indices)
        )
        values = self.state_0.body_qd.numpy()
        rows = np.asarray(
            [values[self._body_indices[index]] for index in selected],
            dtype=np.float32,
        )
        return rows[:, :3].copy(), rows[:, 3:6].copy()

    def zero_body_velocities(self, indices=None) -> None:
        """Zero selected body twists in both double-buffered states."""

        selected = (
            tuple(range(len(self._specs)))
            if indices is None
            else tuple(int(index) for index in indices)
        )
        for state in (self.state_0, self.state_1):
            values = state.body_qd.numpy()
            for index in selected:
                values[self._body_indices[index]] = 0.0
            state.body_qd.assign(values)

    def damp_body_velocity(
        self,
        idx: int,
        *,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
        max_linear_velocity: float | None = None,
        max_angular_velocity: float | None = None,
        dt: float | None = None,
    ) -> None:
        """Apply PhysX-style exponential damping/limits to one free body.

        Newton's MuJoCo path otherwise leaves free-joint passive damping at
        zero.  Manipulation assets imported from USD commonly depend on the
        dimensionless PhysX rigid-body damping rates, so expose their direct
        ``velocity *= exp(-rate * dt)`` equivalent without adding a joint or
        positional constraint.
        """

        idx = int(idx)
        if not 0 <= idx < len(self._specs):
            raise IndexError(f"body index out of range: {idx}")
        rates = np.asarray((linear_damping, angular_damping), dtype=np.float64)
        if not np.isfinite(rates).all() or np.any(rates < 0.0):
            raise ValueError("body damping rates must be finite and non-negative")
        interval = self.dt if dt is None else float(dt)
        if not np.isfinite(interval) or interval < 0.0:
            raise ValueError("body damping interval must be finite and non-negative")
        limits = (max_linear_velocity, max_angular_velocity)
        if any(value is not None and value <= 0.0 for value in limits):
            raise ValueError("body velocity limits must be positive when set")

        body_idx = self._body_indices[idx]
        joint_idx = int(self._body_joint_idx[body_idx])
        qd_start = int(self._joint_qd_starts[joint_idx])
        linear_scale = float(np.exp(-rates[0] * interval))
        angular_scale = float(np.exp(-rates[1] * interval))
        for state in (self.state_0, self.state_1):
            joint_qd = state.joint_qd.numpy()
            linear = joint_qd[qd_start : qd_start + 3] * linear_scale
            angular = joint_qd[qd_start + 3 : qd_start + 6] * angular_scale
            linear_norm = float(np.linalg.norm(linear))
            angular_norm = float(np.linalg.norm(angular))
            if max_linear_velocity is not None and linear_norm > max_linear_velocity:
                linear *= float(max_linear_velocity) / linear_norm
            if max_angular_velocity is not None and angular_norm > max_angular_velocity:
                angular *= float(max_angular_velocity) / angular_norm
            joint_qd[qd_start : qd_start + 3] = linear
            joint_qd[qd_start + 3 : qd_start + 6] = angular
            state.joint_qd.assign(joint_qd)
            self._newton.eval_fk(
                self.model,
                state.joint_q,
                state.joint_qd,
                state,
            )

    def set_kinematic_pose(self, idx: int, pos, quat_xyzw=None) -> None:
        """Overwrite a kinematic body's pose. Raises if not kinematic.

        Newton's MuJoCo solver derives ``state.body_q`` via FK from
        ``state.joint_q`` every substep, so writing to ``body_q`` alone
        gets overwritten on the next iteration. The actual pose backing
        store for a body added via ``add_body`` is its free_joint's 7-DoF
        slice in ``joint_q`` (xyz + xyzw). We write there — and to
        The batched updater writes both state buffers; this convenience path
        also synchronizes FK immediately for callers that update one body.
        """
        if quat_xyzw is None:
            quat_xyzw = self._specs[idx].init_quat_xyzw
        self.set_kinematic_poses(
            (idx,),
            np.asarray(pos, dtype=np.float32)[None],
            np.asarray(quat_xyzw, dtype=np.float32)[None],
        )
        self.sync_kinematics()

    def set_kinematic_poses(
        self,
        indices,
        positions,
        quaternions_xyzw,
        linear_velocities=None,
        angular_velocities=None,
    ) -> None:
        """Batch free-joint updates; call :meth:`sync_kinematics` once after."""

        key = tuple(int(index) for index in indices)
        count = len(key)
        positions = np.asarray(positions, dtype=np.float32)
        quaternions = np.asarray(quaternions_xyzw, dtype=np.float32)
        if positions.shape != (count, 3) or quaternions.shape != (count, 4):
            raise ValueError(
                "batched kinematic poses require (N,3) positions and "
                "(N,4) xyzw quaternions"
            )
        linear = (
            np.zeros((count, 3), dtype=np.float32)
            if linear_velocities is None
            else np.asarray(linear_velocities, dtype=np.float32)
        )
        angular = (
            np.zeros((count, 3), dtype=np.float32)
            if angular_velocities is None
            else np.asarray(angular_velocities, dtype=np.float32)
        )
        if linear.shape != (count, 3) or angular.shape != (count, 3):
            raise ValueError(
                "batched kinematic velocities must both have shape (N,3)"
            )
        poses = np.empty((count, 7), dtype=np.float32)
        poses[:, :3] = positions
        poses[:, 3:7] = quaternions
        velocities = np.concatenate((linear, angular), axis=1)
        if key not in self._kinematic_state_buffers:
            q_starts = []
            qd_starts = []
            for index in key:
                if not self._specs[index].kinematic:
                    raise RuntimeError(
                        f"body {index} ({self._specs[index].name}) is not "
                        "kinematic"
                    )
                body_idx = self._body_indices[index]
                joint_idx = int(self._body_joint_idx[body_idx])
                q_starts.append(int(self._joint_q_starts[joint_idx]))
                qd_starts.append(int(self._joint_qd_starts[joint_idx]))
            self._kinematic_state_buffers[key] = (
                wp.array(q_starts, dtype=wp.int32, device=self._device),
                wp.array(qd_starts, dtype=wp.int32, device=self._device),
                wp.zeros((count, 7), dtype=wp.float32, device=self._device),
                wp.zeros((count, 6), dtype=wp.float32, device=self._device),
            )
        q_starts, qd_starts, pose_buffer, velocity_buffer = (
            self._kinematic_state_buffers[key]
        )
        pose_buffer.assign(poses)
        velocity_buffer.assign(velocities)
        for state in (self.state_0, self.state_1):
            wp.launch(
                _set_free_joint_states,
                dim=count,
                inputs=[
                    state.joint_q,
                    state.joint_qd,
                    q_starts,
                    qd_starts,
                    pose_buffer,
                    velocity_buffer,
                ],
                device=self._device,
            )

    def set_visual_poses(
        self,
        names,
        positions,
        quaternions_xyzw,
    ) -> None:
        """Update non-colliding render shapes without adding physics bodies."""

        requested = tuple(str(name) for name in names)
        count = len(requested)
        positions = np.asarray(positions, dtype=np.float32)
        quaternions = np.asarray(quaternions_xyzw, dtype=np.float32)
        if positions.shape != (count, 3) or quaternions.shape != (count, 4):
            raise ValueError(
                "visual poses require (N,3) positions and (N,4) xyzw quaternions"
            )
        try:
            slots = tuple(self._visual_name_to_slot[name] for name in requested)
        except KeyError as exc:
            raise KeyError(f"unknown render visual: {exc.args[0]!r}") from exc
        norms = np.linalg.norm(quaternions, axis=1)
        if not np.isfinite(positions).all() or not np.isfinite(norms).all():
            raise ValueError("visual poses must be finite")
        if np.any(norms <= 1.0e-8):
            raise ValueError("visual quaternions must be nonzero")
        quaternions = quaternions / norms[:, None]
        poses = np.empty((count, 7), dtype=np.float32)
        poses[:, :3] = positions
        poses[:, 3:7] = quaternions
        if slots not in self._visual_pose_buffers:
            shape_indices = [self._visual_shape_indices[slot] for slot in slots]
            self._visual_pose_buffers[slots] = (
                wp.array(shape_indices, dtype=wp.int32, device=self._device),
                wp.zeros((count, 7), dtype=wp.float32, device=self._device),
            )
        shape_indices, pose_buffer = self._visual_pose_buffers[slots]
        pose_buffer.assign(poses)
        wp.launch(
            _set_render_shape_transforms,
            dim=count,
            inputs=[self.model.shape_transform, shape_indices, pose_buffer],
            device=self._device,
        )

    def sync_kinematics(self) -> None:
        """Refresh body poses after one or more kinematic joint updates.

        Collision detection consumes ``state.body_q`` before the solver's FK
        pass.  Call this once after batching externally driven poses so the
        current substep, rather than the next one, sees their new locations.
        """

        import newton

        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        newton.eval_fk(
            self.model,
            self.state_1.joint_q,
            self.state_1.joint_qd,
            self.state_1,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )


__all__ = [
    "NewtonRigidSolver",
    "RigidBodySpec",
    "RigidContactConfig",
    "RigidContactSensorSpec",
    "RigidModelExtension",
    "RigidVisualSpec",
    "prepare_rigid_vertices",
]
