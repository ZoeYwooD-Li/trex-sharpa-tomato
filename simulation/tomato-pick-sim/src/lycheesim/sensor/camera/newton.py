"""Newton scene-query implementation of the camera sensor contract."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import warnings
from typing import Any, Sequence

import numpy as np
import warp as wp

from ..base import SensorSpec
from .base import CameraSensorBase


_CHANNELS = frozenset(
    {"color", "depth", "forward_depth", "normal", "albedo", "shape_index"}
)


@dataclass(frozen=True)
class PinholeCameraConfig:
    width: int
    height: int
    fov_y_radians: float | Sequence[float] | None = None
    camera_count: int = 1
    channels: tuple[str, ...] = ("color", "depth")
    load_textures: bool = True
    intrinsic_matrices: Any | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.camera_count <= 0:
            raise ValueError("camera_count must be positive")
        unknown = set(self.channels) - _CHANNELS
        if unknown:
            raise ValueError(f"unsupported camera channels: {sorted(unknown)}")
        has_fov = self.fov_y_radians is not None
        has_intrinsics = self.intrinsic_matrices is not None
        if has_fov == has_intrinsics:
            raise ValueError(
                "provide exactly one of fov_y_radians or intrinsic_matrices"
            )
        if has_fov:
            fovs = (
                [float(self.fov_y_radians)]
                if np.isscalar(self.fov_y_radians)
                else [float(value) for value in self.fov_y_radians]
            )
            if len(fovs) not in {1, self.camera_count}:
                raise ValueError("provide one FOV or one FOV per camera")
            if any(
                not math.isfinite(value) or not 0.0 < value < math.pi
                for value in fovs
            ):
                raise ValueError("camera FOV values must lie between 0 and pi")
        else:
            intrinsics = np.asarray(self.intrinsic_matrices, dtype=np.float64)
            if intrinsics.shape == (3, 3):
                intrinsics = intrinsics[None]
            if intrinsics.shape not in {
                (1, 3, 3),
                (self.camera_count, 3, 3),
            }:
                raise ValueError(
                    "intrinsic_matrices must have shape (3,3), (1,3,3), "
                    "or (camera_count,3,3)"
                )
            if not np.isfinite(intrinsics).all():
                raise ValueError("camera intrinsics must be finite")
            if np.any(intrinsics[:, 0, 0] <= 0.0) or np.any(
                intrinsics[:, 1, 1] <= 0.0
            ):
                raise ValueError("camera focal lengths must be positive")
            if np.any(np.abs(np.linalg.det(intrinsics)) <= 1.0e-12):
                raise ValueError("camera intrinsic matrices must be invertible")


@dataclass(frozen=True)
class CameraReading:
    """Device-resident camera outputs.

    Arrays have shape ``(world_count, camera_count, height, width)`` and remain
    on the Newton model device.  Unrequested channels are ``None``.
    """

    color: Any | None
    depth: Any | None
    forward_depth: Any | None
    normal: Any | None
    albedo: Any | None
    shape_index: Any | None


class NewtonCameraSensor(CameraSensorBase):
    """Render pinhole camera observations with Newton ``SensorTiledCamera``."""

    def __init__(
        self,
        model: Any,
        config: PinholeCameraConfig,
        *,
        render_config: Any | None = None,
        _sensor_cls: type | None = None,
    ) -> None:
        if _sensor_cls is None:
            from newton.sensors import SensorTiledCamera

            _sensor_cls = SensorTiledCamera

        self.model = model
        self.config = config
        self.device = model.device
        self.world_count = max(1, int(model.world_count))

        constructor_parameters = inspect.signature(_sensor_cls).parameters
        sensor_kwargs: dict[str, Any] = {"load_textures": config.load_textures}
        if render_config is not None:
            config_name = (
                "default_render_config"
                if "default_render_config" in constructor_parameters
                else "config"
            )
            sensor_kwargs[config_name] = render_config
        self.native_sensor = _sensor_cls(model, **sensor_kwargs)
        self.utils = self.native_sensor.utils

        if config.intrinsic_matrices is not None:
            intrinsics = np.asarray(
                config.intrinsic_matrices, dtype=np.float64
            )
            if intrinsics.shape == (3, 3):
                intrinsics = intrinsics[None]
            self.intrinsic_matrices = np.broadcast_to(
                intrinsics,
                (config.camera_count, 3, 3),
            ).copy()
            self.camera_rays = self._compute_intrinsic_camera_rays(
                self.intrinsic_matrices
            )
        else:
            camera_fovs = (
                [float(config.fov_y_radians)] * config.camera_count
                if np.isscalar(config.fov_y_radians)
                else [float(value) for value in config.fov_y_radians]
            )
            focal_pixels = np.asarray(
                [
                    0.5 * config.height / math.tan(0.5 * fov)
                    for fov in camera_fovs
                ],
                dtype=np.float64,
            )
            self.intrinsic_matrices = np.repeat(
                np.eye(3, dtype=np.float64)[None],
                config.camera_count,
                axis=0,
            )
            self.intrinsic_matrices[:, 0, 0] = focal_pixels
            self.intrinsic_matrices[:, 1, 1] = focal_pixels
            self.intrinsic_matrices[:, 0, 2] = 0.5 * config.width
            self.intrinsic_matrices[:, 1, 2] = 0.5 * config.height

            ray_builder = getattr(
                self.utils, "compute_camera_rays_pinhole", None
            )
            if ray_builder is not None:
                self.camera_rays = ray_builder(
                    config.width,
                    config.height,
                    camera_fovs=camera_fovs,
                )
            else:
                ray_builder = getattr(
                    self.utils,
                    "compute_pinhole_camera_rays",
                    None,
                )
                if ray_builder is None:
                    raise RuntimeError(
                        "Newton tiled camera lacks a pinhole-ray helper"
                    )
                self.camera_rays = ray_builder(
                    config.width,
                    config.height,
                    camera_fovs,
                )

        self._positions = np.zeros(
            (self.world_count, config.camera_count, 3), dtype=np.float32
        )
        # Keep resident and public pose storage in Newton/Warp-native xyzw.
        self._quaternions_xyzw = np.zeros(
            (self.world_count, config.camera_count, 4), dtype=np.float32
        )
        self._quaternions_xyzw[..., 3] = 1.0
        self.camera_transforms = None
        self._upload_camera_transforms()

        channels = set(config.channels)
        self.color = self._create_output("color") if "color" in channels else None
        self.depth = self._create_output("depth") if "depth" in channels else None
        self.forward_depth = (
            self._create_output("depth") if "forward_depth" in channels else None
        )
        self.normal = self._create_output("normal") if "normal" in channels else None
        self.albedo = self._create_output("albedo") if "albedo" in channels else None
        self.shape_index = (
            self._create_output("shape_index")
            if "shape_index" in channels
            else None
        )
        self._ray_depth = (
            self.depth
            if self.depth is not None
            else self._create_output("depth")
            if self.forward_depth is not None
            else None
        )
        self._render_particle_radii = None

    @property
    def spec(self) -> SensorSpec:
        return SensorSpec(
            sensor_type="camera",
            adapter="newton_tiled_raycast",
            input_fields=("state geometry", "camera poses", "camera rays"),
            output_fields=tuple(self.config.channels),
            differentiable=False,
        )

    def _compute_intrinsic_camera_rays(
        self, intrinsics: np.ndarray
    ) -> Any:
        """Build Newton camera rays from OpenCV/ROS intrinsic matrices."""

        pixel_x, pixel_y = np.meshgrid(
            np.arange(self.config.width, dtype=np.float64) + 0.5,
            np.arange(self.config.height, dtype=np.float64) + 0.5,
        )
        pixels = np.stack(
            (pixel_x, pixel_y, np.ones_like(pixel_x)), axis=-1
        )
        rays = np.zeros(
            (
                self.config.camera_count,
                self.config.height,
                self.config.width,
                2,
                3,
            ),
            dtype=np.float32,
        )
        for camera_id, intrinsic in enumerate(intrinsics):
            optical = pixels @ np.linalg.inv(intrinsic).T
            directions = np.stack(
                (optical[..., 0], -optical[..., 1], -optical[..., 2]),
                axis=-1,
            )
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            rays[camera_id, ..., 1, :] = directions.astype(np.float32)
        return wp.array(rays, dtype=wp.vec3f, device=self.device)

    def set_particle_render_radii(
        self, radii: np.ndarray | Sequence[float]
    ) -> None:
        """Override camera-only particle radii without changing simulation.

        Newton normally points its raytracer at ``model.particle_radius``.
        MPM particle radii can be much larger than their useful display size,
        so a separate device buffer keeps camera appearance independent from
        collision and constitutive parameters.
        """
        radii_array = np.asarray(radii, dtype=np.float32).reshape(-1)
        expected_count = int(self.model.particle_radius.shape[0])
        if radii_array.shape[0] != expected_count:
            raise ValueError(
                "camera particle radii must have one value per model particle: "
                f"{radii_array.shape[0]} != {expected_count}"
            )
        if not np.isfinite(radii_array).all() or np.any(radii_array < 0.0):
            raise ValueError(
                "camera particle radii must be finite and nonnegative"
            )
        self._render_particle_radii = wp.array(
            radii_array, dtype=wp.float32, device=self.device
        )
        # Newton 1.5 removed the deprecated public context property without
        # adding a public setter for camera-only particle radii. Keep the
        # version-specific fallback contained inside this compatibility wrapper.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            render_context = getattr(self.native_sensor, "render_context", None)
        if render_context is None:
            render_context = getattr(
                self.native_sensor,
                "_SensorTiledCamera__render_context",
                None,
            )
        if render_context is None:
            raise RuntimeError(
                "Newton tiled camera does not expose a particle-radius render buffer"
            )
        render_context.particles_radius = self._render_particle_radii

    def _create_output(self, channel: str):
        factory = getattr(self.utils, f"create_{channel}_image_output")
        return factory(
            self.config.width,
            self.config.height,
            self.config.camera_count,
        )

    def _upload_camera_transforms(self) -> None:
        transforms = []
        for camera_id in range(self.config.camera_count):
            row = []
            for world_id in range(self.world_count):
                position = self._positions[world_id, camera_id]
                x, y, z, w = self._quaternions_xyzw[world_id, camera_id]
                row.append(
                    wp.transform(
                        wp.vec3(*position),
                        wp.quat(float(x), float(y), float(z), float(w)),
                    )
                )
            transforms.append(row)
        self.camera_transforms = wp.array(
            transforms, dtype=wp.transform, device=self.device
        )

    def set_camera_poses(
        self,
        positions: np.ndarray | Sequence,
        quaternions_xyzw: np.ndarray | Sequence,
    ) -> None:
        """Set camera-to-world poses with ``xyzw`` quaternions.

        Inputs must have shapes ``(world_count, camera_count, 3)`` and
        ``(world_count, camera_count, 4)``.  For a single world and camera,
        plain ``(3,)`` and ``(4,)`` inputs are accepted.
        """
        positions_array = np.asarray(positions, dtype=np.float32)
        quaternions_array = np.asarray(quaternions_xyzw, dtype=np.float32)
        expected_positions = (self.world_count, self.config.camera_count, 3)
        expected_quaternions = (self.world_count, self.config.camera_count, 4)
        try:
            positions_array = np.broadcast_to(
                positions_array, expected_positions
            ).copy()
            quaternions_array = np.broadcast_to(
                quaternions_array, expected_quaternions
            ).copy()
        except ValueError as exc:
            raise ValueError(
                "camera poses must broadcast to "
                f"{expected_positions} and {expected_quaternions}"
            ) from exc

        norms = np.linalg.norm(quaternions_array, axis=-1, keepdims=True)
        if np.any(norms <= 1.0e-8) or not np.all(np.isfinite(norms)):
            raise ValueError("camera quaternions must be finite and nonzero")
        self._positions = positions_array
        self._quaternions_xyzw = quaternions_array / norms
        self._upload_camera_transforms()

    def set_look_at(
        self,
        positions: np.ndarray | Sequence,
        targets: np.ndarray | Sequence,
        up_vectors: np.ndarray | Sequence = (0.0, 1.0, 0.0),
    ) -> None:
        """Aim camera ``-Z`` axes at targets while keeping ``+Y`` near up.

        Inputs broadcast to ``(world_count, camera_count, 3)``. This helper
        follows the same world-major convention as :meth:`set_camera_poses`.
        """
        expected = (self.world_count, self.config.camera_count, 3)
        try:
            positions_array = np.broadcast_to(
                np.asarray(positions, dtype=np.float64),
                expected,
            ).copy()
            targets_array = np.broadcast_to(
                np.asarray(targets, dtype=np.float64),
                expected,
            )
            up_array = np.broadcast_to(
                np.asarray(up_vectors, dtype=np.float64),
                expected,
            )
        except ValueError as exc:
            raise ValueError(
                f"look-at inputs must broadcast to {expected}"
            ) from exc

        quaternions = np.empty(
            (self.world_count, self.config.camera_count, 4),
            dtype=np.float32,
        )
        from scipy.spatial.transform import Rotation

        for world_id in range(self.world_count):
            for camera_id in range(self.config.camera_count):
                forward = (
                    targets_array[world_id, camera_id]
                    - positions_array[world_id, camera_id]
                )
                forward_norm = float(np.linalg.norm(forward))
                if not math.isfinite(forward_norm) or forward_norm <= 1.0e-8:
                    raise ValueError(
                        "camera position and look-at target must differ"
                    )
                forward /= forward_norm
                right = np.cross(
                    forward,
                    up_array[world_id, camera_id],
                )
                right_norm = float(np.linalg.norm(right))
                if not math.isfinite(right_norm) or right_norm <= 1.0e-8:
                    raise ValueError(
                        "camera up vector must not be parallel to view direction"
                    )
                right /= right_norm
                camera_up = np.cross(right, forward)
                rotation = np.column_stack((right, camera_up, -forward))
                x, y, z, w = Rotation.from_matrix(rotation).as_quat()
                quaternions[world_id, camera_id] = (x, y, z, w)

        self.set_camera_poses(
            positions_array.astype(np.float32),
            quaternions,
        )

    def update(self, state: Any, *, refit_bvh: bool = True) -> CameraReading:
        """Render all configured camera channels on the model device."""
        update_parameters = inspect.signature(self.native_sensor.update).parameters

        if refit_bvh and "refit_bvh" not in update_parameters:
            if hasattr(self.model, "bvh_refit_shapes"):
                self.model.bvh_refit_shapes(state)
            if hasattr(self.model, "bvh_refit_particles"):
                self.model.bvh_refit_particles(state)

        kwargs: dict[str, Any] = {}
        output_map = {
            "color_image": self.color,
            "depth_image": self._ray_depth,
            "normal_image": self.normal,
            "albedo_image": self.albedo,
            "shape_index_image": self.shape_index,
        }
        for name, output in output_map.items():
            if output is not None and name in update_parameters:
                kwargs[name] = output

        direct_forward_depth = (
            self.forward_depth is not None
            and "forward_depth_image" in update_parameters
        )
        if direct_forward_depth:
            kwargs["forward_depth_image"] = self.forward_depth
        if "refit_bvh" in update_parameters:
            kwargs["refit_bvh"] = refit_bvh

        self.native_sensor.update(
            state,
            self.camera_transforms,
            self.camera_rays,
            **kwargs,
        )

        if self.forward_depth is not None and not direct_forward_depth:
            converter = getattr(
                self.utils, "convert_ray_depth_to_forward_depth", None
            )
            if converter is None:
                raise RuntimeError(
                    "Newton tiled camera cannot produce forward depth"
                )
            converter(
                self._ray_depth,
                self.camera_transforms,
                self.camera_rays,
                self.forward_depth,
            )
        return self.reading()

    def reading(self) -> CameraReading:
        return CameraReading(
            color=self.color,
            depth=self.depth,
            forward_depth=self.forward_depth,
            normal=self.normal,
            albedo=self.albedo,
            shape_index=self.shape_index,
        )
