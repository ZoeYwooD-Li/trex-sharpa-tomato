"""Dual-FR3 / dual-Wuji2 harvesting with scripted or external policy control."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re

from loguru import logger
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from lycheesim.engine.runner import DemoSetup
from lycheesim.sensor import NewtonCameraSensor, PinholeCameraConfig
from lycheesim.training import CameraSpec, EpisodeFrame

from .backend import CherryTomatoPickBackend
from .camera import packed_rgba_to_rgb
from .collection import apply_exposure, fruit_tint, prepare_collection
from .recording import (
    CherryTomatoEpisodeRecorder,
    next_episode_path,
    world_from_pose,
)
from .robot import interpolate_pose
from .wuji2 import DualWuji2PickRobot
from .stem_pinch import (
    StemPinchTrial, plate_clearance, premature_stem_contact, trial_script_time, stem_tilt_degrees,
    validate_config as validate_stem_pinch, harvest_workspace_ready,
)


PHASES = (
    "support_approach",
    "support_reach",
    "support_close",
    "support_lift",
    "harvest_approach",
    "harvest_reach",
    "harvest_close",
    "settle",
    "pull_twist",
    "carry",
    "complete",
    "support_contact_wait",
    "support_lift_wait",
    "support_lost",
    "support_reach_wait",
    "harvest_contact_wait",
    "hand_collision_blocked",
    "deposit_idle",
    "deposit_transport",
    "deposit_release",
    "deposit_settle",
    "deposit_complete",
    "deposit_failed",
    "policy",
    "policy_grasp_wait",
    "policy_grasp_timeout",
    "stem_pinch_trial_hold",
    "stem_pinch_failed",
)


@dataclass
class _SupportLossGuard:
    """Latch failure after a continuous contact gap, once lift has begun."""

    gap_s: float = 0.0
    lost: bool = False

    def update(self, contact: bool, dt: float, timeout_s: float) -> bool:
        if dt <= 0.0 or timeout_s <= 0.0:
            raise ValueError("support-loss timing must be positive")
        self.gap_s = 0.0 if contact else self.gap_s + dt
        self.lost = self.lost or self.gap_s + 1e-9 >= timeout_s
        return self.lost


def _bounded_recenter_xy(current, desired, anchor, *, max_step, max_offset):
    """One planar step within a fixed reach-stage anchor, without mutating inputs."""
    current, desired, anchor = [np.asarray(v, dtype=float) for v in (current, desired, anchor)]
    if (any(v.shape != (2,) or not np.isfinite(v).all() for v in (current, desired, anchor))
            or not np.isfinite([max_step, max_offset]).all()
            or max_step <= 0 or max_offset <= 0):
        raise ValueError("recenter requires finite XY points and positive step/offset limits")
    offset = desired - anchor
    length = float(np.linalg.norm(offset))
    if length > max_offset:
        offset = offset * (max_offset / length)
    correction = anchor + offset - current
    length = float(np.linalg.norm(correction))
    return current + correction * min(1., max_step / length) if length > 0. else current.copy()


def _support_pose_reached(actual: np.ndarray, target: np.ndarray) -> bool:
    """Keep the approach hand open until IK reaches the grasp TCP."""
    position_error = np.linalg.norm(actual[:3] - target[:3])
    orientation_error = (
        Rotation.from_quat(actual[3:]) * Rotation.from_quat(target[3:]).inv()
    ).magnitude()
    return bool(position_error <= 0.004 and orientation_error <= 0.08)

CAMERA_COLOR_IMAGE = "sensors/gemini_435le/color"
CAMERA_DEPTH_IMAGE = "sensors/gemini_435le/forward_depth"
CAMERA_FRUSTUM = "sensors/gemini_435le/frustum"
WRIST_CAMERA_SIDES = ("left", "right")
WRIST_CAMERA_COLOR_IMAGES = {
    side: f"sensors/gemini_305_{side}/color" for side in WRIST_CAMERA_SIDES
}
WRIST_CAMERA_DEPTH_IMAGES = {
    side: f"sensors/gemini_305_{side}/forward_depth"
    for side in WRIST_CAMERA_SIDES
}
WRIST_CAMERA_FRUSTUMS = {
    side: f"sensors/gemini_305_{side}/frustum" for side in WRIST_CAMERA_SIDES
}


def _episode_seed(base_seed: int, output_path: Path) -> int:
    """Derive a reproducible seed from an auto-numbered episode path."""

    if base_seed < 0:
        raise ValueError("recording.seed must be non-negative")
    match = re.search(r"(\d+)$", output_path.stem)
    episode_index = 1 if match is None else int(match.group(1))
    return base_seed + max(0, episode_index - 1)


def _nonnegative_xyz(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite XYZ values")
    if np.any(result < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    return result


def _randomize_recording_config(
    cfg: dict,
    *,
    seed: int,
) -> tuple[dict, dict[str, object]]:
    """Sample one reproducible tabletop scene for data collection."""

    effective_cfg = deepcopy(cfg)
    recording = dict(effective_cfg.get("recording", {}) or {})
    randomization = dict(recording.get("randomization", {}) or {})
    if not bool(randomization.get("enabled", False)):
        return effective_cfg, {"enabled": False}

    rng = np.random.default_rng(seed)
    xy_range = float(randomization.get("position_xy_m", 0.0))
    z_range = float(randomization.get("position_z_m", 0.0))
    if (
        not np.isfinite(xy_range)
        or not np.isfinite(z_range)
        or xy_range < 0.0
        or z_range < 0.0
    ):
        raise ValueError(
            "recording randomization position ranges must be finite and "
            "non-negative"
        )
    position_offset = rng.uniform(
        (-xy_range, -xy_range, -z_range),
        (xy_range, xy_range, z_range),
    )

    rotation_range = _nonnegative_xyz(
        randomization.get("cluster_rotation_xyz_deg", (0.0, 0.0, 0.0)),
        name="recording.randomization.cluster_rotation_xyz_deg",
    )
    cluster_euler = rng.uniform(-rotation_range, rotation_range)
    cluster_rotation = Rotation.from_euler("xyz", cluster_euler, degrees=True)

    scene_cfg = effective_cfg["scene"]
    nominal_rotation = Rotation.from_quat(
        scene_cfg.get("cluster_quaternion_xyzw", (0.0, 0.0, 0.0, 1.0))
    )
    # Jitter is in world axes, around the authored pose (including stem-up).
    # Anchor/support offsets below are already nominal world-space vectors.
    cluster_quaternion = (cluster_rotation * nominal_rotation).as_quat()
    nominal_target = np.asarray(scene_cfg["tomato_position"], dtype=np.float64)
    nominal_anchor = np.asarray(scene_cfg["stem_anchor"], dtype=np.float64)
    nominal_support_offset = np.asarray(
        scene_cfg["support_tomato_offset_m"], dtype=np.float64
    )
    sampled_target = nominal_target + position_offset
    scene_cfg["tomato_position"] = sampled_target.tolist()
    scene_cfg["stem_anchor"] = (
        sampled_target + cluster_rotation.apply(nominal_anchor - nominal_target)
    ).tolist()
    scene_cfg["support_tomato_offset_m"] = cluster_rotation.apply(
        nominal_support_offset
    ).tolist()
    scene_cfg["cluster_quaternion_xyzw"] = cluster_quaternion.tolist()
    if effective_cfg['robot'].get('right_grasp_follow_scene_rotation', False):
        robot_cfg = effective_cfg['robot']
        robot_cfg['right_cluster_grasp_quat_xyzw'] = (
            cluster_rotation * Rotation.from_quat(robot_cfg['right_cluster_grasp_quat_xyzw'])).as_quat().tolist()
        for key in ('right_cluster_grasp_center_bias_m', 'right_pregrasp_offset_m', 'right_stem_insert_offset_m'):
            if key in robot_cfg:
                robot_cfg[key] = cluster_rotation.apply(robot_cfg[key]).tolist()

    attachment_cfg = effective_cfg["attachment"]
    attachment_cfg["break_direction_world"] = cluster_rotation.apply(
        np.asarray(attachment_cfg["break_direction_world"], dtype=np.float64)
    ).tolist()
    attachment_cfg["rest_quaternion_xyzw"] = (
        cluster_rotation * Rotation.from_quat(
            attachment_cfg.get("rest_quaternion_xyzw", nominal_rotation.as_quat())
        )
    ).as_quat().tolist()

    light_azimuth_range = float(
        randomization.get("light_azimuth_jitter_deg", 0.0)
    )
    light_elevation_range = float(
        randomization.get("light_elevation_jitter_deg", 0.0)
    )
    if (
        not np.isfinite(light_azimuth_range)
        or not np.isfinite(light_elevation_range)
        or light_azimuth_range < 0.0
        or light_elevation_range < 0.0
    ):
        raise ValueError(
            "recording randomization light jitter ranges must be finite and "
            "non-negative"
        )
    nominal_light = np.asarray(
        effective_cfg["camera"]["sensor"].get(
            "light_direction", (-1.0, 1.0, -1.0)
        ),
        dtype=np.float64,
    )
    light_norm = float(np.linalg.norm(nominal_light))
    if (
        nominal_light.shape != (3,)
        or not np.isfinite(nominal_light).all()
        or light_norm <= 0.0
    ):
        raise ValueError(
            "camera.sensor.light_direction must be a nonzero finite XYZ vector"
        )
    nominal_light /= light_norm
    nominal_azimuth = float(np.arctan2(nominal_light[1], nominal_light[0]))
    nominal_elevation = float(np.arcsin(nominal_light[2]))
    azimuth_delta = float(
        rng.uniform(-light_azimuth_range, light_azimuth_range)
    )
    elevation_delta = float(
        rng.uniform(-light_elevation_range, light_elevation_range)
    )
    azimuth = nominal_azimuth + np.deg2rad(azimuth_delta)
    elevation = nominal_elevation + np.deg2rad(elevation_delta)
    if not (-0.5 * np.pi < elevation < 0.0):
        raise ValueError(
            "randomized light elevation must remain between the horizon and "
            "straight down"
        )
    sampled_light = np.asarray(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ),
        dtype=np.float64,
    )
    for camera_key in ("camera", "wrist_cameras"):
        effective_cfg[camera_key]["sensor"]["light_direction"] = (
            sampled_light.tolist()
        )

    metadata: dict[str, object] = {
        "enabled": True,
        "position_offset_m": position_offset.tolist(),
        "tomato_position_m": sampled_target.tolist(),
        "cluster_euler_xyz_deg": cluster_euler.tolist(),
        "cluster_quaternion_xyzw": cluster_quaternion.tolist(),
        "light_azimuth_delta_deg": azimuth_delta,
        "light_elevation_delta_deg": elevation_delta,
        "light_direction": sampled_light.tolist(),
    }
    return effective_cfg, metadata


def _smoothstep(value: float) -> float:
    amount = float(np.clip(value, 0.0, 1.0))
    return amount * amount * (3.0 - 2.0 * amount)


def _axial_pull_offset(displacement, axis) -> np.ndarray:
    """Preserve pull length while aligning with the parent attachment axis."""
    displacement = np.asarray(displacement, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if (displacement.shape != (3,) or axis.shape != (3,)
            or not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(axis))
            or np.linalg.norm(axis) <= 0.0):
        raise ValueError("axial pull requires finite XYZ vectors and a nonzero axis")
    return axis * (np.linalg.norm(displacement) / np.linalg.norm(axis))


def _harvest_pose(
    robot: DualWuji2PickRobot,
    displacement: np.ndarray,
    yaw: float,
    carrier_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Move/twist about the loaded finger centre rather than the wrist TCP."""

    base = robot.left_grasp_tcp
    carrier = np.zeros(3, dtype=np.float64)
    if carrier_offset is not None:
        carrier = np.asarray(carrier_offset, dtype=np.float64)
        if carrier.shape != (3,):
            raise ValueError("harvest carrier offset must contain XYZ")
    delta_rotation = Rotation.from_rotvec((0.0, 0.0, float(yaw)))
    result = base.copy()
    result[:3] += carrier
    result[3:] = (delta_rotation * Rotation.from_quat(base[3:])).as_quat()
    center_offset = robot.left_grasp_center - base[:3]
    desired_center = (
        robot.left_grasp_center
        + carrier
        + np.asarray(displacement, dtype=np.float64)
    )
    result[:3] = desired_center - delta_rotation.apply(center_offset)
    return result


def _scripted_state(
    time_s: float,
    cfg: dict,
    robot: DualWuji2PickRobot,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return TCPs, closures, carried-bunch offset, and the named phase."""

    script = cfg["script"]
    support_approach_end = float(script["support_approach_end_s"])
    support_reach_end = float(script["support_reach_end_s"])
    support_close_end = float(script["support_close_end_s"])
    support_lift_end = float(script["support_lift_end_s"])
    harvest_approach_end = float(script["harvest_approach_end_s"])
    harvest_reach_end = float(script["harvest_reach_end_s"])
    harvest_close_end = float(script["harvest_close_end_s"])
    settle_end = float(script["settle_end_s"])
    pull_end = float(script["pull_end_s"])
    carry_end = float(script["carry_end_s"])

    left_grasp = robot.left_grasp_tcp
    right_grasp = robot.right_grasp_tcp
    zero_offset = np.zeros(3, dtype=np.float64)
    lift_offset = np.asarray(script["support_lift_offset_m"], dtype=np.float64)
    if lift_offset.shape != (3,):
        raise ValueError("script.support_lift_offset_m must contain XYZ")

    if time_s < support_approach_end:
        amount = time_s / support_approach_end
        return (
            robot.default_left_tcp.copy(),
            robot.right_approach_pose(amount),
            np.zeros(2, dtype=np.float32),
            zero_offset,
            "support_approach",
        )
    if time_s < support_reach_end:
        preshape_end = support_approach_end + float(script.get("support_preshape_duration_s", 0.))
        if robot.right_insert_tcp is not None:
            insert_start = float(script['support_insert_start_s'])
            if time_s < insert_start:
                pose = interpolate_pose(robot.right_pregrasp_tcp, robot.right_insert_tcp,
                                        max(0., time_s - preshape_end) / (insert_start - preshape_end))
            else:
                pose = interpolate_pose(robot.right_insert_tcp, right_grasp,
                                        (time_s - insert_start) / (support_reach_end - insert_start))
            return (robot.default_left_tcp.copy(), pose, np.zeros(2, dtype=np.float32),
                    zero_offset, 'support_reach')
        amount = max(0., time_s - preshape_end) / (
            support_reach_end - preshape_end
        )
        return (
            robot.default_left_tcp.copy(),
            interpolate_pose(robot.right_pregrasp_tcp, right_grasp, amount),
            np.zeros(2, dtype=np.float32),
            zero_offset,
            "support_reach",
        )
    if time_s < support_close_end:
        amount = _smoothstep(
            (time_s - support_reach_end)
            / (support_close_end - support_reach_end)
        )
        return (
            robot.default_left_tcp.copy(),
            right_grasp.copy(),
            np.asarray((0.0, amount), dtype=np.float32),
            zero_offset,
            "support_close",
        )

    lifted_right = right_grasp.copy()
    lifted_right[:3] += lift_offset
    if time_s < support_lift_end:
        if cfg['task'].get('bunch_wrap_control', False) and 'support_trial_ramp_duration_s' in script:
            from .bunch_wrap import lift_fraction
            amount = lift_fraction(time_s, script)
            pose = right_grasp.copy()
            pose[:3] += amount * lift_offset
            return (robot.default_left_tcp.copy(), pose,
                    np.asarray((0., 1.), dtype=np.float32), amount * lift_offset, 'support_lift')
        amount = _smoothstep(
            (time_s - support_close_end)
            / (support_lift_end - support_close_end)
        )
        return (
            robot.default_left_tcp.copy(),
            interpolate_pose(right_grasp, lifted_right, amount),
            np.asarray((0.0, 1.0), dtype=np.float32),
            amount * lift_offset,
            "support_lift",
        )

    if cfg['task'].get('bunch_wrap_control', False) and script.get('support_lift_only', True):
        return (robot.default_left_tcp.copy(), lifted_right.copy(),
                np.asarray((0., 1.), dtype=np.float32), lift_offset.copy(), 'support_lift_wait')

    lifted_left_pregrasp = robot.left_pregrasp_tcp.copy()
    lifted_left_pregrasp[:3] += lift_offset
    lifted_left_grasp = left_grasp.copy()
    lifted_left_grasp[:3] += lift_offset
    if time_s < harvest_approach_end:
        amount = (time_s - support_lift_end) / (
            harvest_approach_end - support_lift_end
        )
        return (
            interpolate_pose(
                robot.default_left_tcp, lifted_left_pregrasp, amount
            ),
            lifted_right.copy(),
            np.asarray((0.0, 1.0), dtype=np.float32),
            lift_offset.copy(),
            "harvest_approach",
        )
    if time_s < harvest_reach_end:
        amount = (time_s - harvest_approach_end) / (
            harvest_reach_end - harvest_approach_end
        )
        return (
            interpolate_pose(lifted_left_pregrasp, lifted_left_grasp, amount),
            lifted_right.copy(),
            np.asarray((0.0, 1.0), dtype=np.float32),
            lift_offset.copy(),
            "harvest_reach",
        )
    if time_s < harvest_close_end:
        amount = _smoothstep(
            (time_s - harvest_reach_end)
            / (harvest_close_end - harvest_reach_end)
        )
        return (
            lifted_left_grasp.copy(),
            lifted_right.copy(),
            np.asarray((amount, 1.0), dtype=np.float32),
            lift_offset.copy(),
            "harvest_close",
        )
    if time_s < settle_end:
        return (
            lifted_left_grasp.copy(),
            lifted_right.copy(),
            np.ones(2),
            lift_offset.copy(),
            "settle",
        )

    pull_offset = np.asarray(script["pull_offset_m"], dtype=np.float64)
    pulled = _harvest_pose(
        robot,
        pull_offset,
        float(script["twist_rad"]),
        lift_offset,
    )
    if time_s < pull_end:
        amount = (time_s - settle_end) / (pull_end - settle_end)
        return (
            interpolate_pose(lifted_left_grasp, pulled, amount),
            lifted_right.copy(),
            np.ones(2),
            lift_offset.copy(),
            "pull_twist",
        )

    carry_offset = np.asarray(script["carry_offset_m"], dtype=np.float64)
    carried = _harvest_pose(
        robot,
        pull_offset + carry_offset,
        float(script["twist_rad"]),
        lift_offset,
    )
    if time_s < carry_end:
        amount = (time_s - pull_end) / (carry_end - pull_end)
        return (
            interpolate_pose(pulled, carried, amount),
            lifted_right.copy(),
            np.ones(2),
            lift_offset.copy(),
            "carry",
        )
    return (
        carried,
        lifted_right.copy(),
        np.ones(2),
        lift_offset.copy(),
        "complete",
    )


def build(cfg: dict, config_path: Path | None = None, *, policy=None) -> DemoSetup:
    validate_stem_pinch(cfg)
    stem_pinch_enabled = bool(cfg['task'].get('stem_pinch_control', False))
    wrap_enabled = bool(cfg['task'].get('bunch_wrap_control', False))
    guarded_support = stem_pinch_enabled or wrap_enabled
    if guarded_support and policy is not None:
        raise ValueError('guarded support is a scripted experiment, not an OpenPI action adapter')
    pinch_trial = StemPinchTrial() if guarded_support else None
    pinch_trial_time = trial_script_time(cfg['script']) if guarded_support else None
    recenter_enabled = bool(cfg["script"].get("support_recenter_while_closing", False))
    recenter_speed = float(cfg["script"].get("support_recenter_speed_m_s", .005))
    recenter_limit = float(cfg["script"].get("support_recenter_max_offset_m", .020))
    if recenter_enabled and (not np.isfinite([recenter_speed, recenter_limit]).all()
                             or recenter_speed <= 0 or recenter_limit <= 0):
        raise ValueError("support recenter speed and offset limit must be finite and positive")
    retreat = float(cfg["script"].get("post_detach_retreat_m", 0.0))
    if not np.isfinite(retreat) or retreat < 0.0:
        raise ValueError("post_detach_retreat_m must be finite and non-negative")
    if retreat > 0.0:
        duration = float(cfg["script"].get("post_detach_retreat_duration_s", 0.0))
        axis = np.asarray(cfg["script"]["pull_offset_m"], dtype=float)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("post_detach_retreat_duration_s must be finite and positive")
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0.0:
            raise ValueError("post-detach retreat requires a finite nonzero pull_offset_m")
    recording_cfg = dict(cfg.get("recording", {}) or {})
    recording_enabled = bool(recording_cfg.get("enabled", False))
    selected_output_path: Path | None = None
    recording_seed = int(recording_cfg.get("seed", 42))
    randomization_metadata: dict[str, object] = {"enabled": False}
    target_condition = None
    scene_validation = {"status": "disabled"}
    if recording_enabled:
        output_value = recording_cfg.get("output_path")
        if not output_value:
            raise ValueError(
                "recording.output_path is required when recording is enabled"
            )
        selected_output_path = Path(str(output_value)).expanduser()
        if not selected_output_path.is_absolute():
            selected_output_path = (
                Path(__file__).resolve().parents[2] / selected_output_path
            )
        overwrite = bool(recording_cfg.get("overwrite", False))
        auto_increment = bool(recording_cfg.get("auto_increment", True))
        if auto_increment and not overwrite:
            selected_output_path = next_episode_path(selected_output_path)
        if bool(recording_cfg.get("vary_seed_by_episode", True)):
            recording_seed = _episode_seed(recording_seed, selected_output_path)
        cfg, randomization_metadata, target_condition, scene_validation = prepare_collection(
            cfg, recording_seed, _randomize_recording_config,
        )

    if policy is not None:
        policy.configure(cfg)
    device = str(cfg["sim"].get("device", "cpu"))
    backend = CherryTomatoPickBackend(
        cfg,
        num_envs=1,
        device=device,
        randomize=False,
        config_path=config_path,
    )
    scene = backend.primary_scene
    if guarded_support and cfg['task'].get('minimum_left_tableware_clearance_m'):
        scene.robot.left_tableware = scene.tableware
        scene.robot.minimum_left_tableware_clearance = cfg['task']['minimum_left_tableware_clearance_m']
    deposit = None
    if cfg.get("deposit", {}).get("enabled", False):
        from .tableware import DepositMotion
        if scene.tableware is None or not scene.main_stem_grasp:
            raise ValueError("deposit requires tableware and physical main-stem support")
        deposit = DepositMotion(scene.tableware, cfg["deposit"])
    solver = scene.solver
    camera = scene.camera
    wrist_cameras = scene.wrist_cameras
    camera_cfg = camera.sensor_cfg
    wrist_camera_cfg = wrist_cameras.sensor_cfg
    camera_sensor: NewtonCameraSensor | None = None
    wrist_camera_sensor: NewtonCameraSensor | None = None
    camera_sample_hz = 10.0
    wrist_camera_sample_hz = 30.0
    camera_sample_period_s = 1.0 / camera_sample_hz
    wrist_camera_sample_period_s = 1.0 / wrist_camera_sample_hz
    from newton.sensors import SensorTiledCamera

    if scene.camera_sensor_enabled:
        channels = tuple(
            str(value)
            for value in camera_cfg.get("channels", ("color", "forward_depth"))
        )
        if "color" not in channels or "forward_depth" not in channels:
            raise ValueError(
                "Gemini 435Le requires color and forward_depth channels"
            )
        fixed_render_config = SensorTiledCamera.RenderConfig(
            enable_textures=bool(camera_cfg.get("textures", True)),
            enable_shadows=bool(camera_cfg.get("shadows", True)),
            enable_ambient_lighting=bool(
                camera_cfg.get("ambient_lighting", True)
            ),
            enable_backface_culling=bool(
                camera_cfg.get("backface_culling", True)
            ),
        )
        camera_sensor = NewtonCameraSensor(
            solver.model,
            PinholeCameraConfig(
                width=camera.width,
                height=camera.height,
                camera_count=1,
                channels=channels,
                load_textures=bool(camera_cfg.get("load_textures", True)),
                intrinsic_matrices=camera.intrinsic_matrix,
            ),
            render_config=fixed_render_config,
        )
        camera_sensor.set_camera_poses(camera.position, camera.quaternion_xyzw)
        if bool(camera_cfg.get("default_light", True)):
            light_direction = np.asarray(
                camera_cfg.get("light_direction", (-1.0, 1.0, -1.0)),
                dtype=np.float32,
            )
            if light_direction.shape != (3,) or not np.isfinite(
                light_direction
            ).all():
                raise ValueError("camera sensor light_direction must be finite XYZ")
            length = float(np.linalg.norm(light_direction))
            if length <= 1.0e-8:
                raise ValueError("camera sensor light_direction must be nonzero")
            camera_sensor.utils.create_default_light(
                enable_shadows=bool(camera_cfg.get("shadows", True)),
                direction=wp.vec3(*(light_direction / length)),
            )
        camera_sample_hz = float(camera_cfg.get("sample_hz", 10.0))
        if (
            not np.isfinite(camera_sample_hz)
            or camera_sample_hz <= 0.0
            or camera_sample_hz > 1.0 / solver.dt
        ):
            raise ValueError(
                "camera sensor sample_hz must be positive and no greater "
                "than the physics rate"
            )
        camera_sample_period_s = 1.0 / camera_sample_hz
        logger.info(
            "Gemini 435Le enabled: {}x{} {:.3f}Hz, position={}, target={}",
            camera.width,
            camera.height,
            camera_sample_hz,
            camera.position.tolist(),
            camera.target.tolist(),
        )
    if scene.wrist_camera_sensor_enabled:
        channels = tuple(
            str(value)
            for value in wrist_camera_cfg.get(
                "channels", ("color", "forward_depth")
            )
        )
        if "color" not in channels or "forward_depth" not in channels:
            raise ValueError(
                "Gemini 305 wrist cameras require color and forward_depth channels"
            )
        wrist_render_config = SensorTiledCamera.RenderConfig(
            enable_textures=bool(wrist_camera_cfg.get("textures", True)),
            enable_shadows=bool(wrist_camera_cfg.get("shadows", True)),
            enable_ambient_lighting=bool(
                wrist_camera_cfg.get("ambient_lighting", True)
            ),
            enable_backface_culling=bool(
                wrist_camera_cfg.get("backface_culling", True)
            ),
        )
        wrist_camera_sensor = NewtonCameraSensor(
            solver.model,
            PinholeCameraConfig(
                width=wrist_cameras.width,
                height=wrist_cameras.height,
                camera_count=2,
                channels=channels,
                load_textures=bool(
                    wrist_camera_cfg.get("load_textures", True)
                ),
                intrinsic_matrices=wrist_cameras.intrinsic_matrix,
            ),
            render_config=wrist_render_config,
        )
        wrist_positions, wrist_quaternions = wrist_cameras.camera_poses(
            scene.robot.tcp_poses()
        )
        wrist_camera_sensor.set_camera_poses(
            wrist_positions, wrist_quaternions
        )
        if bool(wrist_camera_cfg.get("default_light", True)):
            light_direction = np.asarray(
                wrist_camera_cfg.get("light_direction", (-1.0, 1.0, -1.0)),
                dtype=np.float32,
            )
            if light_direction.shape != (3,) or not np.isfinite(
                light_direction
            ).all():
                raise ValueError(
                    "wrist camera sensor light_direction must be finite XYZ"
                )
            length = float(np.linalg.norm(light_direction))
            if length <= 1.0e-8:
                raise ValueError(
                    "wrist camera sensor light_direction must be nonzero"
                )
            wrist_camera_sensor.utils.create_default_light(
                enable_shadows=bool(wrist_camera_cfg.get("shadows", True)),
                direction=wp.vec3(*(light_direction / length)),
            )
        wrist_camera_sample_hz = float(
            wrist_camera_cfg.get("sample_hz", 30.0)
        )
        if (
            not np.isfinite(wrist_camera_sample_hz)
            or wrist_camera_sample_hz <= 0.0
            or wrist_camera_sample_hz > 1.0 / solver.dt
        ):
            raise ValueError(
                "wrist camera sensor sample_hz must be positive and no "
                "greater than the physics rate"
            )
        wrist_camera_sample_period_s = 1.0 / wrist_camera_sample_hz
        logger.info(
            "dual Gemini 305 wrist cameras enabled: {}x{} {:.3f}Hz",
            wrist_cameras.width,
            wrist_cameras.height,
            wrist_camera_sample_hz,
        )

    episode_recorder: CherryTomatoEpisodeRecorder | None = None
    robot_record_hz = 30.0
    robot_record_period_s = 1.0 / robot_record_hz
    if recording_enabled:
        if camera_sensor is None or wrist_camera_sensor is None:
            raise ValueError(
                "recording requires the fixed Gemini 435Le and both Gemini 305 "
                "wrist camera sensors to be enabled"
            )
        robot_record_hz = float(
            recording_cfg.get(
                "robot_sample_hz", recording_cfg.get("sample_hz", 30.0)
            )
        )
        if (
            not np.isfinite(robot_record_hz)
            or robot_record_hz <= 0.0
            or robot_record_hz > 1.0 / solver.dt
        ):
            raise ValueError(
                "recording.robot_sample_hz must be positive and no greater "
                "than the physics frequency"
            )
        robot_record_period_s = 1.0 / robot_record_hz
        if selected_output_path is None:
            raise RuntimeError("recording output path was not initialized")
        joint_names = scene.robot.joint_names()
        action_names = (
            "left_tcp_x_m",
            "left_tcp_y_m",
            "left_tcp_z_m",
            "left_tcp_qx",
            "left_tcp_qy",
            "left_tcp_qz",
            "left_tcp_qw",
            "right_tcp_x_m",
            "right_tcp_y_m",
            "right_tcp_z_m",
            "right_tcp_qx",
            "right_tcp_qy",
            "right_tcp_qz",
            "right_tcp_qw",
            f"left_{scene.robot.model_tag}_closure",
            f"right_{scene.robot.model_tag}_closure",
        )
        episode_recorder = CherryTomatoEpisodeRecorder(
            selected_output_path,
            seed=recording_seed,
            object_ids=(
                str(cfg["scene"]["target_name"]),
                str(cfg["scene"]["support_name"]),
                "branch",
            ),
            cameras={
                "gemini_435le": CameraSpec(
                    image_size=(camera.height, camera.width),
                    intrinsics=camera.intrinsic_matrix,
                    distortion=np.zeros(5, dtype=np.float64),
                    world_from_camera=world_from_pose(
                        camera.position, camera.quaternion_xyzw
                    ),
                    metadata={
                        "device": "Orbbec Gemini 435Le",
                        "pose_convention": "opengl",
                        "depth_dataset": "depth_m",
                        "pose_dataset": "world_from_camera",
                        "nominal_sample_hz": camera_sample_hz,
                        "clipping_range_m": list(camera.clipping_range_m),
                    },
                ),
                **{
                    f"gemini_305_{side}": CameraSpec(
                        image_size=(wrist_cameras.height, wrist_cameras.width),
                        intrinsics=wrist_cameras.intrinsic_matrix,
                        distortion=np.zeros(5, dtype=np.float64),
                        metadata={
                            "device": "Orbbec Gemini 305",
                            "mount": f"{side}_{scene.robot.model_tag}_wrist",
                            "pose_convention": "opengl",
                            "depth_dataset": "depth_m",
                            "pose_dataset": "world_from_camera",
                            "nominal_sample_hz": wrist_camera_sample_hz,
                            "optical_position_in_wrist_m": (
                                wrist_cameras.local_positions[side].tolist()
                            ),
                            "optical_quat_in_wrist_xyzw": (
                                wrist_cameras.local_quaternions_xyzw[
                                    side
                                ].tolist()
                            ),
                            "clipping_range_m": list(
                                wrist_cameras.clipping_range_m
                            ),
                        },
                    )
                    for side in WRIST_CAMERA_SIDES
                },
            },
            metadata={
                "coordinate_system": "right_handed_z_up",
                "quaternion_order": "xyzw",
                "robot_sample_hz": robot_record_hz,
                "camera_sample_hz": {
                    "gemini_435le": camera_sample_hz,
                    "gemini_305_left": wrist_camera_sample_hz,
                    "gemini_305_right": wrist_camera_sample_hz,
                },
                "physics_dt_s": solver.dt,
                "unsafe_contact_force_N": float(cfg["task"]["unsafe_contact_force_N"]),
                "timestamp_units": "s",
                "camera_alignment": "latest_not_after_robot_timestamp",
                "joint_names": list(joint_names),
                "joint_position_order": (
                    f"left_fr3_7,left_{scene.robot.model_tag}_{scene.robot.left.hand.nq},"
                    f"right_fr3_7,right_{scene.robot.model_tag}_{scene.robot.right.hand.nq}"
                ),
                "action_names": list(action_names),
                "action_representation": (
                    f"left/right TCP pose targets plus {scene.robot.model_tag} closure targets"
                ),
                "phase_names": list(PHASES),
                "deposit_target_box": deposit.index if deposit is not None else None,
                "success_definition": (
                    "physical_detachment_and_settled_in_selected_box_with_right_support"
                    if deposit is not None else "harvest_and_retention"
                ),
                "support_mode": cfg["scene"].get("support_mode", "fruit"),
                "bunch_motion": (
                    "dynamic_contact_driven" if scene.main_stem_grasp
                    else "kinematic_carrier"
                ),
                "right_contact_source": (
                    "filtered_pad_and_distal_vs_stem_and_selected_fruits" if wrap_enabled else
                    "pad_and_distal_vs_branch_and_support_fruit"
                    if scene.stem_and_fruit else "pad_and_distal_vs_stem"
                    if stem_pinch_enabled else "pad_vs_stem"
                    if scene.main_stem_grasp else "pad_vs_support_fruit"
                ),
                "right_force_closure_order": list(scene.robot.right_force_fingers),
                "support_fruit_names": cfg['scene'].get('support_fruit_names', []),
                "support_lift_only": bool(wrap_enabled and cfg['script'].get('support_lift_only', True)),
                "left_force_closure_order": ["thumb", "index", "middle"],
                "harvest_contact_control": bool(cfg["task"].get("harvest_contact_control", False)),
                "contact_force_order": [
                    *(f"left_{finger}" for finger in scene.robot.tactile_fingers),
                    *(f"right_{finger}" for finger in scene.robot.tactile_fingers),
                ],
                "camera_streams": [
                    "gemini_435le",
                    "gemini_305_left",
                    "gemini_305_right",
                ],
                "depth_units": "m",
                "randomization": randomization_metadata,
                "target_condition": target_condition,
                "scene_validation": scene_validation,
                "quality_filter": recording_cfg.get("quality_filter", {}),
                "effective_config": cfg,
            },
            overwrite=bool(recording_cfg.get("overwrite", False)),
            auto_increment=False,
        )
        logger.info(
            "cherry-tomato multirate recorder enabled: robot={:.3f}Hz, "
            "fixed_camera={:.3f}Hz, wrist_cameras={:.3f}Hz -> {}",
            robot_record_hz,
            camera_sample_hz,
            wrist_camera_sample_hz,
            episode_recorder.output_path,
        )
        if bool(randomization_metadata["enabled"]):
            logger.info(
                "recording episode seed={} randomization={}",
                recording_seed,
                randomization_metadata,
            )
    render = cfg.get("render", {}) or {}
    every = int(render.get("every", 1))
    state = {
        "phase": "support_approach",
        "last_phase": None,
        "detached_hold_left": None,
        "detach_time": None,
        "next_camera_sample_time": 0.0,
        "next_wrist_camera_sample_time": 0.0,
        "next_robot_record_time": 0.0,
        "camera_samples": 0,
        "camera_color": None,
        "camera_depth": None,
        "camera_valid_depth_pixels": 0,
        "wrist_camera_samples": 0,
        "wrist_camera_color": {side: None for side in WRIST_CAMERA_SIDES},
        "wrist_camera_depth": {side: None for side in WRIST_CAMERA_SIDES},
        "wrist_camera_valid_depth_pixels": {
            side: 0 for side in WRIST_CAMERA_SIDES
        },
        "record_previous_joint_q": None,
        "record_previous_time": None,
        "record_pending_action": None,
        "episode_complete": False,
        "script_time": 0.0,
        "stem_lift_start": None,
        "left_retargeted": False,
        "left_contact_target_locked": False,
        "support_loss_guard": _SupportLossGuard(),
        "support_lost_poses": None,
        "stem_pinch_error": "",
        "stem_pinch_trial_passed": False,
        "stem_pinch_min_plate_clearance_m": float('inf'),
        "stem_pinch_lift_local": None,
        "stem_pinch_workspace_wait_s": 0.,
        "support_full_lift_hold_s": 0.,
        "support_lift_verified": False,
    }
    initial_right_grasp = scene.robot.right_grasp_tcp.copy()
    initial_right_pregrasp = scene.robot.right_pregrasp_tcp.copy()
    initial_right_insert = (None if scene.robot.right_insert_tcp is None
                            else scene.robot.right_insert_tcp.copy())
    initial_left_grasp = scene.robot.left_grasp_tcp.copy()
    initial_left_pregrasp = scene.robot.left_pregrasp_tcp.copy()
    stem_in_right_wrist = Rotation.from_quat(initial_right_grasp[3:]).inv().apply(
        scene.stem_grasp_position - initial_right_grasp[:3]
    )
    recenter_anchor = None

    def retarget_left_harvest(position: np.ndarray) -> None:
        # Scripted ground-truth tracking, not camera-based perception. Preserve
        # the grasp attitude and leave lift addition to _scripted_state.
        lift = np.asarray(cfg["script"]["support_lift_offset_m"])
        for attr, original in (("left_grasp_tcp", initial_left_grasp),
                               ("left_pregrasp_tcp", initial_left_pregrasp)):
            pose = original.copy()
            pose[:3] = position + original[:3] - scene.tomato_start - lift
            setattr(scene.robot, attr, pose)
        scene.harvest_reference_tcp = scene.robot.left_grasp_tcp.copy()
        scene.harvest_reference_tcp[:3] += lift

    def stem_script_time() -> float:
        """Gate physical lift on contact, and harvest on measured elevation."""
        nonlocal recenter_anchor
        if scene.robot.hand_collision_blocked:
            return state["script_time"]
        current = state["script_time"]
        if state["support_loss_guard"].lost:
            return current
        candidate = round(current + scene.physics_dt, 9)
        close_end = float(cfg["script"]["support_close_end_s"])
        lift_end = float(cfg["script"]["support_lift_end_s"])
        reach_end = float(cfg["script"]["support_reach_end_s"])
        if ((scene.stem_and_fruit or stem_pinch_enabled)
                and cfg["script"].get("support_close_on_early_contact", True)
                and current < float(cfg["script"]["support_reach_end_s"])
                and scene.right_contact_force_N.max() > 0.3):
            # Stop reaching at first contact, then close remaining fingers.
            # Stem pinch reads only the stem-specific contact matrix here.
            scene.robot.right_grasp_tcp = scene.right_tcp.copy()
            stem_in_right_wrist[:] = Rotation.from_quat(scene.right_tcp[3:]).inv().apply(
                scene.stem_grasp_position - scene.right_tcp[:3]
            )
            current = float(cfg["script"]["support_reach_end_s"])
            candidate = current
        pose_lock = 0.5 * float(cfg['script']['support_approach_end_s'])
        if wrap_enabled:
            from .bunch_wrap import pose_lock_time
            pose_lock = pose_lock_time(cfg['script'])
        if current < pose_lock:
            position, quaternion = solver.body_pose(scene.branch_index)
            delta_rotation = (Rotation.from_quat(quaternion)
                              * Rotation.from_quat(scene.cluster_quaternion_xyzw).inv())
            for attr, original in (("right_grasp_tcp", initial_right_grasp),
                                   ("right_pregrasp_tcp", initial_right_pregrasp),
                                   ("right_insert_tcp", initial_right_insert)):
                if original is None:
                    continue
                pose = original.copy()
                pose[:3] = position + delta_rotation.apply(original[:3] - scene.tomato_start)
                pose[3:] = (delta_rotation * Rotation.from_quat(original[3:])).as_quat()
                setattr(scene.robot, attr, pose)
        if not scene.stem_and_fruit and not stem_pinch_enabled and float(cfg["script"]["support_reach_end_s"]) <= current <= close_end and not scene.support_engaged:
            pose = scene.robot.right_grasp_tcp
            predicted = pose[:3] + Rotation.from_quat(pose[3:]).apply(stem_in_right_wrist)
            correction = scene.stem_grasp_position - predicted
            length = float(np.linalg.norm(correction))
            if length > 0.0:
                pose[:3] += correction * min(1.0, 0.02 * scene.physics_dt / length)
        if (recenter_enabled and (scene.stem_and_fruit or stem_pinch_enabled)
                and reach_end <= current <= close_end and not scene.support_engaged):
            # Scripted ground-truth assistance only. Never lower the hand into
            # the plate, rotate it, chase indefinitely, or modify policy actions.
            pose = scene.robot.right_grasp_tcp
            if recenter_anchor is None:
                recenter_anchor = pose[:2].copy()
            desired = scene.stem_grasp_position - Rotation.from_quat(pose[3:]).apply(stem_in_right_wrist)
            pose[:2] = _bounded_recenter_xy(
                pose[:2], desired[:2], recenter_anchor,
                max_step=recenter_speed * scene.physics_dt, max_offset=recenter_limit,
            )
        approach_end = float(cfg['script']['support_approach_end_s'])
        if (scene.robot.right_reach_q is not None and current <= approach_end < candidate
                and not _support_pose_reached(scene.robot.tcp_poses()[1], scene.robot.right_pregrasp_tcp)):
            candidate = approach_end
        insert_start = float(cfg['script'].get('support_insert_start_s', 0.))
        if (scene.robot.right_insert_tcp is not None and current <= insert_start < candidate
                and not _support_pose_reached(scene.robot.tcp_poses()[1], scene.robot.right_insert_tcp)):
            candidate = insert_start
        if (
            cfg["script"].get("support_require_reached_pose", False)
            and current <= reach_end < candidate
            and not _support_pose_reached(
                scene.robot.tcp_poses()[1], scene.robot.right_grasp_tcp
            )
        ):
            candidate = reach_end
        if current <= close_end and candidate > close_end:
            if not scene.support_engaged:
                candidate = close_end
            elif state["stem_lift_start"] is None:
                state["stem_lift_start"] = scene.stem_grasp_position.copy()
                if guarded_support:
                    tcp = scene.robot.tcp_poses()[1]
                    state['stem_pinch_lift_local'] = Rotation.from_quat(tcp[3:]).inv().apply(
                        scene.stem_grasp_position - tcp[:3])
        if pinch_trial is not None and not pinch_trial.passed and current <= pinch_trial_time < candidate:
            candidate = pinch_trial_time
        if current <= lift_end and candidate > lift_end:
            origin = state["stem_lift_start"]
            lifted = (origin is not None and scene.stem_grasp_position[2] - origin[2]
                      >= float(cfg["task"]["minimum_cluster_lift_m"]))
            tilt = stem_tilt_degrees(solver.body_pose(scene.branch_index)[1],
                                    cfg['scene']['branch_collision_axis'])
            workspace_ready = harvest_workspace_ready(scene.cluster_clearance, tilt, cfg['task'])
            if not scene.support_engaged or not lifted or not workspace_ready:
                candidate = lift_end
                state['support_full_lift_hold_s'] = 0.
                state['support_lift_verified'] = False
                if pinch_trial is not None and 'harvest_workspace_timeout_s' in cfg['task']:
                    state['stem_pinch_workspace_wait_s'] += scene.physics_dt
                    if state['stem_pinch_workspace_wait_s'] >= cfg['task']['harvest_workspace_timeout_s']:
                        pinch_trial.error = 'harvest_workspace_not_clear'
            elif wrap_enabled and (state['support_full_lift_hold_s'] < cfg['script']['support_full_lift_hold_s']
                                   or cfg['script'].get('support_lift_only', True)):
                candidate = lift_end
            elif not state["left_retargeted"]:
                # Freeze a new left approach around the observed raised fruit;
                # planned lift is removed because _scripted_state adds it back.
                position, quaternion = solver.body_pose(scene.tomato_index)
                delta_rotation = (Rotation.from_quat(quaternion)
                                  * Rotation.from_quat(scene.cluster_quaternion_xyzw).inv())
                retarget_left_harvest(position)
                if cfg["script"].get("harvest_pull_along_stem_axis", False):
                    cfg["script"]["pull_offset_m"] = _axial_pull_offset(
                        cfg["script"]["pull_offset_m"], scene.params.break_direction_world
                    ).tolist()
                else:
                    cfg["script"]["pull_offset_m"] = delta_rotation.apply(
                        np.asarray(cfg["script"]["pull_offset_m"])
                    ).tolist()
                state["left_retargeted"] = True
        if (state["left_retargeted"]
                and cfg["script"].get("harvest_track_target_until_contact", False)
                and current <= float(cfg["script"]["harvest_close_end_s"])
                and not state["left_contact_target_locked"]):
            # Slow contact-controlled closure outlasts the nominal close ramp.
            # Follow drift only until first touch; never chase a pushed fruit.
            if scene.left_contact_force_N[:3].max() >= 0.2:
                state["left_contact_target_locked"] = True
            else:
                position, _ = solver.body_pose(scene.tomato_index)
                retarget_left_harvest(position)
        harvest_close_end = float(cfg["script"]["harvest_close_end_s"])
        if (cfg["task"].get("harvest_contact_control", False)
                and current <= harvest_close_end < candidate
                and scene.grasp_hold_steps * scene.physics_dt
                < float(cfg["task"]["minimum_grasp_hold_s"])):
            candidate = harvest_close_end
        state["script_time"] = candidate
        return candidate
    camera_frustum = camera.frustum_lines()

    def sample_due(time_s: float, state_key: str, period_s: float) -> bool:
        """Advance a fractional-rate scheduler on the physics timeline."""

        deadline = float(state[state_key])
        if time_s + 0.5 * solver.dt < deadline:
            return False
        while deadline <= time_s + 0.5 * solver.dt:
            deadline += period_s
        state[state_key] = deadline
        return True

    def register(frontend) -> None:
        if scene.tableware is not None:
            for fixture in scene.tableware.fixtures:
                frontend.add_mesh(f"tableware/{fixture.name}", fixture.mesh.vertices,
                                  fixture.mesh.faces, color=fixture.color, solid=True)
        colors = {
            "branch": (0.16, 0.48, 0.12),
            "tomato": (0.88, 0.035, 0.025),
            "support_tomato": (0.76, 0.045, 0.035),
        }
        visual_names = (
            tuple(scene.visual_meshes)
            if scene.visual_meshes
            else ("branch", "tomato", "support_tomato")
        )
        for name in visual_names:
            visual = scene.visual_meshes.get(name)
            body_name = visual.body_name if visual is not None else name
            index = scene.indices[body_name]
            frontend.add_mesh(
                name,
                (
                    visual.vertices
                    if visual is not None
                    else solver.mesh_local_vertices(index)
                ),
                visual.faces if visual is not None else solver.mesh_local_faces(index),
                color=(fruit_tint(name, cfg) if visual is not None
                       else colors.get(body_name, (0.88, 0.035, 0.025))),
                solid=True,
                normals=visual.normals if visual is not None else None,
                texcoords=visual.uvs if visual is not None else None,
                albedo_texture=visual.texture if visual is not None else None,
            )
            frontend.set_source_metadata(
                name,
                {
                    "role": "environment" if name == "branch" else "object",
                    "dynamic": body_name != "branch",
                    "quaternion_order": "xyzw",
                    "visual_asset": (
                        "user_supplied_cherry_tomato_branch_02"
                        if visual is not None
                        else "procedural_fallback"
                    ),
                },
            )
        for part in scene.robot.visual_parts:
            frontend.add_mesh(
                part.name,
                part.vertices,
                part.faces,
                color=(1.0, 1.0, 1.0),
                solid=True,
                vertex_colors=part.vertex_colors,
            )
            frontend.set_source_metadata(
                part.name,
                {
                    "role": "robot",
                    "kinematic": True,
                    "robot": f"dual_fr3_{scene.robot.model_tag}",
                    "quaternion_order": "xyzw",
                },
            )
        for part in camera.visual_parts:
            frontend.add_mesh(
                part.name,
                part.vertices,
                part.faces,
                color=part.color,
                solid=True,
            )
            frontend.set_source_metadata(
                part.name,
                {
                    "role": "sensor",
                    "dynamic": False,
                    "device": "Orbbec Gemini 435Le",
                    "pose_source": "photo_derived_yaml_calibration",
                    "quaternion_order": "xyzw",
                },
            )
        for part in wrist_cameras.visual_parts:
            frontend.add_mesh(
                part.name,
                part.vertices,
                part.faces,
                color=part.color,
                solid=True,
            )
            frontend.set_source_metadata(
                part.name,
                {
                    "role": "sensor",
                    "dynamic": True,
                    "device": "Orbbec Gemini 305",
                    "mount": f"{part.side}_{scene.robot.model_tag}_wrist",
                    "pose_source": "photo_derived_yaml_extrinsic",
                    "quaternion_order": "xyzw",
                },
            )
        frontend.add_camera(
            "cameras/gemini_435le",
            width=camera.width,
            height=camera.height,
            intrinsic_matrix=camera.intrinsic_matrix,
            horizontal_aperture=1.0,
            vertical_aperture=float(camera.height / camera.width),
            focal_length=float(camera.intrinsic_matrix[0, 0] / camera.width),
            clipping_range_m=camera.clipping_range_m,
            pose_convention="opengl",
        )
        for side in WRIST_CAMERA_SIDES:
            frontend.add_camera(
                f"cameras/gemini_305_{side}",
                width=wrist_cameras.width,
                height=wrist_cameras.height,
                intrinsic_matrix=wrist_cameras.intrinsic_matrix,
                horizontal_aperture=1.0,
                vertical_aperture=float(
                    wrist_cameras.height / wrist_cameras.width
                ),
                focal_length=float(
                    wrist_cameras.intrinsic_matrix[0, 0]
                    / wrist_cameras.width
                ),
                clipping_range_m=wrist_cameras.clipping_range_m,
                pose_convention="opengl",
            )
        frontend.add_lines(
            CAMERA_FRUSTUM,
            n_strips=8,
            points_per_strip=2,
            color=(0.20, 0.72, 1.0),
            radius=0.0007,
        )
        for side in WRIST_CAMERA_SIDES:
            frontend.add_lines(
                WRIST_CAMERA_FRUSTUMS[side],
                n_strips=8,
                points_per_strip=2,
                color=(0.92, 0.65, 0.18),
                radius=0.0005,
            )
        if camera_sensor is not None:
            frontend.add_image(CAMERA_COLOR_IMAGE, kind="color", visible=True)
            frontend.add_image(CAMERA_DEPTH_IMAGE, kind="depth", visible=True)
            frontend.add_scalar(
                "sensors/gemini_435le/valid_depth_pixels", visible=False
            )
        if wrist_camera_sensor is not None:
            for side in WRIST_CAMERA_SIDES:
                frontend.add_image(
                    WRIST_CAMERA_COLOR_IMAGES[side],
                    kind="color",
                    visible=True,
                )
                frontend.add_image(
                    WRIST_CAMERA_DEPTH_IMAGES[side],
                    kind="depth",
                    visible=True,
                )
                frontend.add_scalar(
                    f"sensors/gemini_305_{side}/valid_depth_pixels",
                    visible=False,
                )
        for name in (
            "phase_index",
            "attachment_load",
            "attachment_force_N",
            "attachment_torque_Nm",
            "detached",
            "support_ready",
            "grasped",
            "harvest_distance_m",
            "left_contact_N",
            "right_contact_N",
            "ik_failed",
        ):
            frontend.add_scalar(
                f"task/{name}",
                visible=name in {"attachment_load", "harvest_distance_m"},
            )
        frontend.set_sequence_metadata(
            {
                "demo": "cherry_tomato_pick",
                "coordinate_system": "right_handed_z_up",
                "quaternion_order": "xyzw",
                "robot": f"dual FR3 + left/right {scene.robot.model_tag}",
                "robot_kinematics": "official with-mount roots attached to FR3 link8",
                "contact_geometry": f"ten {scene.robot.model_tag} fingertip contact meshes",
                "cameras": (
                    "fixed Orbbec Gemini 435Le plus left/right wrist-mounted "
                    "Orbbec Gemini 305 RGB-D"
                ),
                "camera_pose_source": "supplied laboratory photographs; YAML-adjustable",
                "camera_resolution": [camera.width, camera.height],
                "camera_intrinsics": camera.intrinsic_matrix.tolist(),
                "camera_sample_hz": camera_sample_hz,
                "camera_clipping_range_m": list(camera.clipping_range_m),
                "wrist_camera_resolution": [
                    wrist_cameras.width,
                    wrist_cameras.height,
                ],
                "wrist_camera_intrinsics": (
                    wrist_cameras.intrinsic_matrix.tolist()
                ),
                "wrist_camera_sample_hz": wrist_camera_sample_hz,
                "wrist_camera_clipping_range_m": list(
                    wrist_cameras.clipping_range_m
                ),
                "source_repository": "git@github.com:yxsdev/cherry_tomato_sim.git",
                "source_revision": "43610761c4f667dc24ac6b5f7d66b71ebf72c1a1",
                "ported_behavior": (
                    "right-arm opposite-end grasp/lift and left-arm "
                    "detachable harvest"
                ),
                "asset_policy": (
                    "local user-supplied detailed mesh; procedural collision geometry"
                    if scene.visual_meshes
                    else "procedural visual and collision geometry"
                ),
            }
        )

    def capture_robot_sample(time_s: float, *, observed_gripper=None) -> None:
        if episode_recorder is None or episode_recorder.capture_finished:
            return

        joint_q = scene.robot.joint_q()
        previous_joint_q = state["record_previous_joint_q"]
        previous_time = state["record_previous_time"]
        if previous_joint_q is None or previous_time is None:
            joint_velocity = np.zeros_like(joint_q)
        else:
            elapsed = float(time_s) - float(previous_time)
            if elapsed <= 0.0:
                raise RuntimeError("recording timestamps must increase")
            joint_velocity = (
                joint_q - np.asarray(previous_joint_q, dtype=np.float32)
            ) / elapsed

        left_tcp, right_tcp = scene.robot.tcp_poses()
        tomato_position, tomato_quaternion = solver.body_pose(
            scene.tomato_index
        )
        support_position, support_quaternion = solver.body_pose(
            scene.support_index
        )
        linear_velocity, angular_velocity = solver.body_velocities(
            (scene.tomato_index, scene.support_index)
        )
        signals = scene.read_signals()
        phase_index = (
            PHASES.index(state["phase"]) if state["phase"] in PHASES else -1
        )
        control_target = np.concatenate(
            (
                scene.left_tcp,
                scene.right_tcp,
                np.asarray((scene.left_grip, scene.right_grip)),
            )
        ).astype(np.float32)
        terminal = bool(
            (signals["success"][0] and (
                not scene.main_stem_grasp
                or time_s >= float(cfg["sim"]["duration"]) - robot_record_period_s
            ))
            or signals["dropped"][0]
            or signals["ik_failed"][0]
            or state["support_loss_guard"].lost
            or scene.robot.hand_collision_blocked
        )
        if deposit is not None:
            # An intentional free fall is not a failed harvest. Success is
            # measured after settling inside the selected box, not at fracture.
            terminal = bool(deposit.placed or deposit.failed or scene.support_lost
                            or scene.robot.hand_collision_blocked or scene.robot.ik_failed)
        terminal = terminal or bool(state['stem_pinch_error'])
        frame = EpisodeFrame(
            time=float(time_s),
            images={},
            joint_pos=joint_q,
            joint_vel=joint_velocity.astype(np.float32),
            ee_pose=np.stack((left_tcp, right_tcp)).astype(np.float32),
            gripper=np.asarray(
                (scene.left_grip, scene.right_grip) if observed_gripper is None
                else observed_gripper, dtype=np.float32
            ),
            extra={
                **({'stem_pinch_failed': np.asarray((bool(state['stem_pinch_error']),), dtype=np.bool_),
                    'stem_pinch_trial_passed': np.asarray((state['stem_pinch_trial_passed'],), dtype=np.bool_)}
                   if guarded_support else {}),
                **({'support_lift_verified': np.asarray((state['support_lift_verified'],), dtype=np.bool_)}
                   if wrap_enabled else {}),
                **({"right_lift_gate_phase": np.asarray((policy.grasp_gate.phase_index,), dtype=np.int16),
                    "right_lift_gate_ceiling_m": np.asarray((policy.grasp_gate.ceiling_m
                                                            if policy.grasp_gate.ceiling_m is not None else 0.,),
                                                           dtype=np.float32)}
                   if policy is not None else {}),
                **({"target_box_one_hot": np.asarray(target_condition["target_box_one_hot"], dtype=np.float32)}
                   if target_condition is not None else {}),
                "placed_in_box": np.asarray((bool(deposit and deposit.placed),), dtype=np.bool_),
                "deposit_failed": np.asarray((bool(deposit and deposit.failed),), dtype=np.bool_),
                "hand_collision_blocked": signals["hand_collision_blocked"].copy(),
                "interhand_distance_m": signals["interhand_distance_m"].copy(),
                "target_tomato_pose": np.concatenate(
                    (tomato_position, tomato_quaternion)
                ).astype(np.float32),
                "target_tomato_velocity": np.concatenate(
                    (linear_velocity[0], angular_velocity[0])
                ).astype(np.float32),
                "support_tomato_pose": np.concatenate(
                    (support_position, support_quaternion)
                ).astype(np.float32),
                "support_tomato_velocity": np.concatenate(
                    (linear_velocity[1], angular_velocity[1])
                ).astype(np.float32),
                "control_target": control_target,
                "policy_command": np.asarray(
                    scene.command, dtype=np.float32
                ).copy(),
                "phase_index": np.asarray((phase_index,), dtype=np.int16),
                "attachment_state": np.asarray(
                    (
                        signals["attachment_raw_load"][0],
                        signals["attachment_load"][0],
                        signals["attachment_force_N"][0],
                        signals["attachment_torque_Nm"][0],
                        scene.attachment_failure.overload_duration_s,
                        signals["detached"][0],
                    ),
                    dtype=np.float32,
                ),
                "task_state": np.asarray(
                    (
                        signals["support_ready"][0],
                        signals["grasped"][0],
                        signals["detached"][0],
                        float(deposit.failed) if deposit is not None else signals["dropped"][0],
                        float(deposit.placed and not deposit.failed and not scene.support_lost
                              and not scene.robot.hand_collision_blocked and not scene.robot.ik_failed)
                        if deposit is not None else signals["success"][0],
                        signals["ik_failed"][0],
                        signals["harvest_distance"][0],
                    ),
                    dtype=np.float32,
                ),
                "contact_force_N": np.asarray(
                    signals["contact_force_N"], dtype=np.float32
                ).copy(),
                "cluster_carrier_offset_m": np.asarray(
                    scene.cluster_offset, dtype=np.float32
                ).copy(),
                "branch_pose": np.asarray(signals["branch_pose"], dtype=np.float32),
                "cluster_clearance_m": np.asarray(signals["cluster_clearance_m"], dtype=np.float32),
                "peak_contact_force_N": np.asarray((scene.peak_contact_force_N,), dtype=np.float32),
                "right_stem_contact_force_N": np.asarray(signals["right_stem_contact_force_N"], dtype=np.float32),
                "right_contact_preload": scene.robot.right_contact_preload.astype(np.float32).copy(),
                "right_force_closure": np.asarray(
                    scene.robot.right_force_closure
                    if scene.robot.right_force_closure is not None
                    else [scene.right_grip] * len(scene.robot.right_force_fingers),
                    dtype=np.float32,
                ).copy(),
                "left_force_closure": np.asarray(
                    scene.robot.left_force_closure
                    if scene.robot.left_force_closure is not None
                    else [scene.left_grip] * 3,
                    dtype=np.float32,
                ).copy(),
                "support_lost": np.asarray(
                    (state["support_loss_guard"].lost,), dtype=np.bool_
                ),
                "left_retention": np.asarray(
                    (scene.left_retention,), dtype=np.float32
                ),
            },
        )
        if episode_recorder.frame_count == 0:
            episode_recorder.append_robot(frame)
        else:
            episode_recorder.append_robot(
                frame,
                transition_action=np.asarray(
                    state["record_pending_action"], dtype=np.float32
                ),
                terminated=terminal,
            )
        state["record_previous_joint_q"] = joint_q.copy()
        state["record_previous_time"] = float(time_s)
        state["record_pending_action"] = control_target

    def step(time_s: float, inputs: dict) -> None:
        if state["episode_complete"]:
            return
        command = inputs.get("tomato_robot_command")
        if policy is not None:
            policy.last_sim_time = float(time_s)
            if command is not None:
                policy.error = "teleoperation input is not allowed during policy evaluation"
                state["episode_complete"] = True
                return
            # Hold the current absolute targets between 30 Hz ticks. Never call
            # scripted harvesting, ground-truth retargeting or deposit.command.
            state["phase"] = "policy"
            if bool(scene.read_signals()["support_ready"][0]) and scene.cluster_clearance >= .005:
                policy.support_engaged = True
            if policy.support_engaged:
                contact = scene._opposed_contact(scene.support_contact_forces()) >= float(
                    cfg["task"].get("minimum_support_retention_contact_N", .2))
                scene.support_lost = state["support_loss_guard"].update(
                    contact, scene.physics_dt, float(cfg["task"].get("support_loss_timeout_s", .08)))
            detach_time = state["detach_time"]
            retention = (0. if detach_time is None else _smoothstep(
                (time_s - detach_time) / float(cfg["script"]["retention_tighten_duration_s"])))
            backend.set_left_retentions(np.asarray((retention,), dtype=np.float32))
        elif command is not None:
            backend.set_robot_command(np.asarray(command, dtype=np.float32))
            state["phase"] = "teleop"
        else:
            if scene.main_stem_grasp and state["stem_lift_start"] is not None:
                contact = scene._opposed_contact(scene.support_contact_forces()) >= float(
                    cfg["task"].get("minimum_support_retention_contact_N", 0.2)
                )
                if stem_pinch_enabled:
                    contact = bool(contact and scene.stem_pinch_opposed() and np.all(
                        scene.support_contact_forces() >= cfg['task']['minimum_support_retention_contact_N']))
                if wrap_enabled:
                    contact = bool(contact and scene.wrap_contacts_ready(retention=True))
                lost = state["support_loss_guard"].update(
                    contact, scene.physics_dt,
                    float(cfg["task"].get("support_loss_timeout_s", 0.08)),
                )
                if lost and state["support_lost_poses"] is None:
                    state["support_lost_poses"] = (
                        tuple(pose.copy() for pose in scene.robot.tcp_poses())
                    )
                    scene.support_lost = True
            left, right, closures, cluster_offset, phase = _scripted_state(
                stem_script_time() if scene.main_stem_grasp else time_s, cfg, scene.robot
            )
            if scene.robot.right_reach_q is not None:
                start = float(cfg['script']['support_approach_end_s'])
                duration = float(cfg['script']['support_preshape_duration_s'])
                scene.robot.right_preshape_amount = _smoothstep(
                    (state['script_time'] - start) / duration)
            if scene.main_stem_grasp:
                if (state["stem_lift_start"] is not None
                        and cfg["script"].get("support_follow_bunch_pitch", True)):
                    # Let the wrist follow the bunch's swing about the pinch
                    # point while translating that point along the lift path.
                    _, branch_quat = solver.body_pose(scene.branch_index)
                    initial_rotation = Rotation.from_quat(scene.cluster_quaternion_xyzw)
                    local_axis = (initial_rotation.inv() * Rotation.from_quat(branch_quat)).apply([0., 1., 0.])
                    pitch = float(np.arctan2(local_axis[2], local_axis[1]))
                    rotation = (initial_rotation * Rotation.from_euler("x", pitch)
                                * initial_rotation.inv()
                                * Rotation.from_quat(scene.robot.right_grasp_tcp[3:]))
                    pinch = right[:3] + Rotation.from_quat(right[3:]).apply(stem_in_right_wrist)
                    right[:3] = pinch - rotation.apply(stem_in_right_wrist)
                    right[3:] = rotation.as_quat()
                if (cfg["script"].get("support_require_reached_pose", False)
                        and state["script_time"] == float(cfg["script"]["support_reach_end_s"])):
                    phase = "support_reach_wait"
                elif state["script_time"] == float(cfg["script"]["support_close_end_s"]):
                    phase = "support_contact_wait"
                elif state["script_time"] == float(cfg["script"]["support_lift_end_s"]):
                    phase = "support_lift_wait"
                elif pinch_trial is not None and state['script_time'] == pinch_trial_time:
                    phase = 'stem_pinch_trial_hold'
                elif (cfg["task"].get("harvest_contact_control", False)
                      and state["script_time"] == float(cfg["script"]["harvest_close_end_s"])):
                    phase = "harvest_contact_wait"
            if state["support_lost_poses"] is not None:
                left, right = state["support_lost_poses"]
                phase = "support_lost"
            if (state["detached_hold_left"] is not None
                    and state["support_lost_poses"] is None
                    and not scene.robot.hand_collision_blocked):
                left = state["detached_hold_left"].copy()
                retreat = float(cfg["script"].get("post_detach_retreat_m", 0.0))
                if retreat > 0.0:
                    duration = float(cfg["script"]["post_detach_retreat_duration_s"])
                    if duration <= 0.0:
                        raise ValueError("post_detach_retreat_duration_s must be positive")
                    axis = np.asarray(cfg["script"]["pull_offset_m"], dtype=float)
                    left[:3] += retreat * _smoothstep((time_s - state["detach_time"]) / duration) * axis / np.linalg.norm(axis)
            if (deposit is not None and scene.detached and not scene.support_lost
                    and not scene.robot.hand_collision_blocked):
                fruit, _ = solver.body_pose(scene.tomato_index)
                left, grip, phase = deposit.command(
                    time_s, scene.robot.tcp_poses()[0], fruit,
                    float(cfg["scene"]["tomato_radius_m"]),
                    bool(scene.read_signals()["grasped"][0]),
                )
                closures = np.asarray((grip, closures[1]))
            backend.set_robot_states(
                left[None, :], right[None, :], closures[None, :]
            )
            if not scene.main_stem_grasp:
                backend.set_cluster_offsets(cluster_offset[None, :])
            detach_time = state["detach_time"]
            retention = 0.0
            if detach_time is not None:
                duration = float(
                    cfg["script"]["retention_tighten_duration_s"]
                )
                if duration <= 0.0:
                    raise ValueError(
                        "script.retention_tighten_duration_s must be positive"
                    )
                retention = _smoothstep((time_s - detach_time) / duration)
            backend.set_left_retentions(
                np.asarray((retention,), dtype=np.float32)
            )
            state["phase"] = phase
        backend.step()
        if pinch_trial is not None:
            origin = state['stem_lift_start']
            tcp = scene.robot.tcp_poses()[1]
            local = Rotation.from_quat(tcp[3:]).inv().apply(scene.stem_grasp_position - tcp[:3])
            pinch_trial.update(
                dt=scene.physics_dt,
                closing=state['script_time'] >= cfg['script']['support_reach_end_s'],
                lifted=origin is not None,
                at_trial=state['script_time'] == pinch_trial_time,
                ready=scene.support_engaged,
                lift_m=0. if origin is None else float(scene.stem_grasp_position[2] - origin[2]),
                slip_m=0. if origin is None else float(np.linalg.norm(local-state['stem_pinch_lift_local'])),
                script=cfg['script'],
            )
            # Keep task force limits, and include incidental fruit contacts in
            # the peak even though they cannot qualify as a stem pinch.
            failures = {'support_lost': scene.support_lost,
                        'left_tableware_clearance': scene.robot.left_tableware_blocked,
                        'ik_failed': scene.robot.ik_failed,
                        'hand_collision': scene.robot.hand_collision_blocked,
                        'unsafe_contact': not np.isfinite(scene.peak_contact_force_N)
                            or scene.peak_contact_force_N > cfg['task']['unsafe_contact_force_N']}
            if stem_pinch_enabled and cfg['script'].get('stem_pinch_open_approach', False):
                failures['stem_contact_before_reach'] = premature_stem_contact(
                    state['script_time'], cfg['script']['support_reach_end_s'],
                    scene.support_contact_forces())
            poses = scene.robot.collision_poses()
            for part in scene.robot.collision_parts:
                if part.side != 'right':
                    continue
                pos, quat = poses[part.name]
                gap = plate_clearance(Rotation.from_quat(quat).apply(part.vertices)+pos, scene.tableware)
                state['stem_pinch_min_plate_clearance_m'] = min(state['stem_pinch_min_plate_clearance_m'], gap)
                if gap < 0.:
                    failures['plate_clearance'] = True
            if any(failures.values()) and not pinch_trial.error:
                pinch_trial.error = ', '.join(k for k, v in failures.items() if v)
            state['stem_pinch_error'] = pinch_trial.error
            state['stem_pinch_trial_passed'] = pinch_trial.passed
            if pinch_trial.error:
                state['phase'] = 'stem_pinch_failed'
                if deposit is not None:
                    deposit.failed = True
        if wrap_enabled:
            from .bunch_wrap import advance_lift_hold
            origin = state['stem_lift_start']
            tilt = stem_tilt_degrees(solver.body_pose(scene.branch_index)[1],
                                    cfg['scene']['branch_collision_axis'])
            valid = bool(not pinch_trial.error and pinch_trial.passed
                         and state['script_time'] >= cfg['script']['support_lift_end_s']
                         and scene.support_engaged and not scene.support_lost
                         and scene.wrap_contacts_ready(retention=True)
                         and origin is not None
                         and scene.stem_grasp_position[2] - origin[2] >= cfg['task']['minimum_cluster_lift_m']
                         and harvest_workspace_ready(scene.cluster_clearance, tilt, cfg['task']))
            state['support_full_lift_hold_s'], state['support_lift_verified'] = advance_lift_hold(
                state['support_full_lift_hold_s'], scene.physics_dt, valid=valid,
                required=cfg['script']['support_full_lift_hold_s'])
        if policy is not None:
            failures = {"support_lost": scene.support_lost,
                        "hand_collision": scene.robot.hand_collision_blocked,
                        "ik_failed": scene.robot.ik_failed,
                        "unsafe_contact": policy.contact_requires_stop(scene.peak_contact_force_N)}
            if any(failures.values()):
                policy.error = "policy safety stop: " + ", ".join(k for k, v in failures.items() if v)
            if not policy.error:
                policy.update_grasp_wait(time_s, scene)
            # Only the model opens the hand. Physical in-box settling is still
            # evaluated by DepositMotion.observe, never its scripted command.
            policy.update_release(deposit, time_s, scene.detached, scene.left_grip)
        if deposit is not None and scene.detached:
            if (scene.support_lost or scene.robot.hand_collision_blocked or scene.robot.ik_failed
                    or (policy is not None and policy.error)):
                deposit.failed = True
            fruit, _ = solver.body_pose(scene.tomato_index)
            velocity, _ = solver.body_velocities((scene.tomato_index,))
            deposit.observe(fruit, velocity[0], float(cfg["scene"]["tomato_radius_m"]), scene.physics_dt)
            if deposit.placed:
                state["phase"] = "deposit_complete"
        if scene.robot.hand_collision_blocked:
            state["phase"] = "hand_collision_blocked"
        # Keep the existing stable-in-box criterion. Capture fresh terminal
        # images before the robot sample freezes the multirate recorder.
        completed = bool(deposit is not None and deposit.placed and not deposit.failed)
        if wrap_enabled and cfg['script'].get('support_lift_only', True) and state['support_lift_verified']:
            completed = True  # Diagnostic lift completion, NOT harvesting success.
        if pinch_trial is not None and pinch_trial.error:
            completed = True  # Finalize a rejected attempt, never a success.
        if policy is not None and policy.error:
            completed = True  # End the attempt, not a successful task label.
        if camera_sensor is not None and (completed or sample_due(
            time_s, "next_camera_sample_time", camera_sample_period_s
        )):
            reading = camera_sensor.update(solver.state_0)
            if reading.color is None or reading.forward_depth is None:
                raise RuntimeError("Gemini 435Le did not return RGB-D")
            depth = np.asarray(
                reading.forward_depth.numpy()[0, 0], dtype=np.float32
            ).copy()
            near, far = camera.clipping_range_m
            valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
            depth[~valid] = np.nan
            state["camera_color"] = apply_exposure(packed_rgba_to_rgb(
                reading.color.numpy()[0, 0]
            ), cfg.get("collection_appearance", {}))
            state["camera_depth"] = depth
            state["camera_valid_depth_pixels"] = int(valid.sum())
            state["camera_samples"] += 1
            if episode_recorder is not None:
                episode_recorder.append_camera(
                    "gemini_435le",
                    time=time_s,
                    color=state["camera_color"],
                    depth_m=depth,
                    world_from_camera=world_from_pose(
                        camera.position, camera.quaternion_xyzw
                    ),
                    valid_depth_pixels=state["camera_valid_depth_pixels"],
                )
            if state["camera_samples"] == 1:
                logger.info(
                    "GEMINI_435LE_FRAME={} valid_depth_pixels={}",
                    state["camera_samples"],
                    state["camera_valid_depth_pixels"],
                )
        if wrist_camera_sensor is not None and (completed or sample_due(
            time_s,
            "next_wrist_camera_sample_time",
            wrist_camera_sample_period_s,
        )):
            wrist_positions, wrist_quaternions = wrist_cameras.camera_poses(
                scene.robot.tcp_poses()
            )
            wrist_camera_sensor.set_camera_poses(
                wrist_positions, wrist_quaternions
            )
            wrist_reading = wrist_camera_sensor.update(solver.state_0)
            if (
                wrist_reading.color is None
                or wrist_reading.forward_depth is None
            ):
                raise RuntimeError("Gemini 305 wrist cameras did not return RGB-D")
            colors = wrist_reading.color.numpy()[0]
            depths = wrist_reading.forward_depth.numpy()[0]
            near, far = wrist_cameras.clipping_range_m
            for index, side in enumerate(WRIST_CAMERA_SIDES):
                depth = np.asarray(depths[index], dtype=np.float32).copy()
                valid = np.isfinite(depth) & (depth >= near) & (depth <= far)
                depth[~valid] = np.nan
                state["wrist_camera_color"][side] = apply_exposure(packed_rgba_to_rgb(
                    colors[index]
                ), cfg.get("collection_appearance", {}))
                state["wrist_camera_depth"][side] = depth
                state["wrist_camera_valid_depth_pixels"][side] = int(
                    valid.sum()
                )
                if episode_recorder is not None:
                    episode_recorder.append_camera(
                        f"gemini_305_{side}",
                        time=time_s,
                        color=state["wrist_camera_color"][side],
                        depth_m=depth,
                        world_from_camera=world_from_pose(
                            wrist_positions[index], wrist_quaternions[index]
                        ),
                        valid_depth_pixels=(
                            state["wrist_camera_valid_depth_pixels"][side]
                        ),
                    )
            state["wrist_camera_samples"] += 1
            if state["wrist_camera_samples"] == 1:
                logger.info(
                    "GEMINI_305_WRIST_FRAME={} valid_depth_pixels={}",
                    state["wrist_camera_samples"],
                    state["wrist_camera_valid_depth_pixels"],
                )
        observed_gripper = (scene.left_grip, scene.right_grip) if policy is not None else None
        if policy is not None and not completed and policy.due(time_s, solver.dt):
            try:
                # The recorded pending action must match the target held over
                # the NEXT observation interval, as in the training converter.
                policy.apply(time_s, solver.dt, scene, state, backend)
            except (RuntimeError, ValueError, TypeError) as exc:
                policy.error = str(exc)
                completed = True
        if policy is not None:
            if policy.grasp_gate.phase in {"waiting", "confirmed"}:
                state["phase"] = "policy_grasp_wait"
            elif policy.grasp_gate.phase == "timed_out":
                state["phase"] = "policy_grasp_timeout"
        if episode_recorder is not None and (completed or sample_due(
            time_s, "next_robot_record_time", robot_record_period_s
        )):
            capture_robot_sample(time_s, observed_gripper=observed_gripper)
        state["episode_complete"] = completed
        if scene.detached and state["detached_hold_left"] is None:
            # The scripted demonstration stops the wrist at fracture and
            # shows the target separated in the real closed hand. RL remains
            # fully action-driven and receives no such phase override.
            state["detached_hold_left"] = scene.left_tcp.copy()
            state["detach_time"] = float(time_s)
        if state["phase"] != state["last_phase"]:
            signals = scene.read_signals()
            logger.info(
                "CHERRY_TOMATO_PHASE={} t={:.3f} detached={} support={} grasped={} load={:.3f}",
                state["phase"],
                time_s,
                bool(signals["detached"][0]),
                bool(signals["support_ready"][0]),
                bool(signals["grasped"][0]),
                float(signals["attachment_load"][0]),
            )
            state["last_phase"] = state["phase"]

    def snapshot(frontend, time_s: float) -> None:
        del time_s
        if scene.visual_meshes:
            for name in scene.visual_meshes:
                frontend.update_pose(name, *scene.visual_pose(name))
        else:
            for name in ("branch", "tomato", "support_tomato"):
                frontend.update_pose(name, *solver.body_pose(scene.indices[name]))
        for part in scene.robot.visual_parts:
            frontend.update_pose(part.name, *scene.robot.visual_pose(part))
        for part in camera.visual_parts:
            frontend.update_pose(
                part.name, part.position, part.quaternion_xyzw
            )
        wrist_poses = scene.robot.tcp_poses()
        for part in wrist_cameras.visual_parts:
            frontend.update_pose(
                part.name,
                *wrist_cameras.visual_pose(part, wrist_poses),
            )
        frontend.update_camera_pose(
            "cameras/gemini_435le",
            camera.position,
            camera.quaternion_xyzw,
        )
        frontend.update_lines(CAMERA_FRUSTUM, camera_frustum)
        wrist_positions, wrist_quaternions = wrist_cameras.camera_poses(
            wrist_poses
        )
        wrist_frustums = wrist_cameras.frustum_lines(wrist_poses)
        for index, side in enumerate(WRIST_CAMERA_SIDES):
            frontend.update_camera_pose(
                f"cameras/gemini_305_{side}",
                wrist_positions[index],
                wrist_quaternions[index],
            )
            frontend.update_lines(
                WRIST_CAMERA_FRUSTUMS[side], wrist_frustums[side]
            )
        if camera_sensor is not None and state["camera_color"] is not None:
            frontend.update_image(CAMERA_COLOR_IMAGE, state["camera_color"])
            frontend.update_image(CAMERA_DEPTH_IMAGE, state["camera_depth"])
            frontend.update_scalar(
                "sensors/gemini_435le/valid_depth_pixels",
                state["camera_valid_depth_pixels"],
            )
        if wrist_camera_sensor is not None:
            for side in WRIST_CAMERA_SIDES:
                if state["wrist_camera_color"][side] is None:
                    continue
                frontend.update_image(
                    WRIST_CAMERA_COLOR_IMAGES[side],
                    state["wrist_camera_color"][side],
                )
                frontend.update_image(
                    WRIST_CAMERA_DEPTH_IMAGES[side],
                    state["wrist_camera_depth"][side],
                )
                frontend.update_scalar(
                    f"sensors/gemini_305_{side}/valid_depth_pixels",
                    state["wrist_camera_valid_depth_pixels"][side],
                )
        signals = scene.read_signals()
        values = {
            "phase_index": PHASES.index(state["phase"]) if state["phase"] in PHASES else -1,
            "attachment_load": signals["attachment_load"][0],
            "attachment_force_N": signals["attachment_force_N"][0],
            "attachment_torque_Nm": signals["attachment_torque_Nm"][0],
            "detached": signals["detached"][0],
            "support_ready": signals["support_ready"][0],
            "grasped": signals["grasped"][0],
            "harvest_distance_m": signals["harvest_distance"][0],
            "left_contact_N": np.max(signals["left_contact_force_N"]),
            "right_contact_N": np.max(signals["right_contact_force_N"]),
            "ik_failed": signals["ik_failed"][0],
        }
        for name, value in values.items():
            frontend.update_scalar(f"task/{name}", float(value))

    def verify() -> None:
        if state['stem_pinch_error']:
            label = 'bunch wrap' if wrap_enabled else 'stem pinch'
            raise RuntimeError(label + ' failed: ' + state['stem_pinch_error'])
        if wrap_enabled and cfg['script'].get('support_lift_only', True):
            if not state['support_lift_verified']:
                raise RuntimeError('bunch wrap full lift/hold not verified')
            logger.info('[verify] bunch wrap lift/hold PASS; harvesting was disabled')
            return
        signals = scene.read_signals()
        result = {
            "hand_collision_blocked": scene.robot.hand_collision_blocked,
            "minimum_interhand_distance_m": scene.robot.minimum_observed_interhand_distance,
            "peak_contact_force_N": float(scene.peak_contact_force_N),
            "cluster_clearance_m": float(scene.cluster_clearance),
            "detached": bool(signals["detached"][0]),
            "support_ready": bool(signals["support_ready"][0]),
            "grasped": bool(signals["grasped"][0]),
            "success": bool(signals["success"][0]),
            "support_lost": state["support_loss_guard"].lost,
            "ik_failed": bool(signals["ik_failed"][0]),
            "harvest_distance_m": float(signals["harvest_distance"][0]),
            "max_attachment_load": float(scene.max_attachment_load),
            "left_harvest_contact_force_N": np.asarray(
                signals["left_contact_force_N"]
            ).tolist(),
            "right_support_contact_force_N": np.asarray(
                signals["right_contact_force_N"]
            ).tolist(),
        }
        result["safe_contact"] = max(
            result["left_harvest_contact_force_N"]
            + result["right_support_contact_force_N"]
        ) <= float(cfg["task"]["unsafe_contact_force_N"])
        if scene.main_stem_grasp:
            result["safe_contact"] = result["safe_contact"] and (
                result["peak_contact_force_N"]
                <= float(cfg["task"]["unsafe_contact_force_N"])
            )
        if camera_sensor is not None:
            result["camera_samples"] = int(state["camera_samples"])
            result["camera_valid_depth_pixels"] = int(
                state["camera_valid_depth_pixels"]
            )
            if result["camera_samples"] <= 0:
                raise RuntimeError("Gemini 435Le produced no camera samples")
            if result["camera_valid_depth_pixels"] <= 0:
                raise RuntimeError("Gemini 435Le produced no valid depth")
        if wrist_camera_sensor is not None:
            result["wrist_camera_samples"] = int(
                state["wrist_camera_samples"]
            )
            result["wrist_camera_valid_depth_pixels"] = dict(
                state["wrist_camera_valid_depth_pixels"]
            )
            if result["wrist_camera_samples"] <= 0:
                raise RuntimeError("Gemini 305 wrist cameras produced no samples")
            if any(
                pixels <= 0
                for pixels in result["wrist_camera_valid_depth_pixels"].values()
            ):
                raise RuntimeError(
                    "a Gemini 305 wrist camera produced no valid depth"
                )
        if episode_recorder is not None:
            result["recording_robot_samples"] = (
                episode_recorder.robot_frame_count
            )
            result["recording_camera_frames"] = (
                episode_recorder.camera_frame_counts
            )
            if result["recording_robot_samples"] < 2:
                raise RuntimeError("episode recorder captured fewer than two frames")
            if any(
                count < 1
                for count in result["recording_camera_frames"].values()
            ):
                raise RuntimeError("episode recorder missed a camera stream")
        holding_bunch = (not scene.main_stem_grasp or (
            result["support_ready"] and result["grasped"]
            and result["cluster_clearance_m"] >= 0.005
            and not result["support_lost"]
        ))
        if deposit is not None:
            result["placed_in_box"] = deposit.placed
            result["deposit_failed"] = deposit.failed
            result["success"] = bool(deposit.placed and result["detached"] and not deposit.failed)
            holding_bunch = (result["support_ready"] and not result["support_lost"]
                             and result["cluster_clearance_m"] >= .005)
        if (not result["success"] or not result["safe_contact"] or result["ik_failed"]
                or result["hand_collision_blocked"] or not holding_bunch):
            raise RuntimeError(f"cherry_tomato_pick verify failed: {result}")
        logger.info("[verify] cherry_tomato_pick {} -> PASS", result)

    def close() -> None:
        if episode_recorder is not None:
            episode_recorder.close()

    setup = DemoSetup(
        dt=backend.physics_dt,
        duration=float(cfg["sim"]["duration"]),
        every=every,
        register=register,
        step=step,
        snapshot=snapshot,
        verify=verify,
        close=close,
        runtime_backend=backend,
        should_stop=lambda: bool(state["episode_complete"]),
    )
    setup._camera_sensor = camera_sensor  # type: ignore[attr-defined]
    setup._wrist_camera_sensor = wrist_camera_sensor  # type: ignore[attr-defined]
    setup._camera_state = state  # type: ignore[attr-defined]
    setup._episode_recorder = episode_recorder  # type: ignore[attr-defined]
    return setup


__all__ = ["PHASES", "build"]
