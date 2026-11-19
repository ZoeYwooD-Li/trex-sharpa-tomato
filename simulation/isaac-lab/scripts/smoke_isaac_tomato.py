#!/usr/bin/env python3
"""Import the real dual-FR3/Sharpa model into Isaac/PhysX and step a tomato scene."""

from __future__ import annotations

import argparse
import atexit
import http.server
import json
import os
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
import webbrowser
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

SIM_ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(os.environ.get("HANDSON_ROOT", SIM_ROOT)).expanduser().resolve()
# Expanded from the tomato-pick task workspaces so both measured episode-54
# home wrists start inside the envelope. Values are world-frame link8 meters.
TREX_LINK8_WORKSPACE_M = {
    "left": ((0.15, -0.30, 0.18), (0.70, 0.55, 0.55)),
    "right": ((0.15, -0.55, 0.18), (0.70, 0.25, 0.55)),
}
PI05_SHARPA_HAND_ORDER = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
sys.path.insert(0, str(PROJECT / "handson"))
sys.path.insert(0, str(SIM_ROOT / "scripts"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mjcf", type=Path, default=SIM_ROOT / "outputs/dual_fr3_sharpa.xml")
parser.add_argument("--output", type=Path, default=SIM_ROOT / "outputs")
parser.add_argument("--tomato-assets", type=Path,
                    default=SIM_ROOT / "assets/fruit",
                    help="Imported tomato and vine mesh/texture directory")
parser.add_argument("--tomato-config", type=Path,
                    default=SIM_ROOT.parent / "tomato-pick-sim/configs/cherry_tomato_pick_sharpa.yaml",
                    help="Source task config for tomato mass, branch geometry and pedicel limits")
parser.add_argument("--table-top-z-m", type=float, default=None,
                    help="Override tabletop height for measured geometry calibration; default is 0.141 m")
parser.add_argument("--table-front-x-m", type=float, default=0.690,
                    help="Far edge of the tabletop in world X; near edge remains 0.120 m")
parser.add_argument("--plate-center-xy-m", type=float, nargs=2, metavar=("X", "Y"),
                    default=(0.4, -0.235), help="Plate center in the robot world frame")
parser.add_argument("--tomato-group-offset-xy-m", type=float, nargs=2, metavar=("DX", "DY"),
                    default=(0.0, -0.05),
                    help="Tomato cluster center offset from the plate center in world XY")
parser.add_argument("--flexible-branch", action="store_true",
                    help="Split the fruit truss into two lightly bending rigid segments")
soft_fruit_group = parser.add_mutually_exclusive_group()
soft_fruit_group.add_argument("--soft-fruit-prototype", choices=(
    "tomato_00", "tomato_01", "tomato_02", "tomato_03", "tomato_04", "tomato_05",
    "tomato_06", "tomato_07", "tomato_08", "tomato_09", "tomato_11"),
    help="Convert only one non-grasp fruit to an attached volume-deformable prototype")
soft_fruit_group.add_argument("--soft-fruit-all", action="store_true",
                              help="Convert all fruit outside the rigid front grasp zone to volume deformables")
soft_fruit_group.add_argument("--soft-grasp-target", action="store_true",
                              help="Give the grasped tomato a deformable skin attached to a small rigid pedicel core")
parser.add_argument("--soft-fruit-on-hand-contact", action="store_true",
                    help="Keep selected soft fruit rigid until a fingertip has sustained contact, then transfer its unchanged mass to the soft body")
parser.add_argument("--branch-lift-probe", action="store_true",
                    help="Apply a slow spring clamp/lift to the branch midpoint after one second of settling")
parser.add_argument("--branch-lift-location", choices=("middle", "front"), default="middle",
                    help="Branch point used by --branch-lift-probe")
parser.add_argument("--basket-first-center-xy-m", type=float, nargs=2, metavar=("X", "Y"),
                    default=(0.245, 0.155), help="First left receiving dish center in the robot world frame")
parser.add_argument("--home-config", type=Path, default=SIM_ROOT / "configs/sharpa_sim_home.json",
                    help="Named radian home positions from the physical Sharpa arm/hand references")
parser.add_argument("--head-camera-pose-json", type=Path, default=None,
                    help="Measured world-frame Isaac camera pose (wxyz), fitted from RGB-D calibration")
parser.add_argument("--steps", type=int, default=24)
parser.add_argument("--render-every", type=int, default=4)
parser.add_argument("--replay-h5", type=Path, default=None,
                    help="Replay named 58-DOF joint positions from a local reference HDF5 episode")
parser.add_argument("--replay-start-sec", type=float, default=0.0)
parser.add_argument("--trex-camera-preview-only", action="store_true",
                    help="Render the T-Rex wrist cameras at the replay pose without calling the policy")
parser.add_argument("--wrist-camera-calibration-json", type=Path, default=None,
                    help="Measured wrist camera intrinsics and link8-local optical poses")
parser.add_argument("--replay-geometry-probe", action="store_true",
                    help="Save sampled wrist and fingertip world positions during a joint replay")
parser.add_argument("--replay-force-limit-n", type=float, default=20.0,
                    help="Stop a real-joint replay when a fingertip exceeds this force; 0 disables the stop")
parser.add_argument("--replay-live-view", action="store_true",
                    help="Open a live local dashboard with replay RGB and ten fingertip tactile images")
parser.add_argument("--record-replay-video", action="store_true",
                    help="Save an overview MP4 during a joint replay without enabling task-eval randomization")
parser.add_argument("--replay-live-view-port", type=int, default=8765)
parser.add_argument("--contact-probe", action="store_true",
                    help="Place a small sphere at the right index link for contact-sensor diagnostics")
parser.add_argument("--sensor-self-test", action="store_true",
                    help="Drop a test block onto the tabletop and report its measured contact force")
parser.add_argument("--policy-ckpt", type=Path, default=None,
                    help="Load a Sharpa BC policy and run inference on each simulated observation")
parser.add_argument("--policy-mode", choices=("tactile-force", "state"), default=None,
                    help="Expected checkpoint input mode; otherwise read it from the checkpoint")
parser.add_argument("--policy-schema", type=Path,
                    default=Path("private-lerobot-data"),
                    help="LeRobot dataset root used to read the named 116-D state/58-D action schema")
parser.add_argument("--policy-apply", action="store_true",
                    help="Apply predicted joint targets; only use after checking the inference-only report")
parser.add_argument("--action-chunk-ckpt", type=Path, default=None,
                    help="Load a train_sharpa_action_chunks.py checkpoint for closed-loop execution")
parser.add_argument("--task-eval", action="store_true",
                    help="Run a full tomato pick-and-place episode and report detachment/basket success")
parser.add_argument("--eval-seed", type=int, default=17,
                    help="Paired initial-condition seed for task evaluation")
parser.add_argument("--episode-seconds", type=float, default=30.0,
                    help="Task rollout duration; 120 physics steps per second")
parser.add_argument("--trex-host", default=None,
                    help="Call a local T-Rex inference server with saved Isaac RGB/tactile observations")
parser.add_argument("--trex-port", type=int, default=5687)
parser.add_argument("--trex-python", type=Path,
                    default=Path(sys.executable),
                    help="Python in the local T-Rex environment, which provides pyzmq")
parser.add_argument("--trex-checkpoint", type=Path,
                    default=Path("private-checkpoint"),
                    help="Checkpoint folder containing its calibrated per-arm URDFs")
parser.add_argument("--trex-apply", action="store_true",
                    help="Convert the returned chunk through URDF IK and execute it in Isaac only")
parser.add_argument("--trex-closed-loop", action="store_true",
                    help="Run a closed-loop T-Rex pick-and-place episode, replanning every four policy frames")
parser.add_argument("--trex-ik-position-tolerance-m", type=float, default=0.008,
                    help="Best-effort T-Rex IK position residual limit (meters)")
parser.add_argument("--trex-ik-rotation-tolerance-deg", type=float, default=3.0,
                    help="Best-effort T-Rex IK rotation residual limit (degrees)")
parser.add_argument("--trex-ik-max-joint-step-rad", type=float, default=0.35,
                    help="Maximum per-policy-step arm joint target change accepted by T-Rex IK")
parser.add_argument("--trex-max-fingertip-force-n", type=float, default=20.0,
                    help="Stop applying T-Rex actions after any fingertip exceeds this contact force")
parser.add_argument("--trex-max-rejection-streak", type=int, default=5,
                    help="Stop T-Rex after this many consecutive IK or workcell-rejected replans")
parser.add_argument("--trex-live-view", action="store_true",
                    help="Open a live local dashboard with three RGB cameras and ten tactile images")
parser.add_argument("--trex-live-view-port", type=int, default=8765)
parser.add_argument("--trex-task", default=None,
                    help="T-Rex instruction; defaults to the task recorded with the real-home episode")
parser.add_argument("--pi05-closed-loop", action="store_true",
                    help="Run the 58D PI05 checkpoint in closed loop in this Isaac scene")
parser.add_argument("--pi05-url", default=None,
                    help="OpenPI websocket endpoint, for example ws://127.0.0.1:8000")
parser.add_argument("--pi05-prompt", default="Pick a tomato from the bunch and place it in the target box.",
                    help="Task prompt passed to the PI05 checkpoint")
parser.add_argument("--pi05-request-timeout", type=float, default=180.0)
parser.add_argument("--pi05-max-fingertip-force-n", type=float, default=20.0,
                    help="Stop applying PI05 actions if a fingertip exceeds this force")
parser.add_argument("--show-isaac-window", action="store_true",
                    help="Show the interactive Isaac Sim window during this rollout")
parser.add_argument("--hold-window-open", action="store_true",
                    help="Keep the Isaac Sim window open at the final frame until it is closed")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.hold_window_open and not args.show_isaac_window:
    parser.error("--hold-window-open requires --show-isaac-window")
if args.branch_lift_probe and not args.flexible_branch:
    parser.error("--branch-lift-probe requires --flexible-branch")
if args.branch_lift_probe and args.replay_h5 is not None:
    parser.error("--branch-lift-probe cannot be combined with --replay-h5")
if (args.soft_fruit_prototype or args.soft_fruit_all) and not args.flexible_branch:
    parser.error("soft fruit modes require --flexible-branch")
if args.soft_fruit_on_hand_contact and not (args.soft_fruit_prototype or args.soft_fruit_all):
    parser.error("--soft-fruit-on-hand-contact requires --soft-fruit-prototype or --soft-fruit-all")
if (args.trex_ik_position_tolerance_m <= 0
        or args.trex_ik_rotation_tolerance_deg <= 0
        or args.trex_ik_max_joint_step_rad <= 0
        or args.trex_max_fingertip_force_n <= 0
        or args.trex_max_rejection_streak <= 0):
    parser.error("T-Rex IK tolerances and safety limits must be positive")
if args.pi05_request_timeout <= 0 or args.pi05_max_fingertip_force_n <= 0:
    parser.error("PI05 timeout and fingertip force limit must be positive")
if args.pi05_closed_loop and not args.pi05_url:
    parser.error("--pi05-closed-loop requires --pi05-url")
if args.pi05_closed_loop and args.trex_closed_loop:
    parser.error("Choose either PI05 or T-Rex for a closed-loop rollout")
home_config_path = args.home_config.expanduser().resolve()
home_config = json.loads(home_config_path.read_text(encoding="utf-8"))
if home_config.get("unit") != "rad":
    parser.error(f"Home config must use radians: {home_config_path}")
real_camera_record_path = Path(home_config.get("sources", {}).get("arms", {}).get("record", ""))
real_camera_calibration = {}
wrist_camera_calibration = {}
if args.wrist_camera_calibration_json is not None:
    calibration_path = args.wrist_camera_calibration_json.expanduser().resolve()
    wrist_camera_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
if real_camera_record_path.is_file():
    real_camera_record = json.loads(real_camera_record_path.read_text(encoding="utf-8"))
    for episode_record in real_camera_record.get("episodes", []):
        if episode_record.get("camera_calibration"):
            real_camera_calibration = episode_record["camera_calibration"]
            break


def _real_camera_intrinsics(camera_name: str, expected_size: tuple[int, int]):
    configured = wrist_camera_calibration.get("cameras", {}).get(camera_name)
    if configured is not None:
        size = (int(configured["width"]), int(configured["height"]))
        if size != expected_size:
            raise ValueError(f"{camera_name} calibration size {size} != expected {expected_size}")
        intrinsic = np.asarray(configured["K"], dtype=float).reshape(3, 3)
        if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError(f"Invalid {camera_name} real camera intrinsics")
        return intrinsic
    entries = real_camera_calibration.get(f"/{camera_name}/color/camera_info", [])
    if not entries:
        return None
    entry = entries[0]
    size = (int(entry["width"]), int(entry["height"]))
    if size != expected_size:
        raise ValueError(f"{camera_name} real camera size {size} != expected {expected_size}")
    intrinsic = np.asarray(entry["K"], dtype=float).reshape(3, 3)
    if not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f"Invalid {camera_name} real camera intrinsics")
    return intrinsic


if args.trex_task is None:
    args.trex_task = home_config.get("sources", {}).get("arms", {}).get("task")
if (args.trex_host or args.trex_closed_loop) and not args.trex_task:
    parser.error("T-Rex task is missing; set --trex-task or sources.arms.task in home config")
home_positions_rad = {
    str(name): float(value)
    for name, value in home_config.get("joint_positions_rad", {}).items()
}
expected_home_joint_names = {
    f"{side}_fr3v2_joint{index}"
    for side in ("left", "right") for index in range(1, 8)
} | {
    f"{side}_{finger}_{joint}"
    for side in ("left", "right")
    for finger, joints in {
        "thumb": ("CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP"),
        "index": ("MCP_FE", "MCP_AA", "PIP", "DIP"),
        "middle": ("MCP_FE", "MCP_AA", "PIP", "DIP"),
        "ring": ("MCP_FE", "MCP_AA", "PIP", "DIP"),
        "pinky": ("CMC", "MCP_FE", "MCP_AA", "PIP", "DIP"),
    }.items() for joint in joints
}
if set(home_positions_rad) != expected_home_joint_names:
    parser.error(
        f"Home config must cover the 58 named dual-FR3/Sharpa joints; "
        f"missing={sorted(expected_home_joint_names - set(home_positions_rad))[:8]}, "
        f"extra={sorted(set(home_positions_rad) - expected_home_joint_names)[:8]}"
    )
if not np.all(np.isfinite(list(home_positions_rad.values()))):
    parser.error(f"Home config contains a non-finite joint value: {home_config_path}")
if args.trex_apply and not args.trex_host:
    parser.error("--trex-apply requires --trex-host to obtain a checkpoint action chunk")
if args.action_chunk_ckpt is not None and not args.policy_apply:
    parser.error("--action-chunk-ckpt requires --policy-apply")
if args.trex_closed_loop and not args.trex_host:
    parser.error("--trex-closed-loop requires --trex-host")
if args.trex_closed_loop:
    args.trex_apply = True
    args.task_eval = True
if args.pi05_closed_loop:
    args.task_eval = True
if args.trex_live_view and not args.trex_closed_loop:
    parser.error("--trex-live-view requires --trex-closed-loop")
if args.replay_live_view and args.replay_h5 is None:
    parser.error("--replay-live-view requires --replay-h5")
if args.trex_camera_preview_only and args.replay_h5 is None:
    parser.error("--trex-camera-preview-only requires --replay-h5")
if (args.task_eval and args.action_chunk_ckpt is None and not args.trex_closed_loop
        and not args.pi05_closed_loop
        and args.replay_h5 is None):
    parser.error("--task-eval requires --action-chunk-ckpt, --trex-closed-loop, or --replay-h5")
if args.task_eval and args.episode_seconds <= 0:
    parser.error("--episode-seconds must be positive")
if args.task_eval:
    args.steps = int(round(args.episode_seconds * 120))
if args.steps <= 0:
    parser.error("--steps must be positive")
if not np.isfinite(args.replay_force_limit_n) or args.replay_force_limit_n < 0:
    parser.error("--replay-force-limit-n must be finite and nonnegative")
if (args.trex_host or args.pi05_closed_loop) and args.steps < 64:
    # T-Rex consumes a 16-sample F6 window at the 30 Hz policy rate.
    args.steps = 64
args.headless = not args.show_isaac_window
args.device = "cuda:0"
app = AppLauncher(args).app

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("isaacsim.asset.importer.mjcf")

import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from handson.sim.isaaclab.compat import make_mjcf_file_cfg  # noqa: E402
from handson.sim.isaaclab.sharpa_tactile import (  # noqa: E402
    TactileGeometry, _contact_arrays, quaternion_wxyz_to_matrix, synthesize_tactile_reading,
)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _state_name_for_dof(dof_name: str, suffix: str) -> str:
    """Map imported MJCF joint names onto the physical LeRobot state schema."""
    if dof_name.startswith(("left_fr3v2_joint", "right_fr3v2_joint")):
        dof_name = dof_name.replace("fr3v2_joint", "fr3_joint")
    return f"{dof_name}.{suffix}"


def _action_name_for_dof(dof_name: str) -> str:
    """Arm commands use `.position`; Sharpa hand commands use `.command`."""
    return _state_name_for_dof(dof_name, "position" if "fr3v2_joint" in dof_name else "command")


def _pool_tactile(images: np.ndarray, grid: int = 8) -> np.ndarray:
    batch, sides, fingers, height, width = images.shape
    if height % grid or width % grid:
        raise ValueError(f"Tactile maps {height}x{width} cannot be pooled to {grid}x{grid}")
    return images.reshape(batch, sides, fingers, grid, height // grid, grid, width // grid).mean(
        axis=(4, 6)).reshape(batch, -1).astype(np.float32) / 255.0


def _look_at_wxyz(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Isaac camera world convention: local +X forward and local +Z up."""
    from scipy.spatial.transform import Rotation

    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    xyzw = Rotation.from_matrix(np.stack((forward, right, up), axis=1)).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _origin_transform(element):
    from scipy.spatial.transform import Rotation

    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    transform[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def _urdf_arm_kinematics(urdf_path: Path, base_link: str, tip_link: str, joint_names: list[str]):
    """Build FK and limits from the exact URDF shipped with this T-Rex checkpoint."""
    by_child = {joint.find("child").get("link"): joint
                for joint in ET.parse(urdf_path).getroot().findall("joint")}
    reverse_chain = []
    link = tip_link
    while link != base_link:
        if link not in by_child:
            raise ValueError(f"No URDF chain {base_link} -> {tip_link} in {urdf_path}")
        joint = by_child[link]
        reverse_chain.append(joint)
        link = joint.find("parent").get("link")
    chain = []
    limits = {}
    for joint in reversed(reverse_chain):
        name, kind = joint.get("name"), joint.get("type")
        if kind not in ("fixed", "revolute", "continuous", "prismatic"):
            raise ValueError(f"Unsupported URDF joint type {kind!r}: {name}")
        index = None
        axis = np.array([0.0, 0.0, 1.0])
        if kind != "fixed":
            if name not in joint_names:
                raise ValueError(f"URDF moving joint missing from checkpoint order: {name}")
            index = joint_names.index(name)
            axis_element = joint.find("axis")
            if axis_element is not None:
                axis = np.fromstring(axis_element.get("xyz", "0 0 1"), sep=" ")
            limit = joint.find("limit")
            if kind == "continuous":
                limits[name] = (-np.pi, np.pi)
            elif limit is None:
                raise ValueError(f"Missing limits for URDF joint {name}")
            else:
                limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        chain.append((kind, index, _origin_transform(joint.find("origin")), axis))
    if len(limits) != len(joint_names):
        raise ValueError(f"URDF chain does not contain every configured joint in {urdf_path}")

    def fk(q):
        transform = np.eye(4)
        for kind, index, origin, axis in chain:
            transform = transform @ origin
            if kind == "fixed":
                continue
            axis_n = axis / np.linalg.norm(axis)
            if kind == "prismatic":
                move = np.eye(4)
                move[:3, 3] = q[index] * axis_n
            else:
                from scipy.spatial.transform import Rotation
                move = np.eye(4)
                move[:3, :3] = Rotation.from_rotvec(axis_n * q[index]).as_matrix()
            transform = transform @ move
        return transform

    lower = np.array([limits[name][0] for name in joint_names], dtype=np.float64)
    upper = np.array([limits[name][1] for name in joint_names], dtype=np.float64)
    return fk, lower, upper


def _rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first = value[:3]
    second = value[3:6]
    first = first / np.linalg.norm(first)
    second = second - first * np.dot(first, second)
    second = second / np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def _solve_arm_chunk(actions: np.ndarray, side: str, chunk_offset: int, q0: np.ndarray,
                     checkpoint: Path, link0_pose_physx_xyzw: np.ndarray,
                     link8_pose_physx_xyzw: np.ndarray,
                     position_tolerance_m: float = 0.008,
                     rotation_tolerance_deg: float = 3.0,
                     max_joint_step_rad: float = 0.35,
                     reference_q: np.ndarray | None = None):
    """Convert 16 chunk-start link8 deltas into bounded FR3 joint targets."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]
    fk, lower, upper = _urdf_arm_kinematics(
        checkpoint / "robot" / f"{side}.urdf", f"{side}_fr3v2_link0", f"{side}_fr3v2_link8", names)
    q0 = np.asarray(q0, dtype=np.float64)
    q0 = np.clip(q0, lower + 1e-6, upper - 1e-6)
    fk0 = fk(q0)
    base_rotation = Rotation.from_quat(link0_pose_physx_xyzw[3:]).as_matrix()
    tip_rotation = Rotation.from_quat(link8_pose_physx_xyzw[3:]).as_matrix()
    measured_relative_pose = np.eye(4)
    measured_relative_pose[:3, :3] = base_rotation.T @ tip_rotation
    measured_relative_pose[:3, 3] = base_rotation.T @ (link8_pose_physx_xyzw[:3] - link0_pose_physx_xyzw[:3])
    if (np.linalg.norm(measured_relative_pose[:3, 3] - fk0[:3, 3]) > 0.002
            or Rotation.from_matrix(fk0[:3, :3].T @ measured_relative_pose[:3, :3]).magnitude()
            > np.deg2rad(1.0)):
        raise RuntimeError(f"{side} checkpoint URDF frame does not match Isaac link0/link8")

    # A fast tactile refinement updates actions within the same 16-frame
    # chunk. Its deltas still refer to the link8 pose at the slow tick.
    base = fk(q0 if reference_q is None else np.asarray(reference_q, dtype=np.float64))
    previous = q0.copy()
    trajectory, errors = [], []
    stopped_after_rejection = False
    for index, action in enumerate(actions):
        if stopped_after_rejection:
            trajectory.append(previous.astype(np.float32))
            errors.append({"position_error_m": None, "rotation_error_deg": None,
                           "solver_success": False, "accepted": False,
                           "rejection_reason": "held_after_prior_action_rejection"})
            continue
        delta_position = action[chunk_offset:chunk_offset + 3]
        delta_rotation = _rot6d_to_matrix(action[chunk_offset + 3:chunk_offset + 9])
        target_position = base[:3, 3] + base[:3, :3] @ delta_position
        target_rotation = base[:3, :3] @ delta_rotation

        def residual(q):
            pose = fk(q)
            position_error = pose[:3, 3] - target_position
            rotation_error = Rotation.from_matrix(target_rotation @ pose[:3, :3].T).as_rotvec()
            return np.concatenate((position_error, 0.35 * rotation_error))

        result = least_squares(residual, previous, bounds=(lower + 1e-6, upper - 1e-6),
                               max_nfev=120, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        pose = fk(result.x)
        tip_world = np.asarray(link0_pose_physx_xyzw[:3]) + base_rotation @ pose[:3, 3]
        workspace_lower, workspace_upper = (
            np.asarray(bound, dtype=np.float64) for bound in TREX_LINK8_WORKSPACE_M[side])
        position_error_m = float(np.linalg.norm(pose[:3, 3] - target_position))
        rotation_error_rad = float(Rotation.from_matrix(target_rotation @ pose[:3, :3].T).magnitude())
        rejection_reason = None
        if (position_error_m > position_tolerance_m
                or rotation_error_rad > np.deg2rad(rotation_tolerance_deg)):
            rejection_reason = (f"residual {position_error_m:.4g} m, "
                                f"{np.rad2deg(rotation_error_rad):.4g} deg")
        elif np.max(np.abs(result.x - previous)) > max_joint_step_rad:
            rejection_reason = "joint target exceeds per-policy-step limit"
        elif np.any(tip_world < workspace_lower) or np.any(tip_world > workspace_upper):
            rejection_reason = f"link8 target outside {side} workcell: {tip_world.round(3).tolist()} m"
        if rejection_reason is not None:
            trajectory.append(previous.astype(np.float32))
            errors.append({"position_error_m": position_error_m,
                           "rotation_error_deg": float(np.rad2deg(rotation_error_rad)),
                           "solver_success": bool(result.success), "accepted": False,
                           "rejection_reason": rejection_reason})
            stopped_after_rejection = True
            continue
        trajectory.append(result.x.astype(np.float32))
        errors.append({"position_error_m": position_error_m,
                       "rotation_error_deg": float(np.rad2deg(rotation_error_rad)),
                       "solver_success": bool(result.success), "accepted": True,
                       "rejection_reason": None})
        previous = result.x
    return np.asarray(trajectory), errors


def _trex_actions_to_joint_targets(actions: np.ndarray, robot, stage, checkpoint: Path,
                                   hand_order: list[str],
                                   position_tolerance_m: float = 0.008,
                                   rotation_tolerance_deg: float = 3.0,
                                   max_joint_step_rad: float = 0.35,
                                   chunk_start_arm_q: dict[str, np.ndarray] | None = None):
    """Map a T-Rex [16,62] chunk to bounded Isaac 58-DOF position targets."""
    current_positions = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
    link_transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
    arm_trajectories, ik_reports = {}, {}
    for side, action_offset in (("left", 0), ("right", 31)):
        names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]
        arm_q = np.asarray([current_positions[robot.dof_names.index(name)] for name in names])
        frame_poses = {
            name: link_transforms[robot._articulation_view.get_link_index(
                f"{side}_fr3v2_{name}")]
            for name in ("link0", "link8")
        }
        arm_trajectories[side], ik_reports[side] = _solve_arm_chunk(
            actions, side, action_offset, arm_q, checkpoint,
            frame_poses["link0"], frame_poses["link8"],
            position_tolerance_m, rotation_tolerance_deg, max_joint_step_rad,
            None if chunk_start_arm_q is None else chunk_start_arm_q[side])

    hand_limits = {}
    for side in ("left", "right"):
        for joint_name in hand_order:
            prim = stage.GetPrimAtPath(f"/World/TomatoRobot/joints/{side}_{joint_name}")
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Missing Sharpa joint in Isaac: {side}_{joint_name}")
            lower = float(prim.GetAttribute("physics:lowerLimit").Get())
            upper = float(prim.GetAttribute("physics:upperLimit").Get())
            hand_limits[f"{side}_{joint_name}"] = (np.deg2rad(lower), np.deg2rad(upper))

    target_chunk = np.repeat(current_positions[None, :], len(actions), axis=0)
    hand_clips = []
    hand_action_offsets = {"left": 9, "right": 40}
    for action_index, action in enumerate(actions):
        for side in ("left", "right"):
            arm_names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]
            for joint_index, name in enumerate(arm_names):
                target_chunk[action_index, robot.dof_names.index(name)] = arm_trajectories[side][action_index, joint_index]
            for hand_index, joint_name in enumerate(hand_order):
                name = f"{side}_{joint_name}"
                lower, upper = hand_limits[name]
                target = float(action[hand_action_offsets[side] + hand_index])
                if not ik_reports[side][action_index]["accepted"]:
                    continue
                bounded_target = float(np.clip(target, lower, upper))
                if bounded_target != target:
                    hand_clips.append({"frame": action_index, "joint": name,
                                       "requested_rad": target, "applied_rad": bounded_target})
                target_chunk[action_index, robot.dof_names.index(name)] = bounded_target
    return target_chunk, ik_reports, hand_limits, hand_clips


def _png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image
    import io

    stream = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def _write_trex_live_preview(
    path: Path,
    frames: dict[str, np.ndarray],
    tactile: np.ndarray,
    f6_history: np.ndarray,
    simulated_time_s: float,
    policy_frame: int,
    replay: bool = False,
) -> None:
    """Write a labeled RGB/tactile dashboard image for the local viewer."""
    from PIL import Image, ImageDraw

    width, height = 1280, 1150
    canvas = Image.new("RGB", (width, height), (20, 23, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16),
              f"Isaac {'joint replay' if replay else 'T-Rex'} live view  |  sim {simulated_time_s:.1f}s  |  frame {policy_frame}",
              fill=(245, 245, 245))

    def add_camera(key: str, label: str, box: tuple[int, int, int, int]) -> None:
        x, y, w, h = box
        image = Image.fromarray(np.asarray(frames[key], dtype=np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        draw.text((x, y - 20), label, fill=(220, 225, 235))
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=(90, 100, 115), width=1)

    add_camera("head", "cam0 Main RGB", (20, 58, 720, 480))
    if replay and not {"wrist_left", "wrist_right"} <= frames.keys():
        draw.text((760, 85), "Real joint replay", fill=(245, 245, 245))
        draw.text((760, 120), "Ten simulated fingertip tactile images", fill=(220, 225, 235))
        draw.text((760, 150), "Updated from Isaac contact sensors", fill=(220, 225, 235))
        draw.text((760, 180), "Gray = no contact; brighter = more deformation", fill=(220, 225, 235))
    else:
        add_camera("wrist_right", "cam1 Right wrist RGB", (760, 58, 480, 300))
        add_camera("wrist_left", "cam2 Left wrist RGB", (760, 390, 480, 300))
    draw.text((20, 720), "Visuotactile images (finger order: left thumb → pinky, right thumb → pinky)",
              fill=(245, 245, 245))
    finger_names = ["L thumb", "L index", "L middle", "L ring", "L pinky",
                    "R thumb", "R index", "R middle", "R ring", "R pinky"]
    touch = np.asarray(tactile, dtype=np.uint8).reshape(10, 240, 240)
    final_f6 = np.asarray(f6_history, dtype=np.float32)[-1].reshape(10, 6)
    for index, (name, image_array) in enumerate(zip(finger_names, touch)):
        row, column = divmod(index, 5)
        x, y = 20 + column * 248, 754 + row * 194
        image = Image.fromarray(image_array).convert("RGB").resize(
            (180, 180), Image.Resampling.NEAREST)
        canvas.paste(image, (x, y))
        force_n = float(np.linalg.norm(final_f6[index, :3]))
        draw.text((x, y - 18), f"{name}  |  {force_n:.2f} N", fill=(220, 225, 235))
        draw.rectangle((x, y, x + 179, y + 179), outline=(90, 100, 115), width=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    canvas.save(temporary, format="PNG")
    os.replace(temporary, path)


def _start_trex_live_view(path: Path, port: int, replay: bool = False) -> tuple[http.server.ThreadingHTTPServer, str]:
    """Serve the local live-preview image to a browser without external access."""
    html = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Isaac Tactile Live View</title><style>body{margin:0;background:#15171b;color:#eee;font:16px sans-serif}header{padding:10px 16px;position:sticky;top:0;background:#202329}img{display:block;width:min(100%,1280px);height:auto;margin:0 auto}</style></head>
<body><header>Isaac 实时观测 · RGB 相机 + 十指视触觉</header><img id=feed><script>
const feed=document.getElementById('feed');function refresh(){feed.src='/latest.png?t='+Date.now()}refresh();setInterval(refresh,250);
</script></body></html>""".encode("utf-8")
    preview = path.resolve()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/latest.png":
                try:
                    payload = preview.read_bytes()
                except FileNotFoundError:
                    self.send_error(404, "Simulation has not produced its first preview yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path.split("?", 1)[0] == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            else:
                self.send_error(404)

        def log_message(self, _format, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="trex-live-view-http", daemon=True).start()
    atexit.register(server.server_close)
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def _static_cube(stage, path: str, position: tuple[float, float, float], size: tuple[float, float, float], color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _visual_mesh(stage, path: str, source: Path, color, translation=None,
                 texture: Path | None = None, face_mask=None):
    with np.load(source) as data:
        points = np.asarray(data["vertices"], dtype=np.float32)
        faces = np.asarray(data["faces"], dtype=np.int32)
        uvs = np.asarray(data["uvs"], dtype=np.float32) if "uvs" in data else None
        normals = np.asarray(data["normals"], dtype=np.float32) if "normals" in data else None
    if face_mask is not None:
        faces = faces[np.asarray(face_mask, dtype=bool)]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateDoubleSidedAttr(True)
    if normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if texture is not None and uvs is not None:
        primvars = UsdGeom.PrimvarsAPI(mesh)
        uv = primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                    UsdGeom.Tokens.vertex)
        uv.Set(Vt.Vec2fArray.FromNumpy(uvs))
        material_path = "/" + path.strip("/").replace("/", "_") + "_Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, material_path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        uv_reader = UsdShade.Shader.Define(stage, material_path + "/UV")
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(stage, material_path + "/BaseColor")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture.resolve())))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    if translation is not None:
        mesh.AddTranslateOp().Set(Gf.Vec3f(*translation))
    return mesh


def _enable_convex_mesh_collision(mesh):
    """Use a source visual fruit mesh as its PhysX convex-hull collider."""
    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    collision.CreateApproximationAttr().Set("convexHull")


def _author_soft_fruit(stage, *, name: str, source: Path, texture: Path,
                       world_position: np.ndarray, fruit_rotation_xyz,
                       cluster_rotation, branch_local_offset: np.ndarray,
                       branch_root_path: str, physics_material_path: str,
                       stem_attachment_local: np.ndarray,
                       initially_enabled: bool = True,
                       mass_kg: float = 0.020):
    """Author a textured, mass-explicit tetrahedral fruit with a pedicel attachment."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

    root_path = f"/World/SoftTomato_{name}"
    root = UsdGeom.Xform.Define(stage, root_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*np.asarray(world_position, dtype=float).tolist()))
    root.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, fruit_rotation_xyz)))
    root_prim = root.GetPrim()
    root_prim.ApplyAPI("OmniPhysicsDeformableBodyAPI")
    root_prim.GetAttribute("omniphysics:mass").Set(float(mass_kg))
    # Touch-gated soft bodies are cooked in place, then disabled before the
    # first rollout step while the original rigid proxies remain active.
    root_prim.GetAttribute("omniphysics:deformableBodyEnabled").Set(True)
    root_prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    for attr_name, value in {
        "physxDeformableBody:selfCollision": False,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": 20,
        "physxDeformableBody:maxLinearVelocity": 2.0,
        "physxDeformableBody:maxDepenetrationVelocity": 0.35,
        "physxDeformableBody:linearDamping": 0.12,
    }.items():
        attr = root_prim.GetAttribute(attr_name)
        if not attr.IsValid():
            attr = root_prim.CreateAttribute(
                attr_name,
                Sdf.ValueTypeNames.Bool if isinstance(value, bool) else
                Sdf.ValueTypeNames.Int if isinstance(value, int) else
                Sdf.ValueTypeNames.Float,
            )
        attr.Set(value)

    visual_path = f"{root_path}/Visual"
    visual = _visual_mesh(stage, visual_path, source, (0.75, 0.035, 0.025), texture=texture)
    if not initially_enabled:
        UsdGeom.Imageable(visual.GetPrim()).MakeInvisible()
    with np.load(source) as source_data:
        visual_points = np.asarray(source_data["vertices"], dtype=np.float32)
    half_extent = np.maximum(np.max(np.abs(visual_points), axis=0), 0.004)

    # The source FBX-derived fruit meshes include disconnected and nearly
    # coplanar triangles. Feed PhysX a clean watertight ellipsoid for cooking,
    # while retaining the textured source as the deforming render surface.
    import trimesh

    cook_path = f"{root_path}/CookingSource"
    cook_mesh = UsdGeom.Mesh.Define(stage, cook_path)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    cook_points = np.asarray(sphere.vertices, dtype=np.float32) * half_extent[None, :]
    cook_mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(cook_points))
    cook_mesh.CreateFaceVertexCountsAttr().Set([3] * len(sphere.faces))
    cook_mesh.CreateFaceVertexIndicesAttr().Set(
        [int(index) for face in sphere.faces for index in face])
    cook_mesh.CreateOrientationAttr("rightHanded")
    UsdGeom.Imageable(cook_mesh).MakeInvisible()

    simulation_path = f"{root_path}/SimulationMesh"
    from omni.physx.scripts import deformableUtils
    ok = deformableUtils.create_auto_volume_deformable_hierarchy(
        stage=stage,
        root_prim_path=root_path,
        simulation_tetmesh_path=simulation_path,
        collision_tetmesh_path=simulation_path,
        cooking_src_mesh_path=cook_path,
        simulation_hex_mesh_enabled=False,
        cooking_src_simplification_enabled=False,
    )
    if not ok:
        raise RuntimeError(f"Isaac Sim could not author a deformable hierarchy for {name}")
    sim_prim = stage.GetPrimAtPath(simulation_path)
    UsdPhysics.CollisionAPI.Apply(sim_prim).CreateCollisionEnabledAttr().Set(True)
    collision_api = sim_prim.ApplyAPI("PhysxCollisionAPI")
    if collision_api:
        sim_prim.GetAttribute("physxCollision:contactOffset").Set(0.002)
        sim_prim.GetAttribute("physxCollision:restOffset").Set(0.0005)
    UsdGeom.Imageable(sim_prim).MakeInvisible()

    from omni.physx.scripts import physicsUtils
    physicsUtils.add_physics_material_to_prim(stage, root_prim, physics_material_path)

    # Isaac's native auto attachment computes a small local tether set between
    # the soft fruit and its owning rigid branch segment from their geometry.
    UsdGeom.Xform.Define(stage, "/World/TomatoFruitAttachments")
    attachment_path = f"/World/TomatoFruitAttachments/SoftTomato_{name}"
    physicsUtils.add_physics_material_to_prim(stage, sim_prim, physics_material_path)
    return {
        "name": name,
        "root_path": root_path,
        "simulation_mesh_path": simulation_path,
        "collision_mesh_path": simulation_path,
        "attachment_path": attachment_path,
        "attachment_branch_path": branch_root_path,
        "stem_attachment_local_m": np.asarray(stem_attachment_local, dtype=float).tolist(),
        "fruit_rotation_xyz_deg": list(map(float, fruit_rotation_xyz)),
        "mass_kg": float(mass_kg),
        "active_at_first_physics_step": bool(initially_enabled),
        "collision_enabled_at_first_physics_step": True,
        "simulation_mesh_generation": "Isaac Sim native deformable cooker from textured source mesh",
        "attachment_method": "runtime-selected native tetra-mesh nodes tethered to owning branch segment",
        "youngs_modulus_pa": 800000.0,
        "poissons_ratio": 0.42,
    }, str(visual.GetPath()), str(sim_prim.GetPath())


def _fruit(stage, path: str, position: tuple[float, float, float], visual_npz: Path,
           fruit_texture: Path, rotation_xyz=(180.0, 0.0, 0.0),
           fruit_radius_m: float = 0.0172, fruit_mass_kg: float = 0.02):
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*position))
    root.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(fruit_mass_kg)
    sphere = UsdGeom.Sphere.Define(stage, f"{path}/Collision")
    sphere.CreateRadiusAttr(fruit_radius_m)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    sphere.MakeInvisible()
    _visual_mesh(stage, f"{path}/Visual", visual_npz, (0.75, 0.035, 0.025),
                 texture=fruit_texture)


def main() -> int:
    args.output = args.output.expanduser().resolve()
    args.mjcf = args.mjcf.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(
        device="cuda:0", dt=1.0 / 120.0, render_interval=4,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(solver_type=1, max_position_iteration_count=32,
                                 max_velocity_iteration_count=4,
                                 gpu_max_rigid_contact_count=2**18,
                                 gpu_max_rigid_patch_count=2**14,
                                 gpu_found_lost_pairs_capacity=2**15,
                                 gpu_found_lost_aggregate_pairs_capacity=2**16,
                                 gpu_total_aggregate_pairs_capacity=2**15,
                                 gpu_collision_stack_size=2**20,
                                 gpu_heap_capacity=2**22,
                                 gpu_temp_buffer_capacity=2**20),
    ))
    light_cfg = sim_utils.DistantLightCfg(intensity=2500.0)
    light_cfg.func("/World/Light", light_cfg)
    fill_light_cfg = sim_utils.DomeLightCfg(intensity=500.0, color=(0.6, 0.6, 0.6))
    fill_light_cfg.func("/World/RoomFillLight", fill_light_cfg)
    import_cfg = make_mjcf_file_cfg(
        sim_utils, asset_path=str(args.mjcf), usd_dir=str(args.output / "robot_usd_v2"),
        fix_base=True, import_sites=True, self_collision=False, make_instanceable=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    )
    print("[Isaac tomato] importing dual FR3 + Sharpa", flush=True)
    import_cfg.func("/World/TomatoRobot", import_cfg)
    stage = omni.usd.get_context().get_stage()
    # Physical tabletop, fruit, vine, support fruit and receiving boxes.
    # Collision geometry follows the tomato-pick source backend: cylinder for
    # the main vine and convex hulls for individual fruit meshes.
    table_top_z = 0.141 if args.table_top_z_m is None else float(args.table_top_z_m)
    if not 0.0 < table_top_z < 1.0:
        raise ValueError("Tabletop height must be between 0 and 1 m")
    table_lift_m = table_top_z - 0.041
    # Episode 54 cam0 puts the plate near image x=440/640 and the receiving
    # tray near x=225/640. Keep the robot's measured real-home pose fixed and
    # shift the tabletop objects toward the right arm to match that layout.
    plate_center_xy = tuple(args.plate_center_xy_m)
    basket_first_center_xy = tuple(args.basket_first_center_xy_m)
    if args.table_front_x_m <= 0.12:
        raise ValueError("Table far edge must be beyond its 0.12 m near edge")
    # The carrier collision mesh reaches x=0.11 m.  Keep the tabletop rear
    # edge beyond it; otherwise PhysX resolves their initial overlap by
    # displacing the whole fixed-base articulation before the first action.
    _static_cube(stage, "/World/Table", ((0.12 + args.table_front_x_m) / 2, 0.05, table_top_z - 0.0125),
                 (args.table_front_x_m - 0.12, 1.0, 0.025), (0.01, 0.018, 0.05))
    # Non-colliding room surfaces give the external replay camera the same
    # visible floor and neutral background as the real cam0 recording.
    for path, position, size, color in (
        ("/World/ReplayFloor", (0.32, 0.0, -0.08), (4.0, 4.0, 0.02), (0.62, 0.64, 0.66)),
        ("/World/ReplayWall", (1.8, 0.0, 2.45), (0.02, 4.0, 5.1), (0.78, 0.78, 0.76)),
        ("/World/ReplayLeftWall", (0.32, 2.0, 2.45), (4.0, 0.02, 5.1), (0.78, 0.78, 0.76)),
        ("/World/ReplayRightWall", (0.32, -2.0, 2.45), (4.0, 0.02, 5.1), (0.78, 0.78, 0.76)),
        ("/World/ReplayRearWall", (-1.65, 0.0, 2.45), (0.02, 4.0, 5.1), (0.78, 0.78, 0.76)),
    ):
        backdrop = UsdGeom.Cube.Define(stage, path)
        backdrop.CreateSizeAttr(1.0)
        backdrop.AddTranslateOp().Set(Gf.Vec3d(*position))
        backdrop.AddScaleOp().Set(Gf.Vec3f(*size))
        backdrop.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    plate = UsdGeom.Cylinder.Define(stage, "/World/TomatoPlate")
    plate.CreateRadiusAttr(0.14)
    plate.CreateHeightAttr(0.004)
    plate.AddTranslateOp().Set(Gf.Vec3d(*plate_center_xy, table_top_z + 0.002))
    plate.CreateDisplayColorAttr([Gf.Vec3f(0.92, 0.92, 0.88)])
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    asset_root = args.tomato_assets.expanduser().resolve()
    import yaml
    source_config = yaml.safe_load(args.tomato_config.expanduser().read_text(encoding="utf-8"))
    source_scene_config = source_config["scene"]
    source_attachment_config = source_config["attachment"]
    fruit_texture = asset_root / "tomato_fruit_basecolor.png"
    stem_texture = asset_root / "tomato_basecolor.png"
    layout_path = asset_root / "isaac_layout.json"
    tomato_layout = json.loads(layout_path.read_text(encoding="utf-8")) if layout_path.is_file() else {}
    # The plate top is z=0.045 m plus the shared tabletop lift; the tomato
    # collision sphere remains seated on it.
    # Restore the original tomato truss and fruit locations from the scene
    # preview: one physical target fruit at the plate center, with the other
    # eleven fruit meshes attached along the truss.
    fruit_offsets = {
        "tomato_00": (-0.0170466043, -0.1040752605, 0.0015868883),
        "tomato_01": (0.0179681778, -0.0814892501, -0.0010500379),
        "tomato_02": (-0.0170465950, -0.1737860739, 0.0021703171),
        "tomato_03": (0.0179681703, -0.1518112868, -0.0004016151),
        "tomato_04": (-0.0174255427, -0.1406981349, -0.0017162894),
        "tomato_05": (0.0191548597, -0.1152091101, 0.0016541561),
        "tomato_06": (-0.0174255483, -0.0703097060, -0.0019317174),
        "tomato_07": (0.0191548727, -0.1849199533, 0.0006805198),
        "tomato_08": (-0.0170463659, -0.0348072425, 0.0009362919),
        "tomato_09": (0.0205952935, -0.0103496760, 0.0006700469),
        "tomato_11": (0.0187244136, -0.0436092839, 0.0011164187),
    }
    tomato_rotation = (180.0, 0.0, 0.0)
    vine_rotation = (180.0, 0.0, 0.0)
    target_attachment = (0.0011886077321249977, 0.000685328077000003, -0.013799792069125003)
    vine_attachment = target_attachment
    if tomato_layout:
        centers = tomato_layout["fruit_centers_relative_to_target_m"]
        fruit_offsets = {
            name: tuple(float(value) for value in center)
            for name, center in centers.items()
            if name != "tomato_10"
        }
        tomato_rotation = tuple(tomato_layout["tomato_root_rotation_xyz_deg"])
        vine_rotation = tuple(tomato_layout["vine_root_rotation_xyz_deg"])
        target_attachment = tuple(tomato_layout["attachment_target_local_m"])
        vine_attachment = tuple(tomato_layout["attachment_cluster_local_m"])
    target_offset = (0.0, 0.0)
    target_key = "tomato_10"
    # The real truss is slightly right of the plate center in cam0.
    tomato_group_offset_xy = tuple(float(value) for value in args.tomato_group_offset_xy_m)
    group_center_xy = (plate_center_xy[0] + tomato_group_offset_xy[0],
                       plate_center_xy[1] + tomato_group_offset_xy[1])
    if args.task_eval:
        rng = np.random.default_rng(args.eval_seed)
        group_center_xy = (group_center_xy[0] + float(rng.uniform(-0.008, 0.008)),
                           group_center_xy[1] + float(rng.uniform(-0.008, 0.008)))
    tomato_start = (group_center_xy[0] + target_offset[0],
                    group_center_xy[1] + target_offset[1], 0.06220 + table_lift_m)
    _fruit(stage, "/World/TargetTomato", tomato_start,
           asset_root / f"{target_key}_visual.npz", fruit_texture,
           rotation_xyz=tomato_rotation,
           fruit_radius_m=(0.006 if args.soft_grasp_target else
                           float(source_scene_config["tomato_radius_m"])),
           fruit_mass_kg=(0.004 if args.soft_grasp_target else
                          float(source_scene_config["tomato_mass_kg"])))
    if args.soft_grasp_target:
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/TargetTomato/Visual")).MakeInvisible()
    cluster = UsdGeom.Xform.Define(stage, "/World/TomatoCluster")
    cluster.AddTranslateOp().Set(Gf.Vec3d(*group_center_xy, 0.06220 + table_lift_m))
    cluster.AddRotateXYZOp().Set(Gf.Vec3f(*vine_rotation))
    distal_cluster = None
    if args.flexible_branch:
        distal_cluster = UsdGeom.Xform.Define(stage, "/World/TomatoClusterDistal")
        distal_cluster.AddTranslateOp().Set(Gf.Vec3d(*group_center_xy, 0.06220 + table_lift_m))
        distal_cluster.AddRotateXYZOp().Set(Gf.Vec3f(*vine_rotation))
        UsdPhysics.RigidBodyAPI.Apply(distal_cluster.GetPrim())
        # Adjacent fruit hulls overlap at rest. They belonged to one actor in
        # the rigid model and must not push the two new segments apart.
        UsdPhysics.FilteredPairsAPI.Apply(cluster.GetPrim()).CreateFilteredPairsRel().AddTarget(
            distal_cluster.GetPath())
    # Match the saved USDA source scene: the full textured vine mesh has its
    # original local transform and is the rigid support for the target fruit.
    # This avoids synthetic rods and the unrelated green support sphere.
    UsdPhysics.RigidBodyAPI.Apply(cluster.GetPrim())
    support_name = str(tomato_layout.get("support_name", "tomato_07"))
    branch_mass_kg = float(source_scene_config["branch_mass_kg"])
    fruit_mass_kg = float(source_scene_config["tomato_mass_kg"])
    branch_split_y_m = float(source_scene_config["branch_length_m"]) / 2.0
    front_rigid_fruit_names = set(sorted(
        (name for name in fruit_offsets if name != support_name),
        key=lambda name: abs(float(fruit_offsets[name][1])),
    )[:3])
    soft_candidate_names = set(fruit_offsets) - front_rigid_fruit_names
    if args.soft_fruit_on_hand_contact:
        # The support fruit has its own spring-connected rigid actor; leave it
        # rigid in touch-gated mode and gate the attached cluster fruit only.
        soft_candidate_names.discard(support_name)
    if args.soft_fruit_prototype:
        if args.soft_fruit_prototype not in soft_candidate_names:
            raise ValueError(f"Soft-fruit prototype {args.soft_fruit_prototype} is in the rigid grasp zone")
        soft_fruit_names = {args.soft_fruit_prototype}
    elif args.soft_fruit_all:
        soft_fruit_names = set(soft_candidate_names)
    else:
        soft_fruit_names = set()
    distal_fruit_names = {name for name, offset in fruit_offsets.items()
                          if args.flexible_branch and name != support_name and
                          abs(float(offset[1])) > branch_split_y_m}
    proximal_rigid_fruit_names = {
        name for name in fruit_offsets
        if name != support_name and name not in soft_fruit_names and
        name not in distal_fruit_names
    }
    distal_rigid_fruit_names = distal_fruit_names - soft_fruit_names
    # Touch-gated fruits use compound rigid proxy colliders on the branch
    # before activation. Their mass must remain on that branch actor until the
    # proxy is replaced by a live deformable body.
    gated_branch_fruit_names = (soft_fruit_names - {support_name}
                                if args.soft_fruit_on_hand_contact else set())
    proximal_fruit_count = len(proximal_rigid_fruit_names) + len(
        gated_branch_fruit_names - distal_fruit_names)
    cluster_mass_kg = (branch_mass_kg * (0.5 if args.flexible_branch else 1.0)
                       + proximal_fruit_count * fruit_mass_kg)
    UsdPhysics.MassAPI.Apply(cluster.GetPrim()).CreateMassAttr(cluster_mass_kg)
    distal_cluster_mass_kg = 0.0
    if distal_cluster is not None:
        distal_cluster_mass_kg = branch_mass_kg * 0.5 + (
            len(distal_rigid_fruit_names) + len(gated_branch_fruit_names & distal_fruit_names)
        ) * fruit_mass_kg
        UsdPhysics.MassAPI.Apply(distal_cluster.GetPrim()).CreateMassAttr(distal_cluster_mass_kg)
    stem_path = asset_root / "stem_visual.npz"
    if distal_cluster is None:
        _visual_mesh(stage, "/World/TomatoCluster/Stem", stem_path,
                     (0.15, 0.42, 0.1), texture=stem_texture)
    else:
        with np.load(stem_path) as stem_data:
            stem_vertices = np.asarray(stem_data["vertices"], dtype=np.float32)
            stem_faces = np.asarray(stem_data["faces"], dtype=np.int32)
        distal_faces = stem_vertices[stem_faces, 1].mean(axis=1) > branch_split_y_m
        _visual_mesh(stage, "/World/TomatoCluster/Stem", stem_path,
                     (0.15, 0.42, 0.1), texture=stem_texture, face_mask=~distal_faces)
        _visual_mesh(stage, "/World/TomatoClusterDistal/Stem", stem_path,
                     (0.15, 0.42, 0.1), texture=stem_texture, face_mask=distal_faces)
    # The original task defines a 20.5 cm, 4 mm-radius branch collision.  The
    # refreshed FBX vine grows along local +Y; use its real cross-axis center
    # so the collider follows the imported asset rather than the older frame.
    with np.load(stem_path) as stem_data:
        stem_points = np.asarray(stem_data["vertices"], dtype=np.float64)
    stem_min, stem_max = stem_points.min(axis=0), stem_points.max(axis=0)
    branch_axis_sign = 1.0 if abs(stem_max[1]) >= abs(stem_min[1]) else -1.0
    branch_length_m = float(source_scene_config["branch_length_m"])
    branch_radius_m = float(source_scene_config["branch_radius_m"])
    branch_collision_paths = []
    for segment_index, segment_path in enumerate(
            ("/World/TomatoCluster", "/World/TomatoClusterDistal")
            if distal_cluster is not None else ("/World/TomatoCluster",)):
        branch_path = f"{segment_path}/BranchCollision"
        branch = UsdGeom.Cylinder.Define(stage, branch_path)
        branch.CreateAxisAttr(UsdGeom.Tokens.y)
        branch.CreateRadiusAttr(branch_radius_m)
        branch.CreateHeightAttr(branch_length_m / (2.0 if distal_cluster is not None else 1.0) -
                                (0.001 if distal_cluster is not None else 0.0))
        center_y = (branch_length_m * (segment_index + 0.5) / 2.0
                    if distal_cluster is not None else branch_length_m * 0.5)
        branch.AddTranslateOp().Set(Gf.Vec3d(
            float((stem_min[0] + stem_max[0]) * 0.5),
            float(branch_axis_sign * center_y),
            float((stem_min[2] + stem_max[2]) * 0.5)))
        UsdPhysics.CollisionAPI.Apply(branch.GetPrim())
        branch.MakeInvisible()
        branch_collision_paths.append(branch_path)
    branch_joint_path = None
    if distal_cluster is not None:
        branch_joint_path = "/World/TomatoBranchFlexJoint"
        flex_joint = UsdPhysics.SphericalJoint.Define(stage, branch_joint_path)
        flex_joint.CreateAxisAttr().Set(UsdPhysics.Tokens.y)
        # Keep the visible bend just under the requested 5 degree ceiling.
        flex_joint.CreateConeAngle0LimitAttr().Set(2.8)
        flex_joint.CreateConeAngle1LimitAttr().Set(2.8)
        flex_joint.CreateBody0Rel().SetTargets([cluster.GetPath()])
        flex_joint.CreateBody1Rel().SetTargets([distal_cluster.GetPath()])
        split_anchor = Gf.Vec3f(
            float((stem_min[0] + stem_max[0]) * 0.5),
            float(branch_axis_sign * branch_split_y_m),
            float((stem_min[2] + stem_max[2]) * 0.5),
        )
        flex_joint.CreateLocalPos0Attr().Set(split_anchor)
        flex_joint.CreateLocalPos1Attr().Set(split_anchor)
    fruit_collider_paths = []
    support_offset = fruit_offsets.get(support_name)
    support_on_distal = bool(args.flexible_branch and support_offset is not None and
                             abs(float(support_offset[1])) > branch_split_y_m)
    if support_offset is None:
        raise RuntimeError(f"Source tomato layout has no support fruit {support_name!r}")
    from scipy.spatial.transform import Rotation
    source_rotation = Rotation.from_euler("xyz", vine_rotation, degrees=True)
    support_world = np.asarray(tomato_start, dtype=np.float64)
    support_world[:2] = np.asarray(group_center_xy, dtype=np.float64)
    support_world += source_rotation.apply(np.asarray(support_offset, dtype=np.float64))
    soft_fruit_metadata = []
    soft_simulation_paths = []
    touch_proxy_metadata = {}
    soft_fruit_youngs_modulus_pa = 800_000.0
    if soft_fruit_names or args.soft_grasp_target:
        from omni.physx.scripts import deformableUtils

        deformable_material_path = "/World/TomatoDeformableMaterial"
        if not deformableUtils.add_deformable_material(
                stage, deformable_material_path, density=700.0,
                static_friction=4.0, dynamic_friction=4.0,
                youngs_modulus=soft_fruit_youngs_modulus_pa, poissons_ratio=0.42):
            raise RuntimeError("Could not create the tomato deformable material")
        deformable_material_prim = stage.GetPrimAtPath(deformable_material_path)
        deformable_material_prim.ApplyAPI("PhysxDeformableMaterialAPI")
        deformable_material_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.25)

    if args.soft_grasp_target:
        target_meta, _target_visual, target_simulation_path = _author_soft_fruit(
            stage, name=target_key, source=asset_root / f"{target_key}_visual.npz",
            texture=fruit_texture, world_position=np.asarray(tomato_start, dtype=float),
            fruit_rotation_xyz=tomato_rotation, cluster_rotation=vine_rotation,
            branch_local_offset=np.zeros(3, dtype=float),
            branch_root_path="/World/TargetTomato",
            physics_material_path=deformable_material_path,
            stem_attachment_local=np.zeros(3, dtype=float), mass_kg=0.016,
        )
        target_meta["world_start_m"] = list(map(float, tomato_start))
        target_meta["attachment_method"] = "central deformable nodes attached to a small rigid pedicel core"
        target_meta["youngs_modulus_pa"] = soft_fruit_youngs_modulus_pa
        soft_fruit_metadata.append(target_meta)
        soft_simulation_paths.append(target_simulation_path)
        # The soft skin surrounds the pedicel core and slightly overlaps the
        # original truss hulls. Those parts belong to the same fruit assembly.
        soft_target_prim = stage.GetPrimAtPath(target_meta["root_path"])
        internal_pairs = UsdPhysics.FilteredPairsAPI.Apply(soft_target_prim)
        for internal_path in ("/World/TargetTomato", "/World/TomatoCluster",
                              "/World/TomatoClusterDistal", "/World/SupportTomato"):
            if stage.GetPrimAtPath(internal_path).IsValid() or internal_path == "/World/SupportTomato":
                internal_pairs.CreateFilteredPairsRel().AddTarget(Sdf.Path(internal_path))

    def author_soft_cluster_fruit(fruit_name: str, offset: np.ndarray, world_position: np.ndarray,
                                  parent_path: str):
        offset = np.asarray(offset, dtype=np.float64)
        # The branch xform and vine use the recorded cluster-local frame; the
        # fruit root gets its own transform so deformation remains independent.
        meta, visual_path, simulation_path = _author_soft_fruit(
            stage, name=fruit_name, source=asset_root / f"{fruit_name}_visual.npz",
            texture=fruit_texture, world_position=world_position,
            fruit_rotation_xyz=tomato_rotation, cluster_rotation=vine_rotation,
            branch_local_offset=offset, branch_root_path=parent_path,
            physics_material_path=deformable_material_path,
            stem_attachment_local=np.asarray([0.0, 0.0, -float(source_scene_config["tomato_radius_m"]) * 0.82]),
            initially_enabled=not args.soft_fruit_on_hand_contact,
        )
        meta["branch_offset_local_m"] = offset.tolist()
        meta["world_start_m"] = world_position.tolist()
        fruit_rot = Rotation.from_euler("xyz", tomato_rotation, degrees=True)
        meta["soft_rotation_relative_to_branch_xyzw"] = (
            source_rotation.inv() * fruit_rot).as_quat().tolist()
        meta["mass_parent_path"] = (None if fruit_name == support_name else parent_path)
        stem_point_world = world_position + fruit_rot.apply(meta["stem_attachment_local_m"])
        branch_origin_world = np.asarray(
            [group_center_xy[0], group_center_xy[1], 0.06220 + table_lift_m], dtype=float)
        meta["branch_anchor_local_m"] = source_rotation.inv().apply(
            stem_point_world - branch_origin_world).tolist()
        meta["youngs_modulus_pa"] = soft_fruit_youngs_modulus_pa
        soft_fruit_metadata.append(meta)
        soft_simulation_paths.append(simulation_path)
        fruit_collider_paths.append(simulation_path)

    if support_name in soft_fruit_names and not args.soft_fruit_on_hand_contact:
        support_branch_path = "/World/TomatoClusterDistal" if support_on_distal else "/World/TomatoCluster"
        author_soft_cluster_fruit(support_name, np.asarray(support_offset, dtype=float),
                                  support_world, support_branch_path)
        support_body = None
    else:
        support_root = UsdGeom.Xform.Define(stage, "/World/SupportTomato")
        support_root.AddTranslateOp().Set(Gf.Vec3d(*support_world.tolist()))
        support_root.AddRotateXYZOp().Set(Gf.Vec3f(*tomato_rotation))
        UsdPhysics.RigidBodyAPI.Apply(support_root.GetPrim())
        UsdPhysics.MassAPI.Apply(support_root.GetPrim()).CreateMassAttr(fruit_mass_kg)
        support_visual = _visual_mesh(
            stage, "/World/SupportTomato/Visual", asset_root / f"{support_name}_visual.npz",
            (0.75, 0.035, 0.025), texture=fruit_texture)
        _enable_convex_mesh_collision(support_visual)
        fruit_collider_paths.append("/World/SupportTomato/Visual")
        if support_name in soft_fruit_names:
            support_branch_path = "/World/TomatoClusterDistal" if support_on_distal else "/World/TomatoCluster"
            author_soft_cluster_fruit(support_name, np.asarray(support_offset, dtype=float),
                                      support_world, support_branch_path)
            if args.soft_fruit_on_hand_contact:
                touch_proxy_metadata[support_name] = {
                    "proxy_root_path": "/World/SupportTomato",
                    "proxy_collision_path": "/World/SupportTomato/Visual",
                    "proxy_visual_path": "/World/SupportTomato/Visual",
                    "proxy_mode": "support_actor",
                    "mass_parent_path": None,
                }
    # Source support attachment is a linear/angular spring-damper, not a weld.
    # Its equal axial/lateral stiffness is implemented as an isotropic spring.
    for fruit_name, (offset_x, offset_y, offset_z) in fruit_offsets.items():
        if fruit_name == support_name:
            continue
        offset = np.asarray([offset_x, offset_y, offset_z], dtype=np.float64)
        fruit_world = np.asarray(tomato_start, dtype=np.float64)
        fruit_world[:2] = np.asarray(group_center_xy, dtype=np.float64)
        fruit_world += source_rotation.apply(offset)
        visual_npz = asset_root / f"{fruit_name}_visual.npz"
        fruit_parent = "/World/TomatoClusterDistal" if fruit_name in distal_fruit_names else "/World/TomatoCluster"
        if fruit_name in soft_fruit_names:
            author_soft_cluster_fruit(fruit_name, offset, fruit_world, fruit_parent)
            if args.soft_fruit_on_hand_contact:
                fruit_visual = _visual_mesh(
                    stage, f"{fruit_parent}/{fruit_name}", visual_npz,
                    (0.75, 0.035, 0.025), (offset_x, offset_y, offset_z),
                    texture=fruit_texture)
                _enable_convex_mesh_collision(fruit_visual)
                proxy_path = str(fruit_visual.GetPath())
                fruit_collider_paths.append(proxy_path)
                touch_proxy_metadata[fruit_name] = {
                    "proxy_root_path": fruit_parent,
                    "proxy_collision_path": proxy_path,
                    "proxy_visual_path": proxy_path,
                    "proxy_mode": "compound_branch_fruit",
                    "mass_parent_path": fruit_parent,
                }
            continue
        fruit_visual = _visual_mesh(stage, f"{fruit_parent}/{fruit_name}", visual_npz,
                                    (0.75, 0.035, 0.025), (offset_x, offset_y, offset_z),
                                    texture=fruit_texture)
        _enable_convex_mesh_collision(fruit_visual)
        fruit_collider_paths.append(str(fruit_visual.GetPath()))
    # The source task models the pedicel as a compliant spring/damper with a
    # filtered, persistent tensile overload rule. A USD FixedJoint cannot
    # reproduce that behavior; the equivalent wrench is applied each PhysX step.
    attachment = target_attachment
    if args.contact_probe:
        probe = UsdGeom.Sphere.Define(stage, "/World/TactileProbe")
        probe.CreateRadiusAttr(0.012)
        probe_translate = probe.AddTranslateOp()
        probe_translate.Set(Gf.Vec3d(1.0, 1.0, 1.0))
        probe.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.8, 0.05)])
        UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
        UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateMassAttr(0.02)
    for box_index, (cx, cy) in enumerate((
            (basket_first_center_xy[0] + 0.095 * index, basket_first_center_xy[1])
            for index in range(3))):
        box_path = "/World/LeftBasket" if box_index == 0 else f"/World/Box{box_index}"
        for name, pos, size in (
            ("Base", (cx, cy, table_top_z + 0.0025), (0.085, 0.085, 0.005)),
            ("LeftWall", (cx - 0.0425, cy, table_top_z + 0.0225), (0.005, 0.085, 0.045)),
            ("RightWall", (cx + 0.0425, cy, table_top_z + 0.0225), (0.005, 0.085, 0.045)),
            ("FrontWall", (cx, cy - 0.0425, table_top_z + 0.0225), (0.085, 0.005, 0.045)),
            ("BackWall", (cx, cy + 0.0425, table_top_z + 0.0225), (0.085, 0.005, 0.045)),
        ):
            _static_cube(stage, f"{box_path}/{name}", pos, size, (0.7, 0.68, 0.52))

    # The source scene uses high-friction fruit/table contacts and explicit
    # velocity damping. Without them a resting bunch slides even with no policy.
    contact_material_path = "/World/TomatoContactMaterial"
    contact_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(source_scene_config["tomato_friction"][0]),
        dynamic_friction=float(source_scene_config["tomato_friction"][0]),
        restitution=0.0,
    )
    contact_material.func(contact_material_path, contact_material)
    contact_collider_paths = [
        "/World/Table", "/World/TomatoPlate", "/World/TargetTomato/Collision",
        *branch_collision_paths, *fruit_collider_paths,
    ]
    if support_name not in soft_fruit_names:
        contact_collider_paths.append("/World/SupportTomato/Visual")
    for collider_path in dict.fromkeys(contact_collider_paths):
        if collider_path in soft_simulation_paths:
            continue
        sim_utils.bind_physics_material(collider_path, contact_material_path)
    damping_bodies = [
        ("/World/TargetTomato", float(source_scene_config["tomato_linear_damping"]),
         float(source_scene_config["tomato_angular_damping"])),
        ("/World/TomatoCluster", 1.0, 1.0),
    ]
    if support_name not in soft_fruit_names or args.soft_fruit_on_hand_contact:
        damping_bodies.append(("/World/SupportTomato",
                               float(source_scene_config.get("support_linear_damping", 5.0)),
                               float(source_scene_config.get("support_angular_damping", 3.0))))
    for body_path, linear_damping, angular_damping in damping_bodies:
        rigid_props = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(body_path))
        rigid_props.CreateLinearDampingAttr().Set(linear_damping)
        rigid_props.CreateAngularDampingAttr().Set(angular_damping)
        rigid_props.CreateMaxLinearVelocityAttr().Set(float(source_scene_config["max_linear_velocity_m_s"]))
        rigid_props.CreateMaxAngularVelocityAttr().Set(float(source_scene_config["max_angular_velocity_rad_s"]))

    sim.set_camera_view(eye=(0.40, -0.70, 1.70), target=(0.34, 0.05, 0.10))
    joint_prims = [prim for prim in Usd.PrimRange(stage.GetPseudoRoot())
                   if prim.IsA(UsdPhysics.RevoluteJoint)]
    articulation_prims = [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPseudoRoot())
                          if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    report = {"backend": "Isaac Sim / PhysX", "joint_count": len(joint_prims),
              "articulation_roots": articulation_prims,
              "joint_paths": [str(prim.GetPath()) for prim in joint_prims],
              "tomato": "/World/TargetTomato", "plate": "/World/TomatoPlate", "basket": "/World/LeftBasket",
              "breakable_pedicel": "runtime compliant spring-damper between TargetTomato and TomatoCluster",
              "pedicel_break_force_N": float(source_attachment_config["break_force_N"]),
              "pedicel_break_torque_Nm": float(source_attachment_config["break_torque_Nm"]),
              "pedicel_bodies": ["/World/TargetTomato", "/World/TomatoCluster"],
              "pedicel_attachment_local_m": list(attachment),
              "support_fruit": (f"/World/SoftTomato_{support_name}" if support_name in soft_fruit_names
                                else "/World/SupportTomato"),
              "support_attachment": ("native deformable vertex-to-xform attachment"
                                     if support_name in soft_fruit_names else
                                     "runtime compliant spring-damper between SupportTomato and its branch segment"),
              "branch_collision": {
                  "path": branch_collision_paths[0],
                  "axis_local": [0.0, branch_axis_sign, 0.0],
                  "length_m": branch_length_m,
                  "radius_m": branch_radius_m,
                  "local_center_m": [float((stem_min[0] + stem_max[0]) * 0.5),
                                      float(branch_axis_sign * branch_length_m * 0.5),
                                      float((stem_min[2] + stem_max[2]) * 0.5)],
              },
              "fruit_collision_paths": fruit_collider_paths,
              "cluster_mass_kg": cluster_mass_kg,
              "distal_cluster_mass_kg": distal_cluster_mass_kg,
              "flexible_branch": bool(args.flexible_branch),
              "branch_collision_paths": branch_collision_paths,
              "branch_flex_joint_path": branch_joint_path,
              "branch_cone_limit_deg_per_axis": 2.8 if distal_cluster is not None else None,
              "branch_resultant_bend_limit_deg_approx": 4.8 if distal_cluster is not None else None,
              "support_fruit_mass_kg": fruit_mass_kg,
              "fruit_grouping": {
                  "grasp_target": target_key,
                  "rigid_front_zone": [target_key, *sorted(
                      front_rigid_fruit_names, key=lambda name: abs(float(fruit_offsets[name][1])))],
                  "soft_fruit_candidates_away_from_grasp": sorted(soft_candidate_names),
                  "soft_fruit_names_enabled": sorted(soft_fruit_names | ({target_key} if args.soft_grasp_target else set())),
                  "soft_attachment_branch_segment": {
                      meta["name"]: meta["attachment_branch_path"] for meta in soft_fruit_metadata
                  },
                  "branch_split_local_y_m": branch_split_y_m,
                  "layout_rule": "target tomato_10 at branch front; the three nearest adjacent fruit remain rigid; soft-grasp-target gives tomato_10 a deformable outer layer",
              },
              "soft_fruit_models": soft_fruit_metadata,
              "soft_fruit_activation_mode": (
                  "rigid proxy until sustained fingertip contact; attached soft collider and visual enabled on touch"
                  if args.soft_fruit_on_hand_contact else "soft body active from first physics step"),
              "rigid_branch_mass_accounting_kg": {
                  "proximal_branch_half": branch_mass_kg * (0.5 if args.flexible_branch else 1.0),
                  "proximal_rigid_fruit_count": proximal_fruit_count,
                  "proximal_total": cluster_mass_kg,
                  "distal_branch_half": branch_mass_kg * 0.5 if distal_cluster is not None else 0.0,
                  "distal_rigid_fruit_count": len(distal_rigid_fruit_names),
                  "distal_total": distal_cluster_mass_kg,
                  "soft_fruit_mass_removed_from_branch_actor_at_start_kg": (
                      len((soft_fruit_names - {support_name}) - gated_branch_fruit_names)
                      * fruit_mass_kg),
                  "support_fruit_mass_remaining_on_separate_actor_kg": (
                      fruit_mass_kg if support_name in soft_fruit_names else 0.0),
                  "touch_proxy_count": len(touch_proxy_metadata),
                  "per_fruit_active_mass_kg": fruit_mass_kg,
                  "initial_total_scene_mass_kg": (
                      cluster_mass_kg + distal_cluster_mass_kg + fruit_mass_kg *
                      (1 + len(fruit_offsets))),
              }}
    report["contact_probe_mode"] = bool(args.contact_probe)
    report["plate_collision"] = True
    report["table_and_contents_lift_m"] = table_lift_m
    report["table_top_z_m"] = table_top_z
    report["real_video_alignment"] = {
        "reference": "A/54_120_data_lerobot episode_000000 cam0 frame 0 (640x400); approximate image-plane alignment",
        "plate_center_xy_m": list(plate_center_xy),
        "tomato_group_offset_from_plate_xy_m": list(tomato_group_offset_xy),
        "truss_center_xy_m": list(group_center_xy),
        "basket_first_center_xy_m": list(basket_first_center_xy),
        "table_front_x_m": args.table_front_x_m,
        "camera_extrinsics_calibrated": False,
    }
    report["tomato_orientation"] = (
        "source FBX coordinates preserved; fruit calyxes face upward"
        if tomato_layout else
        "restored from saved USDA transforms; target fruit attached to textured original vine"
    )
    report["tomato_count_on_plate"] = len(fruit_offsets) + 1
    report["tomato_group_center_world_m"] = [
        float(group_center_xy[0]), float(group_center_xy[1]), 0.06220 + table_lift_m
    ]
    report["tomato_offsets_from_group_center_m"] = {
        target_key: [0.0, 0.0, 0.0],
        **{name: [float(offset[0]), float(offset[1]), float(offset[2])]
           for name, offset in fruit_offsets.items()},
    }
    report["tomato_target_screen_side"] = "plate center"
    report["tomato_visual_stem_mesh"] = True
    report["tomato_support_green_sphere"] = False
    if tomato_layout:
        report["tomato_source_fbx"] = tomato_layout.get("source_fbx")
        report["tomato_source_fbx_sha256"] = tomato_layout.get("source_fbx_sha256")
        report["tomato_source_scale"] = tomato_layout.get("source_scale")
        report["tomato_target_extent_m"] = tomato_layout.get("target_extent_m")
        report["pedicel_attachment_target_local_m"] = list(target_attachment)
        report["pedicel_attachment_cluster_local_m"] = list(vine_attachment)
        report["pedicel_attachment_surface_gap_m"] = tomato_layout.get("attachment_surface_gap_m")
    print(f"[Isaac tomato] revolute joints={len(joint_prims)}, roots={articulation_prims}", flush=True)
    from isaaclab.sensors import ContactSensor as LabContactSensor, ContactSensorCfg
    sensors = []
    external_contact_paths = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if path.startswith("/World/TomatoRobot/"):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            external_contact_paths.append(path)
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # For compound rigid bodies PhysX filters by the actor/root path, not
        # each child collider (GPU contact filtering rejects child mesh paths).
        ancestor = prim.GetParent()
        has_rigid_ancestor = False
        while ancestor and ancestor.IsValid() and ancestor.GetPath() != Sdf.Path.absoluteRootPath:
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                has_rigid_ancestor = True
                break
            ancestor = ancestor.GetParent()
        if not has_rigid_ancestor:
            external_contact_paths.append(path)
    report["external_contact_filter_path_count"] = len(external_contact_paths)
    external_contact_path_indices = {path: index for index, path in enumerate(external_contact_paths)}
    if args.soft_fruit_on_hand_contact:
        missing_proxy_filters = sorted(
            meta["proxy_root_path"] for meta in touch_proxy_metadata.values()
            if meta["proxy_root_path"] not in external_contact_path_indices)
        if missing_proxy_filters:
            raise RuntimeError(f"Touch proxies are absent from fingertip contact filters: {missing_proxy_filters}")
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            body_path = f"/World/TomatoRobot/base_link/{side}_{finger}_DP"
            body = stage.GetPrimAtPath(body_path)
            if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Sharpa fingertip rigid body missing: {body_path}")
            report_api = PhysxSchema.PhysxContactReportAPI.Apply(body)
            report_api.CreateThresholdAttr(0.0)
            sensor = LabContactSensor(ContactSensorCfg(
                prim_path=body_path, update_period=0.0, track_pose=True,
                track_contact_points=bool(external_contact_paths),
                track_friction_forces=bool(external_contact_paths),
                max_contact_data_count_per_prim=256,
                filter_prim_paths_expr=external_contact_paths,
            ))
            sensors.append(sensor)
    report["contact_sensor_count"] = len(sensors)
    report["tactile_shape"] = [2, 5, 240, 240]
    report["wrench_shape"] = [2, 5, 6]
    proof_lab_sensor = None
    proof_body = None
    if args.sensor_self_test:
        # Use Isaac's canonical plane for the sensor self-test so this test
        # isolates contact reporting from the tabletop collider/importer.
        sim_utils.GroundPlaneCfg().func("/World/ContactProofGround", sim_utils.GroundPlaneCfg())
        proof = UsdGeom.Cube.Define(stage, "/World/ContactProof")
        proof.CreateSizeAttr(0.02)
        proof.AddTranslateOp().Set(Gf.Vec3d(0.60, 0.40, 0.14))
        UsdPhysics.CollisionAPI.Apply(proof.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(proof.GetPrim())
        UsdPhysics.MassAPI.Apply(proof.GetPrim()).CreateMassAttr(0.02)
        proof_report_api = PhysxSchema.PhysxContactReportAPI.Apply(proof.GetPrim())
        proof_report_api.CreateThresholdAttr(0.0)
        proof_lab_sensor = LabContactSensor(ContactSensorCfg(
            prim_path="/World/ContactProof", update_period=0.0,
            track_pose=True, track_contact_points=True, track_friction_forces=True,
            filter_prim_paths_expr=["/World/ContactProofGround"],
        ))

    import math
    for joint in joint_prims:
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            raise RuntimeError(f"Missing position drive: {joint.GetPath()}")
        if "fr3v2_joint" in joint.GetName():
            stiffness, damping, max_force = 3000.0, 150.0, 100.0
        else:
            stiffness, damping, max_force = 80.0, 2.0, 12.0
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        drive.GetMaxForceAttr().Set(max_force)
    for joint in joint_prims:
        joint_name = joint.GetName()
        radians = home_positions_rad[joint_name]
        lower_deg = float(joint.GetAttribute("physics:lowerLimit").Get())
        upper_deg = float(joint.GetAttribute("physics:upperLimit").Get())
        if not math.radians(lower_deg) - 1e-4 <= radians <= math.radians(upper_deg) + 1e-4:
            raise ValueError(
                f"Real home for {joint_name}={radians:.5f} rad is outside imported limit "
                f"[{math.radians(lower_deg):.5f}, {math.radians(upper_deg):.5f}] rad"
            )
        state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.GetPositionAttr().Set(math.degrees(radians))
        state.GetVelocityAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        drive.GetTargetPositionAttr().Set(math.degrees(radians))
    sim.reset()
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
    from isaacsim.core.utils.types import ArticulationAction

    # Let PhysX cook each mesh first, then bind the four vertices nearest the
    # pedicel using the actual cooked vertex ordering. This avoids assuming
    # that the cooker preserves source-mesh indices.
    if soft_fruit_metadata:
        from scipy.spatial.transform import Rotation

        pre_view = sim.physics_sim_view.create_volume_deformable_body_view("/World/SoftTomato_*")
        if pre_view is None or int(pre_view.count) != len(soft_fruit_metadata):
            raise RuntimeError("PhysX failed to cook all soft fruits before attachment authoring")
        pre_rest = _numpy(pre_view.get_rest_nodal_positions()).astype(np.float64)
        if pre_rest.ndim != 3 or pre_rest.shape[0] != len(soft_fruit_metadata):
            raise RuntimeError(f"Soft fruit cooked node readback has invalid shape {pre_rest.shape}")
        path_to_index = {str(path): index for index, path in enumerate(pre_view.prim_paths)}
        xform_cache = UsdGeom.XformCache()
        for meta in soft_fruit_metadata:
            body_index = path_to_index[meta["root_path"]]
            rest_nodes = pre_rest[body_index]
            target_world = np.asarray(meta["world_start_m"], dtype=float) + Rotation.from_euler(
                "xyz", meta["fruit_rotation_xyz_deg"], degrees=True
            ).apply(np.asarray(meta["stem_attachment_local_m"], dtype=float))
            chosen = np.argsort(np.linalg.norm(rest_nodes - target_world[None, :], axis=1))[:4]
            owner_prim = stage.GetPrimAtPath(meta["attachment_branch_path"])
            world_to_owner = xform_cache.GetLocalToWorldTransform(owner_prim).GetInverse()
            local_points = np.asarray([
                list(world_to_owner.Transform(Gf.Vec3d(*map(float, rest_nodes[index]))))
                for index in chosen
            ], dtype=float)
            attachment = stage.DefinePrim(meta["attachment_path"], "OmniPhysicsVtxXformAttachment")
            attachment.GetRelationship("omniphysics:src0").SetTargets(
                [Sdf.Path(meta["simulation_mesh_path"])])
            attachment.GetRelationship("omniphysics:src1").SetTargets(
                [Sdf.Path(meta["attachment_branch_path"])])
            attachment.GetAttribute("omniphysics:attachmentEnabled").Set(True)
            attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set(
                [int(index) for index in chosen])
            attachment.GetAttribute("omniphysics:localPositionsSrc1").Set([
                Gf.Vec3f(*map(float, point)) for point in local_points])
            attachment.GetAttribute("omniphysics:stiffness").Set(
                float("inf") if args.soft_grasp_target and meta["name"] == target_key else 1200.0)
            attachment.GetAttribute("omniphysics:damping").Set(
                0.0 if args.soft_grasp_target and meta["name"] == target_key else 2.0)
            meta["attachment_vertex_indices"] = [int(index) for index in chosen]
            meta["attachment_local_positions_in_branch_m"] = local_points.tolist()
            meta["attachment_node_target_distance_m"] = float(
                np.max(np.linalg.norm(rest_nodes[chosen] - target_world[None, :], axis=1)))
        sim.reset()

    robot = SingleArticulation(prim_path="/World/TomatoRobot/base_link/base_link", name="dual_fr3_sharpa")
    robot.initialize()
    soft_body_view = None
    soft_body_names = []
    soft_rest_nodes = None
    soft_body_index_by_name = {}
    if soft_fruit_metadata:
        physics_sim_view = sim.physics_sim_view
        if physics_sim_view is None:
            raise RuntimeError("Isaac did not expose a tensor view for the soft tomato bodies")
        soft_body_view = physics_sim_view.create_volume_deformable_body_view("/World/SoftTomato_*")
        if soft_body_view is None or int(soft_body_view.count) != len(soft_fruit_metadata):
            raise RuntimeError(
                "Soft tomato tensor view count mismatch: "
                f"{0 if soft_body_view is None else int(soft_body_view.count)} != {len(soft_fruit_metadata)}"
            )
        soft_body_names = [str(path).rsplit("/", 1)[-1].removeprefix("SoftTomato_")
                           for path in soft_body_view.prim_paths]
        soft_body_index_by_name = {name: index for index, name in enumerate(soft_body_names)}
        soft_rest_nodes = _numpy(soft_body_view.get_rest_nodal_positions()).astype(np.float32)
        if soft_rest_nodes.ndim != 3 or soft_rest_nodes.shape[0] != len(soft_fruit_metadata):
            raise RuntimeError(f"Soft tomato rest-node readback has invalid shape {soft_rest_nodes.shape}")
        report["soft_fruit_tensor_view"] = {
            "body_count": int(soft_body_view.count),
            "body_paths": [str(path) for path in soft_body_view.prim_paths],
            "simulation_mesh_paths": [str(path) for path in soft_body_view.simulation_mesh_prim_paths],
            "rest_node_shape": list(soft_rest_nodes.shape),
        }
        metadata_by_name = {item["name"]: item for item in soft_fruit_metadata}
        for fruit_name, body_index in soft_body_index_by_name.items():
            metadata_by_name[fruit_name]["soft_body_index"] = int(body_index)
        if args.soft_fruit_on_hand_contact:
            for item in soft_fruit_metadata:
                stage.GetPrimAtPath(item["root_path"]).GetAttribute(
                    "omniphysics:deformableBodyEnabled").Set(False)
                # PhysX still validates attachment actors when an attachment
                # targets a disabled deformable body. Remove it while the
                # corresponding rigid proxy owns the mass; recreate on touch.
                stage.RemovePrim(item["attachment_path"])
                UsdGeom.Imageable(stage.GetPrimAtPath(f"{item['root_path']}/Visual")).MakeInvisible()
            report["soft_fruit_disabled_before_first_physics_step"] = sorted(soft_body_names)
    import torch
    policy = None
    policy_schema = None
    policy_report = None
    policy_action_indices = None
    action_chunk_policy = None
    action_chunk_checkpoint = None
    action_chunk_report = None
    action_chunk_action_indices = None
    task_initial_joint_positions = None
    if args.policy_ckpt is not None:
        import torch
        from handson.data.sharpa_lerobot_bridge import discover_datasets, iter_episode_batches
        from train_sharpa_lerobot_bc import Policy

        checkpoint = torch.load(args.policy_ckpt.expanduser(), map_location="cpu", weights_only=False)
        expected_mode = args.policy_mode or checkpoint.get("mode")
        if checkpoint.get("mode") != expected_mode:
            raise ValueError(f"Checkpoint mode {checkpoint.get('mode')!r} != requested {expected_mode!r}")
        datasets = discover_datasets(args.policy_schema.expanduser())
        policy_schema = datasets[0]
        if len(policy_schema.state_names) != 116 or len(policy_schema.action_names) != 58:
            raise ValueError("Sharpa policy schema must have 116 state and 58 action names")
        if int(checkpoint.get("input_size", -1)) != (816 if expected_mode == "tactile-force" else 116):
            raise ValueError(f"Checkpoint input width does not match {expected_mode}: {checkpoint.get('input_size')}")
        if int(checkpoint.get("action_size", -1)) != 58:
            raise ValueError(f"Expected 58 output actions, got {checkpoint.get('action_size')}")
        policy = Policy(checkpoint["input_size"], checkpoint["action_size"])
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        sim_action_names = [_action_name_for_dof(name) for name in robot.dof_names]
        if set(sim_action_names) != set(policy_schema.action_names):
            missing = sorted(set(policy_schema.action_names) - set(sim_action_names))
            extra = sorted(set(sim_action_names) - set(policy_schema.action_names))
            raise ValueError(f"Imported Isaac DOF names do not match action schema; missing={missing[:12]}, extra={extra[:12]}")
        policy_action_indices = [policy_schema.action_names.index(name) for name in sim_action_names]
        policy_report = {"checkpoint": str(args.policy_ckpt.expanduser().resolve()),
                         "mode": expected_mode, "schema_dataset": str(policy_schema.root),
                         "inference_steps": 0, "finite_outputs": True,
                         "applied_to_robot": bool(args.policy_apply),
                         "input_size": int(checkpoint["input_size"]), "action_size": int(checkpoint["action_size"])}
        report["policy"] = policy_report
    if args.action_chunk_ckpt is not None:
        import torch
        from handson.data.sharpa_lerobot_bridge import discover_datasets, iter_episode_batches
        from train_sharpa_action_chunks import ActionChunkPolicy

        action_chunk_checkpoint = torch.load(args.action_chunk_ckpt.expanduser(),
                                             map_location="cpu", weights_only=False)
        mode = action_chunk_checkpoint.get("mode")
        if mode not in ("rgb-state", "rgb-state-touch"):
            raise ValueError(f"Unsupported action-chunk checkpoint mode: {mode!r}")
        if len(action_chunk_checkpoint.get("state_names", [])) != 116 or len(action_chunk_checkpoint.get("action_names", [])) != 58:
            raise ValueError("Action-chunk checkpoint must carry the 116/58 named joint contract")
        policy_schema = discover_datasets(args.policy_schema.expanduser())[0]
        if tuple(action_chunk_checkpoint["state_names"]) != policy_schema.state_names:
            raise ValueError("Checkpoint state schema differs from the selected LeRobot dataset")
        if tuple(action_chunk_checkpoint["action_names"]) != policy_schema.action_names:
            raise ValueError("Checkpoint action schema differs from the selected LeRobot dataset")
        if args.task_eval:
            initial_batch = next(iter_episode_batches(policy_schema, 0, batch_size=1))
            state_index = {name: index for index, name in enumerate(policy_schema.state_names)}
            init_state_names = [_state_name_for_dof(name, "position") for name in robot.dof_names]
            missing_state = sorted(set(init_state_names) - set(state_index))
            if missing_state:
                raise ValueError(f"LeRobot initialization state is missing joints: {missing_state[:8]}")
            task_initial_joint_positions = np.asarray(
                [initial_batch.state[0, state_index[name]] for name in init_state_names], dtype=np.float32)
            robot.set_joint_positions(torch.as_tensor(task_initial_joint_positions, device="cuda:0"))
            report["task_initialization"] = {
                "source_dataset": str(policy_schema.root), "episode_index": 0,
                "source_frame": 0, "joint_positions_loaded": int(len(task_initial_joint_positions)),
                "joint_velocities": "zeroed after reset",
            }
        touch_size = int(action_chunk_checkpoint["touch_mean"].numel())
        companion_summary = args.action_chunk_ckpt.expanduser().parent / "summary.json"
        if companion_summary.is_file():
            action_chunk_checkpoint.setdefault(
                "sample_stride", json.loads(companion_summary.read_text()).get("sample_stride", 4))
        action_chunk_checkpoint.setdefault("sample_stride", 4)
        replan_frames = min(int(action_chunk_checkpoint["sample_stride"]),
                            int(action_chunk_checkpoint["chunk"]))
        action_chunk_policy = ActionChunkPolicy(
            touch_size, int(action_chunk_checkpoint["chunk"])).to("cuda:0")
        action_chunk_policy.load_state_dict(action_chunk_checkpoint["model_state_dict"], strict=True)
        action_chunk_policy.eval()
        sim_action_names = [_action_name_for_dof(name) for name in robot.dof_names]
        if set(sim_action_names) != set(policy_schema.action_names):
            raise ValueError("Imported Isaac joint names do not match action-chunk checkpoint schema")
        action_chunk_action_indices = [policy_schema.action_names.index(name) for name in sim_action_names]
        action_chunk_report = {
            "checkpoint": str(args.action_chunk_ckpt.expanduser().resolve()),
            "mode": mode, "chunk": int(action_chunk_checkpoint["chunk"]),
            "sample_stride": int(action_chunk_checkpoint.get("sample_stride", 4)),
            "replan_frames": replan_frames,
            "applied_to_robot": True, "inference_steps": 0,
            "finite_outputs": True, "peak_force_n": 0.0,
        }
        report["action_chunk_policy"] = action_chunk_report
    if proof_lab_sensor is not None:
        proof_body = SingleRigidPrim(prim_path="/World/ContactProof", name="contact_proof_body")
        proof_body.initialize()
    link_indices = [robot._articulation_view.get_link_index(
        f"{side}_{finger}_DP")
        for side in ("left", "right")
        for finger in ("thumb", "index", "middle", "ring", "pinky")]
    replay_probe_names = [f"{side}_{finger}_DP"
                          for side in ("left", "right")
                          for finger in ("thumb", "index", "middle", "ring", "pinky")]
    replay_probe_names += [f"{side}_fr3v2_link8" for side in ("left", "right")]
    replay_probe_indices = link_indices + [
        robot._articulation_view.get_link_index(name) for name in replay_probe_names[-2:]]
    replay_probe_times = []
    replay_probe_positions = []
    report["active_dof_count"] = len(robot.dof_names)
    report["active_dof_names"] = list(robot.dof_names)
    report["active_body_names"] = list(robot._articulation_view.body_names)
    # Preserve the exact arm state and FK frames used to validate a T-Rex
    # chunk.  PhysX link quaternions are xyzw; the trained adapter's URDF
    # chains are rooted at each imported link0.
    initial_joint_positions = _numpy(robot.get_joint_positions()).reshape(-1)
    report["home_initialization"] = {
        "config": str(home_config_path),
        "source": home_config.get("sources", {}),
        "joint_count": len(home_positions_rad),
        "requested_positions_rad": home_positions_rad,
        "measured_positions_rad": {
            name: float(initial_joint_positions[index])
            for index, name in enumerate(robot.dof_names)
        },
        "max_abs_error_rad": max(
            abs(float(initial_joint_positions[index]) - home_positions_rad[name])
            for index, name in enumerate(robot.dof_names)
        ),
    }
    link_transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
    from scipy.spatial.transform import Rotation
    report["home_palm_normals_world"] = {}
    for side in ("left", "right"):
        pose = link_transforms[robot._articulation_view.get_link_index(f"{side}_hand_C_MC")]
        # Positive local X is the palmar side: positive finger flexion curls
        # the fingertip toward this side of the authored Sharpa hand.
        normal = Rotation.from_quat(pose[3:]).apply((1.0, 0.0, 0.0))
        report["home_palm_normals_world"][side] = {
            "pose_xyzw": pose.tolist(),
            "normal_xyz": normal.tolist(),
            "angle_from_down_deg": float(np.degrees(np.arccos(np.clip(-normal[2], -1.0, 1.0)))),
        }
    arm_frame_probe = {}
    for side in ("left", "right"):
        q = [float(initial_joint_positions[robot.dof_names.index(f"{side}_fr3v2_joint{i}")])
             for i in range(1, 8)]
        frame_indices = {name: robot._articulation_view.get_link_index(f"{side}_fr3v2_{name}")
                         for name in ("link0", "link8")}
        arm_frame_probe[side] = {
            "joint_positions_rad": q,
            "link0_physx_pose_xyzw": link_transforms[frame_indices["link0"]].tolist(),
            "link8_physx_pose_xyzw": link_transforms[frame_indices["link8"]].tolist(),
        }
    report["arm_frame_probe"] = arm_frame_probe
    report["home_index_fingertip_link_poses_xyzw"] = {
        side: link_transforms[robot._articulation_view.get_link_index(
            f"{side}_index_DP")].tolist()
        for side in ("left", "right")
    }
    report["tactile_link_indices"] = {
        f"{side}_{finger}_DP": link_indices[index]
        for index, (side, finger) in enumerate(
            (side, finger) for side in ("left", "right")
            for finger in ("thumb", "index", "middle", "ring", "pinky"))
    }
    replay = None
    if args.replay_h5 is not None:
        import h5py
        import torch

        with h5py.File(args.replay_h5, "r") as episode:
            metadata = json.loads(episode["metadata"].attrs["extra"])
            names = []
            for name in metadata["joint_names"]:
                group, joint = name.split("/", 1)
                names.append(f"{group.split('_')[0]}_{joint}" if "sharpa" in group else joint)
            samples = np.asarray(episode["robot/joint_pos"], dtype=np.float32)
            timestamps = np.asarray(episode["robot/timestamps"], dtype=np.float64)
        if len(names) != 58 or set(names) != set(robot.dof_names):
            raise ValueError(f"Replay joint contract differs: {set(robot.dof_names) - set(names)}")
        order = [names.index(name) for name in robot.dof_names]
        replay = (timestamps - timestamps[0], samples[:, order])
        initial_index = int(np.clip(np.searchsorted(replay[0], args.replay_start_sec, side="right") - 1,
                                    0, len(replay[0]) - 1))
        robot.set_joint_positions(torch.as_tensor(replay[1][initial_index], device="cuda:0"))
        report["replay_h5"] = str(args.replay_h5)
        report["replay_start_sec"] = args.replay_start_sec
    from isaacsim.sensors.camera import Camera
    from scipy.spatial.transform import Rotation
    from PIL import Image

    import yaml

    tomato_sim_root = PROJECT.parent / "tomato-pick-sim-local"
    sys.path.insert(0, str(tomato_sim_root))
    from demos.cherry_tomato_pick.camera import Gemini305WristCameras, Gemini435LeCamera

    source_cfg_path = tomato_sim_root / "configs/cherry_tomato_pick_sharpa.yaml"
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    predefined_head_camera = Gemini435LeCamera(source_cfg["camera"])
    # The real cam0 view is low and nearly parallel to the tabletop, with both
    # arms entering from opposite sides. Keep the centered rear view direction
    # from the existing Isaac preview, but lower the camera and aim near hand
    # height so the wrists, arm joints, and plate remain in frame. The dataset
    # does not provide calibrated Sharpa camera extrinsics.
    if args.head_camera_pose_json is not None:
        head_camera_pose = json.loads(args.head_camera_pose_json.expanduser().read_text(encoding="utf-8"))
        head_position = np.asarray(head_camera_pose["isaac_camera_position_world_m"], dtype=float)
        head_orientation_isaac = np.asarray(head_camera_pose["isaac_camera_orientation_wxyz"], dtype=float)
        head_target = None
    else:
        head_position = np.array([-0.10, 0.0, 0.62], dtype=float)
        head_target = np.array([1.1257, 0.0, -0.03], dtype=float)
        head_orientation_isaac = _look_at_wxyz(head_position, head_target)
    camera = Camera(
        prim_path="/World/TomatoCamera", name="tomato_head_camera",
        resolution=(predefined_head_camera.width, predefined_head_camera.height),
        position=head_position, orientation=head_orientation_isaac)
    camera.initialize()
    head_usd_camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/TomatoCamera"))
    head_usd_camera.GetHorizontalApertureAttr().Set(1.0)
    head_usd_camera.GetVerticalApertureAttr().Set(
        predefined_head_camera.height / predefined_head_camera.width)
    real_head_k = _real_camera_intrinsics("cam0", (640, 400))
    head_focal_ratio = (float(real_head_k[0, 0] / 640)
                        if real_head_k is not None else
                        float(predefined_head_camera.intrinsic_matrix[0, 0] /
                              predefined_head_camera.width))
    head_usd_camera.GetFocalLengthAttr().Set(head_focal_ratio)
    head_usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(
        *predefined_head_camera.clipping_range_m))
    report["head_camera_pose"] = {
        "position_world_m": head_position.tolist(),
        "orientation_wxyz": head_orientation_isaac.tolist(),
        "calibration_json": (str(args.head_camera_pose_json.expanduser().resolve())
                             if args.head_camera_pose_json is not None else None),
    }
    # Keep cam0 for T-Rex observations. The rollout uses a separate fixed
    # overview resembling the real episode's external 640x400 camera.
    replay_camera = None
    if args.task_eval:
        replay_eye = (head_position.copy() if args.head_camera_pose_json is not None
                      else np.array([-0.68, 0.0, 0.90], dtype=float))
        replay_orientation = (head_orientation_isaac.copy()
                              if args.head_camera_pose_json is not None else None)
        replay_target = np.array([0.35, 0.0, 0.20], dtype=float)
        replay_camera = Camera(
            prim_path="/World/TomatoReplayCamera", name="tomato_replay_camera",
            resolution=(640, 400), position=replay_eye,
            orientation=(replay_orientation if replay_orientation is not None
                         else _look_at_wxyz(replay_eye, replay_target)))
        replay_camera.initialize()
        replay_usd_camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/TomatoReplayCamera"))
        replay_usd_camera.GetHorizontalApertureAttr().Set(1.0)
        replay_usd_camera.GetVerticalApertureAttr().Set(0.625)
        replay_usd_camera.GetFocalLengthAttr().Set(
            head_focal_ratio if args.head_camera_pose_json is not None else 0.55)
        report["replay_camera"] = {
            "position_m": replay_eye.tolist(),
            "orientation_wxyz": (replay_orientation.tolist()
                                 if replay_orientation is not None else None),
            "target_m": (None if replay_orientation is not None else replay_target.tolist()),
            "resolution": [640, 400],
            "source": (str(args.head_camera_pose_json.expanduser().resolve())
                       if args.head_camera_pose_json is not None
                       else "approximate real cam0 overview; extrinsics uncalibrated"),
        }
    rollout_video = None
    task_output_dir = args.output
    if args.replay_live_view:
        live_preview_path = task_output_dir / "replay_live_view" / "latest.png"
        live_view_server, live_view_url = _start_trex_live_view(
            live_preview_path, args.replay_live_view_port, replay=True)
        print(f"[Replay tactile live view] {live_view_url}", flush=True)
        if os.environ.get("REPLAY_LIVE_VIEW_OPEN", "1") == "1":
            webbrowser.open(live_view_url, new=2)
    if args.task_eval or args.record_replay_video:
        import cv2
        if args.trex_closed_loop or args.pi05_closed_loop:
            policy_name = "pi05" if args.pi05_closed_loop else "trex"
            task_output_dir = args.output / f"{policy_name}_eval_seed_{args.eval_seed:04d}"
            task_output_dir.mkdir(parents=True, exist_ok=True)
        if args.trex_live_view:
            live_preview_path = task_output_dir / "trex_live_view" / "latest.png"
            live_view_server, live_view_url = _start_trex_live_view(
                live_preview_path, args.trex_live_view_port)
            print(f"[T-Rex live view] {live_view_url}", flush=True)
            if os.environ.get("TREX_LIVE_VIEW_OPEN", "1") == "1":
                webbrowser.open(live_view_url, new=2)
        video_path = task_output_dir / "rollout.mp4"
        video_writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       7.5, (640, 400))
        if not video_writer.isOpened():
            video_writer.release()
            raise RuntimeError(f"Could not open rollout video writer for {video_path}")
        rollout_video = video_writer
    policy_cameras = {"cam0": camera}
    if action_chunk_policy is not None:
        for side, key in (("left", "cam1"), ("right", "cam2")):
            wrist_index = robot._articulation_view.get_link_index(f"{side}_fr3v2_link8")
            wrist_pose = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0, wrist_index]
            wrist_eye = np.asarray(wrist_pose[:3], dtype=float) + np.array([0.0, 0.0, 0.07])
            wrist_q_wxyz = _look_at_wxyz(wrist_eye, np.asarray(tomato_start))
            wrist_camera = Camera(
                prim_path=f"/World/PolicyCamera{side.title()}", name=f"policy_{key}",
                resolution=(96, 96), position=wrist_eye, orientation=wrist_q_wxyz)
            wrist_camera.initialize()
            policy_cameras[key] = wrist_camera
    tomato_body = SingleRigidPrim(prim_path="/World/TargetTomato", name="target_tomato")
    tomato_body.initialize()
    cluster_body = SingleRigidPrim(prim_path="/World/TomatoCluster", name="tomato_cluster")
    cluster_body.initialize()
    distal_body = None
    if distal_cluster is not None:
        distal_body = SingleRigidPrim(prim_path="/World/TomatoClusterDistal", name="tomato_cluster_distal")
        distal_body.initialize()
    support_body = None
    touch_proxy_bodies = {}
    for fruit_name, proxy_meta in touch_proxy_metadata.items():
        if proxy_meta["proxy_mode"] == "compound_branch_fruit":
            proxy_body = (distal_body if proxy_meta["proxy_root_path"] == "/World/TomatoClusterDistal"
                          else cluster_body)
        else:
            proxy_body = SingleRigidPrim(prim_path=proxy_meta["proxy_root_path"],
                                         name=f"touch_rigid_{fruit_name}")
            proxy_body.initialize()
        touch_proxy_bodies[fruit_name] = proxy_body
    if support_name in touch_proxy_bodies:
        support_body = touch_proxy_bodies[support_name]
    elif stage.GetPrimAtPath("/World/SupportTomato").IsValid():
        support_body = SingleRigidPrim(prim_path="/World/SupportTomato", name="support_tomato")
        support_body.initialize()
    tracked_body_items = [("robot_base", robot), ("tomato", tomato_body),
                          ("cluster", cluster_body)]
    if support_body is not None:
        tracked_body_items.append(("support", support_body))
    if distal_body is not None:
        tracked_body_items.append(("distal_cluster", distal_body))
    report["body_positions_reset"] = {
        name: _numpy(body.get_world_pose()[0]).reshape(-1, 3)[0].astype(float).tolist()
        for name, body in tracked_body_items
    }

    def apply_rigid_wrench(body, force: np.ndarray, torque: np.ndarray, position: np.ndarray) -> None:
        """Apply a world-frame wrench using the GPU tensor API expected by Isaac Sim 5.x."""
        body._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=torch.as_tensor(force, dtype=torch.float32, device="cuda:0").reshape(1, 3),
            torques=torch.as_tensor(torque, dtype=torch.float32, device="cuda:0").reshape(1, 3),
            positions=torch.as_tensor(position, dtype=torch.float32, device="cuda:0").reshape(1, 3),
            is_global=True,
        )

    trex_cameras = {}
    predefined_wrist_cameras = None
    wrist_camera_mount_report = {}
    wrist_camera_local_mounts = {}
    wrist_camera_pose_trace = []
    def trex_wrist_camera_pose(side: str, transforms: np.ndarray):
        """Apply one rigid wrist-camera mount to the current link8 pose."""
        wrist_index = robot._articulation_view.get_link_index(f"{side}_fr3v2_link8")
        wrist_pose = np.asarray(transforms[wrist_index], dtype=float)
        wrist_position = wrist_pose[:3]
        wrist_rotation = Rotation.from_quat(wrist_pose[3:]).as_matrix()
        if side not in wrist_camera_local_mounts:
            camera_key = "wrist_left" if side == "left" else "wrist_right"
            configured_mount = wrist_camera_calibration.get("mounts", {}).get(camera_key)
            if configured_mount is not None:
                local_position = np.asarray(configured_mount["position_link8_m"], dtype=float)
                optical_quat = np.asarray(
                    configured_mount["orientation_link8_opengl_xyzw"], dtype=float)
                mount_source = str(configured_mount.get("source", "calibration JSON"))
            else:
                source_mount = source_cfg["wrist_cameras"].get("mounts", {}).get(side, {})
                if not source_mount:
                    raise ValueError(f"No configured wrist camera mount for {side}")
                local_position = np.asarray(
                    source_mount["optical_position_in_wrist_m"], dtype=float)
                optical_quat = np.asarray(
                    source_mount["optical_quat_in_wrist_xyzw"], dtype=float)
                mount_source = "cherry_tomato_pick_sharpa.yaml mechanical mount"
            # Mount quaternions use OpenGL optical axes (+X right, +Y up,
            # -Z forward). Isaac Camera uses local +X forward and +Z up;
            # construct its right-handed basis from the measured forward/up.
            optical_rotation = Rotation.from_quat(optical_quat)
            forward_local = optical_rotation.apply((0.0, 0.0, -1.0))
            up_local = optical_rotation.apply((0.0, 1.0, 0.0))
            right_local = np.cross(up_local, forward_local)
            right_local /= np.linalg.norm(right_local)
            up_local = np.cross(forward_local, right_local)
            local_rotation = np.column_stack((forward_local, right_local, up_local))
            wrist_camera_local_mounts[side] = (local_position, local_rotation, mount_source)
        local_position, local_rotation, _ = wrist_camera_local_mounts[side]
        eye = wrist_position + wrist_rotation @ local_position
        orientation_xyzw = Rotation.from_matrix(wrist_rotation @ local_rotation).as_quat()
        return eye, orientation_xyzw[[3, 0, 1, 2]]

    def update_trex_wrist_camera_poses(transforms: np.ndarray, episode_time_s: float | None = None):
        """Keep rendered T-Rex sensors rigidly attached to the moving wrist links."""
        frame_poses = {}
        for side, camera_key in (("left", "wrist_left"), ("right", "wrist_right")):
            wrist_camera = trex_cameras.get(camera_key)
            if wrist_camera is None:
                continue
            camera_position, camera_orientation = trex_wrist_camera_pose(side, transforms)
            wrist_camera.set_world_pose(
                position=camera_position, orientation=camera_orientation)
            if camera_key in wrist_camera_mount_report:
                wrist_camera_mount_report[camera_key]["position_world_m"] = (
                    camera_position.astype(float).tolist())
                wrist_camera_mount_report[camera_key]["orientation_world_isaac_wxyz"] = (
                    camera_orientation.astype(float).tolist())
            frame_poses[camera_key] = {
                "position_world_m": camera_position.astype(float).tolist(),
                "orientation_world_isaac_wxyz": camera_orientation.astype(float).tolist(),
            }
        if frame_poses:
            sample = {"cameras": frame_poses}
            if episode_time_s is not None:
                sample["episode_time_s"] = float(episode_time_s)
            wrist_camera_pose_trace.append(sample)

    if args.trex_host or args.pi05_closed_loop or args.trex_camera_preview_only or args.replay_live_view:
        predefined_wrist_cameras = Gemini305WristCameras(source_cfg["wrist_cameras"])
        # The real T-Rex cam1/cam2 recordings are 640x480. Render at half
        # resolution with the same 4:3 aspect before the server's 384x288 resize.
        trex_wrist_resolution = (320, 240)
        initial_transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
        wrist_sensor_cfg = predefined_wrist_cameras.sensor_cfg
        for side, camera_key, real_key in (("left", "wrist_left", "cam2"),
                                           ("right", "wrist_right", "cam1")):
            camera_path = f"/World/Trex{camera_key.title().replace('_', '')}Camera"
            camera_position, camera_orientation = trex_wrist_camera_pose(side, initial_transforms)
            wrist_camera = Camera(
                prim_path=camera_path, name=f"trex_{camera_key}",
                resolution=trex_wrist_resolution,
                position=camera_position, orientation=camera_orientation,
            )
            wrist_camera.initialize()
            if args.trex_camera_preview_only:
                wrist_camera.add_distance_to_image_plane_to_frame()
            usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(camera_path))
            usd_camera.GetHorizontalApertureAttr().Set(1.0)
            usd_camera.GetVerticalApertureAttr().Set(
                trex_wrist_resolution[1] / trex_wrist_resolution[0])
            real_wrist_k = _real_camera_intrinsics(real_key, (640, 480))
            wrist_focal_ratio = (float(real_wrist_k[0, 0] / 640)
                                 if real_wrist_k is not None else
                                 float(predefined_wrist_cameras.intrinsic_matrix[0, 0]
                                       / predefined_wrist_cameras.width))
            usd_camera.GetFocalLengthAttr().Set(wrist_focal_ratio)
            usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(
                *predefined_wrist_cameras.clipping_range_m))
            trex_cameras[camera_key] = wrist_camera
            wrist_sensor_cfg = predefined_wrist_cameras.sensor_cfg
            wrist_camera_mount_report[camera_key] = {
                "position_world_m": camera_position.tolist(),
                "orientation_world_isaac_wxyz": camera_orientation.tolist(),
                "placement": "rigid link8-local optical mount",
                "position_link8_m": wrist_camera_local_mounts[side][0].tolist(),
                "orientation_link8_matrix": wrist_camera_local_mounts[side][1].tolist(),
                "mount_source": wrist_camera_local_mounts[side][2],
                "real_camera": real_key,
                "real_intrinsic_k": real_wrist_k.tolist() if real_wrist_k is not None else None,
                "focal_length_over_horizontal_aperture": wrist_focal_ratio,
                "rendered_fov_deg": [float(np.degrees(2 * np.arctan(0.5 / wrist_focal_ratio))),
                                     float(np.degrees(2 * np.arctan(0.375 / wrist_focal_ratio)))],
            }
        policy_camera_sources = {
            "head": "/World/TomatoCamera",
            "head_camera_model": predefined_head_camera.model,
            "head_camera_intrinsics_source": (
                str(real_camera_record_path) if real_head_k is not None else str(source_cfg_path)),
            "head_camera_real_intrinsic_k": real_head_k.tolist() if real_head_k is not None else None,
            "head_camera_focal_length_over_horizontal_aperture": head_focal_ratio,
            "head_camera_pose_source": (str(args.head_camera_pose_json) if args.head_camera_pose_json
                                        else "low centered rear view guided by real cam0 photo"),
            "head_camera_position_world_m": head_position.astype(float).tolist(),
            "head_camera_target_world_m": (head_target.astype(float).tolist()
                                           if head_target is not None else None),
            "head_camera_orientation_world_isaac_wxyz": (
                head_orientation_isaac.astype(float).tolist()),
            "head_camera_resolution": [predefined_head_camera.width,
                                       predefined_head_camera.height],
            "head_camera_fov_deg": [
                float(np.degrees(2 * np.arctan(0.5 / head_focal_ratio))),
                float(np.degrees(2 * np.arctan(0.3125 / head_focal_ratio)))],
            "wrist_right": "/World/TrexWristRightCamera",
            "wrist_left": "/World/TrexWristLeftCamera",
            "wrist_camera_model": predefined_wrist_cameras.model,
            "wrist_camera_intrinsics_source": (
                str(real_camera_record_path) if real_camera_calibration else str(source_cfg_path)),
            "wrist_camera_pose_source": (str(args.wrist_camera_calibration_json.expanduser().resolve())
                                         if args.wrist_camera_calibration_json else
                                         "configured Gemini305 optical mount in cherry_tomato_pick_sharpa.yaml"),
            "wrist_camera_resolution": list(trex_wrist_resolution),
            "real_wrist_camera_resolution": [640, 480],
            "wrist_camera_fov_deg": {
                key: value["rendered_fov_deg"] for key, value in wrist_camera_mount_report.items()
            },
            "wrist_camera_mounts": wrist_camera_mount_report,
        }
        report["policy_camera_sources"] = policy_camera_sources
        if args.trex_host:
            report["trex_camera_sources"] = policy_camera_sources
    if args.contact_probe:
        sensor_position = _numpy(sensors[6].data.pos_w).reshape(-1, 3)[0]
        sensor_quaternion_wxyz = _numpy(sensors[6].data.quat_w).reshape(-1, 4)[0]
        sensor_rotation = Rotation.from_quat(
            [sensor_quaternion_wxyz[1], sensor_quaternion_wxyz[2],
             sensor_quaternion_wxyz[3], sensor_quaternion_wxyz[0]]).as_matrix()
        # Use the actual initialized ContactSensor frame.  The USD fingertip
        # link origin differs from the reported sensor frame on this asset.
        sensor_surface = sensor_position
        probe_position = sensor_surface + sensor_rotation @ np.array([0.0, 0.0, 0.005])
        probe_translate.Set(Gf.Vec3d(*probe_position.tolist()))
        report["contact_probe_sensor_surface_position"] = sensor_surface.tolist()
        report["contact_probe_drop_position"] = probe_position.tolist()
    # SimulationContext.step() only advances PhysX while the USD timeline is
    # playing.  A reset by itself leaves the timeline stopped, which produced
    # rendered frames but no gravity/contact (and therefore zero sensor data).
    sim.play()
    if task_initial_joint_positions is not None:
        robot.apply_action(ArticulationAction(
            joint_positions=torch.as_tensor(task_initial_joint_positions, dtype=torch.float32, device="cuda:0")))
    geometry = TactileGeometry()
    deformation = np.zeros((2, 5, 240, 240), dtype=np.uint8)
    last_frame_deformation = np.zeros_like(deformation)
    wrench = np.zeros((2, 5, 6), dtype=np.float32)
    sensor_stride = 4 if args.task_eval else 1
    trex_f6_samples = []
    trex_deform_samples = []
    snapshots = []
    peak_sensor_contacts = [0] * len(sensors)
    peak_sensor_force = [0.0] * len(sensors)
    last_sensor_force_norm = np.zeros(len(sensors), dtype=np.float32)
    proof_contact_peak = 0
    proof_min_height = float("inf")
    proof_lab_peak_force = np.zeros(3, dtype=np.float32)
    proof_tactile_peak = 0
    policy_action_min = np.full(58, np.inf, dtype=np.float32)
    policy_action_max = np.full(58, -np.inf, dtype=np.float32)
    policy_action_last = None
    action_chunk_values = None
    action_chunk_peak_by_sensor = np.zeros(10, dtype=np.float32)
    action_chunk_action_min = np.full(58, np.inf, dtype=np.float32)
    action_chunk_action_max = np.full(58, -np.inf, dtype=np.float32)
    action_chunk_tomato_trace = []
    trex_closed_loop_report = None
    trex_active_targets = None
    trex_chunk_start_arm_q = None
    trex_active_chunk_id = None
    trex_smoothed_targets = None
    trex_safety_stop = None
    replay_safety_stop = None
    trex_rejection_streak = 0
    trex_arm_indices = None
    trex_arm_step_limits = 0.4 * np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7]) / 120.0
    trex_force_trace = []
    trex_f6_trace = []
    trex_output_dir = task_output_dir / "trex_observations"
    if args.trex_closed_loop:
        trex_output_dir.mkdir(parents=True, exist_ok=True)
        adapter_metadata = json.loads((args.trex_checkpoint.expanduser() / "adapter_metadata.json").read_text())
        trex_arm_indices = np.asarray([
            robot.dof_names.index(f"{side}_fr3v2_joint{joint}")
            for side in ("left", "right") for joint in range(1, 8)
        ])
        trex_closed_loop_report = {
            "checkpoint": str(args.trex_checkpoint.expanduser().resolve()),
            "task": args.trex_task,
            "physics_rate_hz": 120,
            "policy_rate_hz": 30,
            "arm_command_smoothing": {
                "method": "joint velocity limit at every 120 Hz physics step",
                "source": "T-Rex hardware_code/teleop/arm_hand_control.py SmoothingAndSafetyManager",
                "velocity_scale": 0.4,
                "joint_velocity_limits_rad_s": [2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7],
                "clipped_physics_steps": 0,
                "maximum_requested_arm_step_rad": 0.0,
            },
            "slow_every_policy_frames": 16,
            "tactile_refine_at_policy_frames": [4, 8, 12],
            "action_chunk_frames": 16,
            "slow_requests": 0,
            "fast_refinements": 0,
            "replanning_calls": [],
            "ik_tolerance": {"position_m": args.trex_ik_position_tolerance_m,
                             "rotation_deg": args.trex_ik_rotation_tolerance_deg,
                             "max_joint_step_rad": args.trex_ik_max_joint_step_rad},
            "link8_workspace_world_m": TREX_LINK8_WORKSPACE_M,
            "action_layout": {
                "left_arm_delta_pose": [0, 9], "left_hand_absolute": [9, 31],
                "right_arm_delta_pose": [31, 40], "right_hand_absolute": [40, 62],
                "source": "T-Rex qwen_vla/sharpa_core.py action_chunk + hardware eval_trex_async.py",
            },
            "camera_layout": {"head": "cam0", "wrist_right": "cam1", "wrist_left": "cam2"},
            "ik_failures": [],
            "workcell_rejections": [],
            "hand_joint_limit_clips": [],
            "finite_outputs": True,
            "action_applied": False,
            "applied_policy_chunks": 0,
            "rejected_policy_chunks": 0,
            "fingertip_force_stop_limit_n": args.trex_max_fingertip_force_n,
            "rejection_streak_stop_limit": args.trex_max_rejection_streak,
            "safety_stop": None,
        }
        trex_force_trace = []
    pi05_client = None
    pi05_closed_loop_report = None
    pi05_active_targets = None
    pi05_smoothed_targets = None
    pi05_joint_indices = None
    pi05_joint_limits = None
    pi05_safety_stop = None
    pi05_output_dir = task_output_dir / "pi05_observations"
    pi05_arm_indices = None
    pi05_arm_step_limits = 0.08 * np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7] * 2) / 120.0
    pi05_hand_interpolation_alpha = 0.10
    if args.pi05_closed_loop:
        from pi05_isaac_policy_client import PI05IsaacClient

        pi05_client = PI05IsaacClient(args.pi05_url, timeout_s=args.pi05_request_timeout)
        atexit.register(pi05_client.close)
        pi05_joint_names = (
            [f"left_fr3v2_joint{joint}" for joint in range(1, 8)]
            + [f"left_{name}" for name in PI05_SHARPA_HAND_ORDER]
            + [f"right_fr3v2_joint{joint}" for joint in range(1, 8)]
            + [f"right_{name}" for name in PI05_SHARPA_HAND_ORDER]
        )
        pi05_joint_indices = np.asarray([robot.dof_names.index(name) for name in pi05_joint_names])
        pi05_arm_indices = np.asarray([
            robot.dof_names.index(f"{side}_fr3v2_joint{joint}")
            for side in ("left", "right") for joint in range(1, 8)
        ])
        pi05_joint_limits = _numpy(robot._articulation_view.get_dof_limits()).reshape(-1, 2)
        if pi05_joint_limits.shape[0] != len(robot.dof_names):
            raise RuntimeError(f"Isaac joint limits do not match articulation: {pi05_joint_limits.shape}")
        pi05_output_dir.mkdir(parents=True, exist_ok=True)
        pi05_closed_loop_report = {
            "endpoint": args.pi05_url,
            "server_metadata": pi05_client.metadata,
            "prompt": args.pi05_prompt,
            "physics_rate_hz": 120,
            "action_rate_hz": 30,
            "replan_every_action_frames": 4,
            "action_chunk_frames": 50,
            "action_order": pi05_joint_names,
            "camera_order": ["cam0/base", "cam2/left_wrist", "cam1/right_wrist"],
            "action_representation": "absolute joint positions, post DeltaActions inverse transform",
            "action_applied": False,
            "inference_calls": [],
            "joint_limit_clips": [],
            "arm_command_smoothing": {
                "method": "per-physics-step joint velocity limit",
                "velocity_scale": 0.08,
                "joint_velocity_limits_rad_s": [2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7] * 2,
                "clipped_physics_steps": 0,
            },
            "hand_command_smoothing": {
                "method": "exponential interpolation toward each 30 Hz target",
                "alpha_per_physics_step": pi05_hand_interpolation_alpha,
                "physics_rate_hz": 120,
            },
            "fingertip_force_stop_limit_n": args.pi05_max_fingertip_force_n,
            "safety_stop": None,
        }
    pedicel_dt = 1.0 / 120.0
    pedicel_state = {
        "rest_quaternion_xyzw": None, "break_axis_local": None, "filtered_load": 0.0,
        "overload_duration_s": 0.0, "left_contact_duration_s": 0.0,
        "broken": False, "break_time_s": None, "peak_force_load": 0.0,
        "peak_tensile_force_n": 0.0, "armed": False,
    }
    support_state = {"rest_relative_quaternion_xyzw": None, "peak_force_n": 0.0,
                     "peak_unclamped_force_n": 0.0}
    support_parent_body = distal_body if support_on_distal else cluster_body
    support_parent_mass_kg = distal_cluster_mass_kg if support_on_distal else cluster_mass_kg
    fruit_inverse_mass = 1.0 / fruit_mass_kg + 1.0 / support_parent_mass_kg
    fruit_inertia = 0.4 * fruit_mass_kg * float(source_scene_config["tomato_radius_m"]) ** 2
    fruit_inverse_inertia = 1.0 / fruit_inertia

    def implicit_spring(error: np.ndarray, relative_velocity: np.ndarray,
                        stiffness: float, damping: float, inverse_mass: float) -> np.ndarray:
        """Backward-Euler spring force for the 120 Hz Isaac control interval."""
        denominator = 1.0 + inverse_mass * (
            pedicel_dt * damping + pedicel_dt * pedicel_dt * stiffness)
        return (stiffness * error - (stiffness * pedicel_dt + damping) * relative_velocity) / denominator

    def bound_wrench(value: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(value))
        return value * (limit / norm) if norm > limit else value

    def apply_compliant_support_attachment() -> None:
        """Apply the source support fruit's configured linear/angular compliance."""
        if support_body is None or (args.soft_fruit_on_hand_contact and support_name in soft_fruit_names):
            return
        from scipy.spatial.transform import Rotation

        cluster_pos, cluster_q = support_parent_body.get_world_pose()
        support_pos, support_q = support_body.get_world_pose()
        cluster_pos = _numpy(cluster_pos).reshape(-1, 3)[0].astype(float)
        support_pos = _numpy(support_pos).reshape(-1, 3)[0].astype(float)
        cluster_q = _numpy(cluster_q).reshape(-1, 4)[0].astype(float)
        support_q = _numpy(support_q).reshape(-1, 4)[0].astype(float)
        cluster_rot = Rotation.from_quat(cluster_q[[1, 2, 3, 0]])
        support_rot = Rotation.from_quat(support_q[[1, 2, 3, 0]])
        if support_state["rest_relative_quaternion_xyzw"] is None:
            support_state["rest_relative_quaternion_xyzw"] = (cluster_rot.inv() * support_rot).as_quat()
        anchor = cluster_pos + cluster_rot.apply(np.asarray(support_offset, dtype=float))
        cluster_v = _numpy(support_parent_body.get_linear_velocity()).reshape(-1, 3)[0].astype(float)
        cluster_w = _numpy(support_parent_body.get_angular_velocity()).reshape(-1, 3)[0].astype(float)
        support_v = _numpy(support_body.get_linear_velocity()).reshape(-1, 3)[0].astype(float)
        support_w = _numpy(support_body.get_angular_velocity()).reshape(-1, 3)[0].astype(float)
        anchor_v = cluster_v + np.cross(cluster_w, anchor - cluster_pos)
        params = source_config["support_attachment"]
        stiffness = float(params["linear_stiffness_N_m"])
        damping = float(params["linear_damping_N_s_m"])
        force = implicit_spring(anchor - support_pos, support_v - anchor_v,
                                stiffness, damping, fruit_inverse_mass)
        relative_rotation = cluster_rot.inv() * support_rot
        rotation_error = (
            relative_rotation * Rotation.from_quat(support_state["rest_relative_quaternion_xyzw"]).inv()
        ).as_rotvec()
        relative_w = support_w - cluster_w
        couple = implicit_spring(
            -rotation_error, relative_w,
            float(params["torsional_stiffness_Nm_rad"]),
            float(params["torsional_damping_Nm_s_rad"]), fruit_inverse_inertia)
        support_state["peak_unclamped_force_n"] = max(
            support_state["peak_unclamped_force_n"], float(np.linalg.norm(force)))
        force = bound_wrench(force, 8.0)
        couple = bound_wrench(couple, 0.2)
        support_state["peak_force_n"] = max(support_state["peak_force_n"], float(np.linalg.norm(force)))
        apply_rigid_wrench(support_body, force, couple, support_pos)
        apply_rigid_wrench(support_parent_body, -force, -couple, anchor)

    touch_fruit_runtime = {
        name: {"contact_samples": 0, "activated": False,
               "peak_contact_force_n": 0.0, "activation_step": None,
               "activation_time_s": None, "mass_kg_before_kg": fruit_mass_kg,
               "mass_kg_after_kg": fruit_mass_kg,
               "peak_rigid_attachment_force_n": 0.0,
               "candidate_surface_hit_samples": 0,
               "candidate_surface_peak_force_n": 0.0,
               "branch_anchor_slip_at_activation_m": None}
        for name in touch_proxy_metadata
    }
    touch_activation_events = []
    touch_contact_force_threshold_n = 0.05
    touch_contact_required_samples = 2

    def apply_compliant_touch_proxy_attachments() -> None:
        """Keep the separately modeled support fruit attached until it becomes soft."""
        if not args.soft_fruit_on_hand_contact:
            return
        from scipy.spatial.transform import Rotation

        for fruit_name, state in touch_fruit_runtime.items():
            if state["activated"]:
                continue
            meta = next(item for item in soft_fruit_metadata if item["name"] == fruit_name)
            if touch_proxy_metadata[fruit_name]["proxy_mode"] != "support_actor":
                continue
            proxy = touch_proxy_bodies[fruit_name]
            parent = (distal_body if meta["attachment_branch_path"] == "/World/TomatoClusterDistal"
                      else cluster_body)
            parent_mass = (distal_cluster_mass_kg
                           if parent is distal_body else cluster_mass_kg)
            parent_pos = _numpy(parent.get_world_pose()[0]).reshape(-1, 3)[0].astype(float)
            parent_q = _numpy(parent.get_world_pose()[1]).reshape(-1, 4)[0].astype(float)
            parent_rot = Rotation.from_quat(parent_q[[1, 2, 3, 0]])
            proxy_pos = _numpy(proxy.get_world_pose()[0]).reshape(-1, 3)[0].astype(float)
            proxy_q = _numpy(proxy.get_world_pose()[1]).reshape(-1, 4)[0].astype(float)
            proxy_rot = Rotation.from_quat(proxy_q[[1, 2, 3, 0]])
            branch_anchor = parent_pos + parent_rot.apply(
                np.asarray(meta["branch_anchor_local_m"], dtype=float))
            fruit_local_anchor = np.asarray(meta["stem_attachment_local_m"], dtype=float)
            fruit_anchor = proxy_pos + proxy_rot.apply(fruit_local_anchor)
            parent_v = _numpy(parent.get_linear_velocity()).reshape(-1, 3)[0].astype(float)
            parent_w = _numpy(parent.get_angular_velocity()).reshape(-1, 3)[0].astype(float)
            proxy_v = _numpy(proxy.get_linear_velocity()).reshape(-1, 3)[0].astype(float)
            proxy_w = _numpy(proxy.get_angular_velocity()).reshape(-1, 3)[0].astype(float)
            branch_anchor_v = parent_v + np.cross(parent_w, branch_anchor - parent_pos)
            fruit_anchor_v = proxy_v + np.cross(proxy_w, fruit_anchor - proxy_pos)
            stiffness = float(source_attachment_config["lateral_stiffness_N_m"])
            damping = float(source_attachment_config["lateral_damping_N_s_m"])
            force = implicit_spring(
                branch_anchor - fruit_anchor, fruit_anchor_v - branch_anchor_v,
                stiffness, damping, 1.0 / fruit_mass_kg + 1.0 / parent_mass)
            touch_fruit_runtime[fruit_name]["peak_rigid_attachment_force_n"] = max(
                touch_fruit_runtime[fruit_name]["peak_rigid_attachment_force_n"],
                float(np.linalg.norm(force)))
            force = bound_wrench(force, 8.0)
            apply_rigid_wrench(proxy, force, np.zeros(3), fruit_anchor)
            apply_rigid_wrench(parent, -force, np.zeros(3), branch_anchor)

    def activate_touch_soft_fruit(fruit_name: str, step: int) -> None:
        """Swap a touched rigid proxy for its attached, pre-positioned soft fruit."""
        from scipy.spatial.transform import Rotation

        meta = next(item for item in soft_fruit_metadata if item["name"] == fruit_name)
        proxy = touch_proxy_bodies[fruit_name]
        proxy_mode = touch_proxy_metadata[fruit_name]["proxy_mode"]
        collider_prim = stage.GetPrimAtPath(touch_proxy_metadata[fruit_name]["proxy_collision_path"])
        UsdPhysics.CollisionAPI.Apply(collider_prim).CreateCollisionEnabledAttr().Set(False)
        visual_prim = stage.GetPrimAtPath(touch_proxy_metadata[fruit_name]["proxy_visual_path"])
        UsdGeom.Imageable(visual_prim).MakeInvisible()
        stage.GetPrimAtPath(meta["root_path"]).GetAttribute(
            "omniphysics:deformableBodyEnabled").Set(True)
        attachment = stage.DefinePrim(meta["attachment_path"], "OmniPhysicsVtxXformAttachment")
        attachment.GetRelationship("omniphysics:src0").SetTargets(
            [Sdf.Path(meta["simulation_mesh_path"])])
        attachment.GetRelationship("omniphysics:src1").SetTargets(
            [Sdf.Path(meta["attachment_branch_path"])])
        attachment.GetAttribute("omniphysics:attachmentEnabled").Set(True)
        attachment.GetAttribute("omniphysics:vtxIndicesSrc0").Set(
            [int(index) for index in meta["attachment_vertex_indices"]])
        attachment.GetAttribute("omniphysics:localPositionsSrc1").Set([
            Gf.Vec3f(*map(float, point))
            for point in meta["attachment_local_positions_in_branch_m"]])
        attachment.GetAttribute("omniphysics:stiffness").Set(
            float("inf") if args.soft_grasp_target and meta["name"] == target_key else 1200.0)
        attachment.GetAttribute("omniphysics:damping").Set(
            0.0 if args.soft_grasp_target and meta["name"] == target_key else 2.0)
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{meta['root_path']}/Visual")).MakeVisible()

        state = touch_fruit_runtime[fruit_name]
        state["activated"] = True
        state["activation_step"] = int(step)
        activation_time_s = float(args.replay_start_sec + step / 120.0)
        state["activation_time_s"] = activation_time_s
        parent = (distal_body if meta["attachment_branch_path"] == "/World/TomatoClusterDistal"
                  else cluster_body)
        if touch_proxy_metadata[fruit_name]["proxy_mode"] == "compound_branch_fruit":
            parent.set_mass(max(1e-5, parent.get_mass() - fruit_mass_kg))
        proxy_pos = _numpy(proxy.get_world_pose()[0]).reshape(-1, 3)[0].astype(float)
        proxy_q = _numpy(proxy.get_world_pose()[1]).reshape(-1, 4)[0].astype(float)
        proxy_rot = Rotation.from_quat(proxy_q[[1, 2, 3, 0]])
        parent_pos = _numpy(parent.get_world_pose()[0]).reshape(-1, 3)[0].astype(float)
        parent_q = _numpy(parent.get_world_pose()[1]).reshape(-1, 4)[0].astype(float)
        parent_rot = Rotation.from_quat(parent_q[[1, 2, 3, 0]])
        expected_anchor = parent_pos + parent_rot.apply(
            np.asarray(meta["branch_anchor_local_m"], dtype=float))
        fruit_anchor_now = proxy_pos + proxy_rot.apply(
            np.asarray(meta["branch_offset_local_m"], dtype=float))
        state["branch_anchor_slip_at_activation_m"] = float(
            np.linalg.norm(expected_anchor - fruit_anchor_now))
        event = {"fruit": fruit_name, "physics_step": int(step),
                 "time_s": activation_time_s,
                 "trigger": "sustained fingertip contact with original rigid fruit collider",
                 "contact_force_n": float(state["peak_contact_force_n"]),
                 "mass_kg_before": float(fruit_mass_kg),
                 "mass_kg_after": float(fruit_mass_kg),
                 "mass_parent_before_kg": None,
                 "mass_parent_after_kg": None,
                 "mass_owner_before": "attached volume deformable body",
                 "mass_owner_after": "attached volume deformable body",
                 "branch_anchor_slip_m": state["branch_anchor_slip_at_activation_m"]}
        touch_activation_events.append(event)
        print(f"[Isaac tomato] touch-gated soft activation: {event}", flush=True)

    branch_state = {"rest_relative_quaternion_xyzw": None, "peak_anchor_error_m": 0.0,
                    "peak_bend_deg": 0.0, "peak_force_n": 0.0, "limit_projection_count": 0}
    branch_trace = []
    replay_force_trace = []
    replay_contact_object_peaks = {}
    soft_fruit_trace = []
    soft_fruit_node_position_trace = []
    lift_probe_state = {"initial_anchor_world_m": None, "peak_clamp_force_n": 0.0}
    branch_anchor_local = np.array([float((stem_min[0] + stem_max[0]) * 0.5),
                                    float(branch_axis_sign * branch_split_y_m),
                                    float((stem_min[2] + stem_max[2]) * 0.5)])
    lift_probe_anchor_local = branch_anchor_local.copy()
    if args.branch_lift_location == "front":
        lift_probe_anchor_local[1] = float(branch_axis_sign * min(0.045, branch_split_y_m * 0.55))

    def apply_flexible_branch() -> None:
        """Connect two fruit-bearing rigid truss segments with a damped bend."""
        if distal_body is None:
            return
        from scipy.spatial.transform import Rotation

        proximal_pos, proximal_q = cluster_body.get_world_pose()
        distal_pos, distal_q = distal_body.get_world_pose()
        proximal_pos = _numpy(proximal_pos).reshape(3).astype(float)
        distal_pos = _numpy(distal_pos).reshape(3).astype(float)
        proximal_q = _numpy(proximal_q).reshape(4).astype(float)
        distal_q = _numpy(distal_q).reshape(4).astype(float)
        proximal_rot = Rotation.from_quat(proximal_q[[1, 2, 3, 0]])
        distal_rot = Rotation.from_quat(distal_q[[1, 2, 3, 0]])
        if branch_state["rest_relative_quaternion_xyzw"] is None:
            branch_state["rest_relative_quaternion_xyzw"] = (proximal_rot.inv() * distal_rot).as_quat()
        proximal_anchor = proximal_pos + proximal_rot.apply(branch_anchor_local)
        distal_anchor = distal_pos + distal_rot.apply(branch_anchor_local)
        proximal_v = _numpy(cluster_body.get_linear_velocity()).reshape(3).astype(float)
        distal_v = _numpy(distal_body.get_linear_velocity()).reshape(3).astype(float)
        proximal_w = _numpy(cluster_body.get_angular_velocity()).reshape(3).astype(float)
        distal_w = _numpy(distal_body.get_angular_velocity()).reshape(3).astype(float)
        relative_v = (distal_v + np.cross(distal_w, distal_anchor - distal_pos)
                      - proximal_v - np.cross(proximal_w, proximal_anchor - proximal_pos))
        error = proximal_anchor - distal_anchor
        force = implicit_spring(error, relative_v, 3000.0, 20.0,
                                1.0 / cluster_mass_kg + 1.0 / distal_cluster_mass_kg)
        relative_rotation = ((proximal_rot.inv() * distal_rot) *
                             Rotation.from_quat(branch_state["rest_relative_quaternion_xyzw"]).inv())
        rotation_error = relative_rotation.as_rotvec()
        relative_quat = relative_rotation.as_quat()
        twist_quat = np.array([0.0, relative_quat[1], 0.0, relative_quat[3]])
        twist_norm = float(np.linalg.norm(twist_quat))
        twist_quat = twist_quat / twist_norm if twist_norm > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])
        swing = Rotation.from_quat(relative_quat) * Rotation.from_quat(twist_quat).inv()
        bend_deg = float(np.degrees(np.linalg.norm(swing.as_rotvec())))
        relative_w = distal_w - proximal_w
        # Resist large impact-driven swings while retaining a small gravity sag.
        couple = implicit_spring(-rotation_error, relative_w, 2.5, 0.1,
                                 1.0 / max(1e-5, 0.001 * cluster_mass_kg) +
                                 1.0 / max(1e-5, 0.001 * distal_cluster_mass_kg))
        force = bound_wrench(force, 80.0)
        couple = bound_wrench(couple, 2.0)
        branch_state["peak_anchor_error_m"] = max(branch_state["peak_anchor_error_m"],
                                                    float(np.linalg.norm(error)))
        branch_state["peak_bend_deg"] = max(branch_state["peak_bend_deg"], bend_deg)
        branch_state["peak_force_n"] = max(branch_state["peak_force_n"], float(np.linalg.norm(force)))
        apply_rigid_wrench(distal_body, force, couple, distal_anchor)
        apply_rigid_wrench(cluster_body, -force, -couple, proximal_anchor)

    def apply_branch_lift_probe(step: int) -> None:
        if not args.branch_lift_probe:
            return
        from scipy.spatial.transform import Rotation

        position, quaternion = cluster_body.get_world_pose()
        position = _numpy(position).reshape(3).astype(float)
        quaternion = _numpy(quaternion).reshape(4).astype(float)
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        anchor = position + rotation.apply(lift_probe_anchor_local)
        lift_probe_state["final_anchor_world_m"] = anchor.astype(float).tolist()
        if lift_probe_state["initial_anchor_world_m"] is None:
            lift_probe_state["initial_anchor_world_m"] = anchor.copy()
        time_s = step / 120.0
        if time_s < 1.0:
            return
        lift_m = 0.08 * np.clip((time_s - 1.0) / 2.0, 0.0, 1.0)
        target = lift_probe_state["initial_anchor_world_m"] + np.array([0.0, 0.0, lift_m])
        linear_v = _numpy(cluster_body.get_linear_velocity()).reshape(3).astype(float)
        angular_v = _numpy(cluster_body.get_angular_velocity()).reshape(3).astype(float)
        anchor_v = linear_v + np.cross(angular_v, anchor - position)
        target_v = np.array([0.0, 0.0, 0.04 if 1.0 <= time_s < 3.0 else 0.0])
        force = 600.0 * (target - anchor) + 12.0 * (target_v - anchor_v)
        lift_probe_state["peak_clamp_force_n"] = max(
            lift_probe_state["peak_clamp_force_n"], float(np.linalg.norm(force)))
        apply_rigid_wrench(cluster_body, force, np.zeros(3), anchor)

    def enforce_branch_bend_limit() -> float:
        """Project rare solver overshoot back inside the 4.8 degree swing cone."""
        if distal_body is None or branch_state["rest_relative_quaternion_xyzw"] is None:
            return 0.0
        from scipy.spatial.transform import Rotation

        proximal_q = _numpy(cluster_body.get_world_pose()[1]).reshape(4).astype(float)
        distal_pos, distal_q = distal_body.get_world_pose()
        distal_q = _numpy(distal_q).reshape(4).astype(float)
        proximal_rot = Rotation.from_quat(proximal_q[[1, 2, 3, 0]])
        distal_rot = Rotation.from_quat(distal_q[[1, 2, 3, 0]])
        rest = Rotation.from_quat(branch_state["rest_relative_quaternion_xyzw"])
        delta = proximal_rot.inv() * distal_rot * rest.inv()
        q = delta.as_quat()
        twist_q = np.array([0.0, q[1], 0.0, q[3]])
        norm = float(np.linalg.norm(twist_q))
        twist_q = twist_q / norm if norm > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])
        twist = Rotation.from_quat(twist_q)
        swing = delta * twist.inv()
        swing_vec = swing.as_rotvec()
        angle = float(np.linalg.norm(swing_vec))
        limit = np.deg2rad(4.8)
        if angle > limit:
            swing = Rotation.from_rotvec(swing_vec * (limit / angle))
            corrected_distal = proximal_rot * swing * twist * rest
            quat_xyzw = corrected_distal.as_quat()
            distal_body.set_world_pose(
                position=_numpy(distal_pos).reshape(3),
                orientation=quat_xyzw[[3, 0, 1, 2]],
            )
            branch_state["limit_projection_count"] += 1
            angle = limit
        branch_state["peak_bend_deg"] = max(
            branch_state["peak_bend_deg"], float(np.degrees(angle)))
        return float(np.degrees(angle))

    def apply_compliant_pedicel() -> None:
        """Apply the source task's compliant pedicel wrench to both bodies."""
        from scipy.spatial.transform import Rotation

        target_pos, target_q = tomato_body.get_world_pose()
        cluster_pos, cluster_q = cluster_body.get_world_pose()
        target_pos = _numpy(target_pos).reshape(-1, 3)[0].astype(float)
        cluster_pos = _numpy(cluster_pos).reshape(-1, 3)[0].astype(float)
        target_q = _numpy(target_q).reshape(-1, 4)[0].astype(float)
        cluster_q = _numpy(cluster_q).reshape(-1, 4)[0].astype(float)
        target_rot = Rotation.from_quat(target_q[[1, 2, 3, 0]])
        cluster_rot = Rotation.from_quat(cluster_q[[1, 2, 3, 0]])
        target_offset = target_rot.apply(np.asarray(target_attachment, dtype=float))
        cluster_offset = cluster_rot.apply(np.asarray(vine_attachment, dtype=float))
        target_anchor, cluster_anchor = target_pos + target_offset, cluster_pos + cluster_offset
        if pedicel_state["rest_quaternion_xyzw"] is None:
            # Use the imported reset pose: the USD root rotations differ from
            # the source Newton asset frame and otherwise create preload torque.
            pedicel_state["rest_quaternion_xyzw"] = (cluster_rot.inv() * target_rot).as_quat()
            source_axis = np.asarray(source_attachment_config["break_direction_world"], dtype=float)
            pedicel_state["break_axis_local"] = cluster_rot.inv().apply(source_axis / np.linalg.norm(source_axis))

        force = np.zeros(3, dtype=float)
        couple = np.zeros(3, dtype=float)
        if not pedicel_state["broken"]:
            target_v = _numpy(tomato_body.get_linear_velocity()).reshape(-1, 3)[0].astype(float)
            target_w = _numpy(tomato_body.get_angular_velocity()).reshape(-1, 3)[0].astype(float)
            cluster_v = _numpy(cluster_body.get_linear_velocity()).reshape(-1, 3)[0].astype(float)
            cluster_w = _numpy(cluster_body.get_angular_velocity()).reshape(-1, 3)[0].astype(float)
            point_v = target_v + np.cross(target_w, target_offset) - cluster_v - np.cross(cluster_w, cluster_offset)
            axis = cluster_rot.apply(pedicel_state["break_axis_local"])
            error = cluster_anchor - target_anchor
            axial = float(np.dot(error, axis))
            lateral = error - axial * axis
            axial_v = float(np.dot(point_v, axis))
            lateral_v = point_v - axial_v * axis
            force = (
                implicit_spring(
                    np.asarray([axial]), np.asarray([axial_v]),
                    float(source_attachment_config["axial_stiffness_N_m"]),
                    float(source_attachment_config["axial_damping_N_s_m"]),
                    fruit_inverse_mass)[0] * axis
                + implicit_spring(
                    lateral, lateral_v,
                    float(source_attachment_config["lateral_stiffness_N_m"]),
                    float(source_attachment_config["lateral_damping_N_s_m"]),
                    fruit_inverse_mass)
            )
            relative_rotation = cluster_rot.inv() * target_rot
            rotation_error = (relative_rotation * Rotation.from_quat(pedicel_state["rest_quaternion_xyzw"]).inv()).as_rotvec()
            relative_w = target_w - cluster_w
            twist = float(np.dot(rotation_error, axis))
            bend_vec = rotation_error - twist * axis
            twist_v = float(np.dot(relative_w, axis))
            bend_v = relative_w - twist_v * axis
            couple = (
                implicit_spring(
                    -bend_vec, bend_v,
                    float(source_attachment_config["bending_stiffness_Nm_rad"]),
                    float(source_attachment_config["bending_damping_Nm_s_rad"]),
                    fruit_inverse_inertia)
                + implicit_spring(
                    np.asarray([-twist]), np.asarray([twist_v]),
                    float(source_attachment_config["torsional_stiffness_Nm_rad"]),
                    float(source_attachment_config["torsional_damping_Nm_s_rad"]),
                    fruit_inverse_inertia)[0] * axis
            )
            force = bound_wrench(force, 8.0)
            couple = bound_wrench(couple, 0.2)
            tensile = max(0.0, -float(np.dot(force, axis)))
            raw_load = tensile / float(source_attachment_config["break_force_N"])
            alpha = 1.0 - np.exp(-pedicel_dt / float(source_attachment_config["break_ema_time_constant_s"]))
            pedicel_state["filtered_load"] += alpha * (raw_load - pedicel_state["filtered_load"])
            pedicel_state["peak_force_load"] = max(pedicel_state["peak_force_load"], raw_load)
            pedicel_state["peak_tensile_force_n"] = max(pedicel_state["peak_tensile_force_n"], tensile)

            # The source task arms breakage only after a stable grasp and an
            # intentional harvest motion. Contact and commanded arm motion are
            # the observable Isaac/T-Rex equivalents for this rollout.
            grasp_side = "right" if args.soft_grasp_target else "left"
            grasp_sensor_offset = 5 if grasp_side == "right" else 0
            grasp_contact = float(np.max(
                last_sensor_force_norm[grasp_sensor_offset:grasp_sensor_offset + 5], initial=0.0)) >= 0.03
            pedicel_state["left_contact_duration_s"] = (
                pedicel_state["left_contact_duration_s"] + pedicel_dt if grasp_contact else 0.0
            )
            commanded_motion = 0.0
            if trex_active_targets is not None:
                q_now = _numpy(robot.get_joint_positions()).reshape(-1).astype(float)
                grasp_arm = [robot.dof_names.index(f"{grasp_side}_fr3v2_joint{i}") for i in range(1, 8)]
                commanded_motion = float(np.max(np.abs(trex_active_targets[0, grasp_arm] - q_now[grasp_arm])))
            elif pi05_active_targets is not None:
                q_now = _numpy(robot.get_joint_positions()).reshape(-1).astype(float)
                grasp_arm = [robot.dof_names.index(f"{grasp_side}_fr3v2_joint{i}") for i in range(1, 8)]
                commanded_motion = float(np.max(np.abs(pi05_active_targets[0, grasp_arm] - q_now[grasp_arm])))
            elif replay is not None:
                earlier_index = int(np.searchsorted(
                    replay[0], max(float(replay[0][0]), args.replay_start_sec + index / 120.0 - 0.12),
                    side="right") - 1)
                grasp_arm = [robot.dof_names.index(f"{grasp_side}_fr3v2_joint{i}") for i in range(1, 8)]
                commanded_motion = float(np.max(np.abs(
                    target_positions[grasp_arm] - replay[1][earlier_index, grasp_arm])))
            branch_probe_grasp = bool(args.branch_lift_probe and index >= 120)
            armed = bool(branch_probe_grasp or (
                pedicel_state["left_contact_duration_s"] >= 0.08 and commanded_motion >= 0.02
            ))
            pedicel_state["armed"] = armed
            if armed and pedicel_state["filtered_load"] >= 1.0:
                pedicel_state["overload_duration_s"] += pedicel_dt
            else:
                pedicel_state["overload_duration_s"] = 0.0
            if pedicel_state["overload_duration_s"] + 1e-12 >= float(source_attachment_config["minimum_break_overload_s"]):
                pedicel_state["broken"] = True
                pedicel_state["break_time_s"] = float(index * pedicel_dt)
                force.fill(0.0)
                couple.fill(0.0)

        apply_rigid_wrench(tomato_body, force, couple, target_anchor)
        apply_rigid_wrench(cluster_body, -force, -couple, cluster_anchor)

    for index in range(args.steps):
        frame_deformation = last_frame_deformation.copy() if args.task_eval else np.zeros_like(deformation)
        frame_force_norm = np.zeros(10, dtype=np.float32)
        touch_contact_force_by_name = {name: 0.0 for name in touch_proxy_metadata}
        if replay is not None:
            sample_index = int(np.clip(np.searchsorted(replay[0], args.replay_start_sec + index / 120.0, side="right") - 1,
                                       0, len(replay[0]) - 1))
            target_positions = replay[1][sample_index]
            robot.apply_action(ArticulationAction(
                joint_positions=torch.as_tensor(target_positions, device="cuda:0")))
        apply_flexible_branch()
        apply_branch_lift_probe(index)
        apply_compliant_support_attachment()
        apply_compliant_touch_proxy_attachments()
        apply_compliant_pedicel()
        render_now = index % args.render_every == 0
        # Move the camera sensor prims after physics advances and before the
        # corresponding render, matching the link-attached visual geometry.
        sim.step(render=(render_now and not trex_cameras))
        if replay is not None and index == 0:
            first_transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
            report["replay_first_palm_normals_world"] = {}
            report["replay_first_link_poses_xyzw"] = {}
            for side in ("left", "right"):
                palm_pose = first_transforms[robot._articulation_view.get_link_index(f"{side}_hand_C_MC")]
                palm_normal = Rotation.from_quat(palm_pose[3:]).apply((1.0, 0.0, 0.0))
                report["replay_first_palm_normals_world"][side] = {
                    "normal_xyz": palm_normal.tolist(),
                    "angle_from_down_deg": float(np.degrees(np.arccos(np.clip(-palm_normal[2], -1.0, 1.0)))),
                }
                report["replay_first_link_poses_xyzw"][side] = {
                    link: first_transforms[robot._articulation_view.get_link_index(name)].tolist()
                    for link, name in (
                        ("link8", f"{side}_fr3v2_link8"),
                        ("hand", f"{side}_hand_C_MC"),
                        ("index_fingertip", f"{side}_index_DP"),
                    )
                }
        if render_now and trex_cameras:
            transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
            episode_time_s = (args.replay_start_sec + (index + 1) / 120.0
                              if replay is not None else (index + 1) / 120.0)
            update_trex_wrist_camera_poses(transforms, episode_time_s=episode_time_s)
            sim.render()
        enforce_branch_bend_limit()
        if distal_body is not None and index % 4 == 0:
            proximal_pos = _numpy(cluster_body.get_world_pose()[0]).reshape(3)
            distal_pos = _numpy(distal_body.get_world_pose()[0]).reshape(3)
            target_pos = _numpy(tomato_body.get_world_pose()[0]).reshape(3)
            proximal_quat = _numpy(cluster_body.get_world_pose()[1]).reshape(4)
            distal_quat = _numpy(distal_body.get_world_pose()[1]).reshape(4)
            from scipy.spatial.transform import Rotation
            proximal_rot = Rotation.from_quat(proximal_quat[[1, 2, 3, 0]])
            distal_rot = Rotation.from_quat(distal_quat[[1, 2, 3, 0]])
            relative_rotation = proximal_rot.inv() * distal_rot
            if branch_state["rest_relative_quaternion_xyzw"] is not None:
                relative_rotation *= Rotation.from_quat(branch_state["rest_relative_quaternion_xyzw"]).inv()
            relative_quat = relative_rotation.as_quat()
            twist_quat = np.array([0.0, relative_quat[1], 0.0, relative_quat[3]])
            twist_norm = float(np.linalg.norm(twist_quat))
            twist_quat = twist_quat / twist_norm if twist_norm > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])
            swing = Rotation.from_quat(relative_quat) * Rotation.from_quat(twist_quat).inv()
            swing_bend_deg = float(np.degrees(np.linalg.norm(swing.as_rotvec())))
            branch_trace.append([index / 120.0, *proximal_pos.tolist(), *distal_pos.tolist(),
                                 *target_pos.tolist(), swing_bend_deg])
        if replay is not None and args.replay_geometry_probe and index % 4 == 0:
            transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
            replay_probe_times.append(args.replay_start_sec + index / 120.0)
            replay_probe_positions.append(transforms[replay_probe_indices, :3].copy())
        video_stride = 16 if args.task_eval else 8
        if rollout_video is not None and index % video_stride == 0:
            video_frame = (camera.get_rgba() if args.trex_closed_loop else
                           replay_camera.get_rgba() if replay_camera is not None else camera.get_rgba())
            if video_frame is not None:
                rgb = np.asarray(video_frame)[..., :3].astype(np.uint8)
                if rgb.ndim == 3 and rgb.size:
                    rollout_video.write(cv2.cvtColor(cv2.resize(rgb, (640, 400)), cv2.COLOR_RGB2BGR))
        if index in {0, args.steps - 1}:
            link_transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
            report[f"right_index_body_pose_step_{index}"] = link_transforms[link_indices[6]].tolist()
            report[f"index_and_base_link_poses_step_{index}"] = {
                name: link_transforms[robot._articulation_view.get_link_index(name)].tolist()
                for name in (
                    "left_index_DP", "right_index_DP",
                    "left_fr3v2_link0", "right_fr3v2_link0",
                )
            }
            report[f"body_positions_step_{index}"] = {
                name: _numpy(body.get_world_pose()[0]).reshape(-1, 3)[0].astype(float).tolist()
                for name, body in tracked_body_items
            }
        if proof_lab_sensor is not None:
            proof_lab_sensor.update(1.0 / 120.0, force_recompute=True)
            proof_data = proof_lab_sensor.data
            lab_forces = _numpy(proof_data.net_forces_w).reshape(-1, 3)
            if lab_forces.size:
                proof_lab_peak_force = np.maximum(proof_lab_peak_force, np.max(np.abs(lab_forces), axis=0))
                proof_contact_peak = max(proof_contact_peak, int(np.linalg.norm(lab_forces) > 1e-3))
            proof_position, _ = proof_body.get_world_pose()
            proof_min_height = min(proof_min_height, float(_numpy(proof_position).reshape(-1, 3)[0, 2]))
            if proof_data.contact_pos_w is not None and proof_data.force_matrix_w is not None:
                points = _numpy(proof_data.contact_pos_w).reshape(-1, 3)
                forces = _numpy(proof_data.force_matrix_w).reshape(-1, 3)
                friction = (_numpy(proof_data.friction_forces_w).reshape(-1, 3)
                            if proof_data.friction_forces_w is not None else np.zeros_like(forces))
                forces = forces + friction
                valid = np.isfinite(points).all(axis=1) & np.isfinite(forces).all(axis=1)
                proof_origin = _numpy(proof_data.pos_w).reshape(-1, 3)[0]
                proof_quat = _numpy(proof_data.quat_w).reshape(-1, 4)[0]
                proof_rotation = quaternion_wxyz_to_matrix(proof_quat)
                points_for_map = points[valid]
                forces_for_map = forces[valid]
                if len(points_for_map) == 0 and lab_forces.size and np.linalg.norm(lab_forces[0]) > 1e-3:
                    # Some static ground meshes provide net force but no
                    # filtered point buffer. Use the pad center for this
                    # pipeline self-test; real filtered object contacts use
                    # the reported contact centroid above.
                    points_for_map = proof_origin.reshape(1, 3)
                    forces_for_map = lab_forces[:1]
                if len(points_for_map):
                    proof_reading = synthesize_tactile_reading(
                        positions_world_m=points_for_map,
                        impulses_world_ns=forces_for_map / 120.0,
                        sensor_position_world_m=proof_origin,
                        sensor_rotation_world_from_local=proof_rotation,
                        physics_dt_s=1.0 / 120.0,
                        geometry=geometry,
                    )
                    proof_tactile_peak = max(proof_tactile_peak, int(proof_reading.deformation.max()))
        if index in {0, args.steps // 2, args.steps - 1}:
            frame = camera.get_rgba()
            if frame is not None:
                snapshots.append((index, np.asarray(frame)[..., :3].astype(np.uint8).copy()))
        for sensor_index, sensor in (enumerate(sensors) if index % sensor_stride == 0 else ()):
            sensor.update(1.0 / 120.0, force_recompute=True)
            data = sensor.data
            force_world = _numpy(data.net_forces_w).reshape(-1, 3)[0]
            friction_by_object = (
                _numpy(data.friction_forces_w).reshape(-1, 3)
                if data.friction_forces_w is not None else np.empty((0, 3), dtype=np.float32)
            )
            if len(friction_by_object):
                force_world = force_world + np.nansum(friction_by_object, axis=0)
            force_norm = float(np.linalg.norm(force_world))
            if args.soft_fruit_on_hand_contact and data.force_matrix_w is not None:
                normal_by_contact = _numpy(data.force_matrix_w).reshape(-1, 3)
                friction_by_contact = (
                    _numpy(data.friction_forces_w).reshape(-1, 3)
                    if data.friction_forces_w is not None else np.zeros_like(normal_by_contact)
                )
                object_forces = normal_by_contact + friction_by_contact
                path_force_by_path = {}
                if len(object_forces) == len(external_contact_paths):
                    path_force_by_path = {
                        path: float(np.linalg.norm(force))
                        for path, force in zip(external_contact_paths, object_forces)
                    }
                    if replay is not None or args.trex_closed_loop:
                        for path, path_force in path_force_by_path.items():
                            if path_force < 0.05:
                                continue
                            key = f"{sensor_index}:{path}"
                            entry = replay_contact_object_peaks.setdefault(key, {
                                "sensor_index": int(sensor_index), "path": path,
                                "peak_force_n": 0.0, "first_time_s": float(args.replay_start_sec + index / 120.0),
                                "last_time_s": float(args.replay_start_sec + index / 120.0),
                            })
                            entry["peak_force_n"] = max(float(entry["peak_force_n"]), path_force)
                            entry["last_time_s"] = float(args.replay_start_sec + index / 120.0)
                    for fruit_name, proxy_meta in touch_proxy_metadata.items():
                        if proxy_meta["proxy_mode"] == "support_actor":
                            contact_force = path_force_by_path.get(proxy_meta["proxy_root_path"], 0.0)
                            touch_contact_force_by_name[fruit_name] = max(
                                touch_contact_force_by_name[fruit_name], contact_force)
                if data.contact_pos_w is not None and len(object_forces):
                    contact_positions = _numpy(data.contact_pos_w).reshape(-1, 3)
                    if len(contact_positions) == len(object_forces):
                        valid = (np.isfinite(contact_positions).all(axis=1)
                                 & np.isfinite(object_forces).all(axis=1))
                        point_force_norm = np.linalg.norm(object_forces, axis=1)
                        for fruit_name, proxy_meta in touch_proxy_metadata.items():
                            if (touch_fruit_runtime[fruit_name]["activated"]
                                    or proxy_meta["proxy_mode"] != "compound_branch_fruit"):
                                continue
                            owner = touch_proxy_bodies[fruit_name]
                            owner_pos = _numpy(owner.get_world_pose()[0]).reshape(-1, 3)[0]
                            owner_q = _numpy(owner.get_world_pose()[1]).reshape(-1, 4)[0]
                            from scipy.spatial.transform import Rotation
                            owner_rot = Rotation.from_quat(owner_q[[1, 2, 3, 0]])
                            meta = next(item for item in soft_fruit_metadata
                                        if item["name"] == fruit_name)
                            proxy_pos = owner_pos + owner_rot.apply(
                                np.asarray(meta["branch_offset_local_m"], dtype=float))
                            distance = np.linalg.norm(contact_positions - proxy_pos[None, :], axis=1)
                            radius = float(source_scene_config["tomato_radius_m"])
                            surface_hit = valid & (np.abs(distance - radius) <= 0.005)
                            if np.any(surface_hit):
                                contact_force = float(np.max(point_force_norm[surface_hit]))
                                touch_fruit_runtime[fruit_name]["candidate_surface_hit_samples"] += 1
                                touch_fruit_runtime[fruit_name]["candidate_surface_peak_force_n"] = max(
                                    touch_fruit_runtime[fruit_name]["candidate_surface_peak_force_n"],
                                    contact_force)
                                if (contact_force < touch_contact_force_threshold_n
                                        and proxy_meta["proxy_root_path"] in path_force_by_path):
                                    contact_force = path_force_by_path[proxy_meta["proxy_root_path"]]
                                touch_contact_force_by_name[fruit_name] = max(
                                    touch_contact_force_by_name[fruit_name], contact_force)
                    elif path_force_by_path:
                        # Filtered aggregate forces identify the compound branch actor;
                        # contact points still localize which rigid fruit was touched.
                        valid_points = np.isfinite(contact_positions).all(axis=1)
                        for fruit_name, proxy_meta in touch_proxy_metadata.items():
                            if (touch_fruit_runtime[fruit_name]["activated"]
                                    or proxy_meta["proxy_mode"] != "compound_branch_fruit"):
                                continue
                            owner = touch_proxy_bodies[fruit_name]
                            owner_pos = _numpy(owner.get_world_pose()[0]).reshape(-1, 3)[0]
                            owner_q = _numpy(owner.get_world_pose()[1]).reshape(-1, 4)[0]
                            from scipy.spatial.transform import Rotation
                            owner_rot = Rotation.from_quat(owner_q[[1, 2, 3, 0]])
                            meta = next(item for item in soft_fruit_metadata
                                        if item["name"] == fruit_name)
                            proxy_pos = owner_pos + owner_rot.apply(
                                np.asarray(meta["branch_offset_local_m"], dtype=float))
                            distance = np.linalg.norm(contact_positions - proxy_pos[None, :], axis=1)
                            radius = float(source_scene_config["tomato_radius_m"])
                            surface_hit = valid_points & (np.abs(distance - radius) <= 0.005)
                            if np.any(surface_hit):
                                touch_fruit_runtime[fruit_name]["candidate_surface_hit_samples"] += 1
                                touch_fruit_runtime[fruit_name]["candidate_surface_peak_force_n"] = max(
                                    touch_fruit_runtime[fruit_name]["candidate_surface_peak_force_n"],
                                    path_force_by_path.get(proxy_meta["proxy_root_path"], 0.0))
                                touch_contact_force_by_name[fruit_name] = max(
                                    touch_contact_force_by_name[fruit_name],
                                    path_force_by_path.get(proxy_meta["proxy_root_path"], 0.0))
            if (replay is not None and args.replay_force_limit_n > 0
                    and replay_safety_stop is None
                    and force_norm > args.replay_force_limit_n):
                force_matrix = (_numpy(data.force_matrix_w).reshape(-1, 3)
                                if data.force_matrix_w is not None else np.empty((0, 3)))
                contact_objects = []
                if len(force_matrix) == len(external_contact_paths):
                    contact_objects = [
                        {"path": path, "force_n": float(np.linalg.norm(force))}
                        for path, force in zip(external_contact_paths, force_matrix)
                        if np.linalg.norm(force) > 1e-3
                    ]
                replay_safety_stop = {
                    "physics_step": index,
                    "time_s": (index + 1) / 120.0,
                    "side": ("left", "right")[sensor_index // 5],
                    "finger": ("thumb", "index", "middle", "ring", "pinky")[sensor_index % 5],
                    "force_n": force_norm,
                    "limit_n": args.replay_force_limit_n,
                    "contact_objects": contact_objects,
                }
                print(f"[Isaac tomato] replay force stop: {replay_safety_stop}", flush=True)
            if (args.trex_closed_loop and trex_safety_stop is None
                    and force_norm > args.trex_max_fingertip_force_n):
                trex_safety_stop = {
                    "physics_step": index,
                    "time_s": (index + 1) / 120.0,
                    "side": ("left", "right")[sensor_index // 5],
                    "finger": ("thumb", "index", "middle", "ring", "pinky")[sensor_index % 5],
                    "force_n": force_norm,
                }
                trex_closed_loop_report["safety_stop"] = trex_safety_stop
                hold = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
                trex_active_targets = np.repeat(hold[None, :], 4, axis=0)
                trex_smoothed_targets = hold.copy()
                robot.apply_action(ArticulationAction(
                    joint_positions=torch.as_tensor(hold, dtype=torch.float32, device="cuda:0")))
                print(f"[Isaac tomato] T-Rex safety stop: {trex_safety_stop}", flush=True)
            if (args.pi05_closed_loop and pi05_safety_stop is None
                    and force_norm > args.pi05_max_fingertip_force_n):
                pi05_safety_stop = {
                    "physics_step": index,
                    "time_s": (index + 1) / 120.0,
                    "side": ("left", "right")[sensor_index // 5],
                    "finger": ("thumb", "index", "middle", "ring", "pinky")[sensor_index % 5],
                    "force_n": force_norm,
                    "limit_n": args.pi05_max_fingertip_force_n,
                }
                pi05_closed_loop_report["safety_stop"] = pi05_safety_stop
                hold = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
                pi05_active_targets = np.repeat(hold[None, :], 4, axis=0)
                pi05_smoothed_targets = hold.copy()
                robot.apply_action(ArticulationAction(
                    joint_positions=torch.as_tensor(hold, dtype=torch.float32, device="cuda:0")))
                print(f"[Isaac tomato] PI05 safety stop: {pi05_safety_stop}", flush=True)
            peak_sensor_contacts[sensor_index] = max(peak_sensor_contacts[sensor_index], int(force_norm > 1e-3))
            peak_sensor_force[sensor_index] = max(peak_sensor_force[sensor_index], force_norm)
            frame_force_norm[sensor_index] = force_norm
            last_sensor_force_norm[sensor_index] = force_norm
            position_world = _numpy(data.pos_w).reshape(-1, 3)[0]
            quaternion_wxyz = _numpy(data.quat_w).reshape(-1, 4)[0]
            rotation = quaternion_wxyz_to_matrix(quaternion_wxyz)
            sensor_position = position_world + rotation @ np.array([0.0, 0.0, 0.02])
            positions, impulses = np.empty((0, 3)), np.empty((0, 3))
            if data.contact_pos_w is not None and data.force_matrix_w is not None:
                contact_positions = _numpy(data.contact_pos_w).reshape(-1, 3)
                normal_forces = _numpy(data.force_matrix_w).reshape(-1, 3)
                friction_forces = (
                    _numpy(data.friction_forces_w).reshape(-1, 3)
                    if data.friction_forces_w is not None else np.zeros_like(normal_forces)
                )
                per_object_forces = normal_forces + friction_forces
                valid = np.isfinite(contact_positions).all(axis=1) & np.isfinite(per_object_forces).all(axis=1)
                positions = contact_positions[valid]
                impulses = per_object_forces[valid] * (1.0 / 120.0)
            if len(positions) == 0 and force_norm > 1e-3:
                positions = sensor_position.reshape(1, 3)
                impulses = force_world.reshape(1, 3) * (1.0 / 120.0)
            reading = synthesize_tactile_reading(
                positions_world_m=positions, impulses_world_ns=impulses,
                sensor_position_world_m=sensor_position,
                sensor_rotation_world_from_local=rotation,
                physics_dt_s=1.0 / 120.0, geometry=geometry,
            )
            if force_norm > 1e-3 and not np.any(reading.deformation):
                # This imported fingertip often reports contact outside the
                # proxy pad's 17 mm XY bounds. Show an unlocalized centered
                # contact instead of a misleading black tactile image.
                proxy_impulse = (rotation @ np.array([0.0, 0.0, force_norm / 120.0])).reshape(1, 3)
                reading.deformation[:] = synthesize_tactile_reading(
                    positions_world_m=sensor_position.reshape(1, 3),
                    impulses_world_ns=proxy_impulse,
                    sensor_position_world_m=sensor_position,
                    sensor_rotation_world_from_local=rotation,
                    physics_dt_s=1.0 / 120.0, geometry=geometry,
                ).deformation
            reading.wrench[:3] = force_world @ rotation
            if len(positions):
                torques = np.cross(positions - sensor_position, impulses / (1.0 / 120.0)) @ rotation
                reading.wrench[3:] = torques.sum(axis=0)
            side_index, finger_index = divmod(sensor_index, 5)
            wrench[side_index, finger_index] = reading.wrench
            deformation[side_index, finger_index] = np.maximum(
                deformation[side_index, finger_index], reading.deformation)
            frame_deformation[side_index, finger_index] = reading.deformation
            action_chunk_peak_by_sensor[sensor_index] = max(
                action_chunk_peak_by_sensor[sensor_index], force_norm)
            if index == args.steps - 1 and sensor_index == 6:
                report["right_index_sensor_frame"] = {
                    "net_force_world_n": force_world.tolist(),
                    "net_force_local_n": (force_world @ rotation).tolist(),
                    "contact_point_count": int(len(positions)),
                    "sensor_position_world_m": sensor_position.tolist(),
                }
        if args.soft_fruit_on_hand_contact and index % sensor_stride == 0:
            for fruit_name, contact_force in touch_contact_force_by_name.items():
                state = touch_fruit_runtime[fruit_name]
                if state["activated"]:
                    continue
                state["peak_contact_force_n"] = max(
                    float(state["peak_contact_force_n"]), float(contact_force))
                if contact_force >= touch_contact_force_threshold_n:
                    state["contact_samples"] += 1
                else:
                    state["contact_samples"] = 0
                if state["contact_samples"] >= touch_contact_required_samples:
                    activate_touch_soft_fruit(fruit_name, index)
        if soft_body_view is not None and index % 4 == 0:
            from scipy.spatial.transform import Rotation

            live_nodes = _numpy(soft_body_view.get_simulation_nodal_positions()).astype(np.float32)
            trace_row = [index / 120.0]
            node_frame = []
            metadata_by_name = {item["name"]: item for item in soft_fruit_metadata}
            for body_index, name in enumerate(soft_body_names):
                meta = metadata_by_name[name]
                node_count = int(soft_rest_nodes.shape[1])
                nodes = live_nodes[body_index, :node_count]
                rest_nodes = soft_rest_nodes[body_index, :node_count]
                active_for_metrics = (not args.soft_fruit_on_hand_contact
                                      or touch_fruit_runtime[name]["activated"])
                displacement = (np.linalg.norm(nodes - rest_nodes, axis=-1)
                                if active_for_metrics else np.zeros(node_count, dtype=np.float32))
                owner = (tomato_body if meta["attachment_branch_path"] == "/World/TargetTomato"
                         else cluster_body if meta["attachment_branch_path"] == "/World/TomatoCluster"
                         else distal_body)
                owner_pos, owner_q = owner.get_world_pose()
                owner_pos = _numpy(owner_pos).reshape(3).astype(float)
                owner_q = _numpy(owner_q).reshape(4).astype(float)
                owner_rot = Rotation.from_quat(owner_q[[1, 2, 3, 0]])
                attachment_indices = np.asarray(meta["attachment_vertex_indices"], dtype=np.int64)
                attachment_local = np.asarray(meta["attachment_local_positions_in_branch_m"], dtype=float)
                attachment_world = owner_pos + owner_rot.apply(attachment_local)
                slip = (float(np.max(np.linalg.norm(
                    nodes[attachment_indices] - attachment_world, axis=1)))
                        if active_for_metrics else 0.0)
                trace_row.extend((float(displacement.max(initial=0.0)),
                                  float(displacement.mean()) if displacement.size else 0.0,
                                  slip))
                node_frame.append(nodes.copy())
            trace_row.append(float(np.max(frame_force_norm, initial=0.0)))
            soft_fruit_trace.append(trace_row)
            soft_fruit_node_position_trace.append(node_frame)
        if index % sensor_stride == 0:
            last_frame_deformation = frame_deformation.copy()
            if replay is not None:
                tomato_pose = _numpy(tomato_body.get_world_pose()[0]).reshape(3)
                replay_force_trace.append([index / 120.0, *frame_force_norm.tolist(),
                                           *tomato_pose.tolist()])
        if args.replay_live_view and index % 8 == 0:
            live_rgba = {
                "head": camera.get_rgba(),
                "wrist_left": trex_cameras["wrist_left"].get_rgba(),
                "wrist_right": trex_cameras["wrist_right"].get_rgba(),
            }
            if all(frame is not None for frame in live_rgba.values()):
                _write_trex_live_preview(
                    live_preview_path,
                    {key: np.asarray(frame)[..., :3].astype(np.uint8)
                     for key, frame in live_rgba.items()},
                    frame_deformation, wrench.reshape(1, 10, 6),
                    args.replay_start_sec + (index + 1) / 120.0,
                    index // 4, replay=True)
        if (args.trex_host or args.pi05_closed_loop) and index % 4 == 3:
            # Preserve temporal order at the 30 Hz policy rate.
            trex_f6_samples.append(wrench.copy())
            trex_deform_samples.append(frame_deformation.copy())
            trex_f6_trace.append(wrench.copy())
            trex_force_trace.append({
                "physics_step": index,
                "time_s": (index + 1) / 120.0,
                "f6": wrench.copy().tolist(),
            })
        if policy is not None:
            joint_pos = _numpy(robot.get_joint_positions()).reshape(-1)
            joint_vel = _numpy(robot.get_joint_velocities()).reshape(-1)
            sim_state_names = []
            sim_state_values = []
            for dof_i, dof_name in enumerate(robot.dof_names):
                sim_state_names.append(_state_name_for_dof(dof_name, "position"))
                sim_state_values.append(float(joint_pos[dof_i]))
            for dof_i, dof_name in enumerate(robot.dof_names):
                sim_state_names.append(_state_name_for_dof(dof_name, "velocity"))
                sim_state_values.append(float(joint_vel[dof_i]))
            state_by_name = dict(zip(sim_state_names, sim_state_values))
            if set(state_by_name) != set(policy_schema.state_names):
                raise ValueError("Imported Isaac state names do not match the checkpoint state schema")
            state = np.asarray([state_by_name[name] for name in policy_schema.state_names], dtype=np.float32)
            if expected_mode == "tactile-force":
                model_input = np.concatenate((state[None], _pool_tactile(deformation[None]), wrench.reshape(1, -1)), axis=1)
            else:
                model_input = state[None]
            x_mean = checkpoint["x_mean"].cpu().numpy()
            x_std = checkpoint["x_std"].cpu().numpy()
            normalized_input = np.clip((model_input - x_mean) / x_std, -10, 10).astype(np.float32)
            with torch.no_grad():
                normalized_action = policy(torch.from_numpy(normalized_input)).cpu().numpy()
            predicted = normalized_action[0] * checkpoint["y_std"].cpu().numpy() + checkpoint["y_mean"].cpu().numpy()
            policy_report["inference_steps"] += 1
            finite = bool(np.isfinite(predicted).all())
            policy_report["finite_outputs"] = policy_report["finite_outputs"] and finite
            if not finite:
                raise RuntimeError("Policy produced non-finite joint targets")
            policy_action_min = np.minimum(policy_action_min, predicted)
            policy_action_max = np.maximum(policy_action_max, predicted)
            policy_action_last = predicted.astype(np.float32)
            if args.policy_apply:
                targets_in_robot_order = predicted[policy_action_indices]
                robot.apply_action(ArticulationAction(
                    joint_positions=torch.as_tensor(targets_in_robot_order, device="cuda:0")))
        if args.trex_closed_loop and index % 4 == 0:
            if index % 16 == 0 and trex_safety_stop is None:
                chunk_action_offset = (index // 4) % 16
                request_mode = "slow_and_fast" if chunk_action_offset == 0 else "fast"
                raw_frames = {
                    "head": camera.get_rgba(),
                    "wrist_left": trex_cameras["wrist_left"].get_rgba(),
                    "wrist_right": trex_cameras["wrist_right"].get_rgba(),
                }
                if any(frame is None for frame in raw_frames.values()):
                    raise RuntimeError("Isaac did not return all three cameras during T-Rex closed-loop")
                frames = {key: np.asarray(frame)[..., :3].astype(np.uint8)
                          for key, frame in raw_frames.items()}
                f6_current = wrench.reshape(10, 6).copy()
                history = [sample.reshape(10, 6) for sample in trex_f6_samples[-16:]]
                if not history:
                    history = [f6_current]
                history = [history[0]] * (16 - len(history)) + history
                f6_history = np.stack(history[-16:]).astype(np.float32)
                deformation_frame = (trex_deform_samples[-1].reshape(10, 240, 240).copy()
                                     if trex_deform_samples else frame_deformation.reshape(10, 240, 240).copy())
                if args.trex_live_view:
                    _write_trex_live_preview(
                        task_output_dir / "trex_live_view" / "latest.png",
                        frames, deformation_frame, f6_history,
                        index / 120.0, index // 4)
                call_dir = trex_output_dir / f"policy_frame_{index // 4:05d}"
                call_dir.mkdir(parents=True, exist_ok=True)
                for camera_key, frame in frames.items():
                    Image.fromarray(frame).save(call_dir / f"trex_isaac_{camera_key}.png")
                np.savez_compressed(call_dir / "trex_isaac_tactile_input.npz",
                                    tactile_f6=f6_history, tactile_deform=deformation_frame)
                client_result = subprocess.run([
                    str(args.trex_python.expanduser()),
                    str(SIM_ROOT / "scripts/trex_isaac_observation_client.py"),
                    "--input-dir", str(call_dir.resolve()),
                    "--host", args.trex_host, "--port", str(args.trex_port),
                    "--task", args.trex_task, "--mode", request_mode,
                ], check=True, capture_output=True, text=True, timeout=240)
                trex_result = json.loads((call_dir / "trex_isaac_policy_result.json").read_text())
                if trex_result["mode"] != request_mode:
                    raise RuntimeError(f"T-Rex returned {trex_result['mode']} for {request_mode} request")
                if request_mode == "slow_and_fast":
                    trex_active_chunk_id = trex_result["chunk_id"]
                    trex_closed_loop_report["slow_requests"] += 1
                else:
                    if trex_result["chunk_id"] != trex_active_chunk_id:
                        raise RuntimeError("T-Rex fast refinement refers to a different action chunk")
                    trex_closed_loop_report["fast_refinements"] += 1
                actions = np.load(call_dir / "trex_isaac_action_chunk.npy").astype(np.float64)
                if actions.shape != (16, 62) or not np.isfinite(actions).all():
                    raise RuntimeError(f"Invalid T-Rex action chunk at physics step {index}")
                ik_failure = None
                ik_reports = {}
                current_before_ik = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
                if request_mode == "slow_and_fast":
                    trex_chunk_start_arm_q = {
                        side: np.asarray([current_before_ik[robot.dof_names.index(
                            f"{side}_fr3v2_joint{i}")] for i in range(1, 8)])
                        for side in ("left", "right")
                    }
                side_target_changes = None
                try:
                    target_chunk, ik_reports, hand_limits, hand_clips = _trex_actions_to_joint_targets(
                        actions[chunk_action_offset:chunk_action_offset + 4],
                        robot, stage, args.trex_checkpoint.expanduser(),
                        adapter_metadata["hand_order"],
                        args.trex_ik_position_tolerance_m,
                        args.trex_ik_rotation_tolerance_deg,
                        args.trex_ik_max_joint_step_rad,
                        chunk_start_arm_q=trex_chunk_start_arm_q)
                    rejected_by_side = {
                        side: [item["rejection_reason"] for item in rows if not item["accepted"]]
                        for side, rows in ik_reports.items()
                    }
                    rejected_by_side = {side: reasons for side, reasons in rejected_by_side.items() if reasons}
                    if rejected_by_side:
                        ik_failure = "; ".join(
                            f"{side}: {reasons[0]}" for side, reasons in rejected_by_side.items())
                    trex_active_targets = target_chunk
                    trex_closed_loop_report["hand_joint_limit_clips"].extend(
                        {"policy_frame": index // 4 + clip["frame"], **clip}
                        for clip in hand_clips)
                    side_target_changes = {}
                    for side in ("left", "right"):
                        arm_names = [f"{side}_fr3v2_joint{i}" for i in range(1, 8)]
                        hand_names = [f"{side}_{name}" for name in adapter_metadata["hand_order"]]
                        arm_indices = [robot.dof_names.index(name) for name in arm_names]
                        hand_indices = [robot.dof_names.index(name) for name in hand_names]
                        side_target_changes[side] = {
                            "arm_max_abs_target_delta_rad": float(np.max(np.abs(
                                target_chunk[0, arm_indices] - current_before_ik[arm_indices]))),
                            "hand_max_abs_target_delta_rad": float(np.max(np.abs(
                                target_chunk[0, hand_indices] - current_before_ik[hand_indices]))),
                        }
                except (RuntimeError, ValueError) as error:
                    ik_failure = str(error)
                    current = current_before_ik.astype(np.float32)
                    trex_active_targets = np.repeat(current[None, :], 4, axis=0)
                primary_reasons = [
                    row["rejection_reason"]
                    for rows in ik_reports.values() for row in rows
                    if row["rejection_reason"] not in (None, "held_after_prior_action_rejection")
                ]
                workcell_rejected = any("workcell" in reason for reason in primary_reasons)
                ik_solver_failed = bool(ik_failure is not None and
                                        (not primary_reasons or any(
                                            "workcell" not in reason for reason in primary_reasons)))
                if ik_failure is not None:
                    rejection_record = {
                        "physics_step": index,
                        "policy_frame": index // 4,
                        "error": ik_failure,
                        "rejected_actions_by_side": ({
                            side: [item["rejection_reason"] for item in rows if not item["accepted"]]
                            for side, rows in ik_reports.items()} if ik_reports else None),
                    }
                    if workcell_rejected:
                        trex_closed_loop_report["workcell_rejections"].append(rejection_record)
                    if ik_solver_failed:
                        trex_closed_loop_report["ik_failures"].append(rejection_record)
                trex_closed_loop_report["finite_outputs"] &= bool(np.isfinite(actions).all())
                # This client only asks the T-Rex service for a prediction;
                # in this caller the action chunk is then IK-converted and
                # sent to Isaac below. Keep each saved per-call result honest.
                if ik_reports:
                    action_exec_frames = sum(
                        any(ik_reports[side][frame]["accepted"] for side in ik_reports)
                        for frame in range(min(len(rows) for rows in ik_reports.values())))
                else:
                    action_exec_frames = 0
                action_was_applied = action_exec_frames > 0
                trex_result["action_applied"] = action_was_applied
                trex_result["execution_status"] = (
                    "applied" if action_was_applied and ik_failure is None else
                    "partially_applied_with_workcell_holds" if action_was_applied and not ik_solver_failed else
                    "partially_applied_with_ik_holds" if action_was_applied else
                    "workcell_rejected_hold_pose" if workcell_rejected and not ik_solver_failed else
                    "ik_rejected_hold_pose")
                trex_result["executed_policy_frames"] = action_exec_frames
                trex_closed_loop_report["action_applied"] |= action_was_applied
                if action_was_applied:
                    trex_closed_loop_report["applied_policy_chunks"] += 1
                else:
                    trex_closed_loop_report["rejected_policy_chunks"] += 1
                trex_result["action_contract"] = (
                    "delta-base link8 xyz+rot6d and absolute hand targets; "
                    "converted by Isaac URDF IK and applied to the articulation"
                )
                (call_dir / "trex_isaac_policy_result.json").write_text(
                    json.dumps(trex_result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                trex_closed_loop_report["replanning_calls"].append({
                    "physics_step": index,
                    "policy_frame": index // 4,
                    "mode": request_mode,
                    "chunk_id": trex_active_chunk_id,
                    "chunk_action_offset": chunk_action_offset,
                    "latency_ms": float(trex_result["latency_ms"]),
                    "action_shape": list(actions.shape),
                    "action_abs_max": float(np.abs(actions).max()),
                    "f6_history_shape": list(f6_history.shape),
                    "f6_history_range": [float(f6_history.min()), float(f6_history.max())],
                    "deformation_shape": list(deformation_frame.shape),
                    "camera_nonblack_fraction": {
                        key: float(np.any(frame > 5, axis=-1).mean())
                        for key, frame in frames.items()},
                    "ik_failed": ik_solver_failed,
                    "workcell_rejected": workcell_rejected,
                    "ik_failure": ik_failure,
                    "ik_max_position_error_m": ({
                        side: max(item["position_error_m"] for item in rows if item["position_error_m"] is not None)
                        for side, rows in ik_reports.items()} if ik_reports else None),
                    "ik_max_rotation_error_deg": ({
                        side: max(item["rotation_error_deg"] for item in rows if item["rotation_error_deg"] is not None)
                        for side, rows in ik_reports.items()} if ik_reports else None),
                    "target_change_by_side_rad": side_target_changes,
                    "hand_joint_count": len(hand_limits) if hand_limits and action_was_applied else None,
                    "execution_status": trex_result["execution_status"],
                    "action_applied": action_was_applied,
                    "executed_policy_frames": action_exec_frames,
                    "ik_rejected_actions_by_side": ({
                        side: sum(not item["accepted"] for item in rows)
                        for side, rows in ik_reports.items()} if ik_reports else None)})
                np.save(task_output_dir / "trex_latest_action_chunk.npy", actions.astype(np.float32))
                print(client_result.stdout, end="", flush=True)
                trex_rejection_streak = trex_rejection_streak + 1 if ik_failure is not None else 0
                if trex_rejection_streak >= args.trex_max_rejection_streak:
                    trex_safety_stop = {
                        "physics_step": index,
                        "time_s": index / 120.0,
                        "reason": "consecutive_ik_or_workcell_rejections",
                        "rejection_streak": trex_rejection_streak,
                        "last_rejection": ik_failure,
                    }
                    trex_closed_loop_report["safety_stop"] = trex_safety_stop
                    hold = current_before_ik.copy()
                    trex_active_targets = np.repeat(hold[None, :], 4, axis=0)
                    trex_smoothed_targets = hold.copy()
                    robot.apply_action(ArticulationAction(
                        joint_positions=torch.as_tensor(hold, dtype=torch.float32, device="cuda:0")))
                    print(f"[Isaac tomato] T-Rex safety stop: {trex_safety_stop}", flush=True)
            if trex_active_targets is None:
                raise RuntimeError("T-Rex closed loop did not initialize an action chunk")
            action_index = min((index % 16) // 4, len(trex_active_targets) - 1)
            requested_targets = np.asarray(trex_active_targets[action_index], dtype=np.float64)
            if trex_smoothed_targets is None:
                trex_smoothed_targets = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
            arm_step = requested_targets[trex_arm_indices] - trex_smoothed_targets[trex_arm_indices]
            smoothing_report = trex_closed_loop_report["arm_command_smoothing"]
            smoothing_report["maximum_requested_arm_step_rad"] = max(
                smoothing_report["maximum_requested_arm_step_rad"], float(np.max(np.abs(arm_step))))
            step_limits = np.tile(trex_arm_step_limits, 2)
            if np.any(np.abs(arm_step) > step_limits):
                smoothing_report["clipped_physics_steps"] += 1
            trex_smoothed_targets[trex_arm_indices] += np.clip(arm_step, -step_limits, step_limits)
            hand_indices = np.setdiff1d(np.arange(len(requested_targets)), trex_arm_indices)
            trex_smoothed_targets[hand_indices] = requested_targets[hand_indices]
            robot.apply_action(ArticulationAction(
                joint_positions=torch.as_tensor(trex_smoothed_targets,
                                                dtype=torch.float32, device="cuda:0")))
        if args.pi05_closed_loop and index % 16 == 0 and pi05_safety_stop is None:
            raw_frames = {
                "base": camera.get_rgba(),
                "left_wrist": trex_cameras["wrist_left"].get_rgba(),
                "right_wrist": trex_cameras["wrist_right"].get_rgba(),
            }
            if any(frame is None for frame in raw_frames.values()):
                raise RuntimeError("Isaac did not return all three RGB frames for PI05")
            frames = {name: np.asarray(frame)[..., :3].astype(np.uint8)
                      for name, frame in raw_frames.items()}
            measured_full = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
            measured_state = measured_full[pi05_joint_indices].astype(np.float32)
            actions, inference_latency_s, server_timing = pi05_client.infer(
                base=frames["base"], left_wrist=frames["left_wrist"],
                right_wrist=frames["right_wrist"], state=measured_state,
                prompt=args.pi05_prompt)
            call_index = len(pi05_closed_loop_report["inference_calls"])
            call_dir = pi05_output_dir / f"inference_{call_index:05d}"
            call_dir.mkdir(parents=True, exist_ok=False)
            for name, frame in frames.items():
                Image.fromarray(frame).save(call_dir / f"{name}.png")
            np.save(call_dir / "measured_state_58d.npy", measured_state)
            np.save(call_dir / "actions_50x58.npy", actions)
            target_chunk = np.repeat(measured_full[None, :], 4, axis=0)
            requested_targets = actions[:4]
            raw_targets = requested_targets.copy()
            lower = pi05_joint_limits[pi05_joint_indices, 0]
            upper = pi05_joint_limits[pi05_joint_indices, 1]
            bounded_targets = np.clip(raw_targets, lower[None, :], upper[None, :])
            for action_frame, joint_index in zip(*np.where(bounded_targets != raw_targets), strict=True):
                pi05_closed_loop_report["joint_limit_clips"].append({
                    "inference": call_index,
                    "frame": int(action_frame),
                    "joint": pi05_joint_names[int(joint_index)],
                    "requested_rad": float(raw_targets[action_frame, joint_index]),
                    "applied_rad": float(bounded_targets[action_frame, joint_index]),
                })
            target_chunk[:, pi05_joint_indices] = bounded_targets
            pi05_active_targets = target_chunk
            pi05_closed_loop_report["action_applied"] = True
            inference_report = {
                "physics_step": index,
                "sim_time_s": index / 120.0,
                "latency_ms": inference_latency_s * 1000.0,
                "server_timing": server_timing,
                "state_shape": list(measured_state.shape),
                "action_shape": list(actions.shape),
                "camera_shapes_hwc": {name: list(frame.shape) for name, frame in frames.items()},
                "camera_nonblack_fraction": {
                    name: float(np.any(frame > 5, axis=-1).mean())
                    for name, frame in frames.items()},
                "action_abs_max": float(np.abs(actions).max()),
                "joint_limit_clips": int(np.count_nonzero(bounded_targets != raw_targets)),
            }
            pi05_closed_loop_report["inference_calls"].append(inference_report)
            (call_dir / "inference.json").write_text(
                json.dumps(inference_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
            print(f"[PI05] inference {call_index}: {inference_latency_s * 1000:.1f} ms, "
                  f"actions={actions.shape}, sim_t={index / 120.0:.2f}s", flush=True)

        if args.pi05_closed_loop:
            if pi05_active_targets is None:
                raise RuntimeError("PI05 closed loop did not initialize an action chunk")
            requested = np.asarray(pi05_active_targets[min((index % 16) // 4, 3)], dtype=np.float64)
            if pi05_smoothed_targets is None:
                pi05_smoothed_targets = _numpy(robot.get_joint_positions()).reshape(-1).astype(np.float64)
            arm_step = requested[pi05_arm_indices] - pi05_smoothed_targets[pi05_arm_indices]
            smoothing = pi05_closed_loop_report["arm_command_smoothing"]
            smoothing["clipped_physics_steps"] += int(np.any(np.abs(arm_step) > pi05_arm_step_limits))
            pi05_smoothed_targets[pi05_arm_indices] += np.clip(
                arm_step, -pi05_arm_step_limits, pi05_arm_step_limits)
            hand_indices = np.setdiff1d(np.arange(len(requested)), pi05_arm_indices)
            hand_step = requested[hand_indices] - pi05_smoothed_targets[hand_indices]
            pi05_smoothed_targets[hand_indices] += pi05_hand_interpolation_alpha * hand_step
            robot.apply_action(ArticulationAction(
                joint_positions=torch.as_tensor(pi05_smoothed_targets,
                                                dtype=torch.float32, device="cuda:0")))

        if action_chunk_policy is not None and index % 4 == 0:
            if index % (replan_frames * 4) == 0:
                transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
                for side, key in (("left", "cam1"), ("right", "cam2")):
                    wrist_index = robot._articulation_view.get_link_index(f"{side}_fr3v2_link8")
                    wrist_pose = transforms[wrist_index]
                    wrist_eye = np.asarray(wrist_pose[:3], dtype=float) + np.array([0.0, 0.0, 0.07])
                    wrist_q_wxyz = _look_at_wxyz(wrist_eye, np.asarray(tomato_start))
                    policy_cameras[key].set_world_pose(position=wrist_eye, orientation=wrist_q_wxyz)
                sim.render()
                image_values = []
                from PIL import Image as _PILImage
                for key in ("cam0", "cam1", "cam2"):
                    rgba = policy_cameras[key].get_rgba()
                    if rgba is None:
                        raise RuntimeError(f"Isaac did not return policy camera {key}")
                    rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
                    image = _PILImage.fromarray(rgb).resize(
                        (int(action_chunk_checkpoint["rgb_size"]), int(action_chunk_checkpoint["rgb_size"])),
                        _PILImage.Resampling.BILINEAR)
                    image_values.append(np.transpose(np.asarray(image), (2, 0, 1)))
                image_batch = np.stack(image_values, axis=0)[None]
                joint_pos = _numpy(robot.get_joint_positions()).reshape(-1)
                joint_vel = _numpy(robot.get_joint_velocities()).reshape(-1)
                state_by_name = {}
                for dof_i, dof_name in enumerate(robot.dof_names):
                    state_by_name[_state_name_for_dof(dof_name, "position")] = float(joint_pos[dof_i])
                    state_by_name[_state_name_for_dof(dof_name, "velocity")] = float(joint_vel[dof_i])
                state = np.asarray([state_by_name[name] for name in policy_schema.state_names], dtype=np.float32)
                touch = np.concatenate((_pool_tactile(frame_deformation[None]), wrench.reshape(1, -1)), axis=1)[0]
                state = np.clip((state - action_chunk_checkpoint["state_mean"].numpy()) /
                                action_chunk_checkpoint["state_std"].numpy(), -10, 10)
                touch = np.clip((touch - action_chunk_checkpoint["touch_mean"].numpy()) /
                                action_chunk_checkpoint["touch_std"].numpy(), -10, 10)
                use_touch = action_chunk_checkpoint["mode"] == "rgb-state-touch"
                with torch.no_grad():
                    normalized_chunk = action_chunk_policy(
                        torch.from_numpy(image_batch).to("cuda:0"),
                        torch.from_numpy(state.astype(np.float32)[None]).to("cuda:0"),
                        torch.from_numpy(touch.astype(np.float32)[None]).to("cuda:0"),
                        use_touch).cpu().numpy()[0]
                action_chunk_values = (normalized_chunk * action_chunk_checkpoint["action_std"].numpy()
                                       + action_chunk_checkpoint["action_mean"].numpy())
                action_chunk_action_min = np.minimum(action_chunk_action_min, action_chunk_values.min(axis=0))
                action_chunk_action_max = np.maximum(action_chunk_action_max, action_chunk_values.max(axis=0))
                action_chunk_report["inference_steps"] += 1
                finite = bool(np.isfinite(action_chunk_values).all())
                action_chunk_report["finite_outputs"] &= finite
                if not finite:
                    raise RuntimeError("Action-chunk policy produced non-finite joint targets")
            if action_chunk_values is None:
                raise RuntimeError("Action-chunk policy did not produce an action chunk")
            action_index = min((index % (replan_frames * 4)) // 4,
                               int(action_chunk_checkpoint["chunk"]) - 1)
            robot_targets = action_chunk_values[action_index, action_chunk_action_indices]
            robot.apply_action(ArticulationAction(
                joint_positions=torch.as_tensor(robot_targets, dtype=torch.float32, device="cuda:0")))
        if args.task_eval:
            tomato_position, _ = tomato_body.get_world_pose()
            action_chunk_tomato_trace.append(_numpy(tomato_position).reshape(-1, 3)[0].astype(float).tolist())
        if replay_safety_stop is not None:
            report["replay_safety_stop"] = replay_safety_stop
            report[f"body_positions_step_{index}"] = {
                name: _numpy(body.get_world_pose()[0]).reshape(-1, 3)[0].astype(float).tolist()
                for name, body in tracked_body_items
            }
            break
    executed_physics_steps = index + 1
    report["executed_physics_steps"] = executed_physics_steps
    rgba = camera.get_rgba()
    if args.trex_camera_preview_only:
        if rgba is not None:
            preview_overview = np.asarray(rgba)[..., :3].astype(np.uint8)
            Image.fromarray(preview_overview).save(args.output / "trex_isaac_overview.png")
        transforms = _numpy(robot._articulation_view._physics_view.get_link_transforms())[0]
        assembly_cameras = {}
        assembly_view_report = {}
        if wrist_camera_calibration.get("mounts"):
            from scipy.spatial.transform import Rotation

            # Add a third-person diagnostic view for each assembly. Looking
            # from the normal of the rod/optical-axis plane makes their
            # measured relative angle visible in the rendered image.
            for side in ("left", "right"):
                camera_key = f"wrist_{side}"
                mount = wrist_camera_calibration["mounts"][camera_key]
                wrist_index = robot._articulation_view.get_link_index(
                    f"{side}_fr3v2_link8")
                wrist_pose = np.asarray(transforms[wrist_index], dtype=float)
                wrist_rotation = Rotation.from_quat(wrist_pose[3:]).as_matrix()
                local_position = np.asarray(mount["position_link8_m"], dtype=float)
                local_anchor = np.asarray(mount["support_rod_anchor_link8_m"], dtype=float)
                local_rod = local_position - local_anchor
                local_rod /= np.linalg.norm(local_rod)
                optical_rotation = Rotation.from_quat(
                    np.asarray(mount["orientation_link8_opengl_xyzw"], dtype=float))
                local_forward = optical_rotation.apply((0.0, 0.0, -1.0))
                world_rod = wrist_rotation @ local_rod
                world_forward = wrist_rotation @ local_forward
                view_normal = np.cross(world_rod, world_forward)
                view_normal /= np.linalg.norm(view_normal)
                # Keep inspection cameras above the floor. A mirrored wrist
                # can otherwise put this side-view camera below the tabletop.
                if view_normal[2] < 0.0:
                    view_normal = -view_normal
                view_direction = view_normal + 0.20 * world_forward
                view_direction /= np.linalg.norm(view_direction)
                target = wrist_pose[:3] + wrist_rotation @ (0.5 * local_position)
                target += wrist_rotation @ np.array([0.0, 0.0, 0.018])
                eye = target + 0.43 * view_direction
                path = f"/World/WristAssembly{side.title()}InspectionCamera"
                inspection_camera = Camera(
                    prim_path=path, name=f"wrist_assembly_{side}_inspection",
                    resolution=(800, 600), position=eye,
                    orientation=_look_at_wxyz(eye, target),
                )
                inspection_camera.initialize()
                inspection_usd_camera = UsdGeom.Camera(stage.GetPrimAtPath(path))
                inspection_usd_camera.GetHorizontalApertureAttr().Set(1.0)
                inspection_usd_camera.GetVerticalApertureAttr().Set(0.75)
                inspection_usd_camera.GetFocalLengthAttr().Set(0.80)
                inspection_usd_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 5.0))
                assembly_cameras[side] = inspection_camera
                assembly_view_report[side] = {
                    "target_world_m": target.tolist(),
                    "eye_world_m": eye.tolist(),
                    "rod_optical_angle_deg": float(mount.get("camera_angle_from_rod_deg", 0.0)),
                }
        final_episode_time_s = (args.replay_start_sec + executed_physics_steps / 120.0
                                if replay is not None else executed_physics_steps / 120.0)
        update_trex_wrist_camera_poses(transforms, episode_time_s=final_episode_time_s)
        for _ in range(4):
            sim.render()
        for side, inspection_camera in assembly_cameras.items():
            frame = inspection_camera.get_rgba()
            if frame is None:
                raise RuntimeError(f"Isaac did not return the {side} assembly inspection view")
            Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8)).save(
                args.output / f"wrist_assembly_{side}.png")
        preview_images = {}
        preview_depths = {}
        for camera_key in ("wrist_left", "wrist_right"):
            frame = trex_cameras[camera_key].get_rgba()
            if frame is None:
                raise RuntimeError(f"Isaac did not return {camera_key} during camera preview")
            rgb = np.asarray(frame)[..., :3].astype(np.uint8)
            Image.fromarray(rgb).save(args.output / f"trex_isaac_{camera_key}.png")
            preview_images[camera_key] = list(rgb.shape)
            depth = trex_cameras[camera_key].get_depth()
            if depth is not None:
                depth = np.asarray(depth, dtype=np.float32).squeeze()
                np.save(args.output / f"trex_isaac_{camera_key}_depth.npy", depth)
                preview_depths[camera_key] = {
                    "shape_hw": list(depth.shape),
                    "min_m": float(np.nanmin(depth)),
                    "max_m": float(np.nanmax(depth)),
                }
        camera_projection_check_pixels = {}
        camera_depth_check_m = {}
        tomato_tensor = torch.as_tensor([tomato_start], dtype=torch.float32, device="cuda:0")
        for camera_key in ("wrist_left", "wrist_right"):
            projected = trex_cameras[camera_key].get_image_coords_from_world_points(tomato_tensor)
            pixel = np.asarray(projected.detach().cpu() if hasattr(projected, "detach") else projected)[0]
            camera_projection_check_pixels[camera_key] = [pixel.astype(float).tolist()]
            depth = np.asarray(trex_cameras[camera_key].get_depth(), dtype=np.float32).squeeze()
            x_px, y_px = (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
            if 0 <= y_px < depth.shape[0] and 0 <= x_px < depth.shape[1]:
                camera_depth_check_m[camera_key] = float(depth[y_px, x_px])
            else:
                camera_depth_check_m[camera_key] = None
        preview_report = {
            "replay_h5": str(args.replay_h5.expanduser().resolve()),
            "replay_start_sec": float(args.replay_start_sec),
            "captured_at_episode_sec": float(args.replay_start_sec + executed_physics_steps / 120.0),
            "camera_mapping": {"wrist_right": "cam1", "wrist_left": "cam2"},
            "camera_shapes_hwc": preview_images,
            "calibration_json": (str(args.wrist_camera_calibration_json.expanduser().resolve())
                                 if args.wrist_camera_calibration_json else None),
            "camera_mounts": wrist_camera_mount_report,
            "camera_pose_trace": wrist_camera_pose_trace,
            "assembly_inspection_views": assembly_view_report,
            "tomato_start_world_m": [float(value) for value in tomato_start],
            "robot_link_poses_world_xyzw": {
                f"{side}_{link_name}": np.asarray(
                    transforms[robot._articulation_view.get_link_index(link_name)], dtype=float
                ).tolist()
                for side in ("left", "right")
                for link_name in (
                    f"{side}_fr3v2_link7", f"{side}_fr3v2_link8",
                    f"{side}_thumb_DP", f"{side}_index_DP", f"{side}_middle_DP",
                )
            },
            "camera_projection_check_pixels": camera_projection_check_pixels,
            "camera_depth_check_m": camera_depth_check_m,
            "depth_preview": preview_depths,
        }
        (args.output / "wrist_camera_preview.json").write_text(
            json.dumps(preview_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(json.dumps(preview_report, indent=2, ensure_ascii=False), flush=True)
        sim.stop()
        app.close()
        raise SystemExit(0)
    if rollout_video is not None:
        rollout_video.release()
    if replay_camera is not None:
        replay_rgba = replay_camera.get_rgba()
        if replay_rgba is not None:
            replay_rgb = np.asarray(replay_rgba)[..., :3]
            if replay_rgb.size:
                Image.fromarray(replay_rgb.astype(np.uint8)).save(task_output_dir / "replay_overview.png")
    if rgba is not None:
        rgb = np.asarray(rgba)[..., :3]
        if rgb.size:
            Image.fromarray(rgb.astype(np.uint8)).save(args.output / "isaac_tomato_camera.png")
            report["camera_rgb_shape"] = list(rgb.shape)
    if args.trex_host and not args.trex_closed_loop:
        sim.render()
        head_frame = camera.get_rgba()
        right_frame = trex_cameras["wrist_right"].get_rgba()
        left_frame = trex_cameras["wrist_left"].get_rgba()
        if head_frame is None or right_frame is None or left_frame is None:
            raise RuntimeError("Isaac did not return all three camera frames for T-Rex")
        head_rgb = np.asarray(head_frame)[..., :3].astype(np.uint8)
        right_rgb = np.asarray(right_frame)[..., :3].astype(np.uint8)
        left_rgb = np.asarray(left_frame)[..., :3].astype(np.uint8)
        if len(trex_f6_samples) < 16:
            raise RuntimeError(f"Only collected {len(trex_f6_samples)} of 16 T-Rex tactile frames")
        f6_history = np.stack(trex_f6_samples[-16:]).reshape(16, 10, 6).astype(np.float32)
        tactile_maps = trex_deform_samples[-1].reshape(10, 240, 240).copy()
        Image.fromarray(head_rgb).save(args.output / "trex_isaac_head.png")
        Image.fromarray(right_rgb).save(args.output / "trex_isaac_wrist_right.png")
        Image.fromarray(left_rgb).save(args.output / "trex_isaac_wrist_left.png")
        np.savez_compressed(args.output / "trex_isaac_tactile_input.npz",
                            tactile_f6=f6_history, tactile_deform=tactile_maps)
        client_result = subprocess.run([
            str(args.trex_python.expanduser()),
            str(SIM_ROOT / "scripts/trex_isaac_observation_client.py"),
            "--input-dir", str(args.output.resolve()),
            "--host", args.trex_host, "--port", str(args.trex_port),
            "--task", args.trex_task,
        ], check=True, capture_output=True, text=True, timeout=240)
        print(client_result.stdout, end="", flush=True)
        trex_path = args.output / "trex_isaac_policy_result.json"
        trex_result = json.loads(trex_path.read_text())
        report["trex_probe"] = trex_result
        report["trex_observation_contract"] = {
            "camera_order": ["head", "wrist_left", "wrist_right"],
            "camera_shapes_hwc": [list(head_rgb.shape), list(left_rgb.shape), list(right_rgb.shape)],
            "camera_nonblack_fraction": {
                "head": float(np.any(head_rgb > 5, axis=-1).mean()),
                "wrist_left": float(np.any(left_rgb > 5, axis=-1).mean()),
                "wrist_right": float(np.any(right_rgb > 5, axis=-1).mean()),
            },
            "finger_order": ["left_thumb", "left_index", "left_middle", "left_ring", "left_pinky",
                             "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky"],
            "f6_shape": list(f6_history.shape),
            "f6_units": "N and N*m; force/torque expressed in each fingertip sensor local frame",
            "deformation_shape": list(tactile_maps.shape),
            "deformation_range": [int(tactile_maps.min()), int(tactile_maps.max())],
            "f6_history_sample_rate_hz": 30,
            "physics_rate_hz": 120,
        }
        if args.trex_apply:
            import torch

            actions = np.load(args.output / "trex_isaac_action_chunk.npy").astype(np.float64)
            adapter_metadata = json.loads((args.trex_checkpoint.expanduser() / "adapter_metadata.json").read_text())
            target_chunk, ik_reports, hand_limits, hand_clips = _trex_actions_to_joint_targets(
                actions, robot, stage, args.trex_checkpoint.expanduser(),
                adapter_metadata["hand_order"], args.trex_ik_position_tolerance_m,
                args.trex_ik_rotation_tolerance_deg, args.trex_ik_max_joint_step_rad)
            np.save(args.output / "trex_isaac_joint_targets.npy", target_chunk.astype(np.float32))
            report["trex_hand_joint_limit_clips"] = hand_clips

            rollout_peak_force = np.zeros(len(sensors), dtype=np.float64)
            rollout_steps_per_action = 4  # 120 Hz PhysX, 30 Hz checkpoint actions.
            for action_index, target_positions in enumerate(target_chunk):
                action_target = torch.as_tensor(target_positions, device="cuda:0", dtype=torch.float32)
                robot.apply_action(ArticulationAction(joint_positions=action_target))
                for substep in range(rollout_steps_per_action):
                    sim.step(render=(substep == rollout_steps_per_action - 1))
                    for sensor_index, sensor in enumerate(sensors):
                        sensor.update(1.0 / 120.0, force_recompute=True)
                        forces = _numpy(sensor.data.net_forces_w).reshape(-1, 3)
                        if forces.size:
                            rollout_peak_force[sensor_index] = max(
                                rollout_peak_force[sensor_index], float(np.linalg.norm(forces[0])))
            final_positions = _numpy(robot.get_joint_positions()).reshape(-1)
            report["trex_rollout"] = {
                "executed_in": "Isaac Sim / PhysX only",
                "physics_steps": int(len(actions) * rollout_steps_per_action),
                "policy_actions": int(len(actions)),
                "policy_rate_hz": 30,
                "physics_rate_hz": 120,
                "action_applied": any(item["accepted"] for rows in ik_reports.values() for item in rows),
                "execution_status": ("applied" if all(item["accepted"] for rows in ik_reports.values() for item in rows)
                                     else "partially_applied_with_ik_holds"
                                     if any(item["accepted"] for rows in ik_reports.values() for item in rows)
                                     else "ik_rejected_hold_pose"),
                "ik_rejected_actions_by_side": {
                    side: sum(not item["accepted"] for item in rows)
                    for side, rows in ik_reports.items()},
                "ik_max_position_error_m": {
                    side: max(row["position_error_m"] for row in rows if row["position_error_m"] is not None) for side, rows in ik_reports.items()},
                "ik_max_rotation_error_deg": {
                    side: max(row["rotation_error_deg"] for row in rows if row["rotation_error_deg"] is not None) for side, rows in ik_reports.items()},
                "ik_solver_converged_count": {
                    side: sum(row["solver_success"] for row in rows) for side, rows in ik_reports.items()},
                "ik_residual_tolerance_pass_count": {
                    side: sum(row["accepted"] for row in rows) for side, rows in ik_reports.items()},
                "peak_fingertip_contact_force_n": rollout_peak_force.tolist(),
                "final_arm_joint_positions_rad": {
                    side: [float(final_positions[robot.dof_names.index(f"{side}_fr3v2_joint{i}")])
                           for i in range(1, 8)] for side in ("left", "right")},
            }
            report["trex_probe"]["action_applied"] = report["trex_rollout"]["action_applied"]
            report["trex_probe"]["action_contract"] = (
                "checkpoint URDF FK/IK, chunk-start link8 delta xyz+rot6d, absolute hand targets")
            trex_result["action_applied"] = report["trex_rollout"]["action_applied"]
            trex_result["execution_status"] = report["trex_rollout"]["execution_status"]
            trex_result["action_contract"] = report["trex_probe"]["action_contract"]
            trex_path.write_text(json.dumps(trex_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            sim.render()
            for camera_key, camera_obj in (("head", camera),
                                           ("wrist_right", trex_cameras["wrist_right"]),
                                           ("wrist_left", trex_cameras["wrist_left"])):
                frame = camera_obj.get_rgba()
                if frame is not None:
                    Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8)).save(
                        args.output / f"trex_isaac_{camera_key}_after.png")
    if snapshots:
        from PIL import ImageDraw

        sheet = Image.new("RGB", (960 * len(snapshots), 680), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (step, frame) in enumerate(snapshots):
            sheet.paste(Image.fromarray(frame), (column * 960, 40))
            draw.text((column * 960 + 8, 10), f"Isaac PhysX step {step}", fill="black")
        sheet.save(args.output / "isaac_tomato_progress.png")
    if replay is not None:
        actual = robot.get_joint_positions()
        if hasattr(actual, "detach"):
            actual = actual.detach().cpu().numpy()
        error = np.abs(np.asarray(actual) - target_positions)
        report["replay_stable"] = bool(np.isfinite(error).all())
        report["replay_final_mean_joint_error_rad"] = float(np.mean(error))
        report["replay_final_max_joint_error_rad"] = float(np.max(error))
        report["replay_final_worst_joints"] = [
            {"joint": robot.dof_names[i], "error_rad": float(error[i])}
            for i in np.argsort(error)[-5:][::-1]
        ]
    np.savez_compressed(args.output / "isaac_tomato_tactile.npz", deformation=deformation, wrench=wrench)
    if replay_force_trace:
        np.savez_compressed(args.output / "replay_force_trace.npz",
                            time_peak_finger_force_and_target_position=np.asarray(
                                replay_force_trace, dtype=np.float32))
        report["replay_force_trace"] = str(args.output / "replay_force_trace.npz")
    if distal_body is not None:
        np.savez_compressed(args.output / "branch_trace.npz",
                            time_and_root_positions_world_m=np.asarray(branch_trace, dtype=np.float32))
        report["flexible_branch_diagnostics"] = {
            "peak_anchor_error_m": branch_state["peak_anchor_error_m"],
            "peak_bend_deg": branch_state["peak_bend_deg"],
            "peak_internal_force_n": branch_state["peak_force_n"],
            "limit_projection_count": branch_state["limit_projection_count"],
            "configured_max_bend_deg": 4.8,
            "trace": str(args.output / "branch_trace.npz"),
        }
        if args.branch_lift_probe:
            report["branch_lift_probe"] = {
                "target_lift_m": 0.08,
                "peak_clamp_force_n": lift_probe_state["peak_clamp_force_n"],
                "initial_anchor_world_m": lift_probe_state["initial_anchor_world_m"].tolist(),
                "final_anchor_world_m": lift_probe_state.get("final_anchor_world_m"),
                "location": args.branch_lift_location,
                "modeled_as": "clamped branch point; probe automatically arms the target pedicel break rule at lift onset",
            }
    if soft_body_view is not None and soft_fruit_trace:
        soft_trace_array = np.asarray(soft_fruit_trace, dtype=np.float32)
        soft_node_array = np.asarray(soft_fruit_node_position_trace, dtype=np.float32)
        soft_trace_path = args.output / "soft_fruit_trace.npz"
        # World node travel combines translation, rotation and actual shape
        # change. Align each frame to its first sampled shape before reporting
        # deformation, so a lifted fruit is not mistaken for a liquid one.
        shape_rms_m = np.empty(soft_node_array.shape[:2], dtype=np.float32)
        centroid_travel_m = np.empty_like(shape_rms_m)
        shape_rms_m.fill(np.nan)
        centroid_travel_m.fill(np.nan)
        invalid_node_samples = np.zeros(len(soft_body_names), dtype=np.int32)
        for body_index in range(len(soft_body_names)):
            initial_nodes = soft_node_array[0, body_index].astype(np.float64)
            if not np.isfinite(initial_nodes).all():
                invalid_node_samples[body_index] = len(soft_node_array)
                continue
            initial_center = initial_nodes.mean(axis=0)
            initial_centered = initial_nodes - initial_center
            for sample_index, nodes in enumerate(soft_node_array[:, body_index]):
                current_nodes = nodes.astype(np.float64)
                if not np.isfinite(current_nodes).all():
                    invalid_node_samples[body_index] += 1
                    continue
                current_center = current_nodes.mean(axis=0)
                centered = current_nodes - current_center
                try:
                    u, _, vt = np.linalg.svd(initial_centered.T @ centered)
                    parity = np.sign(np.linalg.det(u @ vt))
                    rotation = u @ np.diag([1.0, 1.0, parity]) @ vt
                except np.linalg.LinAlgError:
                    invalid_node_samples[body_index] += 1
                    continue
                residual = initial_centered @ rotation - centered
                shape_rms_m[sample_index, body_index] = float(
                    np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
                centroid_travel_m[sample_index, body_index] = float(
                    np.linalg.norm(current_center - initial_center))
        np.savez_compressed(
            soft_trace_path,
            time_s=soft_trace_array[:, 0],
            body_names=np.asarray(soft_body_names),
            per_body_max_and_mean_node_displacement_and_anchor_slip_m=soft_trace_array[:, 1:-1].reshape(
                len(soft_fruit_trace), len(soft_body_names), 3),
            peak_fingertip_contact_force_n=soft_trace_array[:, -1],
            nodal_positions_world_m=soft_node_array,
            rigid_aligned_shape_rms_m=shape_rms_m,
            centroid_travel_m=centroid_travel_m,
        )
        soft_diagnostics = {}
        for body_index, name in enumerate(soft_body_names):
            body_metrics = soft_trace_array[:, 1:-1].reshape(
                len(soft_fruit_trace), len(soft_body_names), 3)[:, body_index]
            soft_diagnostics[name] = {
                "peak_node_displacement_m": float(body_metrics[:, 0].max(initial=0.0)),
                "final_max_node_displacement_m": float(body_metrics[-1, 0]),
                "peak_mean_node_displacement_m": float(body_metrics[:, 1].max(initial=0.0)),
                "peak_attachment_slip_m": float(body_metrics[:, 2].max(initial=0.0)),
                "peak_rigid_aligned_shape_rms_m": (float(np.nanmax(shape_rms_m[:, body_index]))
                    if np.isfinite(shape_rms_m[:, body_index]).any() else None),
                "final_rigid_aligned_shape_rms_m": (float(shape_rms_m[-1, body_index])
                    if np.isfinite(shape_rms_m[-1, body_index]) else None),
                "peak_centroid_travel_m": (float(np.nanmax(centroid_travel_m[:, body_index]))
                    if np.isfinite(centroid_travel_m[:, body_index]).any() else None),
                "invalid_node_samples": int(invalid_node_samples[body_index]),
                "initial_shape_extent_xyz_m": (np.ptp(soft_node_array[0, body_index], axis=0).tolist()
                    if np.isfinite(soft_node_array[0, body_index]).all() else None),
                "final_shape_extent_xyz_m": (np.ptp(soft_node_array[-1, body_index], axis=0).tolist()
                    if np.isfinite(soft_node_array[-1, body_index]).all() else None),
                "attachment_branch_segment": next(
                    item["attachment_branch_path"] for item in soft_fruit_metadata
                    if item["name"] == name),
            }
        report["soft_fruit_diagnostics"] = {
            "readback_source": "Isaac PhysX volume-deformable tensor view, world nodal positions",
            "sample_rate_hz": 30,
            "trace": str(soft_trace_path),
            "max_fingertip_contact_force_n": float(soft_trace_array[:, -1].max(initial=0.0)),
            "fruits": soft_diagnostics,
        }
    if replay_probe_positions:
        np.savez_compressed(
            args.output / "replay_geometry_probe.npz",
            time_s=np.asarray(replay_probe_times, dtype=np.float64),
            body_names=np.asarray(replay_probe_names),
            position_world_m=np.asarray(replay_probe_positions, dtype=np.float32),
            target_tomato_start_world_m=np.asarray(tomato_start, dtype=np.float32),
            table_top_z_m=np.float32(table_top_z),
        )
        report["replay_geometry_probe"] = str(args.output / "replay_geometry_probe.npz")
    report["max_tactile_intensity"] = int(deformation.max())
    report["max_force_norm_n"] = float(np.linalg.norm(wrench[..., :3], axis=-1).max())
    report["peak_sensor_contacts"] = peak_sensor_contacts
    report["peak_sensor_force_n"] = peak_sensor_force
    if args.soft_fruit_on_hand_contact:
        activation_report = {
            "trigger": "sustained fingertip contact localized to the original rigid fruit surface only",
            "contact_force_threshold_n": touch_contact_force_threshold_n,
            "required_consecutive_sensor_samples": touch_contact_required_samples,
            "physics_rate_hz": 120,
            "events": touch_activation_events,
            "fruits": touch_fruit_runtime,
            "mass_invariant": all(
                abs(float(state["mass_kg_before_kg"]) - float(state["mass_kg_after_kg"])) < 1e-9
                for state in touch_fruit_runtime.values()),
        }
        report["soft_fruit_touch_activation"] = activation_report
        activation_path = args.output / "soft_fruit_activation_trace.json"
        activation_path.write_text(json.dumps(activation_report, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
        report["soft_fruit_touch_activation_trace"] = str(activation_path)
    if (replay is not None and replay_contact_object_peaks) or args.trex_closed_loop or args.pi05_closed_loop:
        finger_names = ("thumb", "index", "middle", "ring", "pinky")
        contact_objects = sorted(
            (dict(item,
                  side=("left", "right")[item["sensor_index"] // 5],
                  finger=finger_names[item["sensor_index"] % 5])
             for item in replay_contact_object_peaks.values()),
            key=lambda item: item["peak_force_n"], reverse=True)
        contact_object_path = args.output / (
            "trex_contact_object_peaks.json" if args.trex_closed_loop
            else "pi05_contact_object_peaks.json" if args.pi05_closed_loop
            else "replay_contact_object_peaks.json")
        contact_object_path.write_text(
            json.dumps(contact_objects, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.trex_closed_loop:
            report["trex_contact_object_peaks"] = str(contact_object_path)
        elif args.pi05_closed_loop:
            report["pi05_contact_object_peaks"] = str(contact_object_path)
        else:
            report["replay_contact_object_peaks"] = str(contact_object_path)
    report["pedicel"] = {
        "model": "source axial/lateral spring-damper with EMA and persistent overload",
        "axial_stiffness_N_m": float(source_attachment_config["axial_stiffness_N_m"]),
        "lateral_stiffness_N_m": float(source_attachment_config["lateral_stiffness_N_m"]),
        "break_force_N": float(source_attachment_config["break_force_N"]),
        "break_ema_time_constant_s": float(source_attachment_config["break_ema_time_constant_s"]),
        "minimum_break_overload_s": float(source_attachment_config["minimum_break_overload_s"]),
        "broken": bool(pedicel_state["broken"]),
        "break_time_s": pedicel_state["break_time_s"],
        "filtered_force_load": float(pedicel_state["filtered_load"]),
        "peak_raw_force_load": float(pedicel_state["peak_force_load"]),
        "peak_tensile_force_n": float(pedicel_state["peak_tensile_force_n"]),
        "overload_duration_s": float(pedicel_state["overload_duration_s"]),
        "harvest_break_armed": bool(pedicel_state["armed"]),
        "arming_rule": (
            "branch clamp probe arms at 1.0 s; otherwise requires "
            f"{'right' if args.soft_grasp_target else 'left'} fingertip contact >= 0.03 N "
            "for >= 0.08 s plus >= 0.02 rad same-side arm motion in the IK or replay command"
        ),
    }
    report["support_attachment"] = {
        "model": "source linear/angular spring-damper with implicit 120 Hz update",
        "linear_stiffness_N_m": float(source_config["support_attachment"]["linear_stiffness_N_m"]),
        "linear_damping_N_s_m": float(source_config["support_attachment"]["linear_damping_N_s_m"]),
        "peak_force_n": float(support_state["peak_force_n"]),
        "peak_unclamped_force_n": float(support_state["peak_unclamped_force_n"]),
        "maximum_applied_force_n": 8.0,
    }
    if action_chunk_report is not None:
        action_chunk_report["peak_force_by_fingertip_n"] = action_chunk_peak_by_sensor.tolist()
        action_chunk_report["peak_fingertip_force_n"] = float(action_chunk_peak_by_sensor.max(initial=0.0))
        action_chunk_report["predicted_action_min"] = action_chunk_action_min.tolist()
        action_chunk_report["predicted_action_max"] = action_chunk_action_max.tolist()
        report["action_chunk_policy"] = action_chunk_report
    if args.task_eval:
        final_tomato_position, _ = tomato_body.get_world_pose()
        final_tomato_position = _numpy(final_tomato_position).reshape(-1, 3)[0].astype(float)
        initial_tomato_position = np.asarray(tomato_start, dtype=float)
        displacement = float(np.linalg.norm(final_tomato_position - initial_tomato_position))
        final_joint_positions = _numpy(robot.get_joint_positions()).reshape(-1)
        joint_delta = np.abs(final_joint_positions - initial_joint_positions)
        tomato_finite = bool(np.isfinite(final_tomato_position).all())
        tomato_in_workspace = bool(tomato_finite and
                                   0.0 <= final_tomato_position[0] <= 0.8 and
                                   -0.6 <= final_tomato_position[1] <= 0.6 and
                                   0.0 <= final_tomato_position[2] <= 0.5)
        basket_center = np.asarray(basket_first_center_xy, dtype=float)
        in_basket = bool(tomato_in_workspace and
                         abs(final_tomato_position[0] - basket_center[0]) <= 0.040 and
                         abs(final_tomato_position[1] - basket_center[1]) <= 0.040 and
                         table_top_z + 0.005 <= final_tomato_position[2] <= table_top_z + 0.065)
        task_result = {
            "seed": int(args.eval_seed),
            "policy_mode": ("real-joint-replay" if replay is not None else
                            action_chunk_report["mode"] if action_chunk_report is not None else
                            "pi05-closed-loop" if args.pi05_closed_loop else
                            "trex-closed-loop" if args.trex_closed_loop else "no-policy"),
            "episode_seconds": float(executed_physics_steps / 120),
            "initial_tomato_position_m": initial_tomato_position.tolist(),
            "final_tomato_position_m": final_tomato_position.tolist(),
            "tomato_displacement_m": displacement,
            "tomato_in_workspace": tomato_in_workspace,
            "max_joint_change_rad": float(joint_delta.max(initial=0.0)),
            "mean_joint_change_rad": float(joint_delta.mean()),
            "pedicel_broken": bool(pedicel_state["broken"]),
            "harvest_success": bool(pedicel_state["broken"] and tomato_in_workspace and 0.05 <= displacement <= 0.70),
            "basket_success": in_basket,
            "pick_and_place_success": bool(pedicel_state["broken"] and displacement >= 0.05 and in_basket),
            "peak_fingertip_force_n": float(max(
                action_chunk_peak_by_sensor.max(initial=0.0),
                max(peak_sensor_force, default=0.0) if args.pi05_closed_loop else 0.0)),
            "peak_force_by_fingertip_n": (peak_sensor_force if args.pi05_closed_loop
                                          else action_chunk_peak_by_sensor.tolist()),
            "tactile_sample_rate_hz": int(120 // sensor_stride),
            "success_definition": "harvest: pedicel overload break is recorded and the finite tomato center remains in the workcell 5-70 cm from its seeded start; basket: center inside first basket bounds and above the base",
            "calibrated_to_hardware": False,
            "rollout_video": (str(task_output_dir / "rollout.mp4")
                              if (task_output_dir / "rollout.mp4").is_file() else None),
        }
        report["task_evaluation"] = task_result
        (task_output_dir / "task_evaluation.json").write_text(json.dumps(task_result, indent=2) + "\n")
        np.save(task_output_dir / "tomato_position_trace.npy",
                np.asarray(action_chunk_tomato_trace, dtype=np.float32))
    if args.trex_closed_loop:
        report["trex_closed_loop"] = trex_closed_loop_report
        trex_f6_array = (np.stack(trex_f6_trace).astype(np.float32)
                         if trex_f6_trace else np.empty((0, 2, 5, 6), dtype=np.float32))
        np.savez_compressed(task_output_dir / "force_tactile_trace.npz",
                            f6=trex_f6_array,
                            time_s=np.asarray([row["time_s"] for row in trex_force_trace], dtype=np.float32),
                            deformation=np.stack(trex_deform_samples).astype(np.uint8)
                            if trex_deform_samples else np.empty((0, 2, 5, 240, 240), dtype=np.uint8))
        (task_output_dir / "trex_closed_loop_report.json").write_text(
            json.dumps({"task_evaluation": report.get("task_evaluation"),
                        "trex_closed_loop": trex_closed_loop_report}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    if args.pi05_closed_loop:
        report["pi05_closed_loop"] = pi05_closed_loop_report
        pi05_f6_array = (np.stack(trex_f6_trace).astype(np.float32)
                         if trex_f6_trace else np.empty((0, 2, 5, 6), dtype=np.float32))
        np.savez_compressed(task_output_dir / "pi05_force_tactile_trace.npz",
                            f6=pi05_f6_array,
                            time_s=np.asarray([row["time_s"] for row in trex_force_trace], dtype=np.float32),
                            deformation=np.stack(trex_deform_samples).astype(np.uint8)
                            if trex_deform_samples else np.empty((0, 2, 5, 240, 240), dtype=np.uint8))
        (task_output_dir / "pi05_closed_loop_report.json").write_text(
            json.dumps({"task_evaluation": report.get("task_evaluation"),
                        "pi05_closed_loop": pi05_closed_loop_report}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    if policy_report is not None:
        policy_report["predicted_action_min"] = policy_action_min.tolist()
        policy_report["predicted_action_max"] = policy_action_max.tolist()
        policy_report["last_predicted_action"] = policy_action_last.tolist() if policy_action_last is not None else None
    if proof_lab_sensor is not None:
        report["sensor_self_test_peak_force_n"] = float(np.linalg.norm(proof_lab_peak_force))
        report["sensor_self_test_peak_contact_count"] = proof_contact_peak
        report["sensor_self_test_tactile_peak_intensity"] = proof_tactile_peak
        report["sensor_self_test_lab_force_shape"] = list(proof_lab_sensor.data.net_forces_w.shape)
        report["sensor_self_test_lab_contact_position_shape"] = (
            list(proof_lab_sensor.data.contact_pos_w.shape)
            if proof_lab_sensor.data.contact_pos_w is not None else None)
        contact_position = (
            _numpy(proof_lab_sensor.data.contact_pos_w).reshape(-1, 3)[0]
            if proof_lab_sensor.data.contact_pos_w is not None else None
        )
        report["sensor_self_test_lab_contact_position_w_m"] = (
            contact_position.tolist()
            if contact_position is not None and np.isfinite(contact_position).all() else None
        )
        report["sensor_self_test_lab_friction_force_n"] = (
            _numpy(proof_lab_sensor.data.friction_forces_w).reshape(-1, 3)[0].tolist()
            if proof_lab_sensor.data.friction_forces_w is not None else None)
        proof_position, _ = proof_body.get_world_pose()
        report["sensor_self_test_final_position_m"] = _numpy(proof_position).reshape(-1, 3)[0].tolist()
        report["sensor_self_test_min_height_m"] = proof_min_height
    tomato_final_position, _ = tomato_body.get_world_pose()
    tomato_final_position = _numpy(tomato_final_position).reshape(-1, 3)[0].astype(float)
    report["target_tomato_final_world_position_m"] = tomato_final_position.tolist()
    report["target_tomato_displacement_m"] = float(
        np.linalg.norm(tomato_final_position - np.asarray(tomato_start, dtype=float)))
    (args.output / "isaac_scene_import.json").write_text(json.dumps(report, indent=2) + "\n")
    stage.GetRootLayer().Export(str(args.output / "isaac_tomato_scene.usda"))
    print(f"[Isaac tomato] completed {executed_physics_steps} PhysX steps", flush=True)
    self_test_ok = (
        not args.sensor_self_test
        or (report.get("sensor_self_test_peak_force_n", 0.0) > 0.0
            and report.get("sensor_self_test_peak_contact_count", 0) > 0
            and report.get("sensor_self_test_tactile_peak_intensity", 0) > 0)
    )
    if args.sensor_self_test and not self_test_ok:
        print("[Isaac tomato] tactile contact self-test failed; inspect isaac_scene_import.json", flush=True)
    return 0 if len(joint_prims) == 58 and report.get("replay_stable", True) and self_test_ok else 2


if __name__ == "__main__":
    import threading

    result = 1
    try:
        result = main()
        if result == 0 and args.hold_window_open:
            print("[Isaac tomato] final scene is open; close the Isaac Sim window to exit", flush=True)
            while app.is_running():
                app.update()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        close_thread = threading.Thread(target=lambda: app.close(wait_for_replicator=False), daemon=True)
        close_thread.start()
        close_thread.join(timeout=5.0)
        if close_thread.is_alive():
            print("[Isaac tomato] Kit close timed out; ending completed smoke process", flush=True)
            os._exit(result)
    raise SystemExit(result)
