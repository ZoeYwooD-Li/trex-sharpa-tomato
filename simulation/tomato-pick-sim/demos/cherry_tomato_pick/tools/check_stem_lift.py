"""Bounded physics diagnostic without a frontend or dataset writes."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from demos.cherry_tomato_pick.scene import build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/cherry_tomato_pick_main_stem.yaml"))
    parser.add_argument("--duration", type=float, default=26.0)
    parser.add_argument("--verify-lift", action="store_true",
                        help="Check only the physical right-hand lift, before left contact")
    parser.add_argument("--with-cameras", action="store_true",
                        help="Keep the configured camera sensors enabled")
    parser.add_argument("--snapshot-lift", type=Path,
                        help="Save measured lifted poses for offline route checks")
    parser.add_argument("--snapshot-blocked", type=Path,
                        help="Save the accepted hand poses if clearance blocks a command")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0.0:
        parser.error("--duration must be finite and positive")
    cfg = yaml.safe_load(args.config.read_text())
    if not args.with_cameras:
        cfg["camera"]["sensor"]["enabled"] = False
        cfg["wrist_cameras"]["sensor"]["enabled"] = False
    cfg["recording"]["enabled"] = False
    setup = build(cfg)
    scene = setup.runtime_backend.scenes[0]
    try:
        for i in range(round(args.duration / setup.dt)):
            setup.step(i * setup.dt, {})
            if args.verify_lift and (scene.support_lost or scene.robot.ik_failed):
                raise RuntimeError(
                    f"right-hand lift failed in {setup._camera_state['phase']} at {i * setup.dt:.3f} s"
                )
            if i % 50 == 0:
                _, quat = scene.solver.body_pose(scene.branch_index)
                fruit, _ = scene.solver.body_pose(scene.tomato_index)
                print(
                    "CHECK", round(i * setup.dt, 3), setup._camera_state["phase"],
                    "stem", np.round(scene.stem_grasp_position, 4),
                    "rot", np.round(Rotation.from_quat(quat).as_rotvec(), 3),
                    "fruit", np.round(fruit, 4),
                    "right", np.round(scene.right_contact_force_N, 3),
                    "closure", np.round(scene.robot.right_force_closure, 3),
                    "left", np.round(scene.left_contact_force_N[:3], 3),
                    "left_closure", (np.round(scene.robot.left_force_closure, 3)
                                     if scene.robot.left_force_closure is not None else None),
                    "clearance_m", round(scene.cluster_clearance, 4),
                    "ready", scene.support_engaged, "ik", scene.robot.ik_failed,
                    "peak_N", round(scene.peak_contact_force_N, 3),
                    "hand_gap_m", round(scene.robot.interhand_distance, 4),
                    "detached", scene.detached,
                    "load", round(scene.attachment_load, 2), flush=True,
                )
            if scene.robot.hand_collision_blocked:
                if args.snapshot_blocked is not None:
                    np.savez(args.snapshot_blocked,
                             target=scene.solver.body_pose(scene.tomato_index)[0],
                             left_command=scene.left_tcp,
                             left_tcp=scene.robot.tcp_poses()[0],
                             right_tcp=scene.robot.tcp_poses()[1],
                             left_arm_q=scene.robot.arm_q()[0],
                             right_arm_q=scene.robot.arm_q()[1],
                             left_hand_q=scene.robot.left.hand_data.qpos.copy(),
                             right_hand_q=scene.robot.right.hand_data.qpos.copy())
                raise RuntimeError(f"hand clearance guard blocked {scene.robot.hand_clearance.closest_pair}")
            if args.verify_lift and setup._camera_state["phase"] == "harvest_approach":
                if args.snapshot_lift is not None:
                    np.savez(args.snapshot_lift,
                             target=scene.solver.body_pose(scene.tomato_index)[0],
                             right_tcp=scene.robot.tcp_poses()[1],
                             left_arm_q=scene.robot.arm_q()[0],
                             right_arm_q=scene.robot.arm_q()[1],
                             right_hand_q=scene.robot.right.hand_data.qpos.copy())
                lift = float(scene.stem_grasp_position[2]
                             - setup._camera_state["stem_lift_start"][2])
                result = {
                    "time_s": i * setup.dt,
                    "stem_lift_m": lift,
                    "cluster_clearance_m": float(scene.cluster_clearance),
                    "peak_contact_force_N": float(scene.peak_contact_force_N),
                    "support_ready": bool(scene.support_engaged),
                    "support_lost": bool(scene.support_lost),
                    "ik_failed": bool(scene.robot.ik_failed),
                }
                if args.with_cameras:
                    state = setup._camera_state
                    result["camera_samples"] = int(state["camera_samples"])
                    result["wrist_camera_samples"] = int(state["wrist_camera_samples"])
                    if (result["camera_samples"] <= 0 or result["wrist_camera_samples"] <= 0
                            or state["camera_valid_depth_pixels"] <= 0
                            or any(n <= 0 for n in state["wrist_camera_valid_depth_pixels"].values())):
                        raise RuntimeError(f"camera sampling verification failed: {result}")
                if (lift < float(cfg["task"]["minimum_cluster_lift_m"])
                        or scene.cluster_clearance < 0.005
                        or not scene.support_engaged or scene.support_lost
                        or scene.robot.ik_failed
                        or scene.peak_contact_force_N > float(cfg["task"]["unsafe_contact_force_N"])):
                    raise RuntimeError(f"right-hand lift verification failed: {result}")
                print("LIFT PASS", result, flush=True)
                return
        if args.verify_lift:
            raise RuntimeError(f"right-hand lift timed out in {setup._camera_state['phase']}")
        setup.verify()
    finally:
        setup.close()


if __name__ == "__main__":
    main()
