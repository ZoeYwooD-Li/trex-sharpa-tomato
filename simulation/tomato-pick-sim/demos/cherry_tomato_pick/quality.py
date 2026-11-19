"""Conservative success labels and an explicit successful-demonstration file list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


QUALITY_SCHEMA = "lycheesim.cherry_tomato_pick.quality/v1"


def assess_episode(frames, actions, cameras, metadata, terminated):
    """Never equate an episode termination (which can mean failure) with success."""
    settings = metadata.get("quality_filter", {})
    reasons = []
    if len(frames) < 2:
        reasons.append("insufficient_robot_samples")
    if not terminated:
        reasons.append("incomplete_episode")
    if metadata.get("scene_validation", {}).get("status") != "passed":
        reasons.append("initial_scene_not_validated")
    required = {"task_state", "placed_in_box", "deposit_failed", "support_lost",
                "hand_collision_blocked", "target_box_one_hot", "cluster_clearance_m",
                "peak_contact_force_N"}
    if not frames or any(not required <= set(f.extra) for f in frames):
        reasons.append("missing_task_evidence")
    else:
        task = np.stack([f.extra["task_state"] for f in frames])
        if task.shape != (len(frames), 7):
            reasons.append("invalid_task_state")
        else:
            if not (task[-1, 2] == 1 and task[-1, 4] == 1
                    and bool(np.all(frames[-1].extra["placed_in_box"]))):
                reasons.append("not_detached_and_placed")
            if np.any(task[:, 5] != 0):
                reasons.append("ik_failed")
            if task[-1, 0] != 1:
                reasons.append("right_support_not_ready_at_end")
        for key in ("deposit_failed", "support_lost", "hand_collision_blocked"):
            if any(np.any(f.extra[key]) for f in frames):
                reasons.append(key)
        if not np.all(frames[-1].extra["cluster_clearance_m"] >= .005):
            reasons.append("bunch_not_lifted_at_end")
        threshold = float(metadata.get("unsafe_contact_force_N", 0.))
        if not np.isfinite(threshold) or threshold <= 0:
            reasons.append("missing_contact_force_limit")
        elif any(np.any(f.extra["peak_contact_force_N"] > threshold) for f in frames):
            reasons.append("unsafe_contact_force")
        goal = metadata.get("target_condition") or {}
        expected = goal.get("target_box_one_hot")
        if (expected is None or not goal.get("policy_input_required")
                or any(not np.array_equal(f.extra["target_box_one_hot"], expected) for f in frames)):
            reasons.append("missing_or_inconsistent_target_condition")
    for frame in frames:
        if any(value is None for value in (frame.joint_pos, frame.joint_vel, frame.ee_pose, frame.gripper)):
            reasons.append("missing_robot_fields")
        arrays = [frame.joint_pos, frame.joint_vel, frame.ee_pose, frame.gripper,
                  *frame.extra.values()]
        if (not np.isfinite(frame.time)
                or any(a is not None and not np.isfinite(a).all() for a in arrays)):
            reasons.append("nonfinite_robot_state")
            break
    if any(not np.isfinite(a).all() for a in actions):
        reasons.append("nonfinite_action")
    dt = float(metadata.get("physics_dt_s", 0.))
    minimum_depth = float(settings.get("min_valid_depth_fraction", .001))
    if not np.isfinite(minimum_depth) or not 0 <= minimum_depth <= 1:
        raise ValueError("quality_filter.min_valid_depth_fraction must be in [0, 1]")
    expected_cameras = metadata.get("camera_streams", [])
    if not expected_cameras:
        reasons.append("missing_camera_requirements")
    for name in expected_cameras:
        stream = cameras.get(name, [])
        if not stream:
            reasons.append(f"missing_camera:{name}")
            continue
        rate = float(metadata.get("camera_sample_hz", {}).get(name, 0.))
        if not np.isfinite(rate) or rate <= 0:
            reasons.append(f"missing_camera_rate:{name}")
            continue
        period = 1. / rate
        times = np.array([f.time for f in stream])
        if (np.any(np.diff(times) > period + dt + 1e-6)
                or (frames and (times[0] > frames[0].time + dt + 1e-6
                                or frames[-1].time - times[-1] > period + dt + 1e-6))):
            reasons.append(f"camera_time_gap:{name}")
        if any(np.count_nonzero(np.isfinite(f.depth_m) & (f.depth_m > 0))
               / max(1, f.depth_m.size) < minimum_depth for f in stream):
            reasons.append(f"invalid_depth:{name}")
    if len(frames) > 1:
        rate = float(metadata.get("robot_sample_hz", 0.))
        if not np.isfinite(rate) or rate <= 0:
            reasons.append("missing_robot_rate")
        elif np.any(np.diff([f.time for f in frames]) > 1. / rate + dt + 1e-6):
            reasons.append("robot_time_gap")
    reasons = sorted(set(reasons))
    return {"schema": QUALITY_SCHEMA, "accepted": not reasons,
            "status": "accepted" if not reasons else "rejected",
            "reasons": reasons, "robot_frames": len(frames),
            "camera_frames": {name: len(stream) for name, stream in cameras.items()},
            "target_box_id": (metadata.get("target_condition") or {}).get("target_box_id"),
            "scope": "initial_geometry_and_recorded_task_sensor_guards_not_full_robot_collision_certification"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="List only quality-accepted HDF5 demonstrations (read-only).")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    if not args.directory.is_dir():
        parser.error("directory does not exist")
    import h5py
    from collections import Counter

    counts, reasons, goals = Counter(), Counter(), Counter()
    for path in sorted(args.directory.iterdir()):
        if path.suffix.lower() not in (".h5", ".hdf5"):
            continue
        # Read the authoritative HDF5 label, not a possibly stale sidecar.
        with h5py.File(path, "r") as archive:
            metadata = json.loads(archive["metadata"].attrs["extra"])
            report = metadata.get("quality", {})
            accepted = report.get("schema") == QUALITY_SCHEMA and report.get("accepted") is True
            counts["accepted" if accepted else "rejected_or_unassessed"] += 1
            reasons.update(report.get("reasons", ["unassessed"]))
            if accepted:
                goals[str(report.get("target_box_id"))] += 1
                if not args.summary:
                    print(path.resolve())
    if args.summary:
        print(json.dumps({"episodes": dict(counts), "rejection_reasons": dict(reasons),
                          "accepted_by_box": dict(goals)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
