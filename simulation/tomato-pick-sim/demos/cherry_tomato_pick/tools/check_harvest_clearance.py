"""Nominal lifted-hand sweep, for route design (not a physics-success test)."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from demos.cherry_tomato_pick.hand_clearance import HandClearance
from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    measured = np.load(args.snapshot)
    cfg = yaml.safe_load(Path("configs/cherry_tomato_pick_main_stem.yaml").read_text())
    cfg["robot"]["minimum_interhand_distance_m"] = 0.0
    # Fixed reference attitude makes repeated searches comparable.
    cfg["robot"]["left_harvest_quat_xyzw"] = [-.587136552, .606221343, .499539745, .195515716]
    target = np.array(cfg["scene"]["tomato_position"])
    robot = DualWuji2PickRobot(cfg["robot"], repository_root=Path.cwd(),
                             tomato_position=target,
                             support_position=target + cfg["scene"]["stem_grasp_offset_m"])
    check = HandClearance(robot.visual_parts)
    initial = robot.left_grasp_tcp.copy()
    raised_target = measured["target"]
    candidates = []
    for angles in [(x, y, z) for x in (-40, 0, 40, 80) for y in (-60, -30, 0)
                   for z in (-100, -80, -40, 0)]:
        delta = Rotation.from_euler("xyz", angles, degrees=True)
        pose = initial.copy()
        pose[:3] = raised_target + delta.apply(initial[:3] - target)
        pose[3:] = (delta * Rotation.from_quat(initial[3:])).as_quat()
        worst, pair = .05, None
        try:
            arm_q = robot.left.solve_root_pose(pose[:3], pose[3:], measured["left_arm_q"], max_iterations=180)
        except RuntimeError:
            continue
        samples = [(grip, (a, a, a)) for grip in (0., .5, 1.) for a in np.linspace(0., 1.15, 9)]
        samples += [(1., (.95, .5, .95)), (1., (1., .8, 1.))]
        for grip, amounts in samples:
            q = robot.left_open_q + grip * (robot.left_closed_q - robot.left_open_q)
            for finger, amount in zip(("thumb", "index", "middle"), amounts):
                for name, address in robot.left.canonical_qadr.items():
                    if name.startswith(finger):
                        q[address] = robot.left_open_q[address] + amount * (robot.left_closed_q[address] - robot.left_open_q[address])
            robot._set_kinematic_state(arm_q, measured["right_arm_q"], q, measured["right_hand_q"])
            distance = check.distance(robot.visual_pose)
            if distance < worst:
                worst, pair = distance, check.closest_pair
        candidates.append((worst, angles, pair, pose[3:].tolist()))
    for candidate in sorted(candidates, reverse=True)[:10]:
        print(candidate, flush=True)


if __name__ == "__main__":
    main()
