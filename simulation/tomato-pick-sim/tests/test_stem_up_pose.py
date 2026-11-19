"""Nominal stem-up geometry and world-space randomization conventions."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from demos.cherry_tomato_pick.collection import initial_geometry, prepare_collection
from demos.cherry_tomato_pick.scene import _randomize_recording_config, _axial_pull_offset


def config():
    return yaml.safe_load(Path("configs/cherry_tomato_pick_deposit.yaml").read_text())


def check_attachment(cfg):
    scene, attachment = cfg["scene"], cfg["attachment"]
    rotation = Rotation.from_quat(scene["cluster_quaternion_xyzw"])
    target = np.array(scene["tomato_position"])
    anchor = np.array(scene["stem_anchor"])
    np.testing.assert_allclose(
        target + rotation.apply(attachment["tomato_local_point"]), anchor, atol=1e-8
    )
    np.testing.assert_allclose(
        Rotation.from_quat(attachment["rest_quaternion_xyzw"]).as_matrix(),
        rotation.as_matrix(), atol=1e-8,
    )
    assert anchor[2] > target[2] + .012
    assert rotation.apply(scene["stem_grasp_offset_m"])[2] > .008
    pull = _axial_pull_offset(cfg["script"]["pull_offset_m"], attachment["break_direction_world"])
    assert np.dot(pull, target - anchor) > 0  # Pull away, not toward the stem.


def test_nominal_stem_up_with_randomization_disabled():
    cfg = config()
    cfg["recording"]["randomization"]["enabled"] = False
    sampled, _ = _randomize_recording_config(cfg, seed=42)
    assert sampled == cfg
    check_attachment(sampled)


def test_jitter_composes_with_nominal_rotation_and_keeps_connections():
    cfg = config()
    original = deepcopy(cfg)
    for seed in range(20):
        sampled, metadata = _randomize_recording_config(cfg, seed=seed)
        expected = (Rotation.from_euler("xyz", metadata["cluster_euler_xyz_deg"], degrees=True)
                    * Rotation.from_quat(cfg["scene"]["cluster_quaternion_xyzw"]))
        np.testing.assert_allclose(Rotation.from_quat(
            sampled["scene"]["cluster_quaternion_xyzw"]).as_matrix(), expected.as_matrix())
        np.testing.assert_allclose(sampled["scene"]["support_tomato_offset_m"],
                                   expected.apply([.04673, -.18492, .00068]))
        check_attachment(sampled)
    assert cfg == original


def test_footprint_is_body_local_independent_of_nominal_pose():
    cfg = config()
    authored = deepcopy(cfg)
    authored["scene"]["cluster_quaternion_xyzw"] = [0., 0., 0., 1.]
    authored["scene"]["support_tomato_offset_m"] = [.04673, -.18492, .00068]
    np.testing.assert_allclose(initial_geometry(cfg), initial_geometry(authored))


def test_legal_randomized_stem_up_scenes():
    for seed in (42, 3854939909, 1, 2, 3):
        sampled, _, _, report = prepare_collection(config(), seed, _randomize_recording_config)
        assert report["status"] == "passed"
        check_attachment(sampled)


def test_stem_up_robot_initial_grasp_ik_is_reachable():
    from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot

    cfg, _, _, _ = prepare_collection(config(), 3854939909, _randomize_recording_config)
    target = np.asarray(cfg["scene"]["tomato_position"])
    rotation = Rotation.from_quat(cfg["scene"]["cluster_quaternion_xyzw"])
    robot = DualWuji2PickRobot(
        cfg["robot"], repository_root=Path(__file__).resolve().parents[1],
        tomato_position=target,
        support_position=target + rotation.apply(cfg["scene"]["stem_grasp_offset_m"]),
    )
    assert not robot.ik_failed
