"""Historical forward-flexed posture contract, not a physics-success test."""
from pathlib import Path

import numpy as np
import yaml

from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot


ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / 'configs/cherry_tomato_pick_collection_baseline.yaml').read_text())


def test_baseline_is_explicit_stem_down_preview_with_original_gates():
    cfg = config()
    assert cfg['scene']['cluster_quaternion_xyzw'] == [0., 0., 0., 1.]
    assert not cfg['recording']['enabled']
    assert cfg['recording']['auto_increment']
    assert not cfg['recording']['overwrite']
    assert cfg['recording']['output_path'].startswith('outputs/collection_baseline/')
    assert cfg['task']['unsafe_contact_force_N'] == 35.
    assert cfg['task']['minimum_support_contact_fingers'] == 3
    assert cfg['task']['minimum_support_hold_s'] == .25
    assert not cfg['task'].get('bunch_wrap_control', False)
    assert not cfg['task'].get('stem_pinch_control', False)
    assert not cfg['script']['support_close_on_early_contact']
    assert cfg['script']['support_require_reached_pose']


def test_original_five_finger_hand_and_wrist_are_restored():
    robot = config()['robot']
    assert robot['right_pregrasp_hand']['positions'] == [0.] * 20
    assert 'right_reach_hand' not in robot
    assert robot['right_force_fingers'] == ['thumb', 'index', 'middle', 'ring', 'pinky']
    assert robot['right_grasp_opposing_fingers'] == ['index', 'middle', 'ring', 'pinky']
    assert robot['right_distal_first_fingers'] == ['thumb', 'index']
    np.testing.assert_allclose(robot['right_cluster_grasp_quat_xyzw'],
                               [-.114514745, -.414701132, .749464296, .503202358], atol=1e-8)
    hand = robot['right_cluster_grasp_hand']
    for name, position in zip(hand['joint_order'], hand['positions'], strict=True):
        if not name.endswith('_abd'):
            assert 0. <= position <= 1.4, (name, position)


def test_runtime_starts_open_and_never_reverse_flexes_during_closure():
    cfg = config()
    target = np.asarray(cfg['scene']['tomato_position'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=ROOT,
                             tomato_position=target,
                             support_position=target + cfg['scene']['stem_grasp_offset_m'])
    assert not robot.ik_failed
    np.testing.assert_allclose(robot.right.hand_data.qpos, 0.)
    left, right = robot.tcp_poses()
    for closure in (0., .1, .25, .5, .75, 1., 1.15):
        robot.right_force_closure = np.full(5, closure)
        robot.set_targets(left, right, 0., min(1., closure))
        for name, address in robot.right.canonical_qadr.items():
            if not name.endswith('_abd'):
                assert robot.right.hand_data.qpos[address] >= 0., (closure, name)
