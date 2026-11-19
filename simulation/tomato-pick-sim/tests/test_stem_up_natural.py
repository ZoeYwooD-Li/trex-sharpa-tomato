"""Stem-up scene with the historical forward-flexed hand, not success-rate evidence."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot

ROOT = Path(__file__).resolve().parents[1]


def config(name='cherry_tomato_pick_stem_up_natural'):
    return yaml.safe_load((ROOT / 'configs' / (name + '.yaml')).read_text())


def test_stem_up_geometry_and_attachments_remain_consistent():
    cfg = config()
    scene, attachment = cfg['scene'], cfg['attachment']
    assert scene['cluster_quaternion_xyzw'] == [0., 1., 0., 0.]
    rotation = Rotation.from_quat(scene['cluster_quaternion_xyzw'])
    target, anchor = np.asarray(scene['tomato_position']), np.asarray(scene['stem_anchor'])
    np.testing.assert_allclose(target + rotation.apply(attachment['tomato_local_point']), anchor, atol=1e-8)
    np.testing.assert_allclose(Rotation.from_quat(attachment['rest_quaternion_xyzw']).as_matrix(),
                               rotation.as_matrix(), atol=1e-8)
    assert anchor[2] > target[2] + .012
    assert rotation.apply(scene['stem_grasp_offset_m'])[2] > .008


def test_right_hand_shape_is_unchanged_and_grasp_center_height_is_recalibrated():
    cfg = config()
    base = config('cherry_tomato_pick_collection_baseline')
    for field in ('right_cluster_grasp_hand', 'right_pregrasp_hand', 'right_cluster_grasp_quat_xyzw',
                  'right_grasp_opposing_fingers', 'right_force_fingers', 'right_distal_first_fingers'):
        assert cfg['robot'][field] == base['robot'][field]
    np.testing.assert_allclose(
        np.asarray(cfg['robot']['right_cluster_grasp_center_bias_m'])
        - base['robot']['right_cluster_grasp_center_bias_m'], [0., 0., -.020], atol=1e-12)
    assert 'right_reach_hand' not in cfg['robot']


def test_left_harvest_attitude_avoids_stem_without_reversing_finger_flexion():
    cfg = config()['robot']
    baseline = config('cherry_tomato_pick_collection_baseline')['robot']
    expected = (Rotation.from_euler('x', -30., degrees=True)
                * Rotation.from_quat(baseline['left_harvest_quat_xyzw']))
    np.testing.assert_allclose(Rotation.from_quat(cfg['left_harvest_quat_xyzw']).as_matrix(),
                               expected.as_matrix(), atol=1e-9)
    assert cfg['left_harvest_hand'] == baseline['left_harvest_hand']
    assert cfg['left_pregrasp_hand'] == baseline['left_pregrasp_hand']
    assert cfg['left_pregrasp_offset_m'] == baseline['left_pregrasp_offset_m']


def test_left_nominal_full_downward_pull_is_reachable():
    cfg = config()
    target = np.asarray(cfg['scene']['tomato_position'])
    rotation = Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=ROOT, tomato_position=target,
                             support_position=target + rotation.apply(cfg['scene']['stem_grasp_offset_m']),
                             harvest_validation_offset=cfg['script']['support_lift_offset_m'])
    pose = robot.left_grasp_tcp.copy()
    pose[:3] += cfg['script']['support_lift_offset_m']
    q = robot.left.solve_root_pose(pose[:3], pose[3:], robot._left_q)
    # Nominal stem-up pull is downward. Exercise the complete configured
    # displacement, not just the point where detachment happened in one run.
    length = np.linalg.norm(cfg['script']['pull_offset_m'])
    for down in np.linspace(0., length, 71):
        q = robot.left.solve_root_pose(pose[:3] - [0., 0., down], pose[3:], q,
                                      max_iterations=180)


def test_no_physics_or_contact_gates_are_relaxed_and_preview_does_not_record():
    cfg, original = config(), config('cherry_tomato_pick_deposit')
    for section in ('scene', 'attachment', 'support_attachment', 'task', 'script', 'contact', 'solver'):
        assert cfg[section] == original[section]
    assert cfg['task']['unsafe_contact_force_N'] == 35.
    assert not cfg['recording']['enabled']
    assert cfg['recording']['auto_increment'] and not cfg['recording']['overwrite']
    assert cfg['recording']['output_path'].startswith('outputs/stem_up_natural/')


def test_reachable_open_start_and_forward_flexion_with_contact_closure():
    cfg = config()
    target = np.asarray(cfg['scene']['tomato_position'])
    rotation = Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=ROOT, tomato_position=target,
                             support_position=target + rotation.apply(cfg['scene']['stem_grasp_offset_m']),
                             harvest_validation_offset=cfg['script']['support_lift_offset_m'])
    assert not robot.ik_failed
    np.testing.assert_allclose(robot.right.hand_data.qpos, 0.)
    left, right = robot.tcp_poses()
    for closure in (0., .25, .5, .75, 1., 1.15):
        robot.right_force_closure = np.full(5, closure)
        robot.set_targets(left, right, 0., min(1., closure))
        for name, address in robot.right.canonical_qadr.items():
            if not name.endswith('_abd'):
                assert robot.right.hand_data.qpos[address] >= 0., (closure, name)
