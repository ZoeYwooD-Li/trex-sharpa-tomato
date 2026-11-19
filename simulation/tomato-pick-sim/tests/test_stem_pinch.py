"""Stem-only guard semantics; these tests do not establish physical success."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.cherry_tomato_pick.stem_pinch import (
    StemPinchTrial, opposed_stem_tips, plate_clearance, premature_stem_contact,
    trial_script_time, validate_config, stem_tilt_degrees, harvest_workspace_ready,
)
from demos.cherry_tomato_pick.backend import _TomatoScene, _counterpart_force_N


SCRIPT = dict(stem_pinch_timeout_s=1., stem_pinch_trial_height_m=.01,
              stem_pinch_trial_hold_s=.4, stem_pinch_trial_timeout_s=1.,
              stem_pinch_max_slip_m=.008, support_close_end_s=2.,
              support_lift_end_s=5., support_lift_offset_m=[0., 0., .12])


def test_total_fruit_force_is_not_counted_as_stem_contact():
    sensor = SimpleNamespace(total_force=SimpleNamespace(numpy=lambda: np.array([[20., 0., 0.]])),
                             force_matrix=SimpleNamespace(numpy=lambda: np.zeros((1, 1, 3))))
    assert _counterpart_force_N(sensor) == 0.
    sensor.force_matrix = SimpleNamespace(numpy=lambda: np.array([[[3., 4., 0.]]]))
    assert _counterpart_force_N(sensor) == 5.


def test_actual_stem_contact_positions_and_normal_forces_must_both_oppose():
    def array(value):
        return SimpleNamespace(numpy=lambda: np.asarray(value, dtype=float).reshape(1, 1, 3))
    scene = object.__new__(_TomatoScene)
    scene.support_force_indices = [0, 1, 2]
    positions = [[.004, 0., 0.], [-.004, .01, 0.], [-.004, -.01, 0.]]
    scene.right_sensors = [SimpleNamespace(force_matrix=array(force),
                           force_matrix_friction=array([0., 0., 0.]), position_matrix=array(pos))
                           for force, pos in zip(([3., 0., 0.], [-3., 0., 0.], [-3., 0., 0.]), positions)]
    scene.solver = SimpleNamespace(body_pose=lambda _: (np.zeros(3), [0., 0., 0., 1.]))
    scene.branch_index = 0
    scene.cfg = {'scene': {'branch_collision_axis': [0., 1., 0.]}}
    scene.stem_grasp_position = np.zeros(3)
    assert scene.stem_pinch_opposed()
    scene.right_sensors[2].force_matrix = array([3., 0., 0.])
    assert not scene.stem_pinch_opposed()
    scene.right_sensors[2].force_matrix = array([-3., 0., 0.])
    scene.right_sensors[2].position_matrix = array([.004, -.01, 0.])
    assert not scene.stem_pinch_opposed()


def test_distal_contact_is_one_finger_and_zero_force_rows_do_not_shift_contact():
    def array(value):
        return SimpleNamespace(numpy=lambda: np.asarray(value, dtype=float).reshape(2, 1, 3))
    scene = object.__new__(_TomatoScene)
    scene.support_force_indices = [0, 1, 2]
    center = np.array([.4, -.1, .08])
    scene.right_sensors = []
    for sign, y in ((1., 0.), (-1., .01), (-1., -.01)):
        scene.right_sensors.append(SimpleNamespace(
            force_matrix=array([[0., 0., 0.], [3.*sign, 0., 0.]]),
            force_matrix_friction=array(np.zeros((2, 3))),
            position_matrix=array([np.zeros(3), center + [.004*sign, y, 0.]])))
    scene.solver = SimpleNamespace(body_pose=lambda _: (np.zeros(3), [0., 0., 0., 1.]))
    scene.branch_index = 0
    scene.cfg = {'scene': {'branch_collision_axis': [0., 1., 0.]}}
    scene.stem_grasp_position = center
    assert scene.stem_pinch_opposed()
    scene.right_sensors[2].force_matrix = array(np.zeros((2, 3)))
    assert not scene.stem_pinch_opposed()


def update(guard, **kwargs):
    values = dict(dt=.1, closing=True, lifted=False, at_trial=False,
                  ready=False, lift_m=0., slip_m=0., script=SCRIPT)
    values.update(kwargs)
    guard.update(**values)


def test_opposition_requires_both_fingers_across_the_stem():
    tips = np.array([[.01, 0., 0.], [-.01, .01, 0.], [-.01, -.01, 0.]])
    assert opposed_stem_tips(tips, [0., 0., 0.], [0., 1., 0.])
    tips[2, 0] = .01
    assert not opposed_stem_tips(tips, [0., 0., 0.], [0., 1., 0.])
    tips[2] = np.nan
    assert not opposed_stem_tips(tips, [0., 0., 0.], [0., 1., 0.])


def test_fruit_contacts_and_closed_commands_cannot_confirm_stem_grasp():
    scene = object.__new__(_TomatoScene)
    scene.stem_pinch_control = True
    scene.stem_and_fruit = False
    scene.right_grasp_force_ema = np.zeros(3)  # Only actual pad/stem sensors feed EMA.
    scene.task_cfg = dict(minimum_support_bilateral_contact_N=2.,
                          minimum_support_contact_fingers=3, minimum_support_finger_contact_N=1.)
    scene.stem_pinch_opposed = lambda: True
    scene.robot = SimpleNamespace(right_grip=1., right_force_closure=np.ones(3))
    assert not scene.support_contacts_ready()
    scene.right_grasp_force_ema[:] = [3., 3., .9]
    assert not scene.support_contacts_ready()
    scene.right_grasp_force_ema[:] = 3.
    assert scene.support_contacts_ready()
    scene.stem_pinch_opposed = lambda: False
    assert not scene.support_contacts_ready()


def test_close_timeout_is_latched_and_not_an_infinite_wait():
    guard = StemPinchTrial()
    for _ in range(11):
        update(guard)
    assert guard.error == 'stem_pinch_grasp_timeout'
    update(guard, ready=True)
    assert guard.error


def test_trial_requires_actual_lift_and_continuous_hold():
    guard = StemPinchTrial()
    for _ in range(3):
        update(guard, lifted=True, at_trial=True, ready=True, lift_m=.01)
    assert not guard.passed
    update(guard, lifted=True, at_trial=True, ready=False, lift_m=.01)
    assert guard.stable_s == 0.
    for _ in range(4):
        update(guard, lifted=True, at_trial=True, ready=True, lift_m=.01)
    assert guard.passed and not guard.error
    update(guard, lifted=True, slip_m=.009)
    assert guard.error == 'stem_pinch_slip'  # Still protects the subsequent full lift.


def test_commanded_lift_without_bunch_motion_times_out():
    guard = StemPinchTrial()
    for _ in range(11):
        update(guard, lifted=True, at_trial=True, ready=True, lift_m=0.)
    assert guard.error == 'stem_pinch_trial_timeout' and not guard.passed


def test_trial_time_caps_the_original_lift_at_one_centimeter():
    from demos.cherry_tomato_pick.kinematics import minimum_jerk
    t = trial_script_time(SCRIPT)
    fraction = (t - SCRIPT['support_close_end_s']) / 3.
    assert minimum_jerk(fraction**2 * (3-2*fraction)) * .12 == pytest.approx(.01)


def test_pose_interpolation_roundoff_cannot_exceed_slerp_endpoints():
    from demos.cherry_tomato_pick.robot import interpolate_pose
    start = np.array([0., 0., 0., 0., 0., 0., 1.])
    end = np.array([0., 0., .12, 0., 0., 1., 0.])
    for t in np.r_[np.linspace(1.-1e-5, 1., 1000), 1.1, -.1]:
        pose = interpolate_pose(start, end, t)
        assert 0. <= pose[2] <= .12
        assert np.linalg.norm(pose[3:]) == pytest.approx(1.)


def test_plate_guard_catches_faces_spanning_plate_without_inside_vertices():
    plate = SimpleNamespace(plate_center=np.zeros(2), plate_radii=np.ones(2),
                            floor=.045, plate_height=.014)
    vertices = np.array([[-2., -2., .04], [2., 2., .10]])
    assert plate_clearance(vertices, plate) < 0
    assert plate_clearance(vertices + [0., 0., .03], plate) > 0
    # A pad over the flat bowl need not stay above the distant rim.
    assert plate_clearance(np.array([[0., 0., .052], [.5, .5, .055]]), plate) > 0
    assert plate_clearance(np.array([[.95, 0., .052], [1., .05, .055]]), plate) < 0


def test_clearance_outside_plate_uses_actual_tabletop_not_zero():
    plate = SimpleNamespace(plate_center=np.zeros(2), plate_radii=np.ones(2),
                            floor=.045, wall=.004, plate_height=.014,
                            cfg={'tabletop': {'enabled': True}})
    assert plate_clearance(np.array([[2., 0., .040]]), plate) == pytest.approx(-.0012)
    assert plate_clearance(np.array([[2., 0., .050]]), plate) == pytest.approx(.0088)


def test_stem_tilt_is_axis_angle_not_quaternion_euler_pitch():
    from scipy.spatial.transform import Rotation
    assert stem_tilt_degrees([0., 1., 0., 0.], [0., 1., 0.]) == pytest.approx(0.)
    q = Rotation.from_euler('x', 30., degrees=True).as_quat()
    assert stem_tilt_degrees(q, [0., 1., 0.]) == pytest.approx(30.)
    assert stem_tilt_degrees(-q, [0., -1., 0.]) == pytest.approx(30.)


def test_left_release_requires_bunch_clearance_and_tilt_together():
    task = dict(minimum_harvest_cluster_clearance_m=.06, maximum_harvest_stem_tilt_deg=15.)
    assert harvest_workspace_ready(.08, 8., task)
    assert not harvest_workspace_ready(.0068, 8., task)
    assert not harvest_workspace_ready(.08, 37.6, task)
    assert not harvest_workspace_ready(np.nan, 8., task)


def test_existing_configuration_does_not_enable_experiment():
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_deposit.yaml').read_text())
    validate_config(cfg)
    assert not cfg['task'].get('stem_pinch_control', False)
    cfg['task']['stem_pinch_control'] = True
    with pytest.raises(ValueError, match='main_stem'):
        validate_config(cfg)


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf])
def test_invalid_trial_limits_fail_before_runtime(bad):
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_deposit.yaml').read_text())
    cfg['scene']['support_mode'] = 'main_stem'
    cfg['task'].update(stem_pinch_control=True, minimum_support_wrap_closure=0.,
                       minimum_support_contact_fingers=3)
    cfg['robot']['right_force_fingers'] = ['thumb', 'index', 'middle']
    cfg['script'].update(SCRIPT, stem_pinch_max_slip_m=bad)
    with pytest.raises(ValueError, match='stem_pinch_max_slip'):
        validate_config(cfg)


def test_opt_in_preset_is_separate_and_keeps_unused_fingers_tucked():
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    validate_config(cfg)
    np.testing.assert_array_equal(cfg['robot']['right_pregrasp_hand']['positions'][12:],
                                  cfg['robot']['right_cluster_grasp_hand']['positions'][12:])
    assert cfg['scene']['support_mode'] == 'main_stem'
    assert cfg['recording']['output_path'].startswith('outputs/main_stem_pinch/')
    from demos.cherry_tomato_pick.scene import build
    with pytest.raises(ValueError, match='scripted experiment'):
        build(cfg, policy=object())


def test_open_reach_does_not_retarget_at_a_middle_finger_touch():
    assert premature_stem_contact(5., 7., [0., 0., 1.])
    assert not premature_stem_contact(5., 7., [0., 0., 0.])
    assert not premature_stem_contact(7., 7., [0., 0., 1.])


def test_open_approach_rejects_precurved_fingers_and_early_close_shortcut():
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['script'].update(stem_pinch_open_approach=True, support_close_on_early_contact=False)
    cfg['robot']['right_pregrasp_hand']['positions'][:12] = [0.] * 12
    validate_config(cfg)
    cfg['robot']['right_pregrasp_hand']['positions'][0] = .1
    with pytest.raises(ValueError, match='fully open'):
        validate_config(cfg)
    cfg['robot']['right_pregrasp_hand']['positions'][0] = 0.
    cfg['script']['support_close_on_early_contact'] = True
    with pytest.raises(ValueError, match='single-finger'):
        validate_config(cfg)


def test_robot_starts_and_reaches_with_active_fingers_open():
    from scipy.spatial.transform import Rotation
    from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['robot']['right_pregrasp_hand']['positions'][:12] = [0.] * 12
    target = np.asarray(cfg['scene']['tomato_position'])
    grasp = target + Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw']).apply(
        cfg['scene']['stem_grasp_offset_m'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=Path.cwd(),
                               tomato_position=target, support_position=grasp)
    active = [address for name, address in robot.right.canonical_qadr.items()
              if name.startswith(tuple(robot.right_force_fingers))]
    np.testing.assert_allclose(robot.right.hand_data.qpos[active], 0.)
    robot.set_targets(robot.default_left_tcp, robot.right_pregrasp_tcp, 0., 0.)
    np.testing.assert_allclose(robot.right.hand_data.qpos[active], 0.)


def test_two_finger_pinch_still_requires_opposing_measured_contacts():
    assert opposed_stem_tips([[.004, 0., 0.], [-.004, 0., 0.]], np.zeros(3), [0., 1., 0.])
    assert not opposed_stem_tips([[.004, 0., 0.], [.004, .01, 0.]], np.zeros(3), [0., 1., 0.])
    scene = object.__new__(_TomatoScene)
    scene.stem_pinch_control = True
    scene.stem_and_fruit = False
    scene.right_grasp_force_ema = np.array([3., 0.])
    scene.task_cfg = dict(minimum_support_bilateral_contact_N=2., minimum_support_contact_fingers=2,
                          minimum_support_finger_contact_N=1.)
    scene.stem_pinch_opposed = lambda: True
    assert not scene.support_contacts_ready()
    scene.right_grasp_force_ema[:] = 3.
    assert scene.support_contacts_ready()
    scene.stem_pinch_opposed = lambda: False
    assert not scene.support_contacts_ready()


def test_randomized_stem_insertion_rotates_the_whole_calibrated_frame():
    from copy import deepcopy
    from scipy.spatial.transform import Rotation
    from demos.cherry_tomato_pick.scene import _randomize_recording_config
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['robot']['right_grasp_follow_scene_rotation'] = True
    cfg['robot']['right_stem_insert_offset_m'] = [0., -.075, 0.]
    before = deepcopy(cfg)
    sampled, metadata = _randomize_recording_config(cfg, seed=42)
    rotation = Rotation.from_euler('xyz', metadata['cluster_euler_xyz_deg'], degrees=True)
    for key in ('right_cluster_grasp_center_bias_m', 'right_pregrasp_offset_m', 'right_stem_insert_offset_m'):
        np.testing.assert_allclose(sampled['robot'][key], rotation.apply(cfg['robot'][key]))
    expected = rotation * Rotation.from_quat(cfg['robot']['right_cluster_grasp_quat_xyzw'])
    np.testing.assert_allclose(Rotation.from_quat(sampled['robot']['right_cluster_grasp_quat_xyzw']).as_matrix(), expected.as_matrix())
    assert cfg == before


def test_preshape_requires_timed_clearance_hold_and_all_configured_fingers():
    from copy import deepcopy
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['robot']['right_reach_hand'] = deepcopy(cfg['robot']['right_cluster_grasp_hand'])
    cfg['script']['support_preshape_duration_s'] = 1.
    validate_config(cfg)
    cfg['script']['support_preshape_duration_s'] = 0.
    with pytest.raises(ValueError, match='preshape'):
        validate_config(cfg)
    cfg['script']['support_preshape_duration_s'] = 1.
    cfg['task']['minimum_support_contact_fingers'] = 1
    with pytest.raises(ValueError, match='all configured'):
        validate_config(cfg)


def test_preshape_starts_open_and_holds_wrist_away_before_reach():
    from copy import deepcopy
    from scipy.spatial.transform import Rotation
    from demos.cherry_tomato_pick.scene import _scripted_state
    from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['robot']['right_reach_hand'] = deepcopy(cfg['robot']['right_cluster_grasp_hand'])
    cfg['script']['support_preshape_duration_s'] = 1.5
    target = np.asarray(cfg['scene']['tomato_position'])
    grasp = target + Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw']).apply(cfg['scene']['stem_grasp_offset_m'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=Path.cwd(), tomato_position=target, support_position=grasp)
    np.testing.assert_allclose(robot.right.hand_data.qpos, robot.right_open_q)
    t = cfg['script']['support_approach_end_s'] + .75
    _, right, closures, _, _ = _scripted_state(t, cfg, robot)
    np.testing.assert_allclose(right, robot.right_pregrasp_tcp)
    np.testing.assert_allclose(closures, 0.)
    robot.right_preshape_amount = 1.
    robot.right_force_closure = np.zeros(len(robot.right_force_fingers))
    robot.set_targets(robot.default_left_tcp, right, 0., 0.)
    np.testing.assert_allclose(robot.right.hand_data.qpos, robot.right_reach_q)
    # Two-segment insertion holds the wrist outside the stem until preshaped.
    robot.right_insert_tcp = robot.right_grasp_tcp.copy()
    robot.right_insert_tcp[:3] += [0., -.025, 0.]
    cfg['script']['support_insert_start_s'] = 5.5
    _, reached, _, _, _ = _scripted_state(5.5, cfg, robot)
    np.testing.assert_allclose(reached, robot.right_insert_tcp)
    _, reached, _, _, _ = _scripted_state(cfg['script']['support_reach_end_s'], cfg, robot)
    np.testing.assert_allclose(reached, robot.right_grasp_tcp)
    _, trial, _, _, _ = _scripted_state(trial_script_time(cfg['script']), cfg, robot)
    np.testing.assert_allclose(trial[:3] - robot.right_grasp_tcp[:3],
                              [0., 0., cfg['script']['stem_pinch_trial_height_m']], atol=1e-9)


def test_joint_approach_fk_does_not_change_observed_robot_state():
    from scipy.spatial.transform import Rotation
    from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    target = np.asarray(cfg['scene']['tomato_position'])
    grasp = target + Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw']).apply(cfg['scene']['stem_grasp_offset_m'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=Path.cwd(), tomato_position=target, support_position=grasp)
    if robot.right_approach_q is None:
        robot.right_approach_q = robot.right.solve_root_pose(robot.right_pregrasp_tcp[:3], robot.right_pregrasp_tcp[3:], robot._right_q)
    before = robot.tcp_poses()[1].copy()
    start = robot.right_approach_pose(0.)
    np.testing.assert_allclose(start, robot.default_right_tcp, atol=1e-7)
    robot.right_approach_pose(.5)
    np.testing.assert_allclose(robot.tcp_poses()[1], before, atol=1e-7)
    end = robot.right_approach_pose(1.)
    np.testing.assert_allclose(end[:3], robot.right_pregrasp_tcp[:3], atol=2e-5)


def test_left_tableware_guard_restores_joint_state_and_latches_stop():
    from scipy.spatial.transform import Rotation
    from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    target = np.asarray(cfg['scene']['tomato_position'])
    grasp = target + Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw']).apply(cfg['scene']['stem_grasp_offset_m'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=Path.cwd(), tomato_position=target, support_position=grasp)
    # Deliberately impossible elevated table checks rollback, not physics success.
    robot.left_tableware = SimpleNamespace(plate_center=np.zeros(2), plate_radii=np.ones(2)*10,
                                           floor=2., plate_height=.014)
    robot.minimum_left_tableware_clearance = .005
    for part, hull in robot.left_tableware_meshes:
        rotation = Rotation.from_euler('xyz', [.4, -.7, .2])
        np.testing.assert_allclose(rotation.apply(hull).min(0), rotation.apply(part.vertices).min(0))
        np.testing.assert_allclose(rotation.apply(hull).max(0), rotation.apply(part.vertices).max(0))
    before = robot.joint_q().copy()
    robot.set_targets(robot.default_left_tcp, robot.default_right_tcp, .2, .2)
    assert robot.left_tableware_blocked
    np.testing.assert_array_equal(robot.joint_q(), before)
    robot.left_tableware = None
    robot.set_targets(robot.default_left_tcp, robot.default_right_tcp, .5, .5)
    np.testing.assert_array_equal(robot.joint_q(), before)


@pytest.mark.parametrize('key,value', [('minimum_harvest_cluster_clearance_m', -1.),
                                     ('minimum_left_tableware_clearance_m', np.nan),
                                     ('harvest_workspace_timeout_s', 0.),
                                     ('maximum_harvest_stem_tilt_deg', 91.)])
def test_invalid_workspace_limits_fail_early(key, value):
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['task'][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_midpoint_preset_is_centered_and_does_not_relax_physics_or_stop_limits():
    old = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_midpoint.yaml').read_text())
    validate_config(cfg)
    np.testing.assert_allclose(cfg['scene']['stem_grasp_offset_m'],
                               cfg['scene']['branch_collision_center_offset_m'])
    assert old['scene']['stem_grasp_offset_m'][1] == -.195
    for key in ('branch_mass_kg', 'tomato_mass_kg', 'tomato_friction', 'hand_friction'):
        assert cfg['scene'][key] == old['scene'][key]
    assert cfg['task']['unsafe_contact_force_N'] == old['task']['unsafe_contact_force_N'] == 35.
    assert cfg['script']['stem_pinch_max_slip_m'] == old['script']['stem_pinch_max_slip_m']
    assert cfg['task']['minimum_left_tableware_clearance_m'] == .005
    assert cfg['task']['minimum_harvest_cluster_clearance_m'] == .06
    assert cfg['task']['maximum_harvest_stem_tilt_deg'] == 15.
    assert cfg['recording']['quality_filter'] == old['recording']['quality_filter']
