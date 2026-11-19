"""Sensor qualification and mode boundaries; not physical lifting evidence."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.cherry_tomato_pick.bunch_wrap import advance_lift_hold, closure_feedback, contact_evidence, pose_lock_time, support_shapes, wrap_ready
from demos.cherry_tomato_pick.stem_pinch import validate_config


def test_per_finger_targets_preserve_order_and_balance_opposing_pair():
    from demos.cherry_tomato_pick.bunch_wrap import contact_targets, closure_rates
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    task = {'minimum_support_contact_fingers': 3, 'support_contact_target_by_finger_N':
            {'middle': 3., 'index': 3., 'thumb': 6., 'ring': 0., 'pinky': 0.}}
    targets = contact_targets(task, fingers)
    np.testing.assert_array_equal(targets, [6., 3., 3., 0., 0.])
    np.testing.assert_array_equal(closure_rates(targets, targets, free_rate=.25, gain=.1), np.zeros(5))
    np.testing.assert_array_equal(closure_rates(np.zeros(5), targets, free_rate=.25, gain=.1),
                                  [.25, .25, .25, 0., 0.])
    assert closure_rates(np.full(5, 10.), targets, free_rate=.25, gain=.1).max() < 0.
    np.testing.assert_array_equal(contact_targets({}, fingers), np.full(5, 3.))


def test_two_stage_lift_rests_at_trial_without_a_resume_velocity_jump():
    from demos.cherry_tomato_pick.bunch_wrap import lift_fraction
    from demos.cherry_tomato_pick.stem_pinch import trial_script_time
    from demos.cherry_tomato_pick.scene import _scripted_state
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_bunch_wrap.yaml').read_text())
    script=cfg['script']
    script['support_trial_ramp_duration_s']=1.5
    start=script['support_close_end_s']; end=script['support_lift_end_s']
    split=trial_script_time(script)
    assert split == start+1.5
    height=script['stem_pinch_trial_height_m']/script['support_lift_offset_m'][2]
    assert lift_fraction(start-1,script) == 0.
    assert lift_fraction(split,script) == pytest.approx(height)
    assert lift_fraction(end+1,script) == 1.
    times=np.linspace(start,end,1001)
    assert np.all(np.diff([lift_fraction(t,script) for t in times]) >= 0.)
    h=1e-4
    for t in (start,split,end):
        assert abs((lift_fraction(t+h,script)-lift_fraction(t,script))/h) < 1e-6
        assert abs((lift_fraction(t,script)-lift_fraction(t-h,script))/h) < 1e-6
        assert abs((lift_fraction(t+h,script)-2*lift_fraction(t,script)+lift_fraction(t-h,script))/h**2) < .001
    pose=np.array([.3,0.,.2,0.,0.,0.,1.])
    robot=SimpleNamespace(left_grasp_tcp=pose.copy(),right_grasp_tcp=pose.copy(),default_left_tcp=pose.copy())
    left,right,_,offset,_=_scripted_state(split,cfg,robot)
    np.testing.assert_array_equal(left,pose)
    assert right[2]-pose[2] == pytest.approx(script['stem_pinch_trial_height_m'])
    np.testing.assert_allclose(right[:3]-pose[:3],offset)


@pytest.mark.parametrize('duration',[0.,-1.,np.nan,True,'1',100.])
def test_split_lift_rejects_invalid_duration(duration):
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_bunch_wrap.yaml').read_text())
    cfg['script']['support_trial_ramp_duration_s']=duration
    with pytest.raises(ValueError,match='support_trial_ramp_duration_s'):validate_config(cfg)


def test_split_lift_cannot_silently_change_legacy_modes():
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['script']['support_trial_ramp_duration_s']=1.
    with pytest.raises(ValueError,match='bunch wrap'):validate_config(cfg)


@pytest.mark.parametrize('bad', [None, {}, {'thumb': 6.},
    {'thumb': np.nan, 'index': 3., 'middle': 3., 'ring': 0., 'pinky': 0.},
    {'thumb': 0., 'index': 3., 'middle': 3., 'ring': 3., 'pinky': 0.},
    {'thumb': 6., 'index': 3., 'middle': -1., 'ring': 0., 'pinky': 0.},
    {'thumb': 6., 'index': 0., 'middle': 3., 'ring': 0., 'pinky': 0.},
    {'thumb': True, 'index': 3., 'middle': 3., 'ring': 0., 'pinky': 0.}])
def test_invalid_per_finger_force_config_fails_before_simulation(bad):
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_bunch_wrap.yaml').read_text())
    cfg['task']['support_contact_target_by_finger_N']=bad
    if bad is None:  # An omitted/None mapping retains the scalar legacy setting.
        validate_config(cfg)
    else:
        with pytest.raises(ValueError):validate_config(cfg)


def test_per_finger_force_configuration_cannot_silently_affect_other_modes():
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    cfg['task']['support_contact_target_by_finger_N']={'thumb': 6., 'index': 3.}
    with pytest.raises(ValueError,match='bunch_wrap_control'):validate_config(cfg)


def sensor(force, pos, *, total=1000.):
    def array(v):
        return SimpleNamespace(numpy=lambda: np.array(v, dtype=float).reshape(2, 1, 3))
    return SimpleNamespace(force_matrix=array([force, [0.,0.,0.]]),
                           force_matrix_friction=array(np.zeros((2,3))),
                           position_matrix=array([pos,[0.,0.,0.]]),
                           total_force=array([[total,0.,0.],[0.,0.,0.]]))


def evidence():
    supports = [sensor([-3,0,0],[-.03,0,0]), sensor([3,0,0],[.03,0,0]),
                sensor([1,0,-2],[.02,0,-.02]), sensor([0,0,0],[0,0,0]), sensor([0,0,0],[0,0,0])]
    fruits = [sensor([0,0,0],[0,0,0]) for _ in range(5)]
    fruits[2] = supports[2]
    return supports, fruits


def ready(supports, fruits):
    return wrap_ready(supports, fruits, np.zeros(3), ['thumb','index','middle','ring','pinky'], .5)


def test_shape_filter_excludes_harvest_target_and_unknown_fruit():
    mapping={'a':'branch_collision_001','b':'support_tomato_collision_000'}
    assert support_shapes(mapping,['a','b'],'target') == ['branch_collision_000', *mapping.values()]
    for names in ([], ['a','a'], ['target'], ['missing']):
        with pytest.raises(ValueError):
            support_shapes(mapping,names,'target')


def test_only_filtered_contact_contributes_not_large_table_force():
    s=sensor([0,0,0],[0,0,0])
    assert contact_evidence(s)[0] == 0.
    s=sensor([3,0,0],[.03,0,0])
    magnitude, pos, _=contact_evidence(s)
    assert magnitude == 3.
    np.testing.assert_array_equal(pos,[.03,0,0])


def test_unqualified_contact_limits_closure_without_confirming_grasp():
    qualified = np.zeros(5)
    measured = np.array([0., 12., 0., 0., 0.])
    np.testing.assert_array_equal(closure_feedback(qualified, measured), measured)
    np.testing.assert_array_equal(qualified, np.zeros(5))
    s = [sensor([0, 0, 0], [0, 0, 0], total=12.) for _ in range(5)]
    assert not ready(s, s)


def test_pose_lock_is_after_air_preshape_not_halfway_through_initial_approach():
    assert pose_lock_time({'support_approach_end_s': 2., 'support_preshape_duration_s': 1.5}) == 3.5
    assert pose_lock_time({'support_approach_end_s': 2.}) == 2.


def test_obstacle_feedback_includes_contacts_rejected_by_support_filter():
    from demos.cherry_tomato_pick.backend import _TomatoScene
    scene = SimpleNamespace(right_bunch_sensors=[sensor([0, 0, 0], [0, 0, 0], total=4.)] * 5,
                            right_contact_force_N=np.zeros(5), support_force_indices=[0, 2, 4])
    np.testing.assert_array_equal(_TomatoScene.right_obstacle_contact_forces(scene), np.full(5, 4.))
    np.testing.assert_array_equal(_TomatoScene.support_contact_forces(scene), np.zeros(3))


def test_nonfinite_obstacle_force_stops_feedback_clearly():
    with pytest.raises(ValueError, match='nonfinite'):
        closure_feedback(np.zeros(5), [0., np.nan, 0., 0., 0.])


@pytest.mark.parametrize('modified_target', [False, True])
def test_joint_approach_is_consumed_once_only_for_its_matching_tcp(monkeypatch, modified_target):
    from scipy.spatial.transform import Rotation
    from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_stem_pinch.yaml').read_text())
    target = np.asarray(cfg['scene']['tomato_position'])
    grasp = target + Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw']).apply(cfg['scene']['stem_grasp_offset_m'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=Path.cwd(),
                              tomato_position=target, support_position=grasp)
    pose = robot.right_approach_pose(.15)
    expected = robot._right_joint_waypoint[1].copy()
    if modified_target:
        pose[0] += .001
    def fail_ik(*args, **kwargs):
        raise RuntimeError('test sentinel: no redundant IK permitted')
    monkeypatch.setattr(robot.right, 'solve_root_pose', fail_ik)
    robot.set_targets(robot.default_left_tcp, pose, 0., 0.)
    assert robot._right_joint_waypoint is None
    assert robot.ik_failed is modified_target
    if not modified_target:
        np.testing.assert_array_equal(robot._right_q, expected)
        robot.set_targets(robot.default_left_tcp, pose, 0., 0.)
        assert robot.ik_failed  # Identical later targets cannot reuse the waypoint.


def test_wrap_needs_thumb_two_other_fingers_and_lower_fruit_support():
    s,f=evidence()
    assert ready(s,f)
    s[1]=sensor([0,0,0],[0,0,0])
    assert not ready(s,f)
    s,f=evidence();f[2]=sensor([0,0,0],[0,0,0])
    assert not ready(s,f)
    s,f=evidence();s[0]=sensor([0,0,0],[0,0,0])
    assert not ready(s,f)


def test_load_confirmation_filters_force_jitter_but_not_contact_loss_or_bad_geometry():
    s,f=evidence()
    fingers=['thumb','index','middle','ring','pinky']
    s[2]=f[2]=sensor([.4,0,-.8],[.02,0,-.02])
    args=dict(confirmation_forces=[3.,3.,2.,0.,0.],
              fruit_confirmation_forces=[0.,0.,2.,0.,0.],contact_floor=.2)
    assert not wrap_ready(s,f,np.zeros(3),fingers,1.5)
    assert wrap_ready(s,f,np.zeros(3),fingers,1.5,**args)
    # Stale EMA cannot confirm a contact that has disappeared.
    s[2]=f[2]=sensor([0.,0.,0.],[.02,0,-.02])
    assert not wrap_ready(s,f,np.zeros(3),fingers,1.5,**args)
    # Even a loaded finger cannot substitute stem force for a lower fruit contact.
    s,f=evidence(); f[2]=sensor([0.,0.,0.],[.02,0,-.02])
    assert not wrap_ready(s,f,np.zeros(3),fingers,1.5,**args)
    s,f=evidence(); s[2]=f[2]=sensor([2.,0.,1.],[.02,0,-.02])
    assert not wrap_ready(s,f,np.zeros(3),fingers,1.5,**args)


def test_retention_geometry_remains_raw_and_independent_of_confirmation_filter():
    s,f=evidence()
    s[2]=f[2]=sensor([.1,0,-.05],[.02,0,-.02])
    assert not wrap_ready(s,f,np.zeros(3),['thumb','index','middle','ring','pinky'],.2)


def test_backend_uses_load_ema_only_for_confirmation_and_raw_retention_separately():
    from demos.cherry_tomato_pick.backend import _TomatoScene
    s,f=evidence();s[2]=f[2]=sensor([.4,0,-.8],[.02,0,-.02])
    scene=SimpleNamespace(solver=SimpleNamespace(body_pose=lambda _: (np.zeros(3),[0.,0.,0.,1.])),
        branch_index=0,branch_com_local=np.zeros(3),right_sensors=s,right_support_fruit_sensors=f,
        robot=SimpleNamespace(right_force_fingers=['thumb','index','middle','ring','pinky']),
        task_cfg=dict(minimum_support_retention_contact_N=.2,minimum_support_finger_contact_N=1.5,
                      minimum_support_contact_fingers=3),right_grasp_force_ema=[3.,3.,2.,0.,0.],
        right_support_fruit_force_ema=[0.,0.,2.,0.,0.])
    assert _TomatoScene.wrap_contacts_ready(scene)
    assert _TomatoScene.wrap_contacts_ready(scene,retention=True)
    s[2]=f[2]=sensor([0.,0.,0.],[.02,0,-.02])
    assert not _TomatoScene.wrap_contacts_ready(scene)
    assert not _TomatoScene.wrap_contacts_ready(scene,retention=True)


def test_pushing_from_above_or_same_side_is_not_support():
    s,f=evidence();f[2]=sensor([1,0,2],[.02,0,-.02])
    assert not ready(s,f)
    s,f=evidence();f[2]=sensor([1,0,-2],[.02,0,.03])
    assert not ready(s,f)
    s,f=evidence();s[1]=sensor([-3,0,0],[-.03,0,0]);s[2]=sensor([-1,0,-2],[-.02,0,-.02])
    assert not ready(s,f)


def test_nonfinite_sensor_data_never_confirms_support():
    s,f=evidence();s[0]=sensor([np.nan,0,0],[-.03,0,0])
    assert not ready(s,f)


def test_missing_filtered_contact_capability_fails_clearly():
    s = sensor([3, 0, 0], [.03, 0, 0])
    s.position_matrix = None
    with pytest.raises(ValueError, match='requires filtered forces'):
        contact_evidence(s)


def test_full_lift_hold_resets_on_current_physics_failure():
    elapsed, verified = advance_lift_hold(1.99, .01, valid=True, required=2.)
    assert verified
    elapsed, verified = advance_lift_hold(elapsed, .01, valid=False, required=2.)
    assert elapsed == 0. and not verified
    elapsed, verified = advance_lift_hold(elapsed, .01, valid=True, required=2.)
    assert elapsed == .01 and not verified


def test_lift_only_never_releases_left_hand_even_after_nominal_harvest_time():
    from demos.cherry_tomato_pick.scene import _scripted_state
    cfg = yaml.safe_load(Path('configs/cherry_tomato_pick_bunch_wrap.yaml').read_text())
    pose = np.array([0., 0., .3, 0., 0., 0., 1.])
    robot = SimpleNamespace(default_left_tcp=pose.copy(), left_grasp_tcp=pose + .1,
                            right_grasp_tcp=pose.copy())
    for t in (cfg['script']['support_lift_end_s'], 20., 100.):
        left, right, closure, _, phase = _scripted_state(t, cfg, robot)
        np.testing.assert_array_equal(left, pose)
        np.testing.assert_allclose(right[:3], pose[:3] + cfg['script']['support_lift_offset_m'])
        assert closure[0] == 0. and phase == 'support_lift_wait'


def test_opt_in_wrap_keeps_open_start_and_separate_training_paths():
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_bunch_wrap.yaml').read_text())
    validate_config(cfg)
    assert cfg['script']['support_lift_only'] is True
    assert cfg['task']['unsafe_contact_force_N'] == 35.
    np.testing.assert_array_equal(cfg['robot']['right_pregrasp_hand']['positions'],0.)
    from demos.cherry_tomato_pick.scene import build
    with pytest.raises(ValueError,match='scripted experiment'):
        build(cfg,policy=object())
    cfg['task']['stem_pinch_control']=True
    with pytest.raises(ValueError,match='mutually exclusive'):
        validate_config(cfg)


@pytest.mark.parametrize('bad',[0.,-1.,np.inf,np.nan])
def test_wrap_full_lift_dwell_requires_positive_finite_time(bad):
    cfg=yaml.safe_load(Path('configs/cherry_tomato_pick_bunch_wrap.yaml').read_text())
    cfg['script']['support_full_lift_hold_s']=bad
    with pytest.raises(ValueError,match='support_full_lift_hold_s'):
        validate_config(cfg)
