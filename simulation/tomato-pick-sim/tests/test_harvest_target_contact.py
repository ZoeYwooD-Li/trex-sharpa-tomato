"""Target-only grasp evidence, obstacle feedback and raised-stage IK checks."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from demos.cherry_tomato_pick.backend import _TomatoScene
from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot


def scene_stub():
    scene = object.__new__(_TomatoScene)
    scene.solver = SimpleNamespace(dt=.005, body_pose=lambda _: (np.zeros(3), np.array([0., 0., 0., 1.])))
    scene.support_index = 0
    scene.main_stem_grasp = False
    scene.detached = scene.grasp_confirmed = scene.support_engaged = scene.support_lost = False
    scene.task_cfg = dict(minimum_support_grip=.72, maximum_support_distance_m=.03,
                          minimum_support_contact_N=1., minimum_grip=.72,
                          maximum_grasp_distance_m=.032, minimum_bilateral_contact_N=1.,
                          maximum_hold_distance_m=.015, minimum_retention_bilateral_contact_N=.1)
    scene.left_grip, scene.right_grip = 1., 0.
    scene.left_contact_force_N = np.array([5., 5., 0., 0., 0.])
    scene.left_target_contact_force_N = np.zeros(5)
    scene.right_contact_force_N = np.zeros(5)
    scene.robot = SimpleNamespace(nominal_grasp_center=lambda _: np.zeros(3),
                                  left_force_closure=np.full(3, .5), hand_collision_blocked=False)
    return scene


def test_neighbor_contacts_cannot_confirm_or_retain_target_grasp():
    scene = scene_stub()
    assert not scene._grasp_state(np.zeros(3))[1]
    assert not scene.grasp_confirmed
    scene.left_target_contact_force_N[:2] = 2.
    assert scene._grasp_state(np.zeros(3))[1]
    scene.detached = True
    scene.left_target_contact_force_N[:] = 0.
    assert not scene._grasp_state(np.zeros(3))[1]
    np.testing.assert_array_equal(scene.left_contact_force_N, [5., 5., 0., 0., 0.])


def test_obstacle_force_still_releases_fingers_without_target_contact():
    scene = scene_stub()
    scene.task_cfg.update(harvest_contact_control=True, harvest_contact_target_N=3.)
    scene.left_grasp_force_ema = np.array([5., 5., 0.])
    scene._drive_harvest_fingers()
    assert np.all(scene.robot.left_force_closure[:2] < .5)
    assert scene.robot.left_force_closure[2] > .5
    assert not scene._grasp_state(np.zeros(3))[1]


@pytest.mark.parametrize('offset', [[0., 0.], [0., 0., np.nan], [np.inf, 0., 0.]])
def test_invalid_harvest_validation_offset_fails_before_loading_assets(offset):
    with pytest.raises(ValueError, match='harvest_validation_offset'):
        DualWuji2PickRobot({}, repository_root=Path.cwd(), tomato_position=np.zeros(3),
                          support_position=np.zeros(3), harvest_validation_offset=offset)


def test_raised_harvest_ik_does_not_translate_the_stored_grasp_twice():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'configs/cherry_tomato_pick_stem_up_natural.yaml').read_text())
    target = np.asarray(cfg['scene']['tomato_position'])
    rotation = Rotation.from_quat(cfg['scene']['cluster_quaternion_xyzw'])
    robot = DualWuji2PickRobot(cfg['robot'], repository_root=root, tomato_position=target,
                             support_position=target + rotation.apply(cfg['scene']['stem_grasp_offset_m']),
                             harvest_validation_offset=cfg['script']['support_lift_offset_m'])
    expected = robot._calibrate_left_harvest(target)
    np.testing.assert_allclose(robot.left_grasp_tcp, expected)
    assert robot.left_grasp_tcp[2] < target[2] + .05
