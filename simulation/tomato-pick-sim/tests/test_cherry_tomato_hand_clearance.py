from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import mujoco

from demos.cherry_tomato_pick.hand_clearance import HandClearance
from demos.cherry_tomato_pick.backend import _TomatoScene
from demos.cherry_tomato_pick.rl import load_cherry_tomato_pick_config
from demos.cherry_tomato_pick.scene import _axial_pull_offset
from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot


@pytest.mark.parametrize("offset, expected", [(.04, .02), (.026, .006), (.01, -.01)])
def test_hand_hulls_include_compiled_mesh_offsets(offset, expected):
    # Off-centre source geometry exercises MuJoCo's mesh recentering transform.
    mesh = trimesh.creation.box(extents=(.02, .02, .02))
    mesh.vertices += [.03, -.01, .02]
    parts = [SimpleNamespace(name=f"robot/{s}_wuji2/palm", source="hand_geom",
                             vertices=mesh.vertices, faces=mesh.faces)
             for s in ("left", "right")]
    check = HandClearance(parts)
    rotation = Rotation.from_euler("z", 31, degrees=True)

    def pose(part):
        displacement = [offset, 0., 0.] if "/right_" in part.name else [0., 0., 0.]
        return rotation.apply(displacement), rotation.as_quat()

    distance = check.distance(pose)
    if expected < 0.0:
        assert distance <= 0.0
    else:
        assert distance == pytest.approx(expected, abs=1e-6)
    assert check.closest_pair == tuple(p.name for p in parts)


def test_guard_rejects_candidate_and_latches_without_moving_hands():
    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config(root / "configs/cherry_tomato_pick_main_stem.yaml")
    target = np.asarray(cfg["scene"]["tomato_position"])
    robot = DualWuji2PickRobot(cfg["robot"], repository_root=root,
                             tomato_position=target,
                             support_position=target + cfg["scene"]["stem_grasp_offset_m"])
    for name, address in robot.left.canonical_qadr.items():
        if name.startswith(("ring", "pinky")):
            assert robot.left_open_q[address] == robot.left_closed_q[address]
        else:
            assert robot.left_open_q[address] == 0.0
    before = [pose.copy() for pose in robot.tcp_poses()]
    before_joints = robot.left.hand_data.qpos.copy()
    robot.hand_clearance = SimpleNamespace(distance=lambda _: 0.0)
    robot.set_targets(robot.left_grasp_tcp, robot.right_grasp_tcp, .5, .5)
    assert robot.hand_collision_blocked
    for actual, expected in zip(robot.tcp_poses(), before):
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_array_equal(robot.left.hand_data.qpos, before_joints)
    robot.hand_clearance = SimpleNamespace(distance=lambda _: .05)
    robot.set_targets(robot.left_grasp_tcp, robot.right_grasp_tcp, 1., 1.)
    for actual, expected in zip(robot.tcp_poses(), before):
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_ambiguous_zero_query_requires_geometric_separation(monkeypatch):
    mesh = trimesh.creation.box(extents=(.02, .02, .02))
    parts = [SimpleNamespace(name=f"robot/{s}_wuji2/palm", source="hand_geom",
                             vertices=mesh.vertices, faces=mesh.faces)
             for s in ("left", "right")]
    check = HandClearance(parts)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: 0.0)
    for displacement, expected in ((.04, .02), (.01, 0.0)):
        def pose(part):
            return np.array([displacement if "/right_" in part.name else 0., 0., 0.]), [0., 0., 0., 1.]
        assert check.distance(pose) == pytest.approx(expected)


def test_pull_follows_attachment_axis_not_bent_fruit_orientation():
    axis = np.array([.2, -.1, .97])
    result = _axial_pull_offset([0., 0., .07], axis)
    assert np.linalg.norm(result) == pytest.approx(.07)
    np.testing.assert_allclose(result / .07, axis / np.linalg.norm(axis))
    with pytest.raises(ValueError):
        _axial_pull_offset([0., 0., .07], [0., 0., 0.])


def test_zero_query_does_not_mistake_aabb_gap_for_mesh_gap(monkeypatch):
    mesh = trimesh.creation.box(extents=(.02, .02, .02))
    parts = [SimpleNamespace(name=f"robot/{s}_wuji2/palm", source="hand_geom",
                             vertices=mesh.vertices, faces=mesh.faces)
             for s in ("left", "right")]
    check = HandClearance(parts)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: 0.0)
    rotation = Rotation.from_euler("z", 45, degrees=True)
    # .041 gives a positive AABB gap below 5 mm, despite a 21 mm mesh gap.
    for offset, gap in ((.035, .015), (.041, .021), (.024, .004), (.01, 0.)):
        def pose(part):
            xyz = [offset if "/right_" in part.name else 0., 0., 0.]
            return rotation.apply(xyz), rotation.as_quat()
        assert check.distance(pose) == pytest.approx(gap)


def test_post_detach_controller_uses_lower_force_target_and_can_release():
    scene = object.__new__(_TomatoScene)
    scene.task_cfg = {"harvest_contact_control": True, "harvest_contact_target_N": 5.,
                      "harvest_retention_contact_target_N": 2.}
    scene.detached = False
    scene.support_lost = False
    scene.solver = SimpleNamespace(dt=.005)
    scene.left_grip = 1.
    scene.robot = SimpleNamespace(left_force_closure=np.full(3, .5))
    scene.left_grasp_force_ema = np.full(3, 3.)
    scene.left_contact_force_N = np.full(5, 3.)
    scene._drive_harvest_fingers()
    pre_break = scene.robot.left_force_closure.copy()
    assert np.all(pre_break > .5)
    scene.detached = True
    scene._drive_harvest_fingers()
    assert np.all(scene.robot.left_force_closure < pre_break)
    np.testing.assert_allclose(scene.left_grasp_force_ema, 3.)
    scene.left_grip = 0.0
    scene._drive_harvest_fingers()
    np.testing.assert_array_equal(scene.robot.left_force_closure, np.zeros(3))
