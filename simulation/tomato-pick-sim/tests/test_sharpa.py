"""Bilateral Wave geometry/control integration; no harvest-success assertion."""
from pathlib import Path
import mujoco
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from demos.cherry_tomato_pick.sharpa import DualSharpaPickRobot
from demos.cherry_tomato_pick.wuji2 import _distal_first_amount

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="module")
def robot():
    cfg = yaml.safe_load((ROOT / "configs/cherry_tomato_pick_sharpa.yaml").read_text())
    scene = cfg["scene"]
    target = np.asarray(scene["tomato_position"])
    support = target + Rotation.from_quat(scene["cluster_quaternion_xyzw"]).apply(scene["stem_grasp_offset_m"])
    return DualSharpaPickRobot(cfg["robot"], repository_root=ROOT, tomato_position=target,
                              support_position=support,
                              harvest_validation_offset=cfg["script"]["support_lift_offset_m"])


def test_bilateral_models_joint_order_and_contacts(robot):
    assert robot.joint_q().shape == (58,)
    assert len(set(robot.joint_names())) == 58
    assert len(robot.collision_parts) == 20
    assert robot.hand_clearance.distance(robot.visual_pose) > 0.005
    for side in ("left", "right"):
        kin = getattr(robot, side)
        assert kin.hand.nq == 22
        assert all(mujoco.mj_id2name(kin.hand, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(side + "_")
                   for j in range(22))
        for f, part in kin.tactile_parts.items():
            p, q = kin.hand_geom_pose(part.source_id)
            np.testing.assert_allclose(kin.tip_positions()[f], p + Rotation.from_quat(q).apply(part.vertices.mean(axis=0)))
        for part in kin.visual_parts:
            if part.source == "hand_geom":
                expected = kin.hand_geom_pose(part.source_id)
                actual = robot.visual_pose(part)
                np.testing.assert_allclose(actual[0], expected[0], atol=1e-6)
                np.testing.assert_allclose(actual[1], expected[1], atol=1e-6)


def test_sharpa_profiles_and_contact_control(robot):
    for side in ("left", "right"):
        kin = getattr(robot, side)
        for suffix in ("open_q", "closed_q"):
            q = getattr(robot, f"{side}_{suffix}")
            assert np.isfinite(q).all()
            assert np.all(q >= kin.hand.jnt_range[:, 0])
            assert np.all(q <= kin.hand.jnt_range[:, 1])
    for name in ("thumb_MCP_FE", "thumb_IP", "index_PIP", "pinky_DIP"):
        assert _distal_first_amount(name, 0.25) == 0.5
    with pytest.raises(ValueError, match="22"):
        old = robot.cfg["left_harvest_hand"]
        try:
            robot.cfg["left_harvest_hand"] = {"joint_order": old["joint_order"][:20], "positions": old["positions"][:20]}
            robot._configured_hand_q("left_harvest_hand", robot.left)
        finally:
            robot.cfg["left_harvest_hand"] = old
