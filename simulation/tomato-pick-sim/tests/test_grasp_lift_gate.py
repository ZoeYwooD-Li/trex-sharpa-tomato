"""Lift gating contracts. Physical success still requires a GPU rollout."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.cherry_tomato_pick.grasp_lift_gate import GraspLiftGate
from demos.cherry_tomato_pick.openpi_policy import PolicyController


def action(z=.4, closure=1.):
    return np.array([.4, .1, .3, 0, 0, 0, 1, .4, -.1, z, 0, 0, 0, 1, .2, closure])


def gate(timeout=8.):
    result = GraspLiftGate(True, timeout)
    result.configure(.72)
    return result


def test_default_off_and_open_hand_approach_unchanged():
    off = GraspLiftGate()
    np.testing.assert_equal(off.constrain(action(), .2, 0), action())
    assert not off.observe(100, False)
    g = gate()
    np.testing.assert_equal(g.constrain(action(closure=.71), .2, 0), action(closure=.71))
    assert g.phase == "approach"


def test_only_right_upward_motion_is_blocked_and_cap_ratchets_down():
    g = gate()
    original = action()
    output = g.constrain(original, .2, 1.)
    assert output[9] == .2
    np.testing.assert_equal(np.delete(output, 9), np.delete(original, 9))
    assert original[9] == .4
    assert g.constrain(action(z=.15), .2, 1.1)[9] == .15
    assert g.constrain(action(z=.151), .151, 1.2)[9] == .15
    assert g.limited_targets == 2


def test_closure_alone_does_not_confirm_and_timeout_does_not_reset_on_open():
    g = gate(2.)
    g.constrain(action(), .2, 1.)
    assert not g.observe(2., False)
    assert g.constrain(action(closure=0), .2, 2.)[15] == 0
    assert g.observe(3., False)
    assert g.phase == "timed_out"
    assert g.constrain(action(), .2, 3.)[9] == .2


def test_confirmation_requires_fresh_prediction_before_release():
    g = gate()
    g.constrain(action(), .2, 0.)
    assert g.observe(.3, True)
    assert g.phase == "confirmed"
    assert g.constrain(action(), .2, .3, fresh_inference=False)[9] == .2
    assert g.constrain(action(), .2, .33, fresh_inference=True)[9] == .4
    assert g.phase == "released"
    assert [e["phase"] for e in g.events] == ["waiting", "confirmed", "released"]


def test_contact_lost_before_replan_relocks_without_extending_deadline():
    g = gate(1.)
    g.constrain(action(), .2, 0.)
    g.observe(.3, True)
    assert g.observe(.31, False)
    assert g.phase == "waiting"
    assert g.constrain(action(), .2, .33, fresh_inference=True)[9] == .2
    assert g.observe(1., False)
    assert g.phase == "timed_out"


def test_fresh_prediction_opening_confirmed_hand_cannot_release_lift():
    g = gate(1.)
    g.constrain(action(), .2, 0.)
    g.observe(.3, True)
    output = g.constrain(action(closure=.1), .2, .33, fresh_inference=True)
    assert output[9] == .2
    assert output[15] == .1  # Preserve the model's opening command.
    assert g.phase == "waiting"
    g.observe(1., False)
    assert g.phase == "timed_out"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        GraspLiftGate(True, timeout)


def harness(*, timeout=8.):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "configs/cherry_tomato_pick_deposit.yaml").read_text())
    pose = np.array([.4, -.1, .2, 0, 0, 0, 1.])
    scene = SimpleNamespace(robot=SimpleNamespace(tcp_poses=lambda: (pose.copy(), pose.copy())),
                            left_grip=.2, right_grip=0., support_engaged=False)
    image = np.zeros((200, 320, 3), dtype=np.uint8)
    cameras = {"camera_color": image, "wrist_camera_color": {"left": image, "right": image}}
    calls, targets = [], []
    def infer(obs):
        calls.append(obs)
        return {"actions": np.tile(action(z=.3 + len(calls) * .05), (16, 1))}
    client = SimpleNamespace(infer=infer, metadata={}, url="ws://test")
    backend = SimpleNamespace(set_robot_command=lambda a: None,
                              set_robot_states=lambda l, r, c: targets.append((l, r, c)))
    policy = PolicyController(client, grasp_before_lift=True, grasp_wait_timeout_s=timeout)
    policy.configure(cfg)
    return policy, scene, cameras, backend, calls, targets


def test_controller_discards_old_chunk_and_replans_after_backend_confirmation():
    p, scene, cameras, backend, calls, targets = harness()
    p.apply(0., .005, scene, cameras, backend)
    assert p.grasp_gate.phase == "waiting"
    assert targets[-1][1][0, 2] == pytest.approx(.2)
    assert len(p.pending) == 3
    # No contact confirmation despite a full-close command.
    scene.right_grip = 1.
    p.update_grasp_wait(.1, scene)
    assert p.grasp_gate.phase == "waiting"
    scene.support_engaged = True
    p.update_grasp_wait(.105, scene)
    assert not p.pending
    assert p.grasp_gate.phase == "confirmed"
    p.apply(.135, .005, scene, cameras, backend)
    assert len(calls) == 2
    assert targets[-1][1][0, 2] == pytest.approx(.4)  # New, not cached .35 target.
    assert p.grasp_gate.phase == "released"
    assert p.summary()["control_mode"] == "hybrid_grasp_before_lift"


def test_controller_timeout_checked_between_policy_ticks():
    p, scene, cameras, backend, calls, targets = harness(timeout=.05)
    p.apply(0., .005, scene, cameras, backend)
    p.update_grasp_wait(.05, scene)
    assert "grasp confirmation timed out" in p.error
    with pytest.raises(RuntimeError, match="timed out"):
        p.apply(.065, .005, scene, cameras, backend)
    assert len(targets) == 1
    assert len(calls) == 1


def test_existing_confirmation_requires_wrap_opposition_multi_finger_and_hold():
    from demos.cherry_tomato_pick.backend import _TomatoScene
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]
                          / "configs/cherry_tomato_pick_deposit.yaml").read_text())
    scene = _TomatoScene.__new__(_TomatoScene)
    scene.task_cfg = cfg["task"]
    scene.main_stem_grasp = scene.stem_and_fruit = True
    scene.right_grip, scene.left_grip = 1., 0.
    scene.detached = False
    scene.support_index = 0
    scene.stem_grasp_position = np.zeros(3)
    scene.solver = SimpleNamespace(dt=.005, body_pose=lambda index: (np.zeros(3), np.array([0, 0, 0, 1])))
    scene.robot = SimpleNamespace(nominal_grasp_center=lambda side: np.zeros(3),
                                 right_force_fingers=["thumb", "index", "middle", "ring", "pinky"],
                                 right_force_closure=np.full(5, 1.1))
    scene.left_contact_force_N = np.zeros(5)
    scene.left_target_contact_force_N = np.zeros(5)
    scene.right_grasp_force_ema = np.full(5, 5.)
    scene.support_contact_steps = 49
    assert not scene._grasp_state(np.zeros(3))[0]  # Hold time not met.
    scene.support_contact_steps = 50
    assert scene._grasp_state(np.zeros(3))[0]
    scene.robot.right_force_closure[2] = .8
    assert not scene._grasp_state(np.zeros(3))[0]  # Incomplete wrap.
    scene.robot.right_force_closure[:] = 1.1
    scene.right_grasp_force_ema[:] = [0., 5., 5., 5., 5.]
    assert not scene._grasp_state(np.zeros(3))[0]  # No opposing thumb contact.
    scene.right_grasp_force_ema[:] = [5., 5., 0., 0., 0.]
    assert not scene._grasp_state(np.zeros(3))[0]  # Fewer than three fingers.
