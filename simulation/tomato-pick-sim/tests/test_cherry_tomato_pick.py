from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation
import torch
import trimesh

from demos.cherry_tomato_pick.backend import (
    AttachmentParams,
    _AttachmentFailureState,
    _TomatoScene,
    attachment_wrench,
    ensure_procedural_assets,
    load_optional_visual_assets,
)
from demos.cherry_tomato_pick.asset_import import prepare_texture_coordinates
from demos.cherry_tomato_pick.camera import (
    Gemini305WristCameras,
    Gemini435LeCamera,
)
from demos.cherry_tomato_pick.rl import (
    TASK_ID,
    load_cherry_tomato_pick_config,
    make_cherry_tomato_pick_env_cfg,
)
from demos.cherry_tomato_pick.scene import (
    _SupportLossGuard,
    _support_pose_reached,
    _episode_seed,
    _randomize_recording_config,
    _scripted_state,
)
from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot, _distal_first_amount
from lycheesim.core.demo_registry import get_demo
from lycheesim.tasks.cherry_tomato_pick import (
    CherryTomatoAttachmentFacade,
    cherry_tomato_all_fruit_visual_assets,
    cherry_tomato_robot_asset,
    cherry_tomato_visual_assets,
)


def test_support_loss_guard_requires_continuous_gap_and_latches() -> None:
    guard = _SupportLossGuard()
    for _ in range(15):
        assert not guard.update(False, 0.005, 0.08)
    assert not guard.update(True, 0.005, 0.08)
    for _ in range(15):
        assert not guard.update(False, 0.005, 0.08)
    assert guard.update(False, 0.005, 0.08)
    assert guard.update(True, 0.005, 0.08)
    with pytest.raises(ValueError, match="positive"):
        guard.update(False, 0.005, 0.0)


def test_distal_first_wrap_curls_tips_before_finger_bases() -> None:
    for name in ("thumb_mcp", "thumb_ip", "index_finger_pip", "index_finger_dip"):
        assert _distal_first_amount(name, 0.0) == 0.0
        assert _distal_first_amount(name, 0.25) == 0.5
        assert _distal_first_amount(name, 1.0) == 1.0
        assert _distal_first_amount(name, 1.15) == pytest.approx(1.15)
    for name in ("thumb_cmc_flex", "thumb_cmc_abd", "index_finger_mcp_flex"):
        assert _distal_first_amount(name, 0.25) == 0.0
        assert _distal_first_amount(name, 1.0) == 1.0
    for name in ("thumb_ip", "index_finger_mcp_flex"):
        values = [_distal_first_amount(name, a) for a in np.linspace(0, 1.15, 100)]
        assert np.all(np.diff(values) >= 0.0)


def test_left_contact_control_stops_loaded_finger_and_respects_open_and_loss() -> None:
    from types import SimpleNamespace
    scene = object.__new__(_TomatoScene)
    scene.task_cfg = {"harvest_contact_control": True}
    scene.detached = False
    scene.support_lost = False
    scene.solver = SimpleNamespace(dt=0.005)
    scene.left_grip = 1.0
    scene.robot = SimpleNamespace(left_force_closure=np.full(3, 0.5))
    scene.left_grasp_force_ema = np.array([0.0, 10.0, 0.0])
    scene.left_contact_force_N = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    scene._drive_harvest_fingers()
    assert scene.robot.left_force_closure[0] > 0.5
    assert scene.robot.left_force_closure[1] < 0.5
    assert scene.robot.left_force_closure[2] > 0.5
    scene.support_lost = True
    held = scene.robot.left_force_closure.copy()
    scene._drive_harvest_fingers()
    np.testing.assert_array_equal(scene.robot.left_force_closure, held)
    scene.support_lost = False
    scene.left_grip = 0.0
    scene._drive_harvest_fingers()
    np.testing.assert_array_equal(scene.robot.left_force_closure, np.zeros(3))
    scene.robot.left_force_closure = None
    scene.task_cfg = {}
    scene._drive_harvest_fingers()
    assert scene.robot.left_force_closure is None


def test_support_confirmation_uses_ema_without_hiding_raw_contact_loss() -> None:
    scene = object.__new__(_TomatoScene)
    scene.stem_and_fruit = True
    scene.support_force_indices = [0, 1, 2, 3, 4]
    scene.right_contact_force_N = np.zeros(5)
    scene.right_grasp_force_ema = np.array([4., 4., 1., 0., 0.])
    assert scene._opposed_contact(scene.support_confirmation_forces()) == 4.0
    assert scene._opposed_contact(scene.support_contact_forces()) == 0.0
    guard = _SupportLossGuard()
    for _ in range(16):
        guard.update(scene._opposed_contact(scene.support_contact_forces()) >= .2, .005, .08)
    assert guard.lost
    scene.stem_and_fruit = False
    np.testing.assert_array_equal(scene.support_confirmation_forces(), np.zeros(5))


@pytest.mark.parametrize("forces, ready", [
    ([5., 4., 0., 0., 0.], False),
    ([5., 4., .49, 0., 0.], False),
    ([5., 4., .5, 0., 0.], True),
    ([5., 0., 4., .6, 0.], True),
    ([0., 4., 4., 4., 4.], False),
    ([2.9, 4., 4., 0., 0.], False),
])
def test_bundle_support_requires_three_fingers(forces, ready) -> None:
    scene = object.__new__(_TomatoScene)
    scene.stem_and_fruit = True
    scene.right_grasp_force_ema = np.asarray(forces)
    scene.task_cfg = {
        "minimum_support_bilateral_contact_N": 3.,
        "minimum_support_contact_fingers": 3,
        "minimum_support_finger_contact_N": .5,
    }
    assert scene.support_contacts_ready() is ready


def test_support_contact_count_defaults_to_existing_two_finger_gate() -> None:
    scene = object.__new__(_TomatoScene)
    scene.stem_and_fruit = True
    scene.right_grasp_force_ema = np.array([4., 4., 0., 0., 0.])
    scene.task_cfg = {"minimum_support_bilateral_contact_N": 3.}
    assert scene.support_contacts_ready()


def test_middle_grasp_requires_finished_wrap_and_contact() -> None:
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config(root / "configs/cherry_tomato_pick_main_stem.yaml")
    scene = object.__new__(_TomatoScene)
    scene.stem_and_fruit = True
    scene.task_cfg = cfg["task"]
    scene.robot = SimpleNamespace(
        right_force_fingers=("thumb", "index", "middle", "ring", "pinky"),
        right_force_closure=np.array([.26, .3, 1.09, 1.15, 1.15]),
    )
    scene.right_grasp_force_ema = np.array([5., 4., 2., 0., 0.])
    assert not scene.support_contacts_ready()
    scene.robot.right_force_closure[2] = 1.10
    assert scene.support_contacts_ready()
    scene.right_grasp_force_ema[2] = 0.0
    assert not scene.support_contacts_ready()
    scene.robot.right_force_closure = None
    assert not scene.support_contacts_ready()


def test_middle_grasp_keeps_left_harvest_sequence_after_support_lift() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config(root / "configs/cherry_tomato_pick_main_stem.yaml")
    target = np.asarray(cfg["scene"]["tomato_position"])
    robot = DualWuji2PickRobot(
        cfg["robot"], repository_root=root, tomato_position=target,
        support_position=target + cfg["scene"]["stem_grasp_offset_m"],
    )
    assert robot.right_distal_first_fingers == ("thumb", "index")
    for joint, minimum in (("thumb_mcp", .75), ("thumb_ip", 1.10),
                           ("index_finger_pip", 1.07), ("index_finger_dip", 1.28)):
        assert robot.right_closed_q[robot.right.canonical_qadr[joint]] >= minimum
    script = cfg["script"]
    lift = np.asarray(script["support_lift_offset_m"])
    for start, end, expected_phase in (
        ("support_lift_end_s", "harvest_approach_end_s", "harvest_approach"),
        ("harvest_approach_end_s", "harvest_reach_end_s", "harvest_reach"),
        ("harvest_reach_end_s", "harvest_close_end_s", "harvest_close"),
        ("settle_end_s", "pull_end_s", "pull_twist"),
    ):
        left, right, closure, _, phase = _scripted_state(
            (script[start] + script[end]) / 2.0, cfg, robot,
        )
        assert phase == expected_phase
        assert closure[1] == 1.0
        np.testing.assert_allclose(right[:3], robot.right_grasp_tcp[:3] + lift)
        assert not np.allclose(left, robot.default_left_tcp)
        if phase in ("harvest_close", "pull_twist"):
            assert closure[0] > 0.0


def _attachment() -> AttachmentParams:
    return AttachmentParams(
        anchor=np.asarray((0.0, 0.0, 0.365)),
        local_point=np.asarray((0.0, 0.0, 0.025)),
        break_direction_world=np.asarray((0.0, 0.0, 1.0)),
        axial_stiffness=280.0,
        lateral_stiffness=70.0,
        axial_damping=2.5,
        lateral_damping=1.0,
        bending_stiffness=0.10,
        torsional_stiffness=0.50,
        bending_damping=0.002,
        torsional_damping=0.01,
        break_force_N=2.0,
        break_torque_Nm=0.20,
    )


def test_moving_parent_does_not_add_pedicel_damping_load() -> None:
    params = _attachment()
    params.anchor[:] = (0.0, 0.0, 0.3)
    params.local_point[:] = 0.0
    velocity = np.array((0.1, -0.2, 0.3))
    angular = np.array((0.2, 0.0, -0.1))
    force, torque, load, _ = attachment_wrench(
        params, position=params.anchor.copy(),
        quaternion_xyzw=np.array((0., 0., 0., 1.)),
        linear_velocity=velocity, angular_velocity=angular,
        anchor_velocity=velocity, parent_angular_velocity=angular,
    )
    np.testing.assert_allclose(force, 0.0, atol=1e-8)
    np.testing.assert_allclose(torque, 0.0, atol=1e-8)
    assert load == 0.0


def test_bundle_grasp_pads_start_outside_selected_colliders() -> None:
    from demos.cherry_tomato_pick.tools.fit_stem_grasp import cylinder_gap, sphere_gap
    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config(root / "configs/cherry_tomato_pick_end_bundle.yaml")
    target = np.asarray(cfg["scene"]["tomato_position"])
    stem = target + cfg["scene"]["stem_grasp_offset_m"]
    robot = DualWuji2PickRobot(cfg["robot"], repository_root=root,
                              tomato_position=target, support_position=stem)
    robot.set_targets(robot.left_grasp_tcp, robot.right_grasp_tcp, 0., 1.)
    for finger in ("thumb", "index", "middle"):
        part = robot.right.tactile_parts[finger]
        pos, quat = robot.right.hand_geom_pose(part.source_id)
        vertices = Rotation.from_quat(quat).apply(part.vertices) + pos
        if finger == "thumb":
            axis = target + cfg["scene"]["branch_collision_center_offset_m"]
            gap = cylinder_gap(vertices, axis, cfg["scene"]["branch_radius_m"])
        else:
            center = target + (np.array([.010528537, -.173786080, .002170317])
                               if finger == "index" else cfg["scene"]["support_tomato_offset_m"])
            gap = sphere_gap(vertices, center, cfg["scene"]["tomato_radius_m"])
        assert 0.0 <= gap < 5e-4


def test_middle_stem_grasp_approaches_open_then_wraps_all_fingers() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config(root / "configs/cherry_tomato_pick_main_stem.yaml")
    target = np.asarray(cfg["scene"]["tomato_position"])
    offset = np.asarray(cfg["scene"]["stem_grasp_offset_m"])
    np.testing.assert_allclose(offset, cfg["scene"]["branch_collision_center_offset_m"])
    assert np.linalg.norm(offset) > 0.08  # Leave the target end accessible.
    robot = DualWuji2PickRobot(
        cfg["robot"], repository_root=root,
        tomato_position=target, support_position=target + offset,
    )
    assert robot.right_force_fingers == ("thumb", "index", "middle", "ring", "pinky")
    assert cfg["script"]["support_close_on_early_contact"] is False
    assert cfg["script"]["support_require_reached_pose"] is True
    for time_s in (0.4, 1.2, cfg["script"]["support_reach_end_s"]):
        left, right, closure, _, _ = _scripted_state(time_s, cfg, robot)
        robot.right_force_closure = np.zeros(5)
        robot.set_targets(left, right, *closure)
        np.testing.assert_allclose(robot.right.hand_data.qpos, 0.0, atol=1e-9)
        assert not robot.ik_failed
    robot.right_force_closure = np.full(5, 0.25)
    # During the right-hand wrap the left hand stays at its waiting pose;
    # its lower harvesting attitude is used only after the bunch is raised.
    robot.set_targets(robot.default_left_tcp, robot.right_grasp_tcp, 0.0, 1.0)
    assert not robot.ik_failed
    for name, fraction in (("thumb_ip", 0.5), ("index_finger_pip", 0.5),
                           ("index_finger_mcp_flex", 0.0), ("middle_finger_pip", 0.25)):
        address = robot.right.canonical_qadr[name]
        assert robot.right.hand_data.qpos[address] == pytest.approx(
            fraction * robot.right_closed_q[address]
        )
    robot.right_force_closure = np.ones(5)
    robot.set_targets(robot.default_left_tcp, robot.right_grasp_tcp, 0.0, 1.0)
    assert not robot.ik_failed
    for name, address in robot.right.canonical_qadr.items():
        if name.endswith(("pip", "dip", "mcp_flex")) or name in ("thumb_mcp", "thumb_ip"):
            assert robot.right.hand_data.qpos[address] > 0.2
    # Account for the bunch falling onto the table before the approach finishes.
    settled_rotation = Rotation.from_euler("y", 0.09)
    settled = robot.right_grasp_tcp.copy()
    settled[:3] = (
        target + np.array([0.0012, 0.0002, -0.017])
        + settled_rotation.apply(settled[:3] - target)
    )
    settled[3:] = (
        settled_rotation * Rotation.from_quat(settled[3:])
    ).as_quat()
    robot.set_targets(robot.default_left_tcp, settled, 0.0, 1.0)
    assert not robot.ik_failed


def test_support_pose_gate_checks_translation_and_orientation() -> None:
    target = np.array([.4, 0., .2, 0., 0., 0., 1.])
    assert _support_pose_reached(target, target)
    displaced = target.copy()
    displaced[0] += .005
    assert not _support_pose_reached(displaced, target)
    rotated = target.copy()
    rotated[3:] = Rotation.from_euler("z", .1).as_quat()
    assert not _support_pose_reached(rotated, target)


def test_middle_grasp_stays_on_the_right_arm_approach_side() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config(root / "configs/cherry_tomato_pick_main_stem.yaml")
    target = np.asarray(cfg["scene"]["tomato_position"])
    center = target + cfg["scene"]["stem_grasp_offset_m"]
    base = np.asarray(cfg["robot"]["source_mounts"]["right_base"]["pos"])
    robot = DualWuji2PickRobot(
        cfg["robot"], repository_root=root,
        tomato_position=target, support_position=center,
    )
    wrist = robot.right_grasp_tcp[:3]
    pregrasp = robot.right_pregrasp_tcp[:3]
    assert wrist[1] < center[1] - 0.01
    assert wrist[0] < center[0]
    assert np.dot((wrist - center)[:2], (base - center)[:2]) > 0.0
    assert np.linalg.norm((pregrasp - base)[:2]) < np.linalg.norm((wrist - base)[:2])
    assert pregrasp[2] > wrist[2]


def test_support_contact_selection_includes_ring_and_pinky() -> None:
    scene = object.__new__(_TomatoScene)
    scene.support_force_indices = list(range(5))
    scene.right_contact_force_N = np.asarray([2.0, 0.0, 0.0, 3.0, 4.0])
    assert scene._opposed_contact(scene.support_contact_forces()) == 2.0


def test_attachment_reaction_conserves_total_wrench() -> None:
    from types import SimpleNamespace
    scene = object.__new__(_TomatoScene)
    scene.main_stem_grasp = True
    scene.branch_index = 0
    origin = np.array((0.3, -0.1, 0.2))
    scene.branch_com_local = np.array((0.02, -0.08, 0.0))
    scene.solver = SimpleNamespace(body_pose=lambda _: (origin, np.array((0., 0., 0., 1.))))
    scene.branch_reaction_force = np.zeros(3)
    scene.branch_reaction_torque = np.zeros(3)
    child = np.array((0.4, 0.1, 0.3))
    force = np.array((1., -2., 3.))
    torque = np.array((0.1, 0.2, -0.1))
    scene._react_on_branch(child, force, torque)
    np.testing.assert_allclose(scene.branch_reaction_force + force, 0.0)
    total = (scene.branch_reaction_torque + torque + np.cross(child, force)
             + np.cross(origin + scene.branch_com_local, scene.branch_reaction_force))
    np.testing.assert_allclose(total, 0.0, atol=1e-12)


def _physical_attachment_config() -> dict[
    str, float | bool | list[float]
]:
    return {
        "tomato_local_point": [0.0, 0.0, 0.0],
        "break_direction_world": [0.0, 0.0, 1.0],
        "break_load_scale": 1.0,
    }


def _reset_physical_attachment(facade: CherryTomatoAttachmentFacade) -> None:
    facade.reset(
        torch.tensor([0, 1]),
        anchor=torch.zeros((2, 3)),
        break_force=torch.ones(2),
    )


def _update_physical_attachment(
    facade: CherryTomatoAttachmentFacade,
    target_position: torch.Tensor,
    armed: torch.Tensor,
):
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    return facade.update(
        target_position=target_position,
        target_rotation_xyzw=identity,
        harvest_armed=armed,
    )


def test_attachment_break_load_uses_only_axial_tension() -> None:
    params = _attachment()
    force, torque, force_load, torque_load = attachment_wrench(
        params,
        position=np.asarray((0.0, 0.0, 0.330)),
        quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    np.testing.assert_allclose(force, (0.0, 0.0, 2.8), atol=1.0e-6)
    np.testing.assert_allclose(torque, 0.0, atol=1.0e-6)
    assert force_load == 0.0
    assert torque_load == 0.0

    force, _torque, force_load, _torque_load = attachment_wrench(
        params,
        position=np.asarray((0.0, 0.0, 0.350)),
        quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    np.testing.assert_allclose(force, (0.0, 0.0, -2.8), atol=1.0e-6)
    assert force_load == pytest.approx(1.4)

    lateral_force, _torque, lateral_load, _torque_load = attachment_wrench(
        params,
        position=np.asarray((0.01, 0.0, 0.340)),
        quaternion_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    np.testing.assert_allclose(lateral_force, (-0.7, 0.0, 0.0), atol=1.0e-6)
    assert lateral_load == pytest.approx(0.0, abs=1.0e-12)


def test_attachment_break_requires_filtered_persistent_overload() -> None:
    state = _AttachmentFailureState(
        dt=0.005,
        ema_time_constant_s=0.030,
        minimum_overload_s=0.080,
    )
    assert not state.update(100.0, armed=True)
    for _ in range(14):
        assert not state.update(100.0, armed=True)
    assert state.overload_duration_s == pytest.approx(0.075)
    assert state.update(100.0, armed=True)

    state.update(0.0, armed=False)
    assert state.overload_duration_s == 0.0
    assert not state.update(100.0, armed=False)
    assert state.overload_duration_s == 0.0


def test_attachment_twist_uses_xyzw_quaternion() -> None:
    params = _attachment()
    yaw = 0.5
    quaternion = np.asarray((0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)))
    _force, torque, _force_load, torque_load = attachment_wrench(
        params,
        position=np.asarray((0.0, 0.0, 0.340)),
        quaternion_xyzw=quaternion,
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    assert torque[2] == np.float32(-0.25)
    assert torque_load == pytest.approx(1.25)

    pitch_quaternion = np.asarray(
        (0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2))
    )
    pitch_position = params.anchor - Rotation.from_quat(
        pitch_quaternion
    ).apply(params.local_point)
    _force, bending_torque, _force_load, bending_load = attachment_wrench(
        params,
        position=pitch_position,
        quaternion_xyzw=pitch_quaternion,
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    assert bending_torque[1] == np.float32(-0.05)
    assert bending_load == pytest.approx(0.25)


def test_fixed_attachment_requires_solver_runtime_views() -> None:
    with pytest.raises(TypeError):
        CherryTomatoAttachmentFacade(  # type: ignore[call-arg]
            2, "cpu", _physical_attachment_config()
        )


def test_fixed_attachment_selected_reset_is_isolated() -> None:
    active = torch.ones(2, dtype=torch.bool)
    reaction = torch.zeros((2, 6))
    facade = CherryTomatoAttachmentFacade(
        2,
        "cpu",
        _physical_attachment_config(),
        constraint_active=active,
        constraint_wrench=reaction,
    )
    _reset_physical_attachment(facade)
    reaction[:, 2] = -1.2
    _update_physical_attachment(
        facade, torch.zeros((2, 3)), torch.ones(2, dtype=torch.bool)
    )

    facade.reset(
        torch.tensor([1]),
        anchor=torch.tensor([[0.2, 0.0, 0.0]]),
        break_force=torch.tensor([2.0]),
    )
    assert bool(facade.views.detached[0])
    assert not bool(facade.views.detached[1])
    assert facade.views.break_force[1].item() == pytest.approx(2.0)
    torch.testing.assert_close(active, torch.tensor([False, True]))


def test_fixed_attachment_rejects_lateral_break_load() -> None:
    active = torch.ones(2, dtype=torch.bool)
    reaction = torch.zeros((2, 6))
    facade = CherryTomatoAttachmentFacade(
        2,
        "cpu",
        _physical_attachment_config(),
        constraint_active=active,
        constraint_wrench=reaction,
    )
    _reset_physical_attachment(facade)
    reaction[:] = torch.tensor([
        [-1.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.2, 0.0, 0.0, 0.0],
    ])
    views = _update_physical_attachment(
        facade, torch.zeros((2, 3)), torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(
        views.attachment_load[:, 0], torch.tensor([0.0, 1.2])
    )
    torch.testing.assert_close(
        views.attachment_shear_load[:, 0], torch.tensor([1.2, 0.0])
    )
    torch.testing.assert_close(
        views.detached[:, 0], torch.tensor([False, True])
    )


def test_fixed_attachment_breaks_from_solver_reaction_and_disables_weld() -> None:
    config = _physical_attachment_config()
    active = torch.ones(2, dtype=torch.bool)
    # Engine convention: this is the weld reaction applied to the tomato.
    # Upward support is compressive; downward reaction is tensile for +Z pick.
    reaction = torch.tensor([
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.2, 0.0, 0.0, 0.0],
    ])
    facade = CherryTomatoAttachmentFacade(
        2,
        "cpu",
        config,
        constraint_active=active,
        constraint_wrench=reaction,
    )
    _reset_physical_attachment(facade)
    # reset clears the shared reaction view; restore the solver sample.
    reaction[:, 2] = torch.tensor([0.2, -1.2])
    views = _update_physical_attachment(
        facade, torch.zeros((2, 3)), torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(
        views.attachment_load[:, 0], torch.tensor([0.0, 1.2])
    )
    torch.testing.assert_close(
        views.detached[:, 0], torch.tensor([False, True])
    )
    torch.testing.assert_close(active, torch.tensor([True, False]))
    facade.reset(
        torch.tensor([1]),
        anchor=torch.zeros((1, 3)),
        break_force=torch.ones(1),
    )
    assert bool(active[1])
    assert not bool(views.detached[1])


def test_fixed_attachment_reset_can_start_selected_row_detached() -> None:
    config = _physical_attachment_config()
    active = torch.ones(2, dtype=torch.bool)
    reaction = torch.ones((2, 6))
    facade = CherryTomatoAttachmentFacade(
        2,
        "cpu",
        config,
        constraint_active=active,
        constraint_wrench=reaction,
    )
    _reset_physical_attachment(facade)
    facade.reset_detached(torch.tensor([1]))
    torch.testing.assert_close(
        facade.views.detached[:, 0], torch.tensor([False, True])
    )
    torch.testing.assert_close(active, torch.tensor([True, False]))
    torch.testing.assert_close(reaction[1], torch.zeros(6))


def test_dynamic_robot_asset_has_one_worldbody_root() -> None:
    path = cherry_tomato_robot_asset()
    assert isinstance(path, Path) and path.is_file()
    worldbody = ET.parse(path).getroot().find("worldbody")
    assert worldbody is not None
    assert [body.attrib["name"] for body in worldbody.findall("body")] == [
        "base_link"
    ]
    compiler = ET.parse(path).getroot().find("compiler")
    assert compiler is not None
    assert Path(compiler.attrib["meshdir"]).is_absolute()


def test_task_robot_drives_leave_carrier_passive_properties_unchanged() -> None:
    import mujoco
    from lycheesim.tasks.cherry_tomato_pick.assets import _cherry_tomato_robot_source_asset

    carrier = mujoco.MjModel.from_xml_path(str(_cherry_tomato_robot_source_asset()))
    task = mujoco.MjModel.from_xml_path(str(cherry_tomato_robot_asset()))
    assert task.nu == task.nv == 56
    for field in ("dof_damping", "dof_armature", "dof_frictionloss",
                  "jnt_stiffness", "qpos_spring", "qpos0", "jnt_range",
                  "jnt_limited", "eq_data"):
        np.testing.assert_array_equal(getattr(task, field), getattr(carrier, field))
    for index in range(task.nu):
        joint = task.joint(int(task.actuator_trnid[index, 0])).name
        arm = joint.startswith(("left_fr3v2_joint", "right_fr3v2_joint"))
        kp, kd, force = (600.0, 40.0, 87.0) if arm else (2.0, 0.1, 2.0)
        np.testing.assert_allclose(task.actuator_gainprm[index, 0], kp)
        np.testing.assert_allclose(task.actuator_biasprm[index, :3], [0, -kp, -kd])
        np.testing.assert_allclose(task.actuator_forcerange[index], [-force, force])
        np.testing.assert_allclose(
            task.actuator_ctrlrange[index], task.jnt_range[task.actuator_trnid[index, 0]])


def test_external_framework_visual_assets_are_complete() -> None:
    assets = cherry_tomato_visual_assets()
    assert set(assets) == {"target", "support", "cluster", "albedo", "metadata"}
    assert all(path.is_file() for path in assets.values())
    with np.load(assets["target"]) as target:
        assert target["vertices"].shape == (1463, 3)
        assert target["uvs"].shape == (1463, 2)
    with np.load(assets["support"]) as support:
        assert support["vertices"].shape == (1463, 3)
        assert support["uvs"].shape == (1463, 2)
    with np.load(assets["cluster"]) as cluster:
        assert cluster["vertices"].shape == (93543, 3)
        assert cluster["uvs"].shape == (93543, 2)

    all_fruit = cherry_tomato_all_fruit_visual_assets()
    assert set(all_fruit) == {
        "fruits", "stem", "albedo", "fruit_albedo", "metadata"
    }
    assert list(all_fruit["fruits"]) == [
        f"tomato_{index:02d}" for index in range(12)
    ]
    assert all(path.is_file() for path in all_fruit["fruits"].values())
    assert all_fruit["stem"].is_file()
    with Image.open(all_fruit["fruit_albedo"]) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    fruit_red = (rgb[..., 0] > 1.25 * rgb[..., 1]) & (rgb[..., 0] > 70.0)
    assert float(np.mean(fruit_red)) == 1.0
    for path in all_fruit["fruits"].values():
        with np.load(path) as fruit:
            assert fruit["vertices"].shape == (1463, 3)
            assert fruit["uvs"].shape == (1463, 2)
            np.testing.assert_allclose(
                0.5 * (
                    fruit["vertices"].min(axis=0)
                    + fruit["vertices"].max(axis=0)
                ),
                0.0,
                atol=1.0e-7,
            )


def test_demo_assets_are_procedural_and_small(tmp_path: Path) -> None:
    cfg = load_cherry_tomato_pick_config()
    paths = ensure_procedural_assets(tmp_path, cfg)
    assert set(paths) == {"tomato", "branch"}
    assert all(path.is_file() for path in paths.values())
    assert sum(path.stat().st_size for path in paths.values()) < 1_000_000


def test_optional_visual_assets_preserve_uvs_and_shared_texture(tmp_path: Path) -> None:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)),
        dtype=np.float32,
    )
    payload = {
        "vertices": vertices,
        "faces": np.asarray(((0, 1, 2),), dtype=np.int32),
        "normals": np.asarray(((0.0, 0.0, 1.0),) * 3, dtype=np.float32),
        "uvs": np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
    }
    target_path = tmp_path / "target.npz"
    cluster_path = tmp_path / "cluster.npz"
    support_path = tmp_path / "support.npz"
    np.savez_compressed(target_path, **payload)
    np.savez_compressed(cluster_path, **payload)
    np.savez_compressed(support_path, **payload)
    texture_path = tmp_path / "albedo.png"
    Image.new("RGBA", (4, 4), (220, 20, 10, 255)).save(texture_path)

    visuals = load_optional_visual_assets(
        tmp_path,
        {
            "assets": {
                "target_visual_npz": target_path.name,
                "cluster_visual_npz": cluster_path.name,
                "support_visual_npz": support_path.name,
                "albedo_texture": texture_path.name,
            }
        },
    )
    assert set(visuals) == {"tomato", "branch", "support_tomato"}
    assert visuals["tomato"].uvs.shape == (3, 2)
    assert visuals["tomato"].texture.shape == (4, 4, 4)
    assert visuals["tomato"].texture is visuals["branch"].texture
    assert visuals["tomato"].texture is visuals["support_tomato"].texture


def test_camera_visuals_use_separate_stem_and_fruit_materials() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_cherry_tomato_pick_config()
    visuals = load_optional_visual_assets(root, cfg)

    static_fruits = {
        f"cluster_tomato_{index:02d}"
        for index in range(12)
        if index not in {7, 10}
    }
    assert set(visuals) == {
        "branch",
        "tomato",
        "support_tomato",
        "target_pedicel",
        *static_fruits,
    }
    assert visuals["branch"].texture_path.name == "tomato_basecolor.png"
    assert visuals["branch"].body_name == "branch"
    pedicel = visuals["target_pedicel"]
    assert pedicel.texture_path.name == "tomato_basecolor.png"
    assert pedicel.body_name == "tomato"
    attachment = np.asarray(cfg["attachment"]["tomato_local_point"])
    assert np.linalg.norm(pedicel.vertices - attachment, axis=1).min() < 0.002
    for name, visual in visuals.items():
        if name in {"branch", "target_pedicel"}:
            continue
        assert visual.texture_path.name == "tomato_fruit_basecolor.png"
        red = visual.texture[..., 0].astype(np.float32)
        green = visual.texture[..., 1].astype(np.float32)
        assert np.all(red > 1.25 * green)
    assert visuals["tomato"].body_name == "tomato"
    assert visuals["support_tomato"].body_name == "support_tomato"
    assert visuals["cluster_tomato_00"].body_name == "branch"
    np.testing.assert_allclose(
        visuals["cluster_tomato_00"].local_position,
        [0.010528528653625, -0.10407525904115, 0.00158688837865],
    )


def test_imported_split_assets_match_audited_upstream_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_root = root / "demos" / "cherry_tomato_pick" / "assets"
    metadata = json.loads((asset_root / "runtime" / "metadata.json").read_text())
    assert metadata["source_revision"] == (
        "43610761c4f667dc24ac6b5f7d66b71ebf72c1a1"
    )
    assert metadata["target_name"] == "tomato_10"
    assert metadata["support_name"] == "tomato_07"
    assert set(metadata["source_detachable_targets"]) == {
        "tomato_02",
        "tomato_07",
        "tomato_09",
        "tomato_10",
    }
    assert metadata["detachable_targets"] == [
        f"tomato_{index:02d}" for index in range(12)
    ]
    assert set(metadata["attachment_local_points_m"]) == set(
        metadata["detachable_targets"]
    )
    assert set(metadata["attachment_point_sources"].values()) == {
        "source_explicit_anchor", "closest_source_stem_vertex_pair"
    }
    assert len(metadata["split_obj_sha256"]) == 13
    assert metadata["texture_coordinate_transform"] == (
        "OBJ_bottom_left_to_runtime_top_left_v=1-v"
    )
    assert metadata["uv_seams_preserved"] is True
    assert metadata["target_vertices"] > 1346
    support_path = asset_root / "runtime" / "support_tomato_visual.npz"
    assert support_path.is_file()
    with np.load(support_path) as payload:
        vertices = np.asarray(payload["vertices"])
    np.testing.assert_allclose(
        0.5 * (vertices.min(axis=0) + vertices.max(axis=0)),
        0.0,
        atol=1.0e-7,
    )


def test_all_tomato_uvs_sample_fruit_skin_not_leaf_atlas() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "demos"
        / "cherry_tomato_pick"
        / "assets"
        / "source"
        / "upstream_split_obj"
    )
    texture_path = source.parent / "textures" / "tomato_Mat_baseColor.png"
    with Image.open(texture_path) as image:
        texture = np.asarray(image.convert("RGB"))
    height, width = texture.shape[:2]

    for index in range(12):
        raw = trimesh.load(
            source / f"tomato_{index:02d}.obj", force="mesh", process=False
        )
        prepared = prepare_texture_coordinates(raw)
        source_uv = np.asarray(raw.visual.uv)
        runtime_uv = np.asarray(prepared.visual.uv)
        # Every seam-expanded vertex must survive; merging down to the 1,346
        # geometric positions loses atlas assignments on 115 seam vertices.
        assert len(prepared.vertices) == len(raw.vertices) == 1463
        np.testing.assert_allclose(runtime_uv[:, 0], source_uv[:, 0])
        np.testing.assert_allclose(runtime_uv[:, 1], 1.0 - source_uv[:, 1])

        x = np.clip(np.rint(runtime_uv[:, 0] * (width - 1)), 0, width - 1).astype(int)
        y = np.clip(np.rint(runtime_uv[:, 1] * (height - 1)), 0, height - 1).astype(int)
        colors = texture[y, x].astype(np.float32)
        fruit_red = (colors[:, 0] > 1.25 * colors[:, 1]) & (colors[:, 0] > 70.0)
        assert float(np.mean(fruit_red)) > 0.99, f"tomato_{index:02d} samples leaf atlas"


def test_script_reaches_dual_arm_pull_and_completion_phases() -> None:
    cfg = load_cherry_tomato_pick_config()
    tomato = np.asarray(cfg["scene"]["tomato_position"])
    robot = DualWuji2PickRobot(
        cfg["robot"],
        repository_root=Path(__file__).resolve().parents[1],
        tomato_position=tomato,
        support_position=tomato
        + np.asarray(cfg["scene"]["support_tomato_offset_m"]),
    )
    left, right, closure, carrier, phase = _scripted_state(1.85, cfg, robot)
    assert phase == "support_close"
    np.testing.assert_allclose(left, robot.default_left_tcp)
    assert 0.0 < closure[1] < 1.0
    np.testing.assert_allclose(carrier, 0.0)

    left, right, closure, carrier, phase = _scripted_state(2.45, cfg, robot)
    assert phase == "support_lift"
    assert 0.0 < carrier[2] < float(cfg["script"]["support_lift_offset_m"][2])
    np.testing.assert_allclose(closure, (0.0, 1.0))

    left, right, closure, carrier, phase = _scripted_state(5.3, cfg, robot)
    assert phase == "pull_twist"
    lift_offset = np.asarray(cfg["script"]["support_lift_offset_m"])
    assert left[2] > robot.left_grasp_tcp[2] + lift_offset[2]
    np.testing.assert_allclose(right[:3], robot.right_grasp_tcp[:3] + lift_offset)
    np.testing.assert_allclose(carrier, lift_offset)
    assert left.shape == right.shape == (7,)
    np.testing.assert_allclose(closure, 1.0)
    _left, _right, _closure, _carrier, phase = _scripted_state(6.7, cfg, robot)
    assert phase == "complete"
    assert len(robot.visual_parts) > 70
    assert len(robot.collision_parts) == 20
    assert robot.left.hand.nq == robot.right.hand.nq == 20
    assert set(robot.left.canonical_qadr) == set(robot.right.canonical_qadr)
    assert all("wuji2" in part.name for part in robot.collision_parts)
    assert len(robot.joint_names()) == 54
    assert robot.joint_q().shape == (54,)
    assert robot.joint_names()[0] == "left_fr3/left_fr3v2_joint1"
    assert robot.joint_names()[27] == "right_fr3/right_fr3v2_joint1"

    support = tomato + np.asarray(cfg["scene"]["support_tomato_offset_m"])
    radius = float(cfg["scene"]["tomato_radius_m"])
    robot.set_targets(
        robot.left_grasp_tcp, robot.right_grasp_tcp, 1.0, 1.0
    )
    poses = robot.collision_poses()
    checked = {
        "left_wuji2_thumb_patch": tomato,
        "left_wuji2_index_patch": tomato,
        "left_wuji2_middle_patch": tomato,
        "right_wuji2_thumb_patch": support,
        "right_wuji2_index_patch": support,
        "right_wuji2_middle_patch": support,
    }
    for part in robot.collision_parts:
        if part.name not in checked:
            continue
        position, quaternion = poses[part.name]
        vertices = Rotation.from_quat(quaternion).apply(part.vertices) + position
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=part.faces, process=False
        ).convex_hull
        _point, distance, _triangle = trimesh.proximity.closest_point_naive(
            mesh, [checked[part.name]]
        )
        surface_gap = float(distance[0]) - radius
        assert surface_gap >= -5.0e-5
        assert surface_gap <= 5.0e-4

    assert float(cfg["robot"]["left_harvest_depth_m"]) == pytest.approx(0.002)
    assert float(cfg["task"]["maximum_hold_distance_m"]) <= 0.015
    assert float(
        cfg["task"]["minimum_retention_bilateral_contact_N"]
    ) >= 0.10

    robot.set_targets(
        robot.left_grasp_tcp,
        robot.right_grasp_tcp,
        1.0,
        1.0,
        left_retention=1.0,
    )
    for name, offset in cfg["robot"][
        "left_retention_joint_offsets_rad"
    ].items():
        qadr = robot.left.canonical_qadr[name]
        assert robot.left.hand_data.qpos[qadr] == pytest.approx(
            robot.left_closed_q[qadr] + float(offset)
        )

    # The Beta2 model grows from its mount toward local -Z.  Both installed
    # wrists must therefore lie on the link8 +Z side, not inside the forearm.
    for side, kinematics in (("left", robot.left), ("right", robot.right)):
        link8_id = (
            kinematics.left_link8_id
            if side == "left"
            else kinematics.right_link8_id
        )
        link8_position = kinematics.carrier_data.xpos[link8_id]
        link8_outward = kinematics.carrier_data.xmat[link8_id].reshape(3, 3)[:, 2]
        wrist_id = kinematics._body_id(kinematics.hand, f"{side[0]}_wrist")
        wrist_position = kinematics._root_position + kinematics._root_rotation.apply(
            kinematics.hand_data.xpos[wrist_id]
        )
        assert np.dot(wrist_position - link8_position, link8_outward) > 0.02


def test_recording_randomization_is_reproducible_and_rigid() -> None:
    cfg = load_cherry_tomato_pick_config()
    randomized, metadata = _randomize_recording_config(cfg, seed=51)
    repeated, repeated_metadata = _randomize_recording_config(cfg, seed=51)

    assert metadata == repeated_metadata
    assert randomized == repeated
    assert "cluster_quaternion_xyzw" not in cfg["scene"]
    assert metadata["enabled"] is True

    offset = np.asarray(metadata["position_offset_m"])
    assert np.max(np.abs(offset[:2])) <= 0.015
    assert offset[2] == pytest.approx(0.0)
    euler = np.asarray(metadata["cluster_euler_xyz_deg"])
    assert np.all(np.abs(euler) <= np.asarray((2.0, 2.0, 12.0)))

    rotation = Rotation.from_quat(metadata["cluster_quaternion_xyzw"])
    nominal_target = np.asarray(cfg["scene"]["tomato_position"])
    sampled_target = np.asarray(randomized["scene"]["tomato_position"])
    np.testing.assert_allclose(sampled_target, nominal_target + offset)
    np.testing.assert_allclose(
        np.asarray(randomized["scene"]["stem_anchor"]) - sampled_target,
        rotation.apply(
            np.asarray(cfg["scene"]["stem_anchor"]) - nominal_target
        ),
    )
    np.testing.assert_allclose(
        randomized["scene"]["support_tomato_offset_m"],
        rotation.apply(cfg["scene"]["support_tomato_offset_m"]),
    )
    np.testing.assert_allclose(
        randomized["attachment"]["rest_quaternion_xyzw"],
        metadata["cluster_quaternion_xyzw"],
    )

    fixed_light = np.asarray(
        randomized["camera"]["sensor"]["light_direction"]
    )
    wrist_light = np.asarray(
        randomized["wrist_cameras"]["sensor"]["light_direction"]
    )
    np.testing.assert_allclose(fixed_light, wrist_light)
    assert np.linalg.norm(fixed_light) == pytest.approx(1.0)
    assert fixed_light[2] < 0.0

    _different, different_metadata = _randomize_recording_config(cfg, seed=52)
    assert different_metadata != metadata


def test_recording_seed_follows_auto_numbered_episode() -> None:
    assert _episode_seed(42, Path("episode_000001.h5")) == 42
    assert _episode_seed(42, Path("episode_000002.h5")) == 43
    assert _episode_seed(42, Path("episode.h5")) == 42
    with pytest.raises(ValueError, match="non-negative"):
        _episode_seed(-1, Path("episode_000001.h5"))


def test_gemini_435le_matches_photo_mount_and_published_optics() -> None:
    cfg = load_cherry_tomato_pick_config()
    camera = Gemini435LeCamera(cfg["camera"])
    np.testing.assert_allclose(camera.dimensions, [0.1385, 0.0405, 0.0700])
    assert camera.baseline_m == pytest.approx(0.095)
    assert (camera.width, camera.height) == (320, 200)
    assert np.linalg.norm(camera.target - camera.position) > 0.31

    view_direction = Rotation.from_quat(camera.quaternion_xyzw).apply(
        [0.0, 0.0, -1.0]
    )
    expected_direction = camera.target - camera.position
    expected_direction /= np.linalg.norm(expected_direction)
    np.testing.assert_allclose(view_direction, expected_direction, atol=1.0e-7)

    horizontal_fov = 2.0 * np.arctan(
        0.5 * camera.width / camera.intrinsic_matrix[0, 0]
    )
    vertical_fov = 2.0 * np.arctan(
        0.5 * camera.height / camera.intrinsic_matrix[1, 1]
    )
    assert np.degrees(horizontal_fov) == pytest.approx(94.0)
    assert np.degrees(vertical_fov) == pytest.approx(68.0)
    assert camera.frustum_lines().shape == (8, 2, 3)
    parts = {
        part.name.rsplit("/", 1)[-1]: part for part in camera.visual_parts
    }
    assert set(parts) == {
        "housing",
        "front_panel",
        "lens_0",
        "lens_1",
        "lens_2",
        "pedestal",
        "bracket",
    }
    pedestal = parts["pedestal"]
    bracket = parts["bracket"]
    pedestal_top = pedestal.position[2] + 0.5 * np.ptp(
        pedestal.vertices[:, 2]
    )
    bracket_bottom = bracket.position[2] - 0.5 * np.ptp(
        bracket.vertices[:, 2]
    )
    assert bracket_bottom == pytest.approx(pedestal_top, abs=1.0e-6)
    assert np.ptp(bracket.vertices[:, 2]) == pytest.approx(0.042)
    housing = parts["housing"]
    housing_world = Rotation.from_quat(housing.quaternion_xyzw).apply(
        housing.vertices
    ) + housing.position
    bracket_top = bracket.position[2] + 0.5 * np.ptp(bracket.vertices[:, 2])
    assert bracket_top >= float(housing_world[:, 2].min())


def test_dual_gemini_305_wrist_mounts_cover_the_grasp_targets() -> None:
    cfg = load_cherry_tomato_pick_config()
    cameras = Gemini305WristCameras(cfg["wrist_cameras"])
    root = Path(__file__).resolve().parents[1]
    target = np.asarray(cfg["scene"]["tomato_position"], dtype=np.float64)
    support = target + np.asarray(
        cfg["scene"]["support_tomato_offset_m"], dtype=np.float64
    )
    robot = DualWuji2PickRobot(
        cfg["robot"],
        repository_root=root,
        tomato_position=target,
        support_position=support,
    )

    assert cameras.model == "orbbec_gemini_305"
    np.testing.assert_allclose(cameras.dimensions, [0.042, 0.042, 0.023])
    assert cameras.baseline_m == pytest.approx(0.018)
    assert (cameras.width, cameras.height) == (320, 200)
    assert float(cameras.sensor_cfg["sample_hz"]) == pytest.approx(30.0)
    assert cameras.clipping_range_m == (0.04, 1.0)
    assert np.linalg.norm(cameras.local_positions["left"][:2]) == pytest.approx(
        0.080
    )
    assert np.linalg.norm(cameras.local_positions["right"][:2]) == pytest.approx(
        0.080
    )

    wrist_poses = (robot.left_grasp_tcp, robot.right_grasp_tcp)
    positions, quaternions = cameras.camera_poses(wrist_poses)
    for index, (side, wrist_pose, grasp_target) in enumerate(
        zip(cameras.sides, wrist_poses, (target, support), strict=True)
    ):
        recovered_local = Rotation.from_quat(wrist_pose[3:]).inv().apply(
            positions[index] - wrist_pose[:3]
        )
        np.testing.assert_allclose(
            recovered_local, cameras.local_positions[side], atol=1.0e-7
        )
        view_direction = Rotation.from_quat(quaternions[index]).apply(
            [0.0, 0.0, -1.0]
        )
        target_direction = grasp_target - positions[index]
        distance = float(np.linalg.norm(target_direction))
        target_direction /= distance
        view_angle = np.degrees(
            np.arccos(np.clip(np.dot(view_direction, target_direction), -1.0, 1.0))
        )
        assert cameras.clipping_range_m[0] <= distance <= 0.25
        assert view_angle < 35.0

    parts_by_side = {
        side: [part for part in cameras.visual_parts if part.side == side]
        for side in cameras.sides
    }
    assert all(len(parts) == 6 for parts in parts_by_side.values())
    for parts in parts_by_side.values():
        names = {part.name.rsplit("/", 1)[-1] for part in parts}
        assert names == {
            "housing",
            "front_panel",
            "lens_0",
            "lens_1",
            "mount_plate",
            "bracket_arm",
        }
    frustums = cameras.frustum_lines(wrist_poses)
    assert all(lines.shape == (8, 2, 3) for lines in frustums.values())


def test_demo_discovery_and_rl_contract() -> None:
    demo = get_demo("cherry-tomato-pick")
    assert demo.available
    assert TASK_ID == "LycheeSim-CherryTomatoPick-v0"
    cfg = load_cherry_tomato_pick_config()

    class _Backend:
        pass

    env_cfg = make_cherry_tomato_pick_env_cfg(
        lambda: _Backend(), device="cpu", training=False, cfg=cfg
    )
    action = env_cfg.actions["dual_arm_command"]
    assert action.action_dim == 10
    assert tuple(action.scale) == (
        0.10,
        0.10,
        0.10,
        1.5,
        2.0,
        0.10,
        0.10,
        0.10,
        1.5,
        2.0,
    )
    assert set(env_cfg.terminations) == {
        "time_out",
        "success",
        "dropped",
        "invalid_state",
    }
