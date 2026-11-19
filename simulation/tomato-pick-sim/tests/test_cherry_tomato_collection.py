from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from demos.cherry_tomato_pick.collection import (
    apply_exposure, fruit_tint, initial_geometry, prepare_collection,
    sample_appearance_and_goal, validate_initial_scene,
)
from demos.cherry_tomato_pick.recording import CherryTomatoEpisodeRecorder
from demos.cherry_tomato_pick.scene import _randomize_recording_config
from demos.cherry_tomato_pick.tableware import Tableware
from lycheesim.training import CameraSpec, EpisodeFrame


def config():
    return yaml.safe_load(Path("configs/cherry_tomato_pick_deposit.yaml").read_text())


def test_collection_reproducible_and_does_not_mutate_source():
    cfg = config()
    original = deepcopy(cfg)
    a = prepare_collection(cfg, 42, _randomize_recording_config)
    b = prepare_collection(cfg, 42, _randomize_recording_config)
    assert a == b
    assert cfg == original
    assert a[3]["status"] == "passed"
    assert a[3]["attempts"] > 1
    assert not a[3]["full_robot_collision_certified"]
    assert a[0]["deposit"]["target_box"] == a[2]["target_box_id"]
    assert a[0]["camera"]["sensor"]["light_direction"] == a[0]["wrist_cameras"]["sensor"]["light_direction"]


def test_uniform_goals_cover_three_boxes_independent_of_visual_settings():
    ids = set()
    for seed in range(30):
        cfg = config()
        _, goal = sample_appearance_and_goal(cfg, seed)
        ids.add(goal["target_box_id"])
        cfg["recording"]["randomization"]["visual"]["enabled"] = False
        _, other = sample_appearance_and_goal(cfg, seed)
        assert goal == other
    assert ids == {0, 1, 2}


def test_fixed_goal_and_visual_off():
    cfg = config()
    cfg["recording"]["randomization"]["enabled"] = False
    cfg["recording"]["target_condition"]["mode"] = "fixed"
    cfg["deposit"]["target_box"] = 2
    appearance, goal = sample_appearance_and_goal(cfg, 1)
    assert appearance == {"enabled": False}
    assert goal["target_box_one_hot"] == [0, 0, 1]
    assert fruit_tint("tomato", cfg) == (1., 1., 1.)


@pytest.mark.parametrize("indices", [[True], [], [3], [1, 1]])
def test_invalid_goal_choices_fail(indices):
    cfg = config()
    cfg["recording"]["target_condition"]["box_indices"] = indices
    with pytest.raises(ValueError, match="box_indices"):
        sample_appearance_and_goal(cfg, 1)


def test_rejection_budget_exhausted_with_reason():
    cfg = config()
    cfg["tableware"]["plate_center_xy"] = [10., 10.]
    cfg["recording"]["scene_validation"]["max_attempts"] = 2
    with pytest.raises(ValueError, match="after 2 attempts.*plate_overhang"):
        prepare_collection(cfg, 1, _randomize_recording_config)


def test_lift_correction_preserves_anchor_and_support_offsets():
    cfg = config()
    points = initial_geometry(cfg)
    candidate, _, _, _ = prepare_collection(cfg, 42, _randomize_recording_config)
    for key in ("tomato_position", "stem_anchor"):
        candidate["scene"][key][2] -= .012
    anchor_offset = np.array(candidate["scene"]["stem_anchor"]) - candidate["scene"]["tomato_position"]
    reasons, report = validate_initial_scene(candidate, points, cfg["recording"]["scene_validation"])
    assert not reasons
    assert report["height_correction_m"] > 0
    np.testing.assert_allclose(np.array(candidate["scene"]["stem_anchor"]) - candidate["scene"]["tomato_position"], anchor_offset)


def test_workspace_rejection():
    cfg = config()
    candidate, _, _, _ = prepare_collection(cfg, 42, _randomize_recording_config)
    candidate["robot"]["left_workspace_max"][1] = .1
    reasons, _ = validate_initial_scene(candidate, initial_geometry(cfg), cfg["recording"]["scene_validation"])
    assert reasons == ["left_workspace_proxy"]


def test_visual_gains_keep_geometry_and_stem_unchanged():
    cfg = config()
    before = Tableware(cfg["tableware"])
    appearance, _ = sample_appearance_and_goal(cfg, 42)
    after = Tableware(cfg["tableware"])
    after.apply_tabletop_gain(appearance["tabletop_gain"])
    assert before.cfg == after.cfg  # Appearance does not create per-episode mesh caches.
    for old, new in zip(before.fixtures, after.fixtures):
        np.testing.assert_array_equal(old.mesh.vertices, new.mesh.vertices)
        if old.name.startswith("tabletop"):
            assert old.color != new.color
    assert fruit_tint("branch", cfg) == (1., 1., 1.)
    assert fruit_tint("target_pedicel", cfg) == (1., 1., 1.)
    assert fruit_tint("tomato", cfg) == fruit_tint("cluster_tomato_01", cfg)
    rgb = np.array([[[255, 100, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(apply_exposure(rgb, {"exposure_gain": 1.2}), [[[255, 120, 0]]])
    np.testing.assert_array_equal(rgb, [[[255, 100, 0]]])


def recorder(tmp_path, *, failure=None, terminal=True, camera=True):
    out = CherryTomatoEpisodeRecorder(
        tmp_path / "episode_000001.h5", seed=42, object_ids=("tomato_10",),
        cameras={"camera": CameraSpec(image_size=(2, 2))},
        metadata={"quality_filter": {"enabled": True},
                  "scene_validation": {"status": "passed"},
                  "target_condition": {"target_box_one_hot": [0, 1, 0], "target_box_id": 1,
                                       "policy_input_required": True},
                  "camera_streams": ["camera"], "camera_sample_hz": {"camera": 30.},
                  "unsafe_contact_force_N": 30.,
                  "robot_sample_hz": 30., "physics_dt_s": .005},
    )
    for i, t in enumerate([0., .035]):
        if camera:
            out.append_camera("camera", time=t, color=np.ones((2, 2, 3), dtype=np.uint8),
                              depth_m=np.ones((2, 2)), world_from_camera=np.eye(4), valid_depth_pixels=4)
        extra = {"task_state": np.array([1., 1., float(i), 0., float(i), 0., .01]),
                 "placed_in_box": np.array([bool(i)]), "deposit_failed": np.array([False]),
                 "support_lost": np.array([False]), "hand_collision_blocked": np.array([False]),
                 "cluster_clearance_m": np.array([.10]), "peak_contact_force_N": np.array([10.]),
                 "target_box_one_hot": np.array([0., 1., 0.])}
        if failure:
            extra[failure][0] = True
        out.append_robot(EpisodeFrame(time=t, images={}, joint_pos=np.zeros(2), joint_vel=np.zeros(2),
                                      ee_pose=np.zeros((2, 7)), gripper=np.zeros(2), extra=extra),
                         transition_action=np.zeros(2) if i else None, terminated=bool(i and terminal))
    return out


def test_success_filter_writes_authoritative_label_and_sidecar(tmp_path):
    out = recorder(tmp_path)
    out.close()
    with h5py.File(out.output_path) as f:
        metadata = json.loads(f["metadata"].attrs["extra"])
        assert metadata["quality"]["accepted"]
        assert "git_revision" in metadata["source"]
    assert json.loads(out.output_path.with_suffix(".quality.json").read_text())["status"] == "accepted"


@pytest.mark.parametrize("failure", ["support_lost", "deposit_failed", "hand_collision_blocked"])
def test_terminated_failures_are_not_successes(tmp_path, failure):
    out = recorder(tmp_path, failure=failure)
    out.close()
    assert not out.quality_report["accepted"]
    assert failure in out.quality_report["reasons"]
    assert out.output_path.exists()


def test_partial_and_missing_camera_preserved_but_rejected(tmp_path):
    out = recorder(tmp_path, terminal=False, camera=False)
    out.close()
    assert {"incomplete_episode", "missing_camera:camera"} <= set(out.quality_report["reasons"])
    with h5py.File(out.output_path) as f:
        assert f["cameras/camera/color"].shape == (0, 2, 2, 3)
        assert f["robot/camera_index/camera"][:].tolist() == [-1, -1]


@pytest.mark.parametrize("fault,reason", [
    ("force", "unsafe_contact_force"), ("nan", "nonfinite_robot_state"),
    ("goal", "missing_or_inconsistent_target_condition"),
    ("clearance", "bunch_not_lifted_at_end"), ("validation", "initial_scene_not_validated"),
    ("depth", "invalid_depth:camera"), ("camera_rate", "camera_time_gap:camera"),
])
def test_quality_rejects_corrupted_or_unsafe_evidence(tmp_path, fault, reason):
    out = recorder(tmp_path)
    if fault == "force":
        out._robot_frames[-1].extra["peak_contact_force_N"][0] = 31.
    elif fault == "nan":
        out._robot_frames[-1].joint_pos[0] = np.nan
    elif fault == "goal":
        out._robot_frames[-1].extra["target_box_one_hot"][:] = [1, 0, 0]
    elif fault == "clearance":
        out._robot_frames[-1].extra["cluster_clearance_m"][0] = 0.
    elif fault == "validation":
        out.metadata["scene_validation"]["status"] = "disabled"
    elif fault == "depth":
        out._camera_frames["camera"][0].depth_m[:] = np.nan
    elif fault == "camera_rate":
        out.metadata["camera_sample_hz"]["camera"] = 200.
    out.close()
    assert not out.quality_report["accepted"]
    assert reason in out.quality_report["reasons"]


def test_selector_excludes_failed_and_unassessed_files(tmp_path, monkeypatch, capsys):
    from demos.cherry_tomato_pick.quality import main
    good = recorder(tmp_path)
    good.close()
    bad = recorder(tmp_path, failure="support_lost")
    bad.close()
    # A stale accepted sidecar must not override the authoritative rejection.
    bad.output_path.with_suffix(".quality.json").write_text('{"accepted": true}')
    monkeypatch.setattr("sys.argv", ["quality", str(tmp_path)])
    main()
    assert capsys.readouterr().out.strip() == str(good.output_path)
