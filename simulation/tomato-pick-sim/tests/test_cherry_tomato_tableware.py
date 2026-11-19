from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial import ConvexHull
import yaml

from demos.cherry_tomato_pick.tableware import DepositMotion, Tableware


def props():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          "configs/cherry_tomato_pick_deposit.yaml").read_text())
    return Tableware(cfg["tableware"])


def test_deposit_config_records_to_incrementing_non_overwritten_episodes():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          "configs/cherry_tomato_pick_deposit.yaml").read_text())
    recording = cfg["recording"]
    assert recording["enabled"] is True
    assert recording["auto_increment"] is True
    assert recording["overwrite"] is False
    assert recording["output_path"].startswith("outputs/cherry_tomato_pick_deposit/")


def test_open_box_has_a_floor_and_four_walls_not_a_filled_hull(tmp_path):
    table = props()
    for fixture in table.fixtures[1:4]:
        assert len(fixture.colliders) == 5
        point = fixture.mesh.bounds.mean(axis=0)
        for mesh in fixture.colliders:
            assert mesh.is_watertight
            eq = ConvexHull(mesh.vertices).equations
            assert np.any(eq[:, :3] @ point + eq[:, 3] > 1e-6)
    table.export(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(table.asset_directory / "tableware.xml"))
    assert model.ngeom == 49 + 3 * 5 + 1 + 4
    assert (table.asset_directory / "plate.obj").is_file()
    assert (table.asset_directory / "tomato_boxes.obj").is_file()


def test_tabletop_covers_props_without_duplicating_ground_collisions(tmp_path):
    table = props()
    desktop = next(f for f in table.fixtures if f.name == "tabletop")
    assert desktop.mesh.is_watertight
    assert not desktop.colliders
    assert desktop.mesh.bounds[1, 2] == pytest.approx(table.floor-table.wall+.0002)
    for fixture in table.fixtures[:4]:
        assert np.all(fixture.mesh.bounds[0, :2] >= desktop.mesh.bounds[0, :2])
        assert np.all(fixture.mesh.bounds[1, :2] <= desktop.mesh.bounds[1, :2])
    assert all(not f.colliders for f in table.fixtures if f.name.startswith("tabletop"))
    table.export(tmp_path)
    import trimesh
    boxes = trimesh.load(table.asset_directory / "tomato_boxes.obj", force="mesh")
    assert boxes.bounds[0, 1] > .2  # Export must not accidentally include the desk.


def test_tabletop_can_be_disabled_without_changing_container_colliders():
    with_table = props()
    without_table = Tableware({**with_table.cfg, "tabletop": {"enabled": False}})
    assert not any(f.name.startswith("tabletop") for f in without_table.fixtures)
    for a, b in zip(with_table.fixtures[:5], without_table.fixtures, strict=True):
        np.testing.assert_array_equal(a.mesh.vertices, b.mesh.vertices)
        assert len(a.colliders) == len(b.colliders)


def test_release_requires_actual_position_over_selected_box():
    table = props()
    motion = DepositMotion(table, {})
    tcp = np.array([.35, .15, .3, 0., 0., 0., 1.])
    fruit = np.array([.35, .15, .2])
    motion.command(0., tcp, fruit, .02, True)
    _, grip, phase = motion.command(1.6, motion.goal, fruit, .02, True)
    assert grip == 1. and phase == "deposit_transport"
    destination = np.r_[table.centers[1], .2]
    motion.command(1.7, motion.goal, destination, .02, True)
    _, grip, phase = motion.command(2.2, motion.goal, destination, .02, False)
    assert grip == 0. and phase == "deposit_settle"
    for _ in range(70):
        motion.observe(np.r_[table.centers[1], table.floor+.02], np.zeros(3), .02, .005)
    assert motion.placed
    motion.observe(np.r_[table.centers[0], table.floor+.02], np.zeros(3), .02, .005)
    assert not motion.placed


def test_transport_timeout_never_opens_over_wrong_place():
    motion = DepositMotion(props(), {})
    tcp = np.array([.35, .15, .3, 0., 0., 0., 1.])
    fruit = np.array([.35, .15, .2])
    motion.command(0., tcp, fruit, .02, True)
    _, grip, phase = motion.command(3., tcp, fruit, .02, False)
    assert grip == 1. and phase == "deposit_failed" and motion.failed
    motion.release_time = 1.
    for _ in range(100):
        motion.observe(np.r_[motion.tableware.centers[1], .065], np.zeros(3), .02, .005)
    assert not motion.placed


def test_stem_up_preview_raises_deposit_without_changing_xy_or_wrist():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] /
                          "configs/cherry_tomato_pick_stem_up_natural.yaml").read_text())
    assert cfg['deposit']['drop_center_z_m'] == pytest.approx(.21)
    table = Tableware(cfg['tableware'])
    raised = DepositMotion(table, cfg['deposit'])
    old = DepositMotion(table, {**cfg['deposit'], 'drop_center_z_m': .18})
    tcp = np.array([.48, .09, .13, 0., 0., 0., 1.])
    fruit = np.array([.36, .01, .17])
    for motion in (old, raised):
        motion.command(0., tcp, fruit, .02, True)
    np.testing.assert_allclose(raised.goal[:3] - old.goal[:3], [0., 0., .03])
    np.testing.assert_array_equal(raised.goal[3:], old.goal[3:])
    mid, grip, phase = raised.command(.8, tcp, fruit, .02, True)
    np.testing.assert_allclose(mid[:3], (tcp[:3] + raised.goal[:3]) / 2)
    assert grip == 1. and phase == 'deposit_transport'
    destination = np.r_[table.centers[raised.index], .21]
    raised.command(1.6, raised.goal, destination, .02, True)
    assert raised.release_time == 1.6


@pytest.mark.parametrize("index", [-1, 3, True, 1.0])
def test_invalid_box_index(index):
    with pytest.raises(ValueError):
        DepositMotion(props(), {"target_box": index})
