"""Bounded Newton gravity-drop check of all three open boxes, without robot control."""

from pathlib import Path

import numpy as np
import trimesh
import warp as wp
import yaml

from lycheesim.engine.backend.newton_rigid import NewtonRigidSolver, RigidBodySpec
from demos.cherry_tomato_pick.tableware import Tableware


def main():
    cfg = yaml.safe_load(Path("configs/cherry_tomato_pick_deposit.yaml").read_text())
    props = Tableware(cfg["tableware"])
    paths = props.export(Path("outputs/tableware_assets"))
    radius = float(cfg["scene"]["tomato_radius_m"])
    sphere = props.asset_directory / "drop_test_sphere.obj"
    trimesh.creation.icosphere(subdivisions=3, radius=radius).export(sphere)
    bodies = []
    for fixture in props.fixtures:
        visual, hulls = paths[fixture.name]
        if hulls:
            bodies.append(RigidBodySpec(
                name=fixture.name, mesh_path=visual, collision_mesh_paths=hulls,
                collision_geometry_kinds=("convex_hull",)*len(hulls),
                center_origin=False, kinematic=True,
            ))
    fruit_indices = []
    for index, xy in enumerate(props.centers):
        fruit_indices.append(len(bodies))
        bodies.append(RigidBodySpec(name=f"test_fruit_{index}", mesh_path=sphere,
                                   init_pos=(*xy, .20), mass_kg=.012,
                                   collision_geometry_kinds=("convex_hull",)))
    with wp.ScopedDevice(cfg["sim"]["device"]):
        solver = NewtonRigidSolver(bodies, dt=.005, substeps=2,
                                   ground_z=cfg["solver"]["ground_z"],
                                   solver_kind=cfg["solver"]["backend"],
                                   rigid_contact_max=256, use_capture=False)
        for _ in range(300):
            solver.step()
        for index, body in enumerate(fruit_indices):
            position, _ = solver.body_pose(body)
            velocity, _ = solver.body_velocities((body,))
            assert props.inside(position, radius, index), (index, position)
            assert np.linalg.norm(velocity[0]) < .08
            print("BOX DROP PASS", index, position, "speed", np.linalg.norm(velocity[0]))


if __name__ == "__main__":
    main()
