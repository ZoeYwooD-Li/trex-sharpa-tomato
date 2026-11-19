"""Fit three Wuji2 pads around the main stem and neighboring fruit proxies.

Run from the repository root; prints suggested YAML values without editing it.
"""
from pathlib import Path
import json
import argparse

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import yaml
import trimesh

from demos.cherry_tomato_pick.wuji2 import DualWuji2PickRobot, _distal_first_amount


def cylinder_gap(vertices, center, radius):
    """Signed gap to an infinite Y-axis cylinder (not the complete asset)."""
    points = (vertices - center)[:, (0, 2)]
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    edges = np.roll(polygon, -1, axis=0) - polygon
    t = np.clip(-np.sum(polygon * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
    distance = np.linalg.norm(polygon + t[:, None] * edges, axis=1).min()
    inside = np.all(hull.equations[:, 2] <= 0)
    return (-distance if inside else distance) - radius


def sphere_gap(vertices, center, radius):
    """Signed gap from a convex pad to an approximate spherical fruit."""
    hull = ConvexHull(vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=hull.simplices, process=False)
    _, distance, _ = trimesh.proximity.closest_point_naive(mesh, [center])
    inside = np.all(hull.equations[:, :3] @ center + hull.equations[:, 3] <= 0)
    return (-float(distance[0]) if inside else float(distance[0])) - radius


def fit_middle(cfg):
    """Keep a fully open approach while fitting a five-finger wrap."""
    target = np.asarray(cfg["scene"]["tomato_position"])
    grasp = target + cfg["scene"]["stem_grasp_offset_m"]
    robot = DualWuji2PickRobot(
        cfg["robot"], repository_root=Path.cwd(),
        tomato_position=target, support_position=grasp,
    )
    kin = robot.right
    original = robot.right_closed_q.copy()
    names = cfg["robot"]["right_cluster_grasp_hand"]["joint_order"]
    addresses = [kin.canonical_qadr[name] for name in names]
    metadata = json.loads(Path(cfg["assets"]["visual_metadata"]).read_text())
    offsets = metadata["part_centers_relative_to_target_m"]
    centers = np.array([target + offsets[n] for n in metadata["fruit_names"]])
    selected = [target + offsets[n] for n in cfg["robot"]["right_fit_contact_fruits"]]
    parts = [part for part in robot.collision_parts if part.side == "right"]
    hulls = [trimesh.Trimesh(vertices=p.vertices, faces=p.faces, process=False).convex_hull
             for p in parts]
    equations = [ConvexHull(h.vertices).equations for h in hulls]
    radius = float(cfg["scene"]["tomato_radius_m"])
    axis = target + cfg["scene"]["branch_collision_center_offset_m"]

    def residual(delta):
        robot.right_closed_q = original.copy()
        robot.right_closed_q[addresses] += delta[:20]
        robot.cfg["right_cluster_grasp_center_bias_m"] = delta[20:].tolist()
        pose = robot._calibrate_right_cluster_grasp(grasp)
        kin.set_hand_root_from_tcp(pose)
        contact = []
        for finger, center in zip(robot.right_force_fingers, selected):
            part = kin.tactile_parts[finger]
            pos, quat = kin.hand_geom_pose(part.source_id)
            vertices = Rotation.from_quat(quat).apply(part.vertices) + pos
            contact.append(sphere_gap(vertices, center, radius) - 0.0003)
        collisions = []
        # Check the open hand and intermediate closure at the reached TCP too.
        for amount in (0.0, 0.5, 1.0):
            q = robot.right_open_q + amount * (robot.right_closed_q - robot.right_open_q)
            for name, address in kin.canonical_qadr.items():
                if any(name.startswith(finger) for finger in robot.right_distal_first_fingers):
                    q[address] = robot.right_open_q[address] + _distal_first_amount(name, amount) * (
                        robot.right_closed_q[address] - robot.right_open_q[address]
                    )
            kin.set_hand_q(q)
            for part, hull, eq in zip(parts, hulls, equations):
                pos, quat = kin.hand_geom_pose(part.source_id)
                rotation = Rotation.from_quat(quat)
                sweep = (0.0, 0.25, 0.5, 0.75, 1.0) if amount == 0.0 else (0.0,)
                approach = np.asarray(cfg["robot"]["right_pregrasp_offset_m"])
                samples = np.concatenate([centers - t * approach for t in sweep])
                local = rotation.inv().apply(samples - pos)
                _, distances, _ = trimesh.proximity.closest_point_naive(hull, local)
                inside = np.all(local @ eq[:, :3].T + eq[:, 3] <= 0.0, axis=1)
                distances[inside] *= -1.0
                collisions.extend(np.minimum(distances - radius, 0.0))
                vertices = rotation.apply(part.vertices) + pos
                for t in sweep:
                    collisions.append(min(cylinder_gap(vertices, axis - t * approach, 0.004), 0.0))
        return np.r_[contact, np.asarray(collisions) * 4.0, delta[:20] * 2e-4, delta[20:] * 1e-3]

    result = least_squares(
        residual, np.zeros(23), bounds=([-.4] * 20 + [-.03] * 3, [.4] * 20 + [.03] * 3),
        diff_step=1e-4, max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8,
    )
    values = residual(result.x)
    print("contact_gaps_m:", values[:5] + .0003)
    print("worst_proxy_overlap_m:", values[5:-23].min() / 4.0)
    print("bias_m:", result.x[20:].tolist())
    print("positions:", robot.right_closed_q[addresses].tolist())
    print("cost:", result.cost, "evaluations:", result.nfev, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path("configs/cherry_tomato_pick_main_stem.yaml").read_text())
    if cfg["robot"].get("right_fit_contact_fruits"):
        if args.open:
            parser.error("middle grasp keeps the approach fully open; run without --open")
        fit_middle(cfg)
        return
    target = np.asarray(cfg["scene"]["tomato_position"])
    stem = target + cfg["scene"].get("stem_grasp_offset_m", cfg["scene"]["branch_collision_center_offset_m"])
    robot = DualWuji2PickRobot(cfg["robot"], repository_root=Path.cwd(),
                              tomato_position=target, support_position=stem)
    kin = robot.right
    original = robot.right_closed_q.copy()
    names = cfg["robot"]["right_cluster_grasp_hand"]["joint_order"][:12]
    addresses = [kin.canonical_qadr[name] for name in names]

    def geometry(delta):
        robot.right_closed_q = original.copy()
        robot.right_closed_q[addresses] += delta[:12]
        if args.open:
            kin.set_hand_q(robot.right_closed_q)
            pose = robot.right_grasp_tcp
        else:
            robot.cfg["right_cluster_grasp_center_bias_m"] = delta[12:].tolist()
            pose = robot._calibrate_right_cluster_grasp(stem)
        kin.set_hand_root_from_tcp(pose)
        vertices = []
        for finger in ("thumb", "index", "middle"):
            part = kin.tactile_parts[finger]
            pos, quat = kin.hand_geom_pose(part.source_id)
            vertices.append(Rotation.from_quat(quat).apply(part.vertices) + pos)
        return vertices

    axis = target + cfg["scene"]["branch_collision_center_offset_m"]
    fruit02 = target + np.array([.010528537, -.173786080, .002170317])
    fruit07 = target + cfg["scene"]["support_tomato_offset_m"]
    metadata = json.loads(Path(cfg["assets"]["visual_metadata"]).read_text())
    centers = np.array([target + np.array(metadata["part_centers_relative_to_target_m"][n])
                        for n in metadata["fruit_names"]])
    parts = [p for p in robot.collision_parts if p.side == "right"]
    hulls = [trimesh.Trimesh(vertices=p.vertices, faces=p.faces, process=False).convex_hull for p in parts]

    def residual(delta):
        selected = gaps(delta) - .00015
        if args.open:
            selected *= 0.0
        overlaps = []
        for part, hull in zip(parts, hulls):
            pos, quat = kin.hand_geom_pose(part.source_id)
            rotation = Rotation.from_quat(quat)
            sweep = [0., .5, 1.] if args.open else [0.]
            samples = np.concatenate([centers - amount * np.array(cfg["robot"]["right_pregrasp_offset_m"])
                                      for amount in sweep])
            local = rotation.inv().apply(samples - pos)
            _, distances, _ = trimesh.proximity.closest_point_naive(hull, local)
            eq = ConvexHull(hull.vertices).equations
            inside = np.all(local @ eq[:, :3].T + eq[:, 3] <= 0, axis=1)
            distances[inside] *= -1
            clearance = .006 if args.open else 0.0
            overlaps.extend(np.minimum(distances - .0172 - clearance, 0.0))
            vv = rotation.apply(part.vertices) + pos
            overlaps.append(min(cylinder_gap(vv, axis, .004) - clearance, 0.0))
        return np.r_[selected, np.array(overlaps) * 3., delta[:12] * 1e-4, delta[12:] * 1e-3]

    def gaps(delta):
        thumb, index, middle = geometry(delta)
        return np.array([cylinder_gap(thumb, axis, .004),
                         sphere_gap(index, fruit02, .0172),
                         sphere_gap(middle, fruit07, .0172)])

    result = least_squares(residual,
                           np.zeros(15), bounds=([-.4]*12+[-.03]*3, [.4]*12+[.03]*3), diff_step=1e-4,
                           xtol=1e-9, gtol=1e-9, ftol=1e-9, max_nfev=150)
    print("gaps_m:", gaps(result.x))
    print("worst_overlap_m:", min(residual(result.x)[3:-15]) / 3., "cost:", result.cost)
    print("bias_m:", robot.cfg["right_cluster_grasp_center_bias_m"])
    print("positions:", [float(robot.right_closed_q[kin.canonical_qadr[n]])
                         for n in cfg["robot"]["right_cluster_grasp_hand"]["joint_order"]])


if __name__ == "__main__":
    main()
