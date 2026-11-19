#!/usr/bin/env python3
"""First quasi-static tomato FEM trial with two measured Sharpa pad tangent planes.

Reuses the existing 30 mm tetra mesh, linear material assembly, and contact
penalty. The tangent-plane pad approximation is experimental, not calibrated.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import bmat, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
BASELINE = Path(__file__).resolve().parents[2] / "material-studies/cherry_tomato_material_sim"
sys.path.insert(0, str(BASELINE))
import run_cherry_tomato_fem as baseline

config = json.loads((HERE / "selected_initial_state.json").read_text())
center = np.asarray(config["tomato_center_world_m"], dtype=float)
trial = HERE / "fem_pad_fixed_center_refined"
trial.mkdir(exist_ok=True)
records = list(csv.DictReader((HERE / "staged_closure_geometry_refined.csv").open()))

points_local, tets, faces, surface_count, volumes = baseline.make_mesh(subdivisions=3)
points = points_local + center - np.array([0.0, 0.0, baseline.RADIUS])
E = 281_000.0
nu = 0.335
density = baseline.MASS / float(np.sum(volumes))
K, gravity = baseline.assemble(points, tets, E, nu, density)
surface_ids = np.arange(surface_count)
ndof = 3 * len(points)
penalty = baseline.PENALTY
floor_z = center[2] - baseline.RADIUS

# Hold the fruit's mean translation and rotation, while all FEM nodes remain free
# to deform. This is a transparent external fixture for a compression experiment.
gauge = lil_matrix((6, ndof))
offset = points - center
for node, r in enumerate(offset):
    col = 3 * node
    gauge[0, col] = gauge[1, col+1] = gauge[2, col+2] = 1.0 / len(points)
    rx, ry, rz = r / baseline.RADIUS / len(points)
    gauge[3, col+1] = -rz; gauge[3, col+2] = ry
    gauge[4, col] = rz; gauge[4, col+2] = -rx
    gauge[5, col] = -ry; gauge[5, col+1] = rx
gauge = gauge.tocsr()


def contacts(record: dict, displacement: np.ndarray):
    position = points + displacement.reshape(-1, 3)
    result = []
    floor_depth = floor_z - position[surface_ids, 2]
    floor_ids = surface_ids[floor_depth > 0]
    if len(floor_ids):
        result.append(("support", floor_ids, np.array([0.0, 0.0, 1.0]),
                       np.array([0.0, 0.0, floor_z])))
    for side in ("thumb", "index"):
        p = np.array([float(record[f"{side}_nearest_{axis}_m"]) for axis in "xyz"])
        n = np.array([float(record[f"{side}_inward_{axis}"]) for axis in "xyz"])
        n /= np.linalg.norm(n)
        signed = (position[surface_ids] - p) @ n
        active = surface_ids[signed < 0]
        if len(active):
            result.append((side, active, n, p))
    return result


def solve_one(record: dict):
    u = np.zeros(ndof)
    previous = None
    for iteration in range(30):
        active = contacts(record, u)
        signature = tuple((name, tuple(ids)) for name, ids, _, _ in active)
        A = K.tolil()
        rhs = gravity.copy()
        for _, ids, normal, plane_point in active:
            block = penalty * np.outer(normal, normal)
            for node in ids:
                dof = 3 * int(node)
                A[dof:dof+3, dof:dof+3] += block
                rhs[dof:dof+3] += penalty * normal * np.dot(plane_point - points[node], normal)
        constrained = bmat([[A.tocsr(), gauge.T],
                            [gauge, csr_matrix((6, 6))]], format="csr")
        solution = spsolve(constrained, np.r_[rhs, np.zeros(6)])
        next_u = solution[:ndof]
        if not np.isfinite(next_u).all():
            raise RuntimeError("FEM returned non-finite displacement")
        if signature == previous and np.max(np.abs(next_u - u)) < 1e-8:
            u = next_u
            break
        u = next_u
        previous = signature
    else:
        raise RuntimeError("Pad contact active set did not converge in 30 iterations")
    deformed = points + u.reshape(-1, 3)
    force = {}
    penetration = {}
    active_count = {}
    for name, ids, normal, plane_point in contacts(record, u):
        depth = np.maximum(0.0, (plane_point - deformed[ids]) @ normal)
        force[name] = (penalty * depth.sum() * normal).tolist()
        penetration[name] = float(depth.max())
        active_count[name] = int(len(ids))
    volume, quality, signed_volume = baseline.tet_quality(deformed, tets)
    total_contact = np.sum(np.array(list(force.values())), axis=0) if force else np.zeros(3)
    holding_force = -(total_contact + gravity.reshape(-1, 3).sum(axis=0))
    summary = {
        "frame": int(record["frame"]), "iterations": iteration + 1,
        "material_E_Pa": E, "material_nu": nu, "density_kg_m3": density,
        "contact_model": "two measured Sharpa pad tangent planes; nodal unilateral penalty",
        "contact_penalty_N_per_m": penalty,
        "fruit_fixture": "zero mean translation and rotation; local FEM deformation free",
        "contact_force_on_tomato_N": force,
        "max_numerical_penetration_m": penetration,
        "active_contact_nodes": active_count,
        "fixture_holding_force_N": holding_force.tolist(),
        "gauge_residual": (gauge @ u).tolist(),
        "mean_node_displacement_m": u.reshape(-1, 3).mean(axis=0).tolist(),
        "max_node_displacement_m": float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()),
        "inverted_tetrahedra": int(np.count_nonzero(signed_volume <= 0)),
        "min_tetra_quality": float(quality.min()),
        "limitation": "First tangent-plane contact approximation; FEM reaction is not Sharpa Wrench or a calibrated tomato force.",
    }
    return deformed, summary


summaries = []
for frame in range(len(records)):
    deformed, summary = solve_one(records[frame])
    np.savez_compressed(trial / f"frame_{frame:02d}.npz", rest_points_m=points,
                        deformed_points_m=deformed, tetrahedra=tets, surface_faces=faces)
    (trial / f"frame_{frame:02d}.json").write_text(json.dumps(summary, indent=2) + "\n")
    summaries.append(summary)
    print(f"frame {frame:02d}: contacts={summary['active_contact_nodes']} "
          f"max displacement={summary['max_node_displacement_m']*1000:.3f} mm", flush=True)
(trial / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
