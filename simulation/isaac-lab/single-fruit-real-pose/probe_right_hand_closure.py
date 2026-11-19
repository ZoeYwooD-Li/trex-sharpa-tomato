#!/usr/bin/env python3
"""Measure a kinematic right-hand closure around the selected single tomato.

This is a geometry probe. It does not compute tomato or gel contact forces.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(os.environ["HANDSON_ROOT"]) / "handson"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True,
                    help="Private selected_initial_state.json with measured joint positions")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.device = "cuda:0"
app = AppLauncher(args).app

import numpy as np
import torch
import omni.usd
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, ArticulationRootPropertiesCfg, JointDrivePropertiesCfg, RigidBodyPropertiesCfg
from isaacsim.core.prims import SingleArticulation
from handson.sim.isaaclab.compat import make_mjcf_file_cfg

config = json.loads(args.config.read_text())
tomato = np.asarray(config["tomato_center_world_m"], dtype=float)
sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0", dt=1/120, render_interval=1,
                                                gravity=(0.0, 0.0, 0.0)))
import_cfg = make_mjcf_file_cfg(
    sim_utils, asset_path=config["robot_mjcf"], usd_dir=str(HERE / "probe_robot_import"),
    fix_base=True, import_sites=True, self_collision=False, make_instanceable=False,
    articulation_props=ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
    joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
)
import_cfg.func("/World/Robot", import_cfg)
sim.reset()
robot = SingleArticulation(prim_path="/World/Robot/base_link/base_link", name="kinematic_probe")
robot.initialize()
names = list(robot.dof_names)
initial = np.asarray([config["robot_joint_positions_rad"][name] for name in names], dtype=float)
probe_names = ["right_thumb_DP", "right_index_DP", "right_middle_DP", "right_ring_DP", "right_pinky_DP"]
probe_ids = [robot._articulation_view.get_link_index(name) for name in probe_names]
stage = omni.usd.get_context().get_stage()
pad_prims = {}
for side in ("thumb", "index"):
    visual_root = stage.GetPrimAtPath(f"/World/Robot/base_link/right_{side}_DP/visuals")
    candidates = [p for p in Usd.PrimRange(visual_root, Usd.TraverseInstanceProxies())
                  if "elastomer" in p.GetName() and p.IsA(UsdGeom.Mesh)]
    if candidates:
        pad_prims[side] = candidates[-1]
print("pad prims: " + str({k: (str(v.GetPath()), v.GetTypeName()) for k, v in pad_prims.items()}), flush=True)
print("thumb paths: " + str([str(p.GetPath()) for p in stage.Traverse() if "right_thumb" in str(p.GetPath())][:30]), flush=True)
print("thumb visual children: " + str([str(p.GetPath()) for p in stage.GetPrimAtPath("/World/Robot/base_link/right_thumb_DP/visuals").GetChildren()]), flush=True)
print("thumb visual is instance: " + str(stage.GetPrimAtPath("/World/Robot/base_link/right_thumb_DP/visuals").IsInstance()), flush=True)
print("thumb proxy paths: " + str([str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot/base_link/right_thumb_DP/visuals"), Usd.TraverseInstanceProxies())][:30]), flush=True)
pad_vertices_link = {}
for side, prim in pad_prims.items():
    link_prim = stage.GetPrimAtPath(f"/World/Robot/base_link/right_{side}_DP")
    mesh_to_link, _ = UsdGeom.XformCache(Usd.TimeCode.Default()).ComputeRelativeTransform(prim, link_prim)
    local = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    pad_vertices_link[side] = np.asarray([tuple(mesh_to_link.Transform(Gf.Vec3d(*map(float, point))))
                                          for point in local], dtype=float)
print("pad vertex counts: " + str({k: len(v) for k, v in pad_vertices_link.items()}), flush=True)

# Conservative first sweep. These are joint targets, not material parameters.
targets = {
    "right_thumb_CMC_FE": 0.95, "right_thumb_MCP_FE": 0.75, "right_thumb_IP": 0.80,
    "right_index_MCP_FE": 0.85, "right_index_PIP": 0.95, "right_index_DIP": 0.60,
    "right_middle_MCP_FE": 0.85, "right_middle_PIP": 0.95, "right_middle_DIP": 0.60,
}
rows = []
for alpha in np.linspace(0, 1, 11):
    q = initial.copy()
    for name, goal in targets.items():
        i = names.index(name)
        q[i] = initial[i] + alpha * (goal - initial[i])
    robot.set_joint_positions(torch.tensor(q, dtype=torch.float32, device="cuda:0"))
    robot.set_joint_velocities(torch.zeros(len(q), dtype=torch.float32, device="cuda:0"))
    sim.play()
    sim.step(render=False)
    sim.pause()
    transforms = robot._articulation_view._physics_view.get_link_transforms()[0]
    row = {"closure_fraction": float(alpha)}
    for name, idx in zip(probe_names, probe_ids):
        pos = np.asarray(transforms[idx, :3].detach().cpu(), dtype=float)
        row[f"{name}_center_distance_mm"] = float(np.linalg.norm(pos - tomato) * 1000)
        for axis, value in zip("xyz", pos):
            row[f"{name}_{axis}_m"] = float(value)
    for side, idx in (("thumb", probe_ids[0]), ("index", probe_ids[1])):
        pose = np.asarray(transforms[idx].detach().cpu(), dtype=float)
        pad_world = Rotation.from_quat(pose[3:7]).apply(pad_vertices_link[side]) + pose[:3]
        row[f"right_{side}_pad_min_vertex_distance_mm"] = float(
            np.linalg.norm(pad_world - tomato[None, :], axis=1).min() * 1000)
    rows.append(row)
    if abs(alpha) < 1e-9 or abs(alpha - 0.6) < 1e-9:
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        pad_centers = {}
        for name, prim in pad_prims.items():
            bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = bound.GetMin(), bound.GetMax()
            pad_centers[name] = [float((lo[i]+hi[i])/2) for i in range(3)]
        print(f"USD pad centers at alpha={alpha:.1f}: {pad_centers}", flush=True)

out = HERE / "right_hand_closure_probe.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out}", flush=True)
for row in rows:
    print("alpha={:.1f} thumb={:.1f} index={:.1f} middle={:.1f} mm".format(
        row["closure_fraction"], row["right_thumb_DP_center_distance_mm"],
        row["right_index_DP_center_distance_mm"], row["right_middle_DP_center_distance_mm"]), flush=True)

# Check whether thumb abduction/adduction can oppose the index near this fruit.
opposition = []
base = initial.copy()
for name, goal in targets.items():
    i = names.index(name)
    base[i] = initial[i] + 0.6 * (goal - initial[i])
for cmc_aa in np.linspace(-0.349, 0.349, 9):
    for mcp_aa in np.linspace(-0.349, 0.349, 9):
        q = base.copy()
        q[names.index("right_thumb_CMC_AA")] = cmc_aa
        q[names.index("right_thumb_MCP_AA")] = mcp_aa
        robot.set_joint_positions(torch.tensor(q, dtype=torch.float32, device="cuda:0"))
        robot.set_joint_velocities(torch.zeros(len(q), dtype=torch.float32, device="cuda:0"))
        sim.play()
        sim.step(render=False)
        sim.pause()
        transforms = robot._articulation_view._physics_view.get_link_transforms()[0]
        thumb = np.asarray(transforms[probe_ids[0], :3].detach().cpu(), dtype=float)
        opposition.append({"thumb_CMC_AA_rad": float(cmc_aa), "thumb_MCP_AA_rad": float(mcp_aa),
                           "thumb_center_distance_mm": float(np.linalg.norm(thumb-tomato)*1000),
                           "thumb_x_m": float(thumb[0]), "thumb_y_m": float(thumb[1]),
                           "thumb_z_m": float(thumb[2])})
opposition.sort(key=lambda x: x["thumb_center_distance_mm"])
with (HERE / "right_thumb_opposition_probe.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(opposition[0]))
    writer.writeheader()
    writer.writerows(opposition)
print("best thumb opposition: " + str(opposition[:5]), flush=True)

# Small wrist-only adjustment; keep the arm's starting pose close to Episode 54.
wrist_sweep = []
base[names.index("right_thumb_CMC_AA")] = opposition[0]["thumb_CMC_AA_rad"]
base[names.index("right_thumb_MCP_AA")] = opposition[0]["thumb_MCP_AA_rad"]
for d6 in np.linspace(-0.2, 0.2, 9):
    for d7 in np.linspace(-0.2, 0.2, 9):
        q = base.copy()
        q[names.index("right_fr3v2_joint6")] += d6
        q[names.index("right_fr3v2_joint7")] += d7
        robot.set_joint_positions(torch.tensor(q, dtype=torch.float32, device="cuda:0"))
        robot.set_joint_velocities(torch.zeros(len(q), dtype=torch.float32, device="cuda:0"))
        sim.play()
        sim.step(render=False)
        sim.pause()
        transforms = robot._articulation_view._physics_view.get_link_transforms()[0]
        thumb = np.asarray(transforms[probe_ids[0], :3].detach().cpu(), dtype=float)
        index = np.asarray(transforms[probe_ids[1], :3].detach().cpu(), dtype=float)
        td = float(np.linalg.norm(thumb-tomato)*1000)
        idist = float(np.linalg.norm(index-tomato)*1000)
        wrist_sweep.append({"right_joint6_delta_rad": float(d6), "right_joint7_delta_rad": float(d7),
                            "thumb_distance_mm": td, "index_distance_mm": idist,
                            "max_distance_mm": max(td, idist),
                            "thumb_x_m": float(thumb[0]), "index_x_m": float(index[0])})
wrist_sweep.sort(key=lambda x: x["max_distance_mm"])
with (HERE / "right_wrist_sweep.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(wrist_sweep[0]))
    writer.writeheader()
    writer.writerows(wrist_sweep)
print("best wrist sweep: " + str(wrist_sweep[:5]), flush=True)

# Candidate staged closure: thumb approaches first, then the 4F index pad.
staged = []
schedule = [("thumb_approach", a, 0.0) for a in np.linspace(0.0, 0.14, 8)]
schedule += [("index_approach", 0.14, a) for a in np.linspace(0.0, 0.50, 11)]
schedule += [("thumb_squeeze", a, 0.50) for a in (0.145, 0.150, 0.155, 0.160)]
for frame, (phase, thumb_fraction, index_fraction) in enumerate(schedule):
    q = initial.copy()
    for name, goal in targets.items():
        if name.startswith("right_thumb_"):
            fraction = thumb_fraction
        elif name.startswith("right_index_"):
            fraction = index_fraction
        else:
            continue
        i = names.index(name)
        q[i] = initial[i] + fraction * (goal - initial[i])
    robot.set_joint_positions(torch.tensor(q, dtype=torch.float32, device="cuda:0"))
    robot.set_joint_velocities(torch.zeros(len(q), dtype=torch.float32, device="cuda:0"))
    sim.play()
    sim.step(render=False)
    sim.pause()
    transforms = robot._articulation_view._physics_view.get_link_transforms()[0]
    record = {"frame": frame, "phase": phase, "thumb_fraction": float(thumb_fraction),
              "index_fraction": float(index_fraction)}
    for side, idx in (("thumb", probe_ids[0]), ("index", probe_ids[1])):
        pose = np.asarray(transforms[idx].detach().cpu(), dtype=float)
        pad_world = Rotation.from_quat(pose[3:7]).apply(pad_vertices_link[side]) + pose[:3]
        distances = np.linalg.norm(pad_world - tomato[None, :], axis=1)
        nearest_point = pad_world[int(np.argmin(distances))]
        nearest = distances.min()
        record[f"{side}_min_vertex_surface_gap_mm"] = float((nearest - 0.015)*1000)
        normal = (tomato - nearest_point)/nearest
        for axis, point, direction in zip("xyz", nearest_point, normal):
            record[f"{side}_nearest_{axis}_m"] = float(point)
            record[f"{side}_inward_{axis}"] = float(direction)
    staged.append(record)
with (HERE / "staged_closure_geometry_refined.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(staged[0]))
    writer.writeheader()
    writer.writerows(staged)
print("staged closure end: " + str(staged[-1]), flush=True)
app.close()
