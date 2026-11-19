#!/usr/bin/env python3
"""Standalone Isaac Sim 5.1 single-tomato ground-loading scene.

Run with the project's Isaac Sim 5.1 Python interpreter on a CUDA host.
Nothing in the harvesting scene is opened or modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np


RADIUS_M = 0.0172  # project-configured geometry, not measured in this experiment
MASS_KG = 0.020  # project-configured mass, not independently verified
YOUNGS_MODULUS_PA = 800_000.0  # existing Isaac prototype trial value; not calibrated
POISSONS_RATIO = 0.42  # assumed trial value
ELASTICITY_DAMPING = 0.25  # assumed numerical/viscoelastic trial value
DAMAGE_STRAIN_THRESHOLD = 0.15  # assumed screening threshold, not fracture law
DT = 1.0 / 120.0


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _ground(stage):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, "/World/Ground")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    cube.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.1))
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.55, 0.62)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def build_scene(stage, debug_meshes=False):
    import trimesh
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, Vt

    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    UsdGeom.Xform.Define(stage, "/World")
    _ground(stage)

    root_path = "/World/SmallTomato"
    root = UsdGeom.Xform.Define(stage, root_path)
    root.AddTranslateOp().Set(Gf.Vec3d(0, 0, RADIUS_M + 0.0005))

    # Watertight geometry is used for cooking and for the visible tomato.
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=RADIUS_M)
    mesh = UsdGeom.Mesh.Define(stage, root_path + "/CookingSource")
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(sphere.vertices, np.float32)))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(sphere.faces))
    mesh.CreateFaceVertexIndicesAttr().Set(sphere.faces.reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.055, 0.035)])
    sim_path = root_path + "/SimulationMesh"
    collision_path = root_path + "/CollisionMesh"
    if not deformableUtils.create_auto_volume_deformable_hierarchy(
        stage=stage, root_prim_path=root_path,
        simulation_tetmesh_path=sim_path, collision_tetmesh_path=collision_path,
        cooking_src_mesh_path=str(mesh.GetPath()),
        simulation_hex_mesh_enabled=False,
        cooking_src_simplification_enabled=False,
    ):
        raise RuntimeError("PhysX failed to create the tomato tet hierarchy")
    # Cooking rebuilds the body root, so author mass and solver settings afterwards.
    prim = stage.GetPrimAtPath(root_path)
    prim.ApplyAPI("OmniPhysicsDeformableBodyAPI")
    prim.GetAttribute("omniphysics:mass").Set(MASS_KG)
    prim.ApplyAPI("PhysxBaseDeformableBodyAPI")
    for name, value in {
        "physxDeformableBody:selfCollision": False,
        "physxDeformableBody:enableSpeculativeCCD": True,
        "physxDeformableBody:solverPositionIterationCount": 32,
        "physxDeformableBody:maxLinearVelocity": 2.0,
        "physxDeformableBody:maxDepenetrationVelocity": 0.35,
        "physxDeformableBody:linearDamping": 0.12,
    }.items():
        attr = prim.GetAttribute(name)
        if not attr.IsValid():
            raise RuntimeError(f"Missing PhysX 5.1 deformable attribute: {name}")
        attr.Set(value)
    sim_prim = stage.GetPrimAtPath(sim_path)
    collision_prim = stage.GetPrimAtPath(collision_path)
    UsdPhysics.CollisionAPI.Apply(collision_prim).CreateCollisionEnabledAttr().Set(True)
    PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    collision_prim.GetAttribute("physxCollision:contactOffset").Set(0.002)
    collision_prim.GetAttribute("physxCollision:restOffset").Set(0.001)
    for path, color in ((sim_prim, Gf.Vec3f(0.1, 0.45, 1.0)),
                        (collision_prim, Gf.Vec3f(1.0, 0.7, 0.05))):
        imageable = UsdGeom.Imageable(path)
        if debug_meshes:
            imageable.MakeVisible()
            UsdGeom.Gprim(path).CreateDisplayColorAttr([color])
            UsdGeom.Gprim(path).CreateDisplayOpacityAttr([0.22])
        else:
            imageable.MakeInvisible()

    density = MASS_KG / ((4.0 / 3.0) * math.pi * RADIUS_M**3)
    material_path = "/World/TomatoDeformableMaterial"
    if not deformableUtils.add_deformable_material(
        stage, material_path, density=density,
        static_friction=0.5, dynamic_friction=0.5,
        youngs_modulus=YOUNGS_MODULUS_PA, poissons_ratio=POISSONS_RATIO,
    ):
        raise RuntimeError("PhysX failed to create deformable material")
    material = stage.GetPrimAtPath(material_path)
    material.ApplyAPI("PhysxDeformableMaterialAPI")
    material.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(ELASTICITY_DAMPING)
    physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
    physicsUtils.add_physics_material_to_prim(stage, mesh.GetPrim(), material_path)
    physicsUtils.add_physics_material_to_prim(stage, sim_prim, material_path)
    return density


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(""))
    parser.add_argument("--force-n", type=float, default=0.2)
    parser.add_argument("--load-seconds", type=float, default=0.5)
    parser.add_argument("--release-seconds", type=float, default=0.5)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--live", action="store_true", help="Keep an interactive Isaac viewport running")
    parser.add_argument("--live-period-seconds", type=float, default=30.0,
                        help="Seconds for one gentle loading and unloading cycle")
    parser.add_argument("--debug-meshes", action="store_true",
                        help="Show collision mesh and tetrahedral simulation mesh")
    args = parser.parse_args()
    if args.live and args.headless:
        parser.error("--live requires a visible Isaac window")
    if args.force_n <= 0 or args.load_seconds <= 0 or args.release_seconds < 0:
        parser.error("force and load duration must be positive; release duration cannot be negative")
    if args.live and (args.live_period_seconds <= 0 or args.force_n < MASS_KG * 9.81):
        parser.error("live period must be positive and live peak force must exceed tomato weight")
    args.output.mkdir(parents=True, exist_ok=True)

    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    if gpu.returncode != 0 or "GPU" not in gpu.stdout:
        raise RuntimeError("Isaac Sim volume-deformable validation requires a working NVIDIA driver/GPU")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "width": 1280 if args.live else 640,
                         "height": 800 if args.live else 480})
    try:
        import omni.usd
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationContext

        sim = SimulationContext(sim_utils.SimulationCfg(
            device="cuda:0", dt=DT, gravity=(0.0, 0.0, 0.0),
            physx=sim_utils.PhysxCfg(solver_type=1, max_position_iteration_count=32),
        ))
        stage = omni.usd.get_context().get_stage()
        density = build_scene(stage, debug_meshes=args.debug_meshes)
        if args.live:
            sim.set_camera_view(eye=(0.075, -0.11, 0.060), target=(0.0, 0.0, RADIUS_M))
        scene_path = args.output / "small_tomato_compression.usda"
        stage.Export(str(scene_path))
        # Auto-deformable cooking is asynchronous, especially with a distinct
        # collision tetra mesh. Let Kit process stage changes before creating views.
        for _ in range(10):
            app.update()
        sim.reset()
        sim.step(render=False)
        view = sim.physics_sim_view.create_volume_deformable_body_view("/World/SmallTomato")
        if view is None or int(view.count) != 1:
            raise RuntimeError("PhysX did not cook exactly one volume deformable body")
        rest = _numpy(view.get_rest_nodal_positions())[0]
        if not np.isfinite(rest).all() or len(rest) < 4:
            raise RuntimeError("Invalid cooked rest nodes")
        rest_height = float(np.ptp(rest[:, 2]))
        if args.live:
            base_force_n = MASS_KG * 9.81
            print(f"LIVE_READY: one tomato and ground; commanded equivalent load "
                  f"cycles {base_force_n:.3f}–{args.force_n:g} N over "
                  f"{args.live_period_seconds:g} s. Close the Isaac window to stop.", flush=True)
            step = 0
            while app.is_running():
                tick = time.monotonic()
                phase = (step * DT / args.live_period_seconds) % 1.0
                blend = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
                command = base_force_n + (args.force_n - base_force_n) * blend
                sim.get_physics_context().set_gravity(-command / MASS_KG)
                sim.step(render=True)
                if step % 120 == 0:
                    nodes = _numpy(view.get_simulation_nodal_positions())[0]
                    strain = max(0.0, rest_height - float(np.ptp(nodes[:, 2]))) / rest_height
                    print(f"LIVE frame={step} load_N={command:.3f} compression_strain={strain:.5f}",
                          flush=True)
                step += 1
                time.sleep(max(0.0, DT - (time.monotonic() - tick)))
            return
        columns = ["time_s", "commanded_force_N", "measured_contact_force_N",
                   "tomato_height_m", "compression_m", "compression_strain",
                   "damage_threshold_exceeded", "tomato_min_z_m"]
        rows = []
        total_steps = round((args.load_seconds + args.release_seconds) / DT)
        for step in range(total_steps):
            time_s = step * DT
            command = args.force_n if time_s < args.load_seconds else 0.0
            sim.get_physics_context().set_gravity(-command / MASS_KG)
            sim.step(render=not args.headless)
            nodes = _numpy(view.get_simulation_nodal_positions())[0]
            if not np.isfinite(nodes).all():
                raise RuntimeError(f"Nonfinite deformable nodes at step {step}")
            height = float(np.ptp(nodes[:, 2]))
            compression = max(0.0, rest_height - height)
            strain = compression / rest_height
            rows.append([round((step + 1) * DT, 6), command, "", height,
                         compression, strain, int(strain >= DAMAGE_STRAIN_THRESHOLD),
                         float(np.min(nodes[:, 2]))])
        with (args.output / "compression_trace.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        summary = {
            "scene": str(scene_path), "steps": len(rows), "dt_s": DT,
            "peak_commanded_force_N": args.force_n,
            "peak_compression_m": max(row[4] for row in rows),
            "peak_compression_strain": max(row[5] for row in rows),
            "end_of_run_compression_m": rows[-1][4],
            "damage_threshold_exceeded": any(row[6] for row in rows),
            "damage_measure": "bounding-height compression strain screening proxy; no PhysX fracture law",
            "measured_contact_force_N": None,
            "contact_force_limit": "Isaac Sim 5.1 public deformable view does not expose rigid-deformable contact manifold; equivalent weight load is recorded separately",
            "rest_node_count": len(rest), "derived_density_kg_m3": density,
            "parameter_status": {
                "radius_m": "existing project configuration; calibration unverified",
                "mass_kg": "existing project configuration; calibration unverified",
                "youngs_modulus_pa": "assumed trial value",
                "poissons_ratio": "assumed trial value",
                "elasticity_damping": "assumed trial value",
                "friction": "assumed trial value",
                "damage_strain_threshold": "assumed screening threshold",
            },
        }
        (args.output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
