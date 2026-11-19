#!/usr/bin/env python3
"""Open a single-tomato position inspection scene in Isaac Sim.

Terminal commands while the viewer is open:
  xyz X Y Z   set tomato center in world meters
  dx VALUE    shift world X in meters
  dy VALUE    shift world Y in meters
  dz VALUE    shift world Z in meters
  reset       return to the initial center
  status      report paused robot displacement
  quit        close Isaac Sim
"""
from __future__ import annotations

import argparse
import csv
import queue
import os
import sys
import threading
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(os.environ["HANDSON_ROOT"]) / "handson"))
MATERIAL_ROOT = Path(__file__).resolve().parents[2] / "material-studies/cherry_tomato_material_sim"
ROBOT_MJCF = Path(os.environ["EPISODE54_ALIGNMENT"]) / "camera_assembly_blue_home_palm_left/dual_fr3_sharpa_camera_visuals.xml"
TOMATO_USD = MATERIAL_ROOT / "cherry_tomato.usda"
INITIAL_STATE_CSV = Path(os.environ["EPISODE54_REAL_JOINT_CSV"])

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tomato-xyz", nargs=3, type=float, default=(0.687, -0.265, 0.230), metavar=("X", "Y", "Z"), help="Tomato center in world meters")
parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_tomato_sharpa_inspection/scene.usda")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.device = "cuda:0"
args.output = args.output.expanduser().resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
if not ROBOT_MJCF.is_file():
    raise FileNotFoundError(f"Robot MJCF is missing: {ROBOT_MJCF}")
if not TOMATO_USD.is_file():
    raise FileNotFoundError(f"Tomato FEM USD is missing: {TOMATO_USD}")
if not INITIAL_STATE_CSV.is_file():
    raise FileNotFoundError(f"Episode 54 initial joint state is missing: {INITIAL_STATE_CSV}")

launcher = AppLauncher(args)
app = launcher.app
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sim import (ArticulationRootPropertiesCfg, JointDrivePropertiesCfg,
                          RigidBodyPropertiesCfg)
from isaacsim.core.prims import SingleArticulation
from handson.sim.isaaclab.compat import make_mjcf_file_cfg

sim = SimulationContext(sim_utils.SimulationCfg(
    device="cuda:0", dt=1/120, render_interval=1, gravity=(0.0, 0.0, 0.0),
    physx=sim_utils.PhysxCfg(solver_type=1, max_position_iteration_count=64,
                             max_velocity_iteration_count=8),
))
import_cfg = make_mjcf_file_cfg(
    sim_utils, asset_path=str(ROBOT_MJCF), usd_dir=str(args.output.parent / "robot_import"),
    fix_base=True, import_sites=True, self_collision=False, make_instanceable=False,
    articulation_props=ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
    joint_drive_props=JointDrivePropertiesCfg(drive_type="force"),
)
import_cfg.func("/World/Robot", import_cfg)
stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    damping_attr = prim.GetAttribute("drive:angular:physics:damping")
    if damping_attr.IsValid():
        damping_attr.Set(max(float(damping_attr.Get() or 0.0), 300.0))
stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("upAxis", UsdGeom.Tokens.z)
with INITIAL_STATE_CSV.open(newline="") as state_file:
    initial_joint_state = next(csv.DictReader(state_file))
sim.reset()
robot = SingleArticulation(prim_path="/World/Robot/base_link/base_link", name="episode54_fr3_sharpa")
robot.initialize()
if len(robot.dof_names) != 58:
    raise RuntimeError(f"Expected 58 robot joints, found {len(robot.dof_names)}")
initial_positions = torch.tensor([float(initial_joint_state[name]) for name in robot.dof_names],
                                 dtype=torch.float32, device="cuda:0")
robot.set_joint_positions(initial_positions)
robot.set_joint_velocities(torch.zeros_like(initial_positions))
world = stage.GetPrimAtPath("/World")
if not world.IsValid():
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
tomato = UsdGeom.Xform.Define(stage, "/World/TomatoFEM")
tomato.GetPrim().GetReferences().AddReference(str(TOMATO_USD), Sdf.Path("/World/SmallTomato"))
translate = tomato.AddTranslateOp()
initial = tuple(args.tomato_xyz)
translate.Set(Gf.Vec3d(*initial))
# A small visible support plane sits 15 mm below the tomato center.
plane = UsdGeom.Cube.Define(stage, "/World/SupportPlane")
plane.CreateSizeAttr(1.0)
plane.AddScaleOp().Set(Gf.Vec3f(0.06, 0.06, 0.001))
plane.AddTranslateOp().Set(Gf.Vec3d(initial[0], initial[1], initial[2] - 0.0155))
plane.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.26, 0.31)])
camera = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
camera.CreateFocalLengthAttr(18.0)
eye = Gf.Vec3d(1.08, -0.80, 0.57)
target = Gf.Vec3d(0.68, -0.27, 0.23)
camera_to_world = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
camera.AddTransformOp().Set(camera_to_world)
light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
light.CreateIntensityAttr(2500)
stage.GetRootLayer().Export(str(args.output))
sim.play()
sim.step(render=True)
sim.pause()
if sim.is_playing():
    raise RuntimeError("Isaac physics timeline did not pause")
print("物理时间线已暂停；观察模式不再执行物理步。", flush=True)
probe_names = ["right_thumb_DP", "right_index_DP", "right_middle_DP", "right_fr3v2_link8"]
probe_indices = [robot._articulation_view.get_link_index(name) for name in probe_names]
frozen_link_positions = None
try:
    import omni.kit.viewport.utility as viewport_utility
    viewport = viewport_utility.get_active_viewport()
    if viewport is not None:
        viewport.set_active_camera("/World/InspectionCamera")
        link_transforms = robot._articulation_view._physics_view.get_link_transforms()[0]
        frozen_link_positions = link_transforms[probe_indices, :3].clone()
        print("初始右手指尖/腕部世界坐标：" + str({
            name: [round(float(v), 4) for v in link_transforms[idx, :3]]
            for name, idx in zip(probe_names, probe_indices)
        }), flush=True)
        joint_velocity = robot.get_joint_velocities().detach().cpu().numpy()
        print(f"暂停后最大关节角速度：{abs(joint_velocity).max():.4f} rad/s", flush=True)
        capture = viewport_utility.capture_viewport_to_file(
            viewport, file_path=str(args.output.parent / "inspection_view.png"))
        for _ in range(30):
            app.update()
        print(f"视口截图：{args.output.parent / 'inspection_view.png'}", flush=True)
except Exception as exc:
    print(f"视口自动取景未设置：{exc}", flush=True)
commands: queue.Queue[str] = queue.Queue()
def read_terminal() -> None:
    print("\n交互命令：xyz X Y Z | dx 米 | dy 米 | dz 米 | reset | status | quit", flush=True)
    while True:
        try:
            commands.put(input("tomato> ").strip())
        except EOFError:
            commands.put("quit")
            return
threading.Thread(target=read_terminal, daemon=True).start()
position = list(initial)
print(f"场景已打开。机器人 MJCF：{ROBOT_MJCF}\n姿态：Episode 54 第 0 帧 58 关节位置\n番茄初始中心：{initial} m\n场景文件：{args.output}", flush=True)
while app.is_running():
    if sim.is_playing():
        sim.pause()
        print("检测到时间线播放，已重新暂停观察场景。", flush=True)
    while not commands.empty():
        command = commands.get()
        parts = command.split()
        try:
            if not parts:
                continue
            if parts[0] == "quit":
                app.close()
                break
            if parts[0] == "status":
                current_links = robot._articulation_view._physics_view.get_link_transforms()[0]
                drift = torch.abs(current_links[probe_indices, :3] - frozen_link_positions).max()
                print(f"物理时间线暂停：{not sim.is_playing()}；指尖/腕部最大位移：{float(drift):.9f} m", flush=True)
                continue
            if parts[0] == "reset":
                position = list(initial)
            elif parts[0] == "xyz" and len(parts) == 4:
                position = [float(v) for v in parts[1:]]
            elif parts[0] in {"dx", "dy", "dz"} and len(parts) == 2:
                position[{"dx": 0, "dy": 1, "dz": 2}[parts[0]]] += float(parts[1])
            else:
                print("用法：xyz X Y Z | dx 米 | dy 米 | dz 米 | reset | quit", flush=True)
                continue
            translate.Set(Gf.Vec3d(*position))
            plane.GetPrim().GetAttribute("xformOp:translate").Set(
                Gf.Vec3d(position[0], position[1], position[2] - 0.0155))
            stage.GetRootLayer().Export(str(args.output))
            print(f"番茄中心 -> ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) m", flush=True)
        except (ValueError, RuntimeError) as exc:
            print(f"命令错误：{exc}", flush=True)
    app.update()
app.close()
