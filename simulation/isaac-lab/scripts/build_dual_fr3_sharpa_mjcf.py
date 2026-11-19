#!/usr/bin/env python3
"""Compose local dual-FR3 carrier and authored Sharpa hands for Isaac MJCF import.

The source checkout is read-only.  All mesh paths point into the private local
copy and the generated MJCF lives in this HandsOn project's outputs directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def _named_joints(element: ET.Element) -> list[str]:
    return [joint.attrib["name"] for joint in element.findall(".//joint[@name]")]


def _rotate_xyzw(quaternion: tuple[float, float, float, float],
                 vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a vector by an XYZW unit quaternion without extra dependencies."""
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def _add_wrist_camera_visuals(carrier: ET.Element, side: str, mount: dict,
                              housing_dimensions_m: tuple[float, float, float],
                              hand_root_offset_m: tuple[float, float, float]) -> dict:
    """Add massless, non-colliding mount visuals under the moving link8 body."""
    link8 = carrier.find(f".//body[@name='{side}_fr3v2_link8']")
    if link8 is None:
        raise ValueError(f"Missing {side} FR3 link8")

    camera_center = tuple(float(v) for v in mount["position_link8_m"])
    rod_anchor = tuple(float(v) for v in mount["support_rod_anchor_link8_m"])
    rod_delta = tuple(p - a for p, a in zip(camera_center, rod_anchor))
    rod_length = math.sqrt(sum(v * v for v in rod_delta))
    if rod_length < 1e-6:
        raise ValueError(f"{side} camera rod has zero length")
    rod_direction = tuple(v / rod_length for v in rod_delta)
    optical_xyzw = tuple(float(v) for v in mount["orientation_link8_opengl_xyzw"])
    quaternion_norm = math.sqrt(sum(v * v for v in optical_xyzw))
    if quaternion_norm < 1e-8:
        raise ValueError(f"{side} camera orientation has zero norm")
    optical_xyzw = tuple(v / quaternion_norm for v in optical_xyzw)
    half_sizes = tuple(float(v) * 0.5 for v in housing_dimensions_m)

    # Locate the rod's endpoint at the first intersection with the oriented
    # camera housing, so the rod visibly meets the camera instead of passing
    # through its center.
    qx, qy, qz, qw = optical_xyzw
    rod_direction_camera = _rotate_xyzw((-qx, -qy, -qz, qw), rod_direction)
    intersections = [half_size / abs(component)
                     for half_size, component in zip(half_sizes, rod_direction_camera)
                     if abs(component) > 1e-9]
    rod_surface_distance = min(intersections)
    rod_end = tuple(center - direction * rod_surface_distance
                    for center, direction in zip(camera_center, rod_direction))
    fmt = lambda values: " ".join(f"{float(v):.12g}" for v in values)
    visual_common = {"contype": "0", "conaffinity": "0", "mass": "0", "group": "1"}

    # 6 cm arm-to-hand connector, directly under the link8 wrist. It is kept
    # visual-only because the hand already starts at this offset in the model.
    connector = ET.SubElement(link8, "geom", {
        "name": f"{side}_arm_hand_connector_visual", "type": "capsule",
        "fromto": fmt((0.0, 0.0, 0.0, *hand_root_offset_m)),
        "size": "0.017", "rgba": "0.10 0.12 0.14 1", **visual_common,
    })

    ET.SubElement(link8, "geom", {
        "name": f"{side}_wrist_camera_support_rod_visual", "type": "capsule",
        "fromto": fmt((*rod_anchor, *rod_end)), "size": "0.0045",
        "rgba": "0.08 0.09 0.11 1", **visual_common,
    })

    q_wxyz = (optical_xyzw[3], optical_xyzw[0], optical_xyzw[1], optical_xyzw[2])
    ET.SubElement(link8, "geom", {
        "name": f"{side}_wrist_camera_housing_visual", "type": "box",
        "pos": fmt(camera_center), "quat": fmt(q_wxyz), "size": fmt(half_sizes),
        "rgba": "0.26 0.30 0.34 1", **visual_common,
    })

    front_panel_center = _rotate_xyzw(
        optical_xyzw, (0.0, 0.0, -half_sizes[2] - 0.0004))
    front_panel_pos = tuple(center + offset
                            for center, offset in zip(camera_center, front_panel_center))
    ET.SubElement(link8, "geom", {
        "name": f"{side}_wrist_camera_front_panel_visual", "type": "box",
        "pos": fmt(front_panel_pos), "quat": fmt(q_wxyz),
        "size": fmt((half_sizes[0] * 0.88, half_sizes[1] * 0.88, 0.0005)),
        "rgba": "0.035 0.045 0.055 1", **visual_common,
    })
    for lens_index, lens_y in enumerate((-0.008, 0.008)):
        lens_offset = _rotate_xyzw(
            optical_xyzw, (0.0, lens_y, -half_sizes[2] - 0.0011))
        lens_pos = tuple(center + offset for center, offset in zip(camera_center, lens_offset))
        ET.SubElement(link8, "geom", {
            "name": f"{side}_wrist_camera_lens_{lens_index}_visual", "type": "cylinder",
        "pos": fmt(lens_pos), "quat": fmt(q_wxyz), "size": "0.0055 0.001",
            "rgba": "0.04 0.38 0.70 1", **visual_common,
        })
    return {
        "camera_center_link8_m": list(camera_center),
        "rod_anchor_link8_m": list(rod_anchor),
        "rod_surface_end_link8_m": list(rod_end),
        "rod_center_distance_m": rod_length,
        "camera_angle_from_rod_deg": mount.get("camera_angle_from_rod_deg"),
        "housing_dimensions_m": list(housing_dimensions_m),
        "connector_endpoint_link8_m": list(hand_root_offset_m),
    }


def _append_hand(carrier: ET.Element, side: str, source: Path, mount_offset: list[float],
                 mesh_dir: Path, root_quat_wxyz: tuple[float, float, float, float]) -> None:
    hand = ET.parse(source).getroot()
    side_prefix = f"{side}_sharpa_"
    mesh_names = {}
    for mesh in hand.findall("./asset/mesh"):
        old_name = mesh.attrib["name"]
        new_mesh = copy.deepcopy(mesh)
        new_mesh.attrib["name"] = side_prefix + old_name
        source_mesh = (source.parent / "meshes" / mesh.attrib["file"]).resolve()
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        target_mesh = mesh_dir / f"{side_prefix}{old_name}{source_mesh.suffix.lower()}"
        shutil.copy2(source_mesh, target_mesh)
        new_mesh.attrib["file"] = str(target_mesh.relative_to(mesh_dir.parent))
        carrier.find("asset").append(new_mesh)
        mesh_names[old_name] = new_mesh.attrib["name"]

    class_names = {}
    hand_default = hand.find("default")
    for default in list(hand_default):
        if default.tag != "default" or not default.get("class"):
            continue
        copied = copy.deepcopy(default)
        for nested in copied.iter("default"):
            old_class = nested.get("class")
            if old_class:
                class_names[old_class] = side_prefix + old_class
                nested.set("class", class_names[old_class])
        carrier.find("default").append(copied)

    link8 = carrier.find(f".//body[@name='{side}_fr3v2_link8']")
    if link8 is None:
        raise ValueError(f"Missing {side} FR3 link8")
    root_body = copy.deepcopy(hand.find("./worldbody/body"))
    root_body.set("pos", " ".join(str(x) for x in mount_offset))
    root_body.set("quat", " ".join(f"{value:.12g}" for value in root_quat_wxyz))
    for element in root_body.iter():
        if element.tag in {"body", "geom", "site", "camera"} and element.get("name") and not element.get("name").startswith(f"{side}_"):
            element.set("name", side_prefix + element.get("name"))
        if element.get("mesh") in mesh_names:
            element.set("mesh", mesh_names[element.get("mesh")])
        if element.get("class") in class_names:
            element.set("class", class_names[element.get("class")])
    link8.append(root_body)
    for actuator in list(hand.find("actuator")):
        carrier.find("actuator").append(copy.deepcopy(actuator))


def build(source_root: Path, output: Path,
          mount_xy_correction_m: tuple[float, float] = (0.0, 0.0),
          right_mount_xy_correction_m: tuple[float, float] = (0.0, 0.0),
          left_mount_xyz_correction_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
          right_mount_xyz_correction_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
          left_hand_root_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
          right_hand_root_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
          hand_root_offset_m: tuple[float, float, float] | None = None,
          wrist_camera_calibration_json: Path | None = None) -> dict:
    hand_quats = {}
    for side, values in (("left", left_hand_root_quat_wxyz),
                         ("right", right_hand_root_quat_wxyz)):
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{side} hand root quaternion must contain four finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 1e-8:
            raise ValueError(f"{side} hand root quaternion has zero norm")
        hand_quats[side] = tuple(value / norm for value in values)
    source_root = source_root.resolve()
    robot_dir = source_root / "demos/assets/robots"
    carrier_file = robot_dir / "dual_fr3_linker_l20/model.collision.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = output.parent / "mesh_sources"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((source_root / "configs/cherry_tomato_pick_sharpa.yaml").read_text())
    resolved_hand_root_offset = tuple(float(value) for value in (
        hand_root_offset_m if hand_root_offset_m is not None
        else config["robot"]["mount_offset_m"]
    ))
    if len(resolved_hand_root_offset) != 3 or not all(math.isfinite(value) for value in resolved_hand_root_offset):
        raise ValueError("hand root offset must contain three finite values")
    carrier = ET.parse(carrier_file).getroot()
    for name, pose in config["robot"]["source_mounts"].items():
        body = carrier.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"Carrier mount missing: {name}")
        position = [float(x) for x in pose["pos"]]
        if name in ("left_base", "right_base"):
            position[0] += mount_xy_correction_m[0]
            position[1] += mount_xy_correction_m[1]
            correction = (left_mount_xyz_correction_m if name == "left_base"
                          else right_mount_xyz_correction_m)
            position = [value + delta for value, delta in zip(position, correction)]
        if name == "right_base":
            position[0] += right_mount_xy_correction_m[0]
            position[1] += right_mount_xy_correction_m[1]
        body.set("pos", " ".join(str(x) for x in position))
        x, y, z, w = pose["quat_xyzw"]
        body.set("quat", f"{w} {x} {y} {z}")
    worldbody = carrier.find("worldbody")
    old_support = worldbody.find("./body[@name='support_box']")
    if old_support is not None:
        worldbody.remove(old_support)
    asset = carrier.find("asset")
    for mesh in asset.findall("mesh"):
        source_mesh = (carrier_file.parent / mesh.get("file")).resolve()
        if not source_mesh.is_file():
            raise FileNotFoundError(source_mesh)
        target_mesh = mesh_dir / f"carrier_{mesh.get('name')}{source_mesh.suffix.lower()}"
        shutil.copy2(source_mesh, target_mesh)
        mesh.set("file", str(target_mesh.relative_to(mesh_dir.parent)))
    for side in ("left", "right"):
        old_hand_name = "L_hand_base_link" if side == "left" else "R_hand_base_link"
        old_hand = carrier.find(f".//body[@name='{old_hand_name}']")
        if old_hand is None:
            raise ValueError(f"Old hand subtree missing: {old_hand_name}")
        parent = next(item for item in carrier.iter("body") if old_hand in list(item))
        parent.remove(old_hand)

    wrist_visual_report = {}
    if wrist_camera_calibration_json is not None:
        calibration_path = wrist_camera_calibration_json.expanduser().resolve()
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        wrist_config = config.get("wrist_cameras", {})
        dimensions = tuple(float(value) for value in wrist_config.get(
            "housing_dimensions_m", (0.042, 0.042, 0.023)))
        if len(dimensions) != 3 or not all(value > 0 for value in dimensions):
            raise ValueError("wrist camera housing dimensions must be three positive values")
        for side in ("left", "right"):
            mount_key = f"wrist_{side}"
            mount = calibration.get("mounts", {}).get(mount_key)
            if mount is None:
                raise ValueError(f"Calibration is missing mount {mount_key}")
            wrist_visual_report[side] = _add_wrist_camera_visuals(
                carrier, side, mount, dimensions, resolved_hand_root_offset)
    # These source constraints and keys describe the removed Wuji hand.
    for name in ("contact", "equality", "keyframe"):
        node = carrier.find(name)
        if node is not None:
            carrier.remove(node)
    actuator_group = carrier.find("actuator")
    for actuator in list(actuator_group):
        if not actuator.get("joint", "").startswith(("left_fr3v2_joint", "right_fr3v2_joint")):
            actuator_group.remove(actuator)
    # Delete assets that were used only by the removed hand; Isaac's importer
    # should never see their obsolete visual/collision mesh paths.
    used_meshes = {geom.get("mesh") for geom in carrier.findall(".//geom[@mesh]")}
    for mesh in list(asset):
        if mesh.tag == "mesh" and mesh.get("name") not in used_meshes:
            asset.remove(mesh)

    for side in ("left", "right"):
        hand_file = source_root / config["robot"][f"sharpa_{side}_mjcf"]
        _append_hand(carrier, side, hand_file, resolved_hand_root_offset,
                     mesh_dir, hand_quats[side])
    for body in carrier.findall(".//body"):
        if body.find("inertial") is None and body.find("geom") is None:
            body.insert(0, ET.Element("inertial", {
                "pos": "0 0 0", "mass": "0.01", "diaginertia": "0.000001 0.000001 0.000001",
            }))
    joint_names = _named_joints(carrier)
    if len(joint_names) != 58 or len(set(joint_names)) != 58:
        raise ValueError(f"Expected 58 unique joints; got {len(joint_names)}")
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            if not carrier.find(f".//geom[@name='{side}_{finger}_elastomer']") is not None:
                raise ValueError(f"Missing tactile geometry: {side}_{finger}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(carrier)
    ET.ElementTree(carrier).write(output, encoding="utf-8", xml_declaration=True)
    report = {"source_root": str(source_root), "mjcf": str(output), "joint_count": len(joint_names),
              "joint_names": joint_names, "hand": "Sharpa Wave", "arm": "dual FR3", "tactile_mesh_count": 10}
    report["mount_xy_correction_m"] = list(mount_xy_correction_m)
    report["right_mount_xy_correction_m"] = list(right_mount_xy_correction_m)
    report["left_mount_xyz_correction_m"] = list(left_mount_xyz_correction_m)
    report["right_mount_xyz_correction_m"] = list(right_mount_xyz_correction_m)
    report["hand_root_quat_wxyz"] = {side: list(quat) for side, quat in hand_quats.items()}
    report["hand_root_offset_m"] = list(resolved_hand_root_offset)
    report["wrist_camera_visuals"] = wrist_visual_report
    report_file = output.with_suffix(".json")
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "tomato-pick-sim-local")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/isaac_tomato/dual_fr3_sharpa.xml")
    parser.add_argument("--mount-xy-correction-m", type=float, nargs=2, metavar=("X", "Y"),
                        default=(0.0, 0.0),
                        help="Experimental world-frame XY correction applied to both FR3 mounting points")
    parser.add_argument("--right-mount-xy-correction-m", type=float, nargs=2, metavar=("X", "Y"),
                        default=(0.0, 0.0),
                        help="Additional world-frame XY correction applied only to the right FR3 mounting point")
    parser.add_argument("--left-mount-xyz-correction-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=(0.0, 0.0, 0.0),
                        help="Measured world-frame XYZ correction applied only to the left FR3 mount")
    parser.add_argument("--right-mount-xyz-correction-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=(0.0, 0.0, 0.0),
                        help="Measured world-frame XYZ correction applied only to the right FR3 mount")
    parser.add_argument("--left-hand-root-quat-wxyz", type=float, nargs=4, metavar=("W", "X", "Y", "Z"),
                        default=(1.0, 0.0, 0.0, 0.0),
                        help="Fixed left link8-to-Sharpa-hand orientation")
    parser.add_argument("--right-hand-root-quat-wxyz", type=float, nargs=4, metavar=("W", "X", "Y", "Z"),
                        default=(1.0, 0.0, 0.0, 0.0),
                        help="Fixed right link8-to-Sharpa-hand orientation")
    parser.add_argument("--hand-root-offset-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=None,
                        help="Override the source-config link8-to-hand translation in meters")
    parser.add_argument("--wrist-camera-calibration-json", type=Path, default=None,
                        help="Add visual wrist rods and camera housings at calibrated link8-local poses")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output,
                           tuple(args.mount_xy_correction_m),
                           tuple(args.right_mount_xy_correction_m),
                           tuple(args.left_mount_xyz_correction_m),
                           tuple(args.right_mount_xyz_correction_m),
                           tuple(args.left_hand_root_quat_wxyz),
                           tuple(args.right_hand_root_quat_wxyz),
                           tuple(args.hand_root_offset_m) if args.hand_root_offset_m is not None else None,
                           args.wrist_camera_calibration_json), indent=2))
