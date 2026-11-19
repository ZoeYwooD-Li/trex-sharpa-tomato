"""Task-owned asset helpers for cherry-tomato harvesting."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Mapping
import xml.etree.ElementTree as ET


def _cherry_tomato_robot_source_asset() -> Path:
    repository_root = Path(__file__).resolve().parents[4]
    path = (
        repository_root
        / "demos"
        / "assets"
        / "robots"
        / "dual_fr3_linker_l20"
        / "model.collision.xml"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def cherry_tomato_visual_assets() -> Mapping[str, Path]:
    """Return the tracked detailed bunch meshes and shared albedo texture."""

    root = (
        Path(__file__).resolve().parents[4]
        / "demos"
        / "cherry_tomato_pick"
        / "assets"
        / "runtime"
    )
    paths = {
        "target": root / "target_tomato_visual.npz",
        "support": root / "support_tomato_visual.npz",
        "cluster": root / "tomato_cluster_visual.npz",
        "albedo": root / "tomato_basecolor.png",
        "metadata": root / "metadata.json",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def cherry_tomato_all_fruit_visual_assets() -> Mapping[str, object]:
    """Return separate stem and fruit meshes for the all-pickable scene."""

    root = (
        Path(__file__).resolve().parents[4]
        / "demos"
        / "cherry_tomato_pick"
        / "assets"
        / "runtime"
    )
    fruit_names = tuple(f"tomato_{index:02d}" for index in range(12))
    fruits = {name: root / f"{name}_visual.npz" for name in fruit_names}
    paths: dict[str, object] = {
        "fruits": fruits,
        "stem": root / "stem_visual.npz",
        "albedo": root / "tomato_basecolor.png",
        "fruit_albedo": root / "tomato_fruit_basecolor.png",
        "metadata": root / "metadata.json",
    }
    flat_paths = [
        *fruits.values(),
        paths["stem"],
        paths["albedo"],
        paths["fruit_albedo"],
        paths["metadata"],
    ]
    missing = [path for path in flat_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


@lru_cache(maxsize=1)
def cherry_tomato_robot_asset() -> Path:
    """Return the single-root carrier with the tomato task's authored drives."""

    source = _cherry_tomato_robot_source_asset()
    source_bytes = source.read_bytes()
    profile = (Path(__file__).resolve().parents[4]
               / "demos/cherry_tomato_pick/assets/robot_actuators.xml")
    profile_bytes = profile.read_bytes()
    digest = hashlib.sha256(
        source_bytes
        + b"\0"
        + profile_bytes
        + b"\0"
        + os.fspath(source.parent.resolve()).encode("utf-8")
        + b"\0single-root-task-drives-v4"
    ).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "lycheesim-task-assets"
    output = cache_dir / f"dual_fr3_sharpa_tomato-{digest}.xml"
    if output.is_file():
        return output

    root = ET.fromstring(source_bytes)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Carrier MJCF has no worldbody: {source}")
    support = [
        body
        for body in worldbody.findall("body")
        if body.attrib.get("name") == "support_box"
    ]
    if len(support) != 1:
        raise ValueError(
            "Carrier MJCF must contain exactly one support_box root"
        )
    worldbody.remove(support[0])
    roots = [body.attrib.get("name") for body in worldbody.findall("body")]
    if roots != ["base_link"]:
        raise ValueError(f"Unexpected carrier worldbody roots: {roots}")
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError(f"Carrier MJCF has no compiler element: {source}")
    compiler.set("meshdir", str(source.parent.resolve()))

    # Keep geometry, passive joint properties and coupling constraints from
    # the carrier. Only this task view replaces the actuator section.
    defaults = root.find("default")
    defaults.extend(ET.fromstring(profile_bytes).find("default"))
    root.remove(root.find("actuator"))
    actuators = ET.SubElement(root, "actuator")
    for joint in worldbody.iter("joint"):
        name = joint.attrib["name"]
        arm = name.startswith(("left_fr3v2_joint", "right_fr3v2_joint"))
        ET.SubElement(actuators, "position", {
            "name": f"{name}_servo", "joint": name,
            "class": "tomato_arm" if arm else "tomato_hand",
        })

    cache_dir.mkdir(parents=True, exist_ok=True)
    candidate = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        ET.ElementTree(root).write(
            candidate, encoding="utf-8", xml_declaration=True
        )
        os.replace(candidate, output)
    finally:
        candidate.unlink(missing_ok=True)
    return output


__all__ = [
    "cherry_tomato_all_fruit_visual_assets",
    "cherry_tomato_robot_asset",
    "cherry_tomato_visual_assets",
]
