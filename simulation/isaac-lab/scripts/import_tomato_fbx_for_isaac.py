#!/usr/bin/env python3
"""Rebuild the Isaac tomato visuals directly from the source FBX models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
import trimesh

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SIM_SOURCE = PROJECT.parent / "tomato-pick-sim-local"
sys.path.insert(0, str(DEFAULT_SIM_SOURCE))

from demos.cherry_tomato_pick.asset_import import prepare_texture_coordinates
from lycheesim.real2sim.binary_fbx import load_visual_meshes
from demos.cherry_tomato_pick.tools.import_assets import _fruit_only_texture


def save_mesh(path: Path, mesh: trimesh.Trimesh) -> None:
    np.savez_compressed(
        path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        uvs=np.asarray(mesh.visual.uv, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source FBX")
    parser.add_argument("--texture", type=Path, required=True, help="Source base-color PNG")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diameter-m", type=float, default=0.0344,
                        help="Match the existing 34.4 mm Isaac collision sphere")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    texture_path = args.texture.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    meshes = {
        name: prepare_texture_coordinates(mesh)
        for name, mesh in load_visual_meshes(source)
    }
    fruit_names = tuple(f"tomato_{index:02d}" for index in range(12))
    missing = [name for name in ("stem", *fruit_names) if name not in meshes]
    if missing:
        raise ValueError(f"FBX is missing expected models: {missing}")

    target = meshes["tomato_10"]
    target_center = np.asarray(target.bounds.mean(axis=0), dtype=np.float64)
    target_diameter = float(np.max(target.extents))
    scale = args.diameter_m / target_diameter
    centers: dict[str, list[float]] = {}
    centered_meshes: dict[str, trimesh.Trimesh] = {}
    for name in fruit_names:
        transformed = meshes[name].copy()
        transformed.vertices = (np.asarray(transformed.vertices) - target_center) * scale
        center = np.asarray(transformed.bounds.mean(axis=0), dtype=np.float64)
        centers[name] = center.tolist()
        transformed.vertices = np.asarray(transformed.vertices) - center
        centered_meshes[name] = transformed
        save_mesh(output / f"{name}_visual.npz", transformed)

    stem = meshes["stem"].copy()
    stem.vertices = (np.asarray(stem.vertices) - target_center) * scale
    save_mesh(output / "stem_visual.npz", stem)

    # Locate the physical connection from the actual FBX surfaces, rather than
    # reusing a point measured on the split-OBJ conversion.
    target_surface = centered_meshes["tomato_10"]
    stem_tree = cKDTree(np.asarray(stem.vertices, dtype=np.float64))
    distances, stem_indices = stem_tree.query(
        np.asarray(target_surface.vertices, dtype=np.float64), k=1
    )
    target_index = int(np.argmin(distances))
    stem_index = int(stem_indices[target_index])
    target_attach = np.asarray(target_surface.vertices[target_index], dtype=np.float64)
    stem_attach = np.asarray(stem.vertices[stem_index], dtype=np.float64)

    from PIL import Image

    with Image.open(texture_path) as image:
        atlas = image.convert("RGBA")
        atlas.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        atlas.save(output / "tomato_basecolor.png", optimize=True)
        _fruit_only_texture(atlas).save(output / "tomato_fruit_basecolor.png", optimize=True)

    metadata = {
        "source_fbx": str(source),
        "source_fbx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_texture_sha256": hashlib.sha256(texture_path.read_bytes()).hexdigest(),
        "source_model": "FBX connected mesh models with source local transforms",
        "source_orientation": "FBX model coordinates preserved; root rotation is identity",
        "source_scale": scale,
        "target_diameter_m": args.diameter_m,
        "target_extent_m": np.asarray(centered_meshes["tomato_10"].extents).tolist(),
        "fruit_centers_relative_to_target_m": centers,
        "stem_extent_m": np.asarray(stem.extents).tolist(),
        "attachment_target_local_m": target_attach.tolist(),
        "attachment_cluster_local_m": stem_attach.tolist(),
        "attachment_surface_gap_m": float(distances[target_index]),
        "fruit_vertices": {name: len(centered_meshes[name].vertices) for name in fruit_names},
        "fruit_triangles": {name: len(centered_meshes[name].faces) for name in fruit_names},
        "tomato_root_rotation_xyz_deg": [0.0, 0.0, 0.0],
        "vine_root_rotation_xyz_deg": [0.0, 0.0, 0.0],
        "visual_tomato_count": len(fruit_names),
    }
    (output / "isaac_layout.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
