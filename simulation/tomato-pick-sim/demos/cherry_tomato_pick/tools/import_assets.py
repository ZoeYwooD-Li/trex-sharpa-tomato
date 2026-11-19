#!/usr/bin/env python3
"""Prepare source meshes for the ``cherry_tomato_pick`` demo.

The preferred input is the named ``stem``/``tomato_00..11`` OBJ split from the
latest ``cherry_tomato_sim`` revision.  The older FBX/ZIP path remains supported
so existing purchased-source archives are still reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh

from demos.cherry_tomato_pick.asset_import import prepare_texture_coordinates
from lycheesim.real2sim.binary_fbx import load_visual_meshes


UPSTREAM_SOURCE_REVISION = "43610761c4f667dc24ac6b5f7d66b71ebf72c1a1"
UPSTREAM_SOURCE_SCALE = 0.00869485
UPSTREAM_DETACHABLE_ANCHORS = {
    "tomato_07": (2.403886, 3.157519, -11.047812),
    "tomato_02": (-1.670465, 3.135315, -9.738387),
    "tomato_10": (-2.748041, 3.443140, 10.353864),
    "tomato_09": (2.655365, 3.445630, 9.084721),
}


def _archive_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in archive.namelist() if name.lower().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {suffix} member, found {matches}"
        )
    return matches[0]


def _orient_vertices(
    vertices: np.ndarray, *, target_center: np.ndarray, scale: float
) -> np.ndarray:
    """Convert source +Y branch direction to a hanging Z-up arrangement."""

    centered = (np.asarray(vertices, dtype=np.float64) - target_center) * scale
    # +90 degrees about X: source +Y becomes world +Z and source +Z becomes -Y.
    return np.column_stack((centered[:, 0], -centered[:, 2], centered[:, 1]))


def _orient_upstream_vertices(
    vertices: np.ndarray, *, target_center: np.ndarray, scale: float
) -> np.ndarray:
    """Bake the latest upstream ``-90 deg X`` assembled pose into a target frame."""

    centered = (np.asarray(vertices, dtype=np.float64) - target_center) * scale
    return np.column_stack((centered[:, 0], centered[:, 2], -centered[:, 1]))


def _load_split_visual_meshes(source_dir: Path) -> dict[str, trimesh.Trimesh]:
    names = ("stem", *(f"tomato_{index:02d}" for index in range(12)))
    missing = [name for name in names if not (source_dir / f"{name}.obj").is_file()]
    if missing:
        raise FileNotFoundError(
            f"split OBJ source is missing {missing!r} under {source_dir}"
        )
    result: dict[str, trimesh.Trimesh] = {}
    for name in names:
        loaded = trimesh.load(
            source_dir / f"{name}.obj", force="mesh", process=False
        )
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"expected one mesh in {name}.obj, got {type(loaded)!r}")
        result[name] = prepare_texture_coordinates(loaded)
    return result


def prepare_split_all_fruit_visual_meshes(
    source_dir: Path,
    *,
    target_name: str = "tomato_10",
    support_name: str = "tomato_07",
    source_scale: float = UPSTREAM_SOURCE_SCALE,
) -> tuple[dict[str, trimesh.Trimesh], trimesh.Trimesh, dict[str, object]]:
    """Prepare one centred render mesh and one pedicel point per fruit.

    The source contribution contains exact attachment coordinates for four
    endpoints.  For the other eight fruit, the closest source stem/fruit
    vertex pair is the reproducible geometric attachment approximation.  The
    distinction is retained in metadata rather than presented as source truth.
    """

    named = _load_split_visual_meshes(source_dir)
    if target_name not in UPSTREAM_DETACHABLE_ANCHORS:
        raise KeyError(
            f"target must be an upstream detachable endpoint, got {target_name!r}; "
            f"choose from {sorted(UPSTREAM_DETACHABLE_ANCHORS)}"
        )
    if support_name not in named or support_name == target_name:
        raise KeyError(
            f"support must be a different named tomato, got {support_name!r}"
        )
    if source_scale <= 0.0:
        raise ValueError("source scale must be positive")

    fruit_names = tuple(f"tomato_{index:02d}" for index in range(12))
    source_target_center = np.asarray(
        named[target_name].bounds.mean(axis=0), dtype=np.float64
    )
    transformed: dict[str, trimesh.Trimesh] = {}
    part_centers_m: dict[str, list[float]] = {}
    for name, mesh in named.items():
        result = mesh.copy()
        result.vertices = _orient_upstream_vertices(
            result.vertices,
            target_center=source_target_center,
            scale=float(source_scale),
        )
        transformed[name] = result
        part_centers_m[name] = np.asarray(result.bounds.mean(axis=0)).tolist()

    stem = transformed["stem"]
    stem_tree = cKDTree(np.asarray(stem.vertices, dtype=np.float64))
    fruit_visuals: dict[str, trimesh.Trimesh] = {}
    attachment_local_points: dict[str, list[float]] = {}
    attachment_point_sources: dict[str, str] = {}
    fruit_extents: dict[str, list[float]] = {}
    fruit_vertices: dict[str, int] = {}
    fruit_triangles: dict[str, int] = {}
    for name in fruit_names:
        fruit = transformed[name]
        center = np.asarray(fruit.bounds.mean(axis=0), dtype=np.float64)
        centred = fruit.copy()
        centred.vertices = np.asarray(centred.vertices) - center
        fruit_visuals[name] = centred

        if name in UPSTREAM_DETACHABLE_ANCHORS:
            source_anchor = np.asarray(
                UPSTREAM_DETACHABLE_ANCHORS[name], dtype=np.float64
            )
            world_anchor = _orient_upstream_vertices(
                source_anchor[None, :],
                target_center=source_target_center,
                scale=float(source_scale),
            )[0]
            local_point = world_anchor - center
            point_source = "source_explicit_anchor"
        else:
            distances, _ = stem_tree.query(
                np.asarray(fruit.vertices, dtype=np.float64), k=1
            )
            closest_fruit_vertex = int(np.argmin(distances))
            local_point = (
                np.asarray(fruit.vertices[closest_fruit_vertex], dtype=np.float64)
                - center
            )
            point_source = "closest_source_stem_vertex_pair"
        attachment_local_points[name] = local_point.tolist()
        attachment_point_sources[name] = point_source
        fruit_extents[name] = np.asarray(centred.extents).tolist()
        fruit_vertices[name] = int(len(centred.vertices))
        fruit_triangles[name] = int(len(centred.faces))

    target = fruit_visuals[target_name]
    support = fruit_visuals[support_name]
    metadata: dict[str, object] = {
        "source_repository": "git@github.com:yxsdev/cherry_tomato_sim.git",
        "source_revision": UPSTREAM_SOURCE_REVISION,
        "source_layout": "working/split_obj",
        "target_name": target_name,
        "support_name": support_name,
        "fruit_names": list(fruit_names),
        "support_local_origin_m": part_centers_m[support_name],
        "source_detachable_targets": sorted(UPSTREAM_DETACHABLE_ANCHORS),
        "detachable_targets": list(fruit_names),
        "source_target_center": source_target_center.tolist(),
        "source_to_metres_scale": float(source_scale),
        "orientation": "-90_deg_about_x_xyzw_[-0.70710678,0,0,0.70710678]",
        "attachment_local_point_m": attachment_local_points[target_name],
        "attachment_local_points_m": attachment_local_points,
        "attachment_point_sources": attachment_point_sources,
        "part_centers_relative_to_target_m": part_centers_m,
        "fruit_extents_m": fruit_extents,
        "fruit_vertices": fruit_vertices,
        "fruit_triangles": fruit_triangles,
        "target_extents_m": np.asarray(target.extents).tolist(),
        "support_extents_m": np.asarray(support.extents).tolist(),
        "target_vertices": int(len(target.vertices)),
        "support_vertices": int(len(support.vertices)),
        "target_triangles": int(len(target.faces)),
        "support_triangles": int(len(support.faces)),
        "texture_coordinate_transform": "OBJ_bottom_left_to_runtime_top_left_v=1-v",
        "uv_seams_preserved": True,
        "split_obj_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_dir.glob("*.obj"))
        },
    }
    return fruit_visuals, stem, metadata


def prepare_split_visual_meshes(
    source_dir: Path,
    *,
    target_name: str = "tomato_10",
    support_name: str = "tomato_07",
    source_scale: float = UPSTREAM_SOURCE_SCALE,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh, dict[str, object]]:
    """Prepare the latest upstream split asset without an FBX SDK dependency."""
    fruit_visuals, stem, metadata = prepare_split_all_fruit_visual_meshes(
        source_dir,
        target_name=target_name,
        support_name=support_name,
        source_scale=source_scale,
    )
    part_centers_m = metadata["part_centers_relative_to_target_m"]
    target = fruit_visuals[target_name]
    support = fruit_visuals[support_name]
    cluster = trimesh.util.concatenate(
        [
            stem,
            *[
                mesh.copy().apply_translation(part_centers_m[name])
                for name, mesh in fruit_visuals.items()
                if name not in {target_name, support_name}
            ],
        ]
    )
    metadata["cluster_extents_m"] = np.asarray(cluster.extents).tolist()
    metadata["cluster_vertices"] = int(len(cluster.vertices))
    metadata["cluster_triangles"] = int(len(cluster.faces))
    return target, support, cluster, metadata


def prepare_visual_meshes(
    fbx_path: Path,
    *,
    target_name: str = "tomato_10",
    support_name: str = "tomato_07",
    target_diameter_m: float = 0.05,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh, dict[str, object]]:
    named = {
        name: prepare_texture_coordinates(mesh)
        for name, mesh in load_visual_meshes(fbx_path)
    }
    tomatoes = {name: mesh for name, mesh in named.items() if name.startswith("tomato_")}
    if len(tomatoes) != 12 or "stem" not in named:
        raise ValueError(
            "expected twelve tomato meshes and one stem mesh, got "
            f"{sorted(named)}"
        )
    if target_name not in tomatoes:
        raise KeyError(f"unknown target {target_name!r}; choose from {sorted(tomatoes)}")
    if support_name not in tomatoes or support_name == target_name:
        raise KeyError(
            f"support must be a different named tomato, got {support_name!r}"
        )
    if target_diameter_m <= 0.0:
        raise ValueError("target diameter must be positive")

    source_target = tomatoes[target_name]
    target_center = np.asarray(source_target.bounds.mean(axis=0), dtype=np.float64)
    scale = float(target_diameter_m) / float(np.max(source_target.extents))

    transformed: dict[str, trimesh.Trimesh] = {}
    for name, mesh in named.items():
        result = mesh.copy()
        result.vertices = _orient_vertices(
            result.vertices, target_center=target_center, scale=scale
        )
        transformed[name] = result

    target = transformed[target_name]
    support = transformed[support_name]
    cluster = trimesh.util.concatenate(
        [
            mesh
            for name, mesh in transformed.items()
            if name not in {target_name, support_name}
        ]
    )
    support_local_origin = np.asarray(support.bounds.mean(axis=0), dtype=np.float64)
    support.vertices = np.asarray(support.vertices) - support_local_origin
    metadata: dict[str, object] = {
        "target_name": target_name,
        "support_name": support_name,
        "support_local_origin_m": support_local_origin.tolist(),
        "target_diameter_m": float(target_diameter_m),
        "source_target_center": target_center.tolist(),
        "source_to_metres_scale": scale,
        "orientation": "+90_deg_about_x_xyzw_[0.70710678,0,0,0.70710678]",
        "target_extents_m": np.asarray(target.extents).tolist(),
        "support_extents_m": np.asarray(support.extents).tolist(),
        "cluster_extents_m": np.asarray(cluster.extents).tolist(),
        "target_vertices": int(len(target.vertices)),
        "support_vertices": int(len(support.vertices)),
        "cluster_vertices": int(len(cluster.vertices)),
        "target_triangles": int(len(target.faces)),
        "support_triangles": int(len(support.faces)),
        "cluster_triangles": int(len(cluster.faces)),
        "texture_coordinate_transform": "FBX_bottom_left_to_runtime_top_left_v=1-v",
        "uv_seams_preserved": True,
    }
    return target, support, cluster, metadata


def _save_visual(path: Path, mesh: trimesh.Trimesh) -> None:
    uvs = getattr(mesh.visual, "uv", None)
    if uvs is None:
        raise ValueError(f"visual mesh has no UVs: {path.name}")
    np.savez_compressed(
        path,
        vertices=np.ascontiguousarray(mesh.vertices, dtype=np.float32),
        faces=np.ascontiguousarray(mesh.faces, dtype=np.int32),
        normals=np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32),
        uvs=np.ascontiguousarray(uvs, dtype=np.float32),
    )


def _fruit_only_texture(texture: Image.Image) -> Image.Image:
    """Keep source fruit texels and make every other atlas region red-safe.

    Newton may reuse mesh attributes while compiling several shapes that share
    one texture atlas.  A fruit can then sample a stem UV after it rotates even
    though the checked source UVs are entirely inside red fruit islands.  The
    stem continues to use the untouched source atlas; this derived fruit-only
    copy preserves every source red texel and maps the unused green atlas
    regions to a luminance-preserving red material.
    """

    rgba = np.asarray(texture.convert("RGBA"), dtype=np.uint8).copy()
    rgb = rgba[..., :3].astype(np.float32)
    source_red = (rgb[..., 0] > 1.25 * rgb[..., 1]) & (rgb[..., 0] > 70.0)
    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    replacement = np.stack(
        (
            np.clip(80.0 + 0.75 * luminance, 0.0, 255.0),
            np.clip(8.0 + 0.11 * luminance, 0.0, 255.0),
            np.clip(4.0 + 0.045 * luminance, 0.0, 255.0),
        ),
        axis=-1,
    ).astype(np.uint8)
    rgba[..., :3][~source_red] = replacement[~source_red]
    return Image.fromarray(rgba)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path(
            "demos/cherry_tomato_pick/assets/source/upstream_split_obj"
        ),
        help="latest split-OBJ directory, legacy FBX, or original ZIP archive",
    )
    parser.add_argument(
        "--basecolor",
        type=Path,
        help="base-color map for FBX input; defaults beside the demo source",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demos/cherry_tomato_pick/assets/runtime"),
    )
    parser.add_argument("--target", default="tomato_10")
    parser.add_argument("--support", default="tomato_07")
    parser.add_argument("--target-diameter", type=float, default=0.05)
    parser.add_argument("--source-scale", type=float, default=UPSTREAM_SOURCE_SCALE)
    parser.add_argument("--texture-size", type=int, default=2048)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(source_path: Path, texture_source: Path) -> dict[str, object]:
        if source_path.is_dir():
            target, support, cluster, metadata = prepare_split_visual_meshes(
                source_path,
                target_name=args.target,
                support_name=args.support,
                source_scale=args.source_scale,
            )
            fruit_visuals, stem_visual, all_fruit_metadata = (
                prepare_split_all_fruit_visual_meshes(
                    source_path,
                    target_name=args.target,
                    support_name=args.support,
                    source_scale=args.source_scale,
                )
            )
            metadata.update(all_fruit_metadata)
        else:
            target, support, cluster, metadata = prepare_visual_meshes(
                source_path,
                target_name=args.target,
                support_name=args.support,
                target_diameter_m=args.target_diameter,
            )
        _save_visual(output_dir / "target_tomato_visual.npz", target)
        _save_visual(output_dir / "support_tomato_visual.npz", support)
        _save_visual(output_dir / "tomato_cluster_visual.npz", cluster)
        if source_path.is_dir():
            _save_visual(output_dir / "stem_visual.npz", stem_visual)
            for name, visual in fruit_visuals.items():
                _save_visual(output_dir / f"{name}_visual.npz", visual)

        texture_path = output_dir / "tomato_basecolor.png"
        fruit_texture_path = output_dir / "tomato_fruit_basecolor.png"
        with Image.open(texture_source) as image:
            texture = image.convert("RGBA")
            texture.thumbnail(
                (args.texture_size, args.texture_size), Image.Resampling.LANCZOS
            )
            texture.save(texture_path, optimize=True)
            _fruit_only_texture(texture).save(fruit_texture_path, optimize=True)
        metadata.update(
            {
                "source_basecolor": texture_source.name,
                "source_basecolor_sha256": hashlib.sha256(
                    texture_source.read_bytes()
                ).hexdigest(),
                "asset_status": "purchased asset supplied by the project owner",
                "license_notice": "redistribution is governed by the purchase license",
                "texture_size": list(texture.size),
                "fruit_texture": fruit_texture_path.name,
                "fruit_texture_derivation": (
                    "source red texels unchanged; non-fruit atlas regions remapped "
                    "to luminance-preserving red for renderer attribute isolation"
                ),
            }
        )
        if source_path.is_dir():
            metadata["source_directory"] = source_path.name
        else:
            metadata["source_fbx"] = source_path.name
            metadata["source_fbx_sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
        return metadata

    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(
            prefix="lycheesim-cherry-tomato-"
        ) as temporary:
            fbx_member = _archive_member(archive, ".fbx")
            texture_member = _archive_member(archive, "tomato_mat_basecolor.png")
            fbx_path = Path(temporary) / "cherry_tomato_branch_02.fbx"
            texture_source = Path(temporary) / "tomato_Mat_baseColor.png"
            fbx_path.write_bytes(archive.read(fbx_member))
            texture_source.write_bytes(archive.read(texture_member))
            metadata = prepare(fbx_path, texture_source)
        metadata["source_archive"] = source.name
        metadata["source_archive_sha256"] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
    else:
        texture_source = (
            args.basecolor.expanduser().resolve()
            if args.basecolor is not None
            else source.parent / "textures" / "tomato_Mat_baseColor.png"
        )
        metadata = prepare(source, texture_source)

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
