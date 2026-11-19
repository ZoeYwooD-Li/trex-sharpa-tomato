"""Texture-coordinate preparation shared by the tomato asset importer/tests."""

from __future__ import annotations

import numpy as np
import trimesh


def prepare_texture_coordinates(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Preserve UV seams and convert OBJ/FBX V coordinates for runtime images.

    Trimesh expands vertices at UV seams when loading OBJ. Merging those
    vertices by position discards one side of every seam and makes unrelated
    atlas islands bleed across a fruit. OBJ/FBX UVs also use a bottom-left
    origin, while Rerun/Newton texture arrays use a top-left image origin.
    """

    result = mesh.copy()
    source_uv = getattr(result.visual, "uv", None)
    if source_uv is None:
        raise ValueError("cherry-tomato mesh has no texture coordinates")
    runtime_uv = np.asarray(source_uv, dtype=np.float64).copy()
    runtime_uv[:, 1] = 1.0 - runtime_uv[:, 1]
    result.visual.uv = runtime_uv
    return result


__all__ = ["prepare_texture_coordinates"]
