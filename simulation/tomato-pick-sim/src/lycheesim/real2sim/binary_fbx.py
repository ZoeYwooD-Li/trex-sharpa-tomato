"""Minimal binary-FBX mesh reader for offline asset conversion.

LycheeSim runtime code consumes normalized meshes and does not depend on an
FBX SDK.  This module keeps the small, reusable subset needed by demo asset
importers: geometry/model connections, local transforms, polygon
triangulation, and texture coordinates.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


@dataclass
class Node:
    name: str
    properties: list[object]
    children: list["Node"]

    def child(self, name: str) -> "Node | None":
        return next((item for item in self.children if item.name == name), None)


class BinaryFbx:
    """Parse the node/property subset used by LycheeSim source assets."""

    def __init__(self, data: bytes):
        if not data.startswith(b"Kaydara FBX Binary  \x00\x1a\x00"):
            raise ValueError("input is not a binary FBX")
        self.data = data
        self.version = struct.unpack_from("<I", data, 23)[0]
        self.offset = 27
        self.wide = self.version >= 7500

    def _property(self) -> object:
        kind = chr(self.data[self.offset])
        self.offset += 1
        scalars = {
            "Y": ("<h", 2),
            "C": ("<?", 1),
            "I": ("<i", 4),
            "F": ("<f", 4),
            "D": ("<d", 8),
            "L": ("<q", 8),
        }
        if kind in scalars:
            fmt, size = scalars[kind]
            value = struct.unpack_from(fmt, self.data, self.offset)[0]
            self.offset += size
            return value
        if kind in {"S", "R"}:
            length = struct.unpack_from("<I", self.data, self.offset)[0]
            self.offset += 4
            value = self.data[self.offset : self.offset + length]
            self.offset += length
            return value.decode("utf-8", errors="replace") if kind == "S" else value
        if kind in {"f", "d", "l", "i", "b", "c"}:
            length, encoding, byte_count = struct.unpack_from(
                "<III", self.data, self.offset
            )
            self.offset += 12
            raw = self.data[self.offset : self.offset + byte_count]
            self.offset += byte_count
            if encoding == 1:
                raw = zlib.decompress(raw)
            elif encoding != 0:
                raise ValueError(f"unsupported FBX array encoding {encoding}")
            dtype = {
                "f": "<f4",
                "d": "<f8",
                "l": "<i8",
                "i": "<i4",
                "b": "<?",
                "c": "i1",
            }[kind]
            return np.frombuffer(raw, dtype=dtype, count=length).copy()
        raise ValueError(f"unsupported FBX property type {kind!r}")

    def _node(self) -> Node | None:
        if self.wide:
            end_offset, property_count, _property_bytes = struct.unpack_from(
                "<QQQ", self.data, self.offset
            )
            self.offset += 24
        else:
            end_offset, property_count, _property_bytes = struct.unpack_from(
                "<III", self.data, self.offset
            )
            self.offset += 12
        name_length = self.data[self.offset]
        self.offset += 1
        if end_offset == 0:
            return None
        name = self.data[self.offset : self.offset + name_length].decode(
            "utf-8", errors="replace"
        )
        self.offset += name_length
        properties = [self._property() for _ in range(property_count)]
        children: list[Node] = []
        while self.offset < end_offset:
            child = self._node()
            if child is None:
                break
            children.append(child)
        self.offset = end_offset
        return Node(name, properties, children)

    def roots(self) -> list[Node]:
        result: list[Node] = []
        while self.offset < len(self.data):
            node = self._node()
            if node is None:
                break
            result.append(node)
        return result


def _clean_name(value: object) -> str:
    return str(value).split("\x00", 1)[0].split("::")[-1]


def _polygons(indices: np.ndarray) -> list[list[int]]:
    result: list[list[int]] = []
    polygon: list[int] = []
    for raw in np.asarray(indices, dtype=np.int64):
        value = int(raw)
        polygon.append(-value - 1 if value < 0 else value)
        if value < 0:
            result.append(polygon)
            polygon = []
    if polygon:
        raise ValueError("unterminated FBX polygon")
    return result


def _triangulate(indices: np.ndarray) -> np.ndarray:
    triangles = [
        (polygon[0], polygon[corner], polygon[corner + 1])
        for polygon in _polygons(indices)
        for corner in range(1, len(polygon) - 1)
    ]
    return np.asarray(triangles, dtype=np.int32)


def _child_value(node: Node, name: str) -> object:
    child = node.child(name)
    if child is None or not child.properties:
        raise ValueError(f"FBX node {node.name!r} has no {name!r} value")
    return child.properties[0]


def _property_vector(
    model: Node,
    name: str,
    default: tuple[float, float, float],
) -> np.ndarray:
    properties = model.child("Properties70")
    if properties is None:
        return np.asarray(default, dtype=np.float64)
    for item in properties.children:
        if item.name == "P" and item.properties and item.properties[0] == name:
            return np.asarray(item.properties[-3:], dtype=np.float64)
    return np.asarray(default, dtype=np.float64)


def _geometry_mesh(geometry: Node, model: Node) -> trimesh.Trimesh:
    source_vertices = np.asarray(
        _child_value(geometry, "Vertices"), dtype=np.float64
    ).reshape(-1, 3)
    source_polygons = _polygons(
        np.asarray(_child_value(geometry, "PolygonVertexIndex"))
    )
    scaling = _property_vector(model, "Lcl Scaling", (1.0, 1.0, 1.0))
    rotation_deg = _property_vector(model, "Lcl Rotation", (0.0, 0.0, 0.0))
    translation = _property_vector(model, "Lcl Translation", (0.0, 0.0, 0.0))
    transformed = Rotation.from_euler("XYZ", rotation_deg, degrees=True).apply(
        source_vertices * scaling
    ) + translation

    layer_uv = geometry.child("LayerElementUV")
    direct_uv: np.ndarray | None = None
    corner_uv_indices: np.ndarray | None = None
    if layer_uv is not None:
        uv_node = layer_uv.child("UV")
        uv_index_node = layer_uv.child("UVIndex")
        if uv_node is not None and uv_index_node is not None:
            direct_uv = np.asarray(uv_node.properties[0], dtype=np.float64).reshape(
                -1, 2
            )
            corner_uv_indices = np.asarray(
                uv_index_node.properties[0], dtype=np.int64
            )

    vertices: list[np.ndarray] = []
    texcoords: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    corner_cursor = 0
    for polygon in source_polygons:
        polygon_uv = None
        if direct_uv is not None and corner_uv_indices is not None:
            polygon_uv = direct_uv[
                corner_uv_indices[corner_cursor : corner_cursor + len(polygon)]
            ]
        for corner in range(1, len(polygon) - 1):
            face: list[int] = []
            for polygon_corner in (0, corner, corner + 1):
                face.append(len(vertices))
                vertices.append(transformed[polygon[polygon_corner]])
                if polygon_uv is not None:
                    texcoords.append(polygon_uv[polygon_corner])
            faces.append(tuple(face))
        corner_cursor += len(polygon)

    visual = None
    if texcoords:
        visual = trimesh.visual.TextureVisuals(
            uv=np.asarray(texcoords, dtype=np.float64)
        )
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        visual=visual,
        process=False,
    )


def load_meshes(path: Path) -> list[tuple[str, trimesh.Trimesh]]:
    """Load geometry without baking model transforms or splitting UV seams."""

    roots = BinaryFbx(path.read_bytes()).roots()
    objects = next(node for node in roots if node.name == "Objects")
    connections = next(node for node in roots if node.name == "Connections")
    geometries: dict[int, tuple[str, trimesh.Trimesh]] = {}
    model_names: dict[int, str] = {}
    for obj in objects.children:
        if len(obj.properties) < 2:
            continue
        object_id = int(obj.properties[0])
        if obj.name == "Model":
            model_names[object_id] = _clean_name(obj.properties[1])
        elif obj.name == "Geometry" and len(obj.properties) >= 3:
            if str(obj.properties[2]) != "Mesh":
                continue
            vertex_node = obj.child("Vertices")
            index_node = obj.child("PolygonVertexIndex")
            if vertex_node is None or index_node is None:
                continue
            vertices = np.asarray(
                vertex_node.properties[0], dtype=np.float64
            ).reshape(-1, 3)
            faces = _triangulate(np.asarray(index_node.properties[0]))
            geometries[object_id] = (
                _clean_name(obj.properties[1]),
                trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            )

    geometry_to_model: dict[int, int] = {}
    for connection in connections.children:
        props = connection.properties
        if connection.name == "C" and len(props) >= 3 and props[0] == "OO":
            child_id, parent_id = int(props[1]), int(props[2])
            if child_id in geometries and parent_id in model_names:
                geometry_to_model[child_id] = parent_id

    names_seen: dict[str, int] = {}
    meshes: list[tuple[str, trimesh.Trimesh]] = []
    for geometry_id, (geometry_name, mesh) in geometries.items():
        base_name = model_names.get(
            geometry_to_model.get(geometry_id, -1), geometry_name
        )
        ordinal = names_seen.get(base_name, 0) + 1
        names_seen[base_name] = ordinal
        unique_name = base_name if ordinal == 1 else f"{base_name}_{ordinal}"
        meshes.append((unique_name, mesh))
    return meshes


def load_visual_meshes(path: Path) -> list[tuple[str, trimesh.Trimesh]]:
    """Load connected mesh models with local transforms and UV seam vertices."""

    roots = BinaryFbx(path.read_bytes()).roots()
    objects = next(node for node in roots if node.name == "Objects")
    connections = next(node for node in roots if node.name == "Connections")
    geometries = {
        int(node.properties[0]): node
        for node in objects.children
        if node.name == "Geometry"
        and len(node.properties) >= 3
        and node.properties[2] == "Mesh"
    }
    models = {
        int(node.properties[0]): node
        for node in objects.children
        if node.name == "Model"
        and len(node.properties) >= 3
        and node.properties[2] == "Mesh"
    }
    geometry_to_model: dict[int, int] = {}
    for connection in connections.children:
        props = connection.properties
        if connection.name == "C" and len(props) >= 3 and props[0] == "OO":
            child_id, parent_id = int(props[1]), int(props[2])
            if child_id in geometries and parent_id in models:
                geometry_to_model[child_id] = parent_id

    result: list[tuple[str, trimesh.Trimesh]] = []
    for geometry_id, geometry in geometries.items():
        model_id = geometry_to_model.get(geometry_id)
        if model_id is None:
            continue
        model = models[model_id]
        result.append(
            (_clean_name(model.properties[1]), _geometry_mesh(geometry, model))
        )
    if not result:
        raise ValueError(f"no connected mesh geometry found in {path}")
    return result
