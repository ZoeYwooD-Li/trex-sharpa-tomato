#!/usr/bin/env python3
"""Export the coupled local FEM geometry and synchronized Sharpa map indices to Isaac USD."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

import numpy as np

from elastic_block_fem_reference import surface_triangles
from sharpa_local_contact_fem import GEL_THICKNESS, HEIGHT, PAD_WIDTH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fem', type=Path, required=True)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fem_file = args.fem/'synchronized_fields.npz'
    map_file = args.maps/'tacmap_frames.npz'
    with np.load(fem_file) as fem, np.load(map_file) as maps:
        points, displacements = fem['points_m'], fem['displacement_m']
        travel, force = fem['platen_travel_mm'], fem['force_N']
        tets = (fem['sample_tets'], fem['gel_tets'])
        if not np.array_equal(fem['frame'], maps['fem_frame']) or not np.array_equal(force, maps['force_N']):
            raise RuntimeError('force and map arrays have incompatible frame IDs')
    args.output.mkdir(parents=True, exist_ok=False)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 640, 'height': 480})
    try:
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.SetTimeCodesPerSecond(30)
        stage.SetFramesPerSecond(30)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(15*(len(force)-1))
        world = UsdGeom.Xform.Define(stage, '/World')
        stage.SetDefaultPrim(world.GetPrim())
        world.GetPrim().CreateAttribute('fem:solver', Sdf.ValueTypeNames.String).Set(
            'External quasi-static elastic FEM; Sharpa 4F local profile and assumed gel')
        for name, file in (('fem:fields', fem_file), ('fem:tacmap', map_file)):
            world.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(os.path.relpath(file, args.output)))

        def box(name, z, dimensions, color):
            cube = UsdGeom.Cube.Define(stage, '/World/'+name)
            cube.CreateSizeAttr(1.)
            translation = cube.AddTranslateOp()
            translation.Set(Gf.Vec3d(0, 0, z))
            cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            return cube.GetPrim(), translation

        box('Ground', -.001, (.016, .016, .002), (.36, .38, .43))
        backing, backing_pose = box('SensorBacking', HEIGHT+GEL_THICKNESS+.0005,
                                    (PAD_WIDTH, PAD_WIDTH, .001), (.81, .46, .13))
        force_attr = backing.CreateAttribute('fem:normalForce_N', Sdf.ValueTypeNames.Double)
        frame_attr = backing.CreateAttribute('fem:fieldFrame', Sdf.ValueTypeNames.Int)
        geometries = []
        for name, indices, color in zip(('Sample', 'Gel'), tets, ((.16, .43, .7), (.19, .63, .48))):
            vertices, local = np.unique(indices, return_inverse=True)
            local = local.reshape(-1, 4)
            faces = surface_triangles(local)
            skin_vertices, skin_faces = np.unique(faces, return_inverse=True)
            tetmesh = UsdGeom.TetMesh.Define(stage, '/World/'+name+'Tet')
            tetmesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray.FromNumpy(local.astype(np.int32)))
            tetmesh.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray.FromNumpy(faces.astype(np.int32)))
            tetmesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
            skin = UsdGeom.Mesh.Define(stage, '/World/'+name+'Surface')
            skin.CreateFaceVertexCountsAttr([3]*len(faces))
            skin.CreateFaceVertexIndicesAttr(skin_faces.reshape(-1).tolist())
            skin.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            skin.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            geometries.append((tetmesh, skin, vertices, skin_vertices))

        for frame, (u, reaction, delta) in enumerate(zip(displacements, force, travel)):
            solved = points+u
            codes = [Usd.TimeCode(frame*15)]
            if frame == 0:
                codes.append(Usd.TimeCode.Default())
            for code in codes:
                for tetmesh, skin, vertices, skin_vertices in geometries:
                    xyz = solved[vertices].astype(np.float32)
                    tetmesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz), code)
                    skin.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz[skin_vertices]), code)
                backing_pose.Set(Gf.Vec3d(0, 0, HEIGHT+GEL_THICKNESS+.0005-delta/1000), code)
                force_attr.Set(float(reaction), code)
                frame_attr.Set(frame, code)

        path = args.output/'sharpa_local_contact.usda'
        if not stage.Export(str(path)):
            raise RuntimeError('USD export failed')
        exported = Usd.Stage.Open(str(path))
        backed = exported.GetPrimAtPath('/World/SensorBacking')
        node_error, force_error, pose_error = 0., 0., 0.
        for frame, (u, reaction, delta) in enumerate(zip(displacements, force, travel)):
            code = Usd.TimeCode(frame*15)
            for tetmesh, skin, vertices, skin_vertices in geometries:
                for geometry, ids in ((tetmesh, vertices), (skin, vertices[skin_vertices])):
                    positions = exported.GetPrimAtPath(geometry.GetPath()).GetAttribute('points').Get(code)
                    node_error = max(node_error, float(np.max(np.abs(np.asarray(positions)-(points+u)[ids]))))
            pose_z = backed.GetAttribute('xformOp:translate').Get(code)[2]
            pose_error = max(pose_error, abs(pose_z-(HEIGHT+GEL_THICKNESS+.0005-delta/1000)))
            force_error = max(force_error, abs(backed.GetAttribute('fem:normalForce_N').Get(code)-reaction))
            if backed.GetAttribute('fem:fieldFrame').Get(code) != frame:
                raise RuntimeError('USD map frame mismatch')
        result = dict(frames=len(force), max_position_error_m=node_error, max_pose_error_m=pose_error,
                      max_force_error_N=force_error, passed=max(node_error, pose_error, force_error)<1e-8)
        (args.output/'usd_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('SHARPA_USD', result, flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
