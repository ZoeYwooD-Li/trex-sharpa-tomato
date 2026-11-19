#!/usr/bin/env python3
"""Quasi-static tetrahedral FEM baseline for one tomato and rigid platens.

The mesh is generated from the project's 17.2 mm radius, then solved with
linear isotropic elasticity and unilateral penalty contact.  USD is a baked
visualization of FEM node positions; no visual-only deformation is authored.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
from pathlib import Path
import sys


def _prepare_usd_runtime():
    """Find the USD modules shipped with this Isaac Sim Python environment."""
    if importlib.util.find_spec("pxr") is not None:
        return
    pattern = str(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" /
                  "site-packages" / "isaacsim" / "extscache" / "omni.usd.libs-*")
    candidates = glob.glob(pattern)
    if not candidates:
        return
    usd_root = Path(candidates[0])
    sys.path.insert(0, str(usd_root))
    if os.environ.get("TOMATO_FEM_USD_REEXEC") != "1":
        env = os.environ.copy()
        env["TOMATO_FEM_USD_REEXEC"] = "1"
        bins = str(usd_root / "bin")
        env["LD_LIBRARY_PATH"] = ":".join(filter(None, (bins, str(Path(sys.prefix) / "lib"),
                                                          env.get("LD_LIBRARY_PATH", ""))))
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_prepare_usd_runtime()

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

RADIUS = 0.015
MASS = 0.006644
POISSON = 0.335
MODULI = (100_000.0, 200_000.0, 281_000.0, 500_000.0, 1_000_000.0)
G = 9.81
PENALTY = 1.0e7  # N/m; numerical contact penalty, convergence-checked


def make_mesh(subdivisions=3):
    surf = trimesh.creation.icosphere(subdivisions=subdivisions, radius=RADIUS)
    boundary = np.asarray(surf.vertices, dtype=np.float64)
    # Add interior points so the tetrahedral FEM is not a surface-only shell.
    spacing = RADIUS * 0.40
    coords = np.arange(-RADIUS + spacing, RADIUS, spacing)
    interior = np.array([(x, y, z) for x in coords for y in coords for z in coords
                         if x*x + y*y + z*z < (RADIUS * 0.90) ** 2], dtype=float)
    if not len(interior):
        interior = np.zeros((1, 3), dtype=float)
    points = np.vstack([boundary, interior])
    tri = Delaunay(points)
    tet = tri.simplices.copy()
    tet = tet[np.linalg.norm(points[tet].mean(axis=1), axis=1) <= RADIUS * 1.001]
    tet_xyz = points[tet]
    signed6 = np.linalg.det(tet_xyz[:, 1:, :] - tet_xyz[:, :1, :])
    swap = signed6 < 0
    tet[swap, 0], tet[swap, 1] = tet[swap, 1].copy(), tet[swap, 0].copy()
    vol = np.abs(signed6) / 6.0
    valid = vol > 1.0e-15
    tet, vol = tet[valid], vol[valid]
    hull = ConvexHull(boundary)
    faces = hull.simplices.copy()
    # World z starts at the floor; the sphere's lower pole is exactly z=0.
    points[:, 2] += RADIUS
    return points, tet, faces, len(boundary), vol


def tet_quality(points, tet):
    xyz = points[tet]
    signed6 = np.linalg.det(xyz[:, 1:, :] - xyz[:, :1, :])
    vol6 = np.abs(signed6)
    vol = vol6 / 6.0
    edge2 = np.zeros(len(tet))
    for i in range(4):
        for j in range(i + 1, 4):
            edge2 += np.sum((xyz[:, i] - xyz[:, j]) ** 2, axis=1)
    q = 12.0 * np.maximum(3.0 * vol, 0.0) ** (2.0 / 3.0) / np.maximum(edge2, 1e-30)
    return vol, q, signed6 / 6.0


def assemble(points, tet, modulus, nu, rho):
    lam = modulus * nu / ((1 + nu) * (1 - 2 * nu))
    mu = modulus / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    np.fill_diagonal(D[:3, :3], lam + 2 * mu)
    D[3:, 3:] = np.eye(3) * mu
    rows, cols, vals = [], [], []
    fg = np.zeros((len(points), 3))
    for ids in tet:
        xyz = points[ids]
        X = np.column_stack([np.ones(4), xyz])
        inv = np.linalg.inv(X)
        grad = inv[1:, :].T
        B = np.zeros((6, 12))
        for i, (dx, dy, dz) in enumerate(grad):
            j = 3 * i
            B[0, j] = dx
            B[1, j + 1] = dy
            B[2, j + 2] = dz
            B[3, j] = dy; B[3, j + 1] = dx
            B[4, j + 1] = dz; B[4, j + 2] = dy
            B[5, j] = dz; B[5, j + 2] = dx
        v = abs(np.linalg.det(xyz[1:] - xyz[:1])) / 6.0
        ke = v * (B.T @ D @ B)
        dofs = np.array([[3*n, 3*n+1, 3*n+2] for n in ids]).ravel()
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel()); cols.extend(cc.ravel()); vals.extend(ke.ravel())
        fg[ids, 2] -= rho * G * v / 4.0
    K = coo_matrix((vals, (rows, cols)), shape=(3*len(points), 3*len(points))).tocsr()
    return K, fg.ravel()


def solve_contact(K, fext, points, faces, top_z=None, penalty=PENALTY):
    n = len(points)
    u = np.zeros((n, 3), dtype=float)
    bottom = int(np.argmin(points[:, 2]))
    # Original baseline constraints. This legacy solver is retained unchanged;
    # see the audit report for why its force output is not a trusted label.
    fixed = [3*bottom, 3*bottom+1, 3*bottom+2]
    fixed2 = int(np.argmin(np.linalg.norm(points - np.array([RADIUS, 0.0, RADIUS]), axis=1)))
    fixed.append(3*fixed2 + 1)
    top_nodes = np.unique(faces)
    floor_active = np.zeros(n, dtype=bool)
    top_active = np.zeros(n, dtype=bool)
    force_history = []
    previous = None
    for iteration in range(100):
        pos = points + u
        floor_active |= pos[:, 2] < 0.0
        if top_z is not None:
            top_active[top_nodes] |= pos[top_nodes, 2] > top_z
        active = floor_active | top_active
        diag = np.zeros(3*n)
        rhs = fext.copy()
        floor_ids = np.flatnonzero(floor_active)
        top_ids = np.flatnonzero(top_active)
        diag[3*floor_ids+2] += penalty
        rhs[3*floor_ids+2] += penalty * (0.0 - points[floor_ids, 2])
        if top_z is not None:
            diag[3*top_ids+2] += penalty
            rhs[3*top_ids+2] += penalty * (top_z - points[top_ids, 2])
        A = (K + diags(diag, format="csr")).tolil()
        for dof in fixed:
            A[:, dof] = 0.0
            A[dof, :] = 0.0
            A[dof, dof] = 1.0
            rhs[dof] = 0.0
        next_u = spsolve(A.tocsr(), rhs).reshape((-1, 3))
        if not np.isfinite(next_u).all():
            raise RuntimeError("FEM solve returned non-finite nodal positions")
        change = float(np.max(np.abs(next_u - u)))
        u = next_u
        pos = points + u
        top_force = float(penalty * np.maximum(pos[top_nodes, 2] - top_z, 0.0).sum()) if top_z is not None else 0.0
        force_history.append(top_force)
        violated_floor = pos[:, 2] < 0.0
        violated_top = np.zeros(n, dtype=bool)
        if top_z is not None:
            violated_top[top_nodes] = pos[top_nodes, 2] > top_z
        new_floor = violated_floor & ~floor_active
        new_top = violated_top & ~top_active
        if previous is not None and np.array_equal(active, previous) and not new_floor.any() and not new_top.any() and change < 1e-10:
            if len(force_history) < 2 or abs(force_history[-1] - force_history[-2]) < 1e-4:
                return pos, iteration + 1, force_history
        previous = active.copy()
        floor_active |= new_floor
        top_active |= new_top
    raise RuntimeError(f"FEM contact active set did not converge: "
                       f"max position change={change:g} m, reaction={force_history[-1]:g} N")

def contact_area(points, faces, plane_z, side="top", tol=2.0e-4):
    xyz = points[faces]
    cent_z = xyz[:, :, 2].mean(axis=1)
    if side == "top":
        active = cent_z >= plane_z - tol
    else:
        active = cent_z <= plane_z + tol
    xy = xyz[active, :, :2]
    if len(xy) == 0:
        return 0.0
    return float(np.sum(np.abs((xy[:, 1, 0]-xy[:, 0, 0])*(xy[:, 2, 1]-xy[:, 0, 1]) -
                                (xy[:, 1, 1]-xy[:, 0, 1])*(xy[:, 2, 0]-xy[:, 0, 0])) / 2.0))


def solve_cases(points, tet, faces, surface_count):
    initial_vol, initial_q, initial_signed = tet_quality(points, tet)
    density = MASS / float(np.sum(initial_vol))
    rows, snapshots, tests = [], {}, {}
    for E in MODULI:
        K, fg = assemble(points, tet, E, POISSON, density)
        settled, siters, shist = solve_contact(K, fg, points, faces)
        floor_depth = max(0.0, -float(np.min(settled[:, 2])))
        settled_h = float(np.ptp(settled[:, 2]))
        sv, sq, signed_settled = tet_quality(settled, tet)
        tests[f"settle_{int(E)}Pa"] = {
            "youngs_modulus_Pa": E, "minimum_node_z_m": float(np.min(settled[:, 2])),
            "ground_penetration_depth_m": floor_depth, "deformed_height_m": settled_h,
            "height_change_from_rest_m": float(np.ptp(points[:, 2]) - settled_h),
            "active_ground_contact_area_m2": contact_area(settled, faces, 0.0, "bottom"),
            "volume_change_fraction": float(np.sum(sv) / np.sum(initial_vol) - 1.0),
            "tet_count": len(tet), "node_count": len(points),
            "min_tet_quality_rest": float(initial_q.min()), "min_tet_quality_settled": float(sq.min()),
            "min_tet_volume_m3": float(sv.min()), "inverted_tets": int(np.sum(signed_settled <= 0)),
            "solver_iterations": siters,
            "solver_converged": True,
        }
        snapshots.setdefault("settled", (settled, None, 0.0))
        for displacement in (0.0005, 0.001, 0.002, 0.003):
            target_z = float(np.max(settled[:, 2]) - displacement)
            compressed, iters, history = solve_contact(K, fg, points, faces, target_z)
            vol, quality, signed = tet_quality(compressed, tet)
            reaction = history[-1]
            height = float(np.ptp(compressed[:, 2]))
            actual = max(0.0, settled_h - height)
            area = contact_area(compressed, faces, target_z, "top")
            rows.append({
                "youngs_modulus_Pa": E, "displacement_command_m": displacement,
                "tomato_height_displacement_m": actual, "reaction_force_N": reaction,
                "compression_rate": actual / settled_h, "top_contact_area_m2": area,
                "volume_change_fraction_vs_rest": float(np.sum(vol) / np.sum(initial_vol) - 1.0),
                "minimum_node_z_m": float(np.min(compressed[:, 2])),
                "ground_penetration_depth_m": max(0.0, -float(np.min(compressed[:, 2]))),
                "min_tet_quality": float(quality.min()), "mean_tet_quality": float(quality.mean()),
                "min_tet_volume_m3": float(vol.min()), "inverted_tets": int(np.sum(signed <= 0)),
                "solver_iterations": iters,
                "reaction_stable": len(history) > 1 and abs(history[-1]-history[-2]) < 1e-5,
            })
            if E == MODULI[2]:
                key = "compress_0_5mm" if displacement == 0.0005 else f"compress_{int(displacement*1000)}mm"
                snapshots[key] = (compressed, target_z, reaction)
    return rows, tests, snapshots, density, initial_q


def write_usd(path, points, tet, faces, surface_count, snapshots):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt
    stage = Usd.Stage.CreateNew(str(path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Z")
    stage.SetMetadata("framesPerSecond", 24.0)
    stage.SetMetadata("timeCodesPerSecond", 24.0)
    stage.GetRootLayer().customLayerData = {
        "solver": "external quasi-static linear tetrahedral FEM (SciPy); USD contains sampled FEM results, not live PhysX deformable simulation",
        "visualization": "time-sampled FEM node positions; geometry is solved externally, not a render-only deformation",
        "material": "homogeneous isotropic linear elastic; E=281000 Pa, nu=0.335; uncalibrated literature reference",
    }
    world = UsdGeom.Xform.Define(stage, "/World")
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(0.5, 0.5, 0.1))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.55, 0.62)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    tomato = UsdGeom.Xform.Define(stage, "/World/SmallTomato")
    skin = UsdGeom.Mesh.Define(stage, "/World/SmallTomato/ComputedFEMSurface")
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=RADIUS)
    skin.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
    skin.CreateFaceVertexIndicesAttr(sphere.faces.reshape(-1).tolist())
    skin.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    skin.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.055, 0.035)])
    mesh_pos = UsdGeom.PointBased(skin)
    tetra_edges = sorted({tuple(sorted((int(t[i]), int(t[j])))) for t in tet
                          for i in range(4) for j in range(i+1, 4)})
    edge_curves = UsdGeom.BasisCurves.Define(stage, "/World/SmallTomato/Debug/TetraEdges")
    edge_curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    edge_curves.CreateCurveVertexCountsAttr([2] * len(tetra_edges))
    edge_curves.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.25, 0.95)])
    edge_curves.CreateDisplayOpacityAttr([0.55])
    edge_points_attr = edge_curves.CreatePointsAttr()
    coll_curves = UsdGeom.BasisCurves.Define(stage, "/World/SmallTomato/Debug/CollisionSurfaceEdges")
    coll_edges = sorted({tuple(sorted((int(a), int(b)))) for f in faces
                         for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))})
    coll_curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    coll_curves.CreateCurveVertexCountsAttr([2] * len(coll_edges))
    coll_curves.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.65, 0.0)])
    coll_curves.CreateDisplayOpacityAttr([0.8])
    coll_points_attr = coll_curves.CreatePointsAttr()
    plate = UsdGeom.Cube.Define(stage, "/World/CompressionPlate")
    plate.CreateSizeAttr(1.0)
    plate.AddScaleOp().Set(Gf.Vec3f(0.045, 0.045, 0.001))
    plate.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.3, 0.36)])
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    plate_translate = plate.AddTranslateOp()
    # Frame 0 is the uncompressed rest state; later frames are converged FEM states.
    output_states = [(0, points, None), (24, snapshots["settled"][0], None)]
    output_states += [(48 + 24*i, snapshots[key][0], snapshots[key][1])
                      for i, key in enumerate(("compress_0_5mm", "compress_1mm", "compress_2mm", "compress_3mm"))]
    for frame, xyz, top_z in output_states:
        tc = Usd.TimeCode(frame)
        mesh_pos.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(xyz[:surface_count], dtype=np.float32)), tc)
        ep = np.asarray([[*xyz[a], *xyz[b]] for a, b in tetra_edges], dtype=np.float32).reshape(-1, 3)
        edge_points_attr.Set(Vt.Vec3fArray.FromNumpy(ep), tc)
        cp = np.asarray([[*xyz[a], *xyz[b]] for a, b in coll_edges], dtype=np.float32).reshape(-1, 3)
        coll_points_attr.Set(Vt.Vec3fArray.FromNumpy(cp), tc)
        if top_z is None:
            plate_translate.Set(Gf.Vec3d(0, 0, 0.08), tc)
        else:
            plate_translate.Set(Gf.Vec3d(0, 0, top_z + 0.001), tc)
    UsdGeom.Imageable(edge_curves).MakeInvisible()
    UsdGeom.Imageable(coll_curves).MakeInvisible()
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(120)
    stage.GetRootLayer().Save()


def write_plots(outdir, points, faces, snapshots, rows):
    os.environ.setdefault("MPLCONFIGDIR", str(outdir / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states = [("Rest", points), ("Settled", snapshots["settled"][0])]
    states += [(f"{mm:g} mm", snapshots[key][0]) for mm, key in
               ((0.5, "compress_0_5mm"), (1, "compress_1mm"), (2, "compress_2mm"), (3, "compress_3mm"))]
    fig, axes = plt.subplots(1, len(states), figsize=(15, 4), sharey=True)
    for ax, (title, xyz) in zip(axes, states):
        ax.triplot(xyz[:, 0] * 1000, xyz[:, 2] * 1000, faces, color="#aa3b2b", lw=0.25)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_aspect("equal")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("z (mm)")
    fig.suptitle("Computed tetrahedral FEM tomato cross-section")
    fig.tight_layout()
    fig.savefig(outdir / "fem_deformation_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for E in MODULI:
        group = [r for r in rows if r["youngs_modulus_Pa"] == E]
        ax.plot([r["tomato_height_displacement_m"] * 1000 for r in group],
                [r["reaction_force_N"] for r in group], marker="o", label=f"E={E/1e6:g} MPa")
    ax.set_xlabel("Measured tomato height reduction (mm)")
    ax.set_ylabel("Computed platen reaction (N)")
    ax.set_title("Quasi-static FEM force–displacement")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fem_force_displacement.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--show-meshes", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points, tet, faces, surface_count, _ = make_mesh()
    rows, tests, snapshots, density, q0 = solve_cases(points, tet, faces, surface_count)
    _, _, signed_rest = tet_quality(points, tet)
    with (args.output / "force_displacement.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    metrics = {
        "engine": "custom quasi-static tetrahedral FEM using SciPy sparse direct solve",
        "material_model": "homogeneous isotropic linear elastic tetrahedra with unilateral penalty contact",
        "units": {"length": "m", "mass": "kg", "force": "N", "youngs_modulus": "Pa"},
        "provenance": {"radius_m": "30 mm diameter project target; assumed sphere", "mass_kg": "derived from 470 kg/m3 literature density and spherical geometry",
                       "density_kg_m3": "470 kg/m3 literature-reference value; not calibrated for this fruit", "poissons_ratio": "0.335 primary-paper reference value; not calibrated", "youngs_modulus_Pa": "requested sensitivity sweep; none calibrated",
                       "contact_penalty_N_per_m": "numerical assumption; penetration reported"},
        "mesh": {"node_count": len(points), "tetrahedron_count": len(tet), "surface_triangle_count": len(faces),
                 "density_kg_m3": density, "min_quality_rest": float(q0.min()), "mean_quality_rest": float(q0.mean()),
                 "all_rest_tets_positive": bool(np.all(signed_rest > 0))},
        "settle_tests": tests,
        "compression_rows": len(rows),
        "physics_notes": ["Contact forces are computed from the FEM contact penalty reaction, not measured fruit data.",
                          "This quasi-static baseline does not model viscoelasticity, fracture, skin/pulp layering, or large-strain hyperelasticity."],
    }
    (args.output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_plots(args.output, points, faces, snapshots, rows)
    write_usd(args.output / "tomato_fem_playback.usda", points, tet, faces, surface_count, snapshots)
    if args.show_meshes:
        # The baked scene includes both meshes and their edge overlays. Enable them
        # in the layer by removing their invisibility opinions for inspection.
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(str(args.output / "tomato_fem_playback.usda"))
        for path in ("/World/SmallTomato/Debug/TetraEdges", "/World/SmallTomato/Debug/CollisionSurfaceEdges"):
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
        stage.GetRootLayer().Save()
    print(json.dumps({"output": str(args.output), "mesh_nodes": len(points), "tetrahedra": len(tet),
                      "settle": tests[f"settle_{int(MODULI[2])}Pa"], "force_points": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
