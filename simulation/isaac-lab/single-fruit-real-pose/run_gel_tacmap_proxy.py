#!/usr/bin/env python3
"""One-way 4F gel response to fixed-tomato FEM index loads.

This applies each FEM index reaction magnitude to the existing Sharpa local gel
model over an assumed central Gaussian patch, then uses the official TacMap
depth quantizer. It is not the full vendor ray-cast sensor or a two-way solve.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "material-studies"
sys.path.insert(0, str(SOURCE))
from sharpa_local_contact_fem import ContactPair, sensor_geometry, volume_ratio

assets = Path(os.environ["SHARPA_TACMAP_ASSETS"])
curvature, xy, directions, keep, profile_rmse_um = sensor_geometry(assets)
pair = ContactPair(12, 120_000.0, curvature)
summaries = json.loads((HERE / "fem_pad_fixed_center_refined/summary.json").read_text())
out = HERE / "gel_tacmap_proxy_fixed_center"
out.mkdir(exist_ok=True)

offset_dof = 3 * pair.offset
gel_k = pair.k[offset_dof:, offset_dof:].tocsc()
backing = pair.backing - offset_dof
free = np.setdiff1d(np.arange(gel_k.shape[0]), backing)
factor = splu(gel_k[free][:, free])
bottom = pair.gel_bottom - pair.offset
xy_bottom = pair.gel_points[bottom, :2]
sigma_m = 0.001
weights = np.exp(-0.5 * np.sum(xy_bottom**2, axis=1) / sigma_m**2)
weights /= weights.sum()

depths, pressures, quantized, rows = [], [], [], []
for summary in summaries:
    force_vector = summary["contact_force_on_tomato_N"].get("index", [0, 0, 0])
    force_N = float(np.linalg.norm(force_vector))
    f = np.zeros(gel_k.shape[0])
    f[3*bottom+2] = force_N*weights
    u = np.zeros_like(f)
    u[free] = factor.solve(f[free])
    gel_displacement = u.reshape(-1, 3)
    full_displacement = np.zeros((pair.offset + len(pair.gel_points), 3))
    full_displacement[pair.offset:] = gel_displacement
    depth, pressure = pair.sample_map(full_displacement, force_N*weights, 0.0,
                                      xy, directions, keep)
    # Vendor deform_quantize, in millimetres; no blur or official ray casting.
    mm = depth*1000
    image = np.where(mm < 0.5, mm/0.005, (mm-0.5)/0.03+100)
    image = np.clip(image, 0, 255).astype(np.uint8)
    min_volume_ratio = volume_ratio(pair.gel_points, pair.gel_tets,
                                    pair.gel_vol, gel_displacement)
    row = {"frame": summary["frame"], "tomato_FEM_index_reaction_N": force_N,
           "gel_peak_depth_um": float(depth.max()*1e6),
           "gel_min_tetra_volume_ratio": min_volume_ratio,
           "tacmap_proxy_peak_uint8": int(image.max()),
           "tacmap_proxy_nonzero_pixels": int(np.count_nonzero(image)),
           "within_previously_validated_0p18mm_force_range": force_N <= 0.01414}
    rows.append(row)
    depths.append(depth.astype(np.float32))
    pressures.append(pressure.astype(np.float32))
    quantized.append(image)

np.savez_compressed(out / "frames.npz", frame=np.arange(len(rows)),
                    gel_depth_m=np.stack(depths), gel_pressure_Pa=np.stack(pressures),
                    tacmap_quantized_proxy=np.stack(quantized))
(out / "summary.json").write_text(json.dumps({
    "method": "one-way force transfer to existing 4F local gel FEM, central Gaussian load; official TacMap quantization",
    "full_vendor_tacmap_runtime": False,
    "two_way_tomato_gel_coupling": False,
    "gel_E_Pa_assumed": 120_000.0,
    "force_patch_sigma_m_assumed": sigma_m,
    "sensor_geometry_profile_fit_rmse_um": profile_rmse_um,
    "rows": rows}, indent=2) + "\n")
for row in rows[-6:]:
    print(row, flush=True)
