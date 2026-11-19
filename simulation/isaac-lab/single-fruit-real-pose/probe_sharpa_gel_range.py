#!/usr/bin/env python3
"""Check whether existing 4F local gel response spans the fixed-tomato loads."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "material-studies"
sys.path.insert(0, str(SOURCE))
from sharpa_local_contact_fem import ContactPair, sensor_geometry, volume_ratio

assets = Path(os.environ["SHARPA_TACMAP_ASSETS"])
curvature, xy, directions, keep, rmse_um = sensor_geometry(assets)
pair = ContactPair(12, 120_000.0, curvature)
rows = []
for mm in (0.0, 0.06, 0.12, 0.18, 0.30, 0.50, 0.80, 1.00):
    u, load, gap, reactions = pair.solve(mm/1000)
    depth, pressure = pair.sample_map(u, load, mm/1000, xy, directions, keep)
    rows.append({"backing_travel_mm": mm, "local_model_force_N": float(load.sum()),
                 "peak_gel_depth_um": float(depth.max()*1e6),
                 "min_gel_volume_ratio": volume_ratio(pair.gel_points, pair.gel_tets,
                                                       pair.gel_vol, u[pair.offset:]),
                 "min_sample_volume_ratio": volume_ratio(pair.lower_points, pair.lower_tets,
                                                          pair.lower_vol, u[:pair.offset]),
                 "max_complementarity_residual_Nm": float(np.max(np.abs(gap*load)))})
out = HERE / "sharpa_gel_force_range.json"
out.write_text(json.dumps({"method": "existing 4F local synthetic-sample/gel model",
                           "assumed_gel_E_Pa": 120_000.0,
                           "profile_fit_rmse_um": rmse_um,
                           "rows": rows}, indent=2) + "\n")
for row in rows:
    print(row, flush=True)
