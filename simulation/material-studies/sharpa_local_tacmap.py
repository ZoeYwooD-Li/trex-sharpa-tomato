#!/usr/bin/env python3
"""Compare geometric TacMap depth with FEM gel displacement at matched force states."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import warp as wp

from elastic_block_fem_reference import surface_triangles
from sharpa_deformed_mesh_probe import two_pass_depth

DEPTH_THRESHOLD_M = 2e-6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fem', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with np.load(args.fem/'synchronized_fields.npz') as data:
        origin_ref = data['ray_origin_reference_m']
        direction = data['ray_inward_direction']
        valid = data['ray_valid']
        points = data['points_m']
        tets = data['sample_tets']
        displacements = data['displacement_m']
        travel_mm = data['platen_travel_mm']
        force = data['force_N']
        fem_maps = data['depth_m']
        pressure = data['pressure_Pa']
    device = args.device
    wp.init()
    count = int(tets.max())+1
    vertices = wp.array(points[:count].astype(np.float32), dtype=wp.vec3, device=device)
    triangles = surface_triangles(tets)
    indices = wp.array(triangles.astype(np.int32).ravel(), dtype=int, device=device)
    mesh = wp.Mesh(points=vertices, indices=indices)
    ray_start = wp.zeros(len(valid), dtype=wp.vec3, device=device)
    ray_direction = wp.array(direction.astype(np.float32), dtype=wp.vec3, device=device)
    ray_valid = wp.array(valid.astype(np.int32), dtype=int, device=device)
    ray_depth = wp.zeros(len(valid), dtype=float, device=device)
    maps, rows = [], []
    for i, (disp, delta_mm, load, fem, pressure_map) in enumerate(zip(displacements, travel_mm, force, fem_maps, pressure)):
        frame = (points[:count]+disp[:count]).astype(np.float32)
        wp.copy(vertices, wp.array(frame, dtype=wp.vec3, device=device))
        mesh.refit()
        origin = origin_ref - (0, 0, delta_mm/1000)
        wp.copy(ray_start, wp.array(origin.astype(np.float32), dtype=wp.vec3, device=device))
        wp.launch(two_pass_depth, len(valid), [mesh.id, ray_start, ray_direction, ray_valid, ray_depth], device=device)
        tacmap = ray_depth.numpy().reshape(240, 240)
        maps.append(tacmap)
        simulated_contact = pressure_map > 0.01*pressure_map.max()
        visible = tacmap > DEPTH_THRESHOLD_M
        pressure_union = np.count_nonzero(visible | simulated_contact)
        gel_depth = fem > DEPTH_THRESHOLD_M
        depth_union = visible | gel_depth
        rows.append(dict(frame=i, backing_travel_mm=float(delta_mm), fem_force_N=float(load),
                         max_fem_gel_depth_m=float(fem.max()), max_tacmap_depth_m=float(tacmap.max()),
                         pressure_iou=float(np.count_nonzero(visible & simulated_contact)/pressure_union)
                         if pressure_union else None,
                         depth_iou=float(np.count_nonzero(visible & gel_depth)/np.count_nonzero(depth_union))
                         if depth_union.any() else None,
                         depth_mae_in_union_m=float(np.mean(np.abs(tacmap[depth_union]-fem[depth_union])))
                         if depth_union.any() else None))
    with (args.output/'frames.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output/'tacmap_frames.npz', fem_frame=np.arange(len(force)),
                        force_N=force, depth_m=np.stack(maps))
    peak = int(np.argmax(travel_mm))
    opened = [r for r in rows if r['backing_travel_mm'] < 0]
    validation = dict(method='vendor two-pass geometric projection on per-frame refitted FEM mesh',
                      full_sensor_runtime=False, depth_threshold_m=DEPTH_THRESHOLD_M,
                      frame_count=len(rows), peak=rows[peak],
                      max_open_gap_tacmap_depth_m=max(r['max_tacmap_depth_m'] for r in opened),
                      max_open_gap_force_N=max(r['fem_force_N'] for r in opened),
                      synchronized=all(r['frame'] == i and np.isfinite(r['fem_force_N']) for i, r in enumerate(rows)))
    (args.output/'validation.json').write_text(json.dumps(validation, indent=2, allow_nan=False)+'\n')
    print(json.dumps(validation, indent=2), flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    y, x = np.nonzero(valid.reshape(240, 240))
    roi = np.s_[max(0, y.min()-6):y.max()+7, max(0, x.min()-6):x.max()+7]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4), constrained_layout=True)
    vmax = max(fem_maps[peak].max(), maps[peak].max())*1e6
    for ax, field, title in ((axes[0], fem_maps[peak], 'FEM gel displacement'),
                             (axes[1], maps[peak], 'TacMap-style geometric depth')):
        im = ax.imshow(1e6*field[roi], cmap='viridis', vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes[:2], label='Normal depth (µm)', shrink=.72)
    im2 = axes[2].imshow(pressure[peak][roi]/1000, cmap='magma')
    axes[2].set_title('FEM contact pressure estimate')
    axes[2].set_axis_off()
    fig.colorbar(im2, ax=axes[2], label='kPa', shrink=.72)
    fig.suptitle(f'Frame {peak} · {travel_mm[peak]:.2f} mm travel · {force[peak]:.4f} N FEM reaction')
    fig.savefig(args.output/'synchronized_maps.png', dpi=165)


if __name__ == '__main__':
    main()
