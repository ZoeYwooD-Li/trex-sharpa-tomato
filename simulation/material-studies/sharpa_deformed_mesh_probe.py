#!/usr/bin/env python3
"""Exercise Sharpa rays on a refreshed FEM surface and on a cached rest surface."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from isaac_ecoflex_coupon_view import revolve_boundary

ASSET_COMMIT = 'e4f8827407e15625368ef0b9cf5a72c44304e16b'
POINTS = 'tactileSensor_map_4F_point_origin.npy'
NORMALS = 'tactileSensor_map_4F_normal_origin.npy'
CLEARANCE_M = 25e-6


@wp.kernel
def two_pass_depth(mesh_id: wp.uint64, starts: wp.array(dtype=wp.vec3),
                   directions: wp.array(dtype=wp.vec3), valid: wp.array(dtype=int),
                   depths: wp.array(dtype=float)):
    i = wp.tid()
    depths[i] = 0.0
    if valid[i] == 0:
        return
    origin, direction = starts[i], directions[i]
    forward = wp.mesh_query_ray(mesh_id, origin, direction, 0.015)
    if forward.result:
        # Same advance/backtrack rule as the released TacMap CPD implementation.
        new_origin = origin + direction*(forward.t-0.0001)
        backward = wp.mesh_query_ray(mesh_id, new_origin, -direction, 0.5)
        if backward.result:
            depths[i] = forward.t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--coupon', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    points = np.load(args.assets/POINTS, allow_pickle=False).reshape(-1, 3)*.001
    normals = np.load(args.assets/NORMALS, allow_pickle=False).reshape(-1, 3)
    norm = np.linalg.norm(normals, axis=1)
    valid = norm > 1e-6
    if points.shape != (240*240, 3) or not np.any(valid):
        raise ValueError('invalid Sharpa 4F point/normal assets')
    inward = np.zeros_like(normals)
    inward[valid] = -normals[valid]/norm[valid, None]
    lowest = int(np.argmin(np.where(valid, points[:, 2], np.inf)))
    points -= points[lowest].copy()
    with np.load(args.coupon/'axisymmetric_nodes.npz') as fem:
        ref = fem['reference_rz_m']
        states = fem['points_rz_m']
        strain = fem['strain']
    ids, cosines, sines, triangles = revolve_boundary(ref, 48)

    def coordinates(state):
        r, z = state[ids].T
        return np.c_[r*cosines, r*sines, z].astype(np.float32)

    wp.init()
    device = args.device
    faces = wp.array(triangles.ravel().astype(np.int32), dtype=int, device=device)
    initial = coordinates(ref)
    moving_vertices = wp.array(initial, dtype=wp.vec3, device=device)
    stale_vertices = wp.array(initial, dtype=wp.vec3, device=device)
    moving = wp.Mesh(points=moving_vertices, indices=faces)
    stale = wp.Mesh(points=stale_vertices, indices=faces)
    starts = wp.zeros(len(points), dtype=wp.vec3, device=device)
    dirs = wp.array(inward, dtype=wp.vec3, device=device)
    ray_valid = wp.array(valid.astype(np.int32), dtype=int, device=device)
    depth = wp.zeros(len(points), dtype=float, device=device)
    current_maps, stale_maps, rows = [], [], []
    for i, state in enumerate(states):
        frame = coordinates(state)
        wp.copy(moving_vertices, wp.array(frame, dtype=wp.vec3, device=device))
        moving.refit()
        pose = np.array([0., 0., float(frame[:, 2].max())+CLEARANCE_M], dtype=np.float32)
        wp.copy(starts, wp.array(points+pose, dtype=wp.vec3, device=device))
        maps = []
        for mesh in (moving, stale):
            wp.launch(two_pass_depth, len(points), [mesh.id, starts, dirs, ray_valid, depth], device=device)
            maps.append(depth.numpy().reshape(240, 240))
        current, cached = maps
        current_maps.append(current)
        stale_maps.append(cached)
        rows.append(dict(fem_frame=i, coupon_strain=float(strain[i]),
                         ray_frame_z_m=float(pose[2]), live_max_depth_m=float(current.max()),
                         stale_max_depth_m=float(cached.max()), live_positive_rays=int(np.count_nonzero(current)),
                         stale_positive_rays=int(np.count_nonzero(cached))))
    np.savez_compressed(args.output/'maps.npz', fem_frame=np.arange(len(states)),
                        strain=strain, live_depth_m=np.stack(current_maps), stale_depth_m=np.stack(stale_maps))
    with (args.output/'frames.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    loaded = int(np.argmax(strain))
    stats = dict(asset_source='https://github.com/sharpa-robotics/sharpa-tacmap',
                 asset_commit=ASSET_COMMIT,
                 sha256={p: hashlib.sha256((args.assets/p).read_bytes()).hexdigest()
                         for p in (POINTS, NORMALS)},
                 device=device, frame_count=len(states), usable_rays=int(valid.sum()),
                 mount='diagnostic placement of the vendor 4F ray grid above the solved upper surface',
                 clearance_m=CLEARANCE_M, full_sensor_runtime=False, finger_force_available=False,
                 max_live_depth_m=max(row['live_max_depth_m'] for row in rows),
                 max_stale_depth_m=max(row['stale_max_depth_m'] for row in rows),
                 peak_frame=rows[loaded])
    stats['refresh_test_passed'] = stats['max_live_depth_m'] < 1e-6 and rows[loaded]['stale_max_depth_m'] > .001
    (args.output/'validation.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    vmax = max(stale_maps[loaded].max()*1000, .001)
    for ax, depth_map, title in zip(axes, (current_maps[loaded], stale_maps[loaded]),
                                   ('Refit to solved FEM surface', 'Cached undeformed surface')):
        im = ax.imshow(1000*depth_map, vmin=0, vmax=vmax, cmap='viridis')
        ax.set(title=title, xlabel='Sharpa pixel column', ylabel='Sharpa pixel row')
    fig.colorbar(im, ax=axes, label='Apparent geometric depth (mm)', shrink=.8)
    fig.savefig(args.output/'stale_vs_refit.png', dpi=160)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == '__main__':
    main()
