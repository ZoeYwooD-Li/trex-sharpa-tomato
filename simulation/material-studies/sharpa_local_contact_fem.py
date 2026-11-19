#!/usr/bin/env python3
"""Localized quasistatic contact with a Sharpa-derived pad profile and an assumed gel."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import nnls
from scipy.sparse import block_diag, coo_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

from elastic_block_fem_reference import assemble
from isaac_elastic_block_probe import H, regular_tets

WIDTH = .008
HEIGHT = .006
PAD_WIDTH = WIDTH/2
GEL_THICKNESS = .003
SAMPLE_E = 50_000.
SAMPLE_NU = .30
GEL_NU = .45
TRAVEL_MM = np.array([-.06, 0., .06, .12, .18, .18, .12, .06, 0., -.06])


def sensor_geometry(path):
    raw = np.load(path/'tactileSensor_map_4F_point_origin.npy', allow_pickle=False)
    normals = np.load(path/'tactileSensor_map_4F_normal_origin.npy', allow_pickle=False)
    center = raw[119, 119]
    outward = normals[119, 119]/np.linalg.norm(normals[119, 119])
    rotation, _ = Rotation.align_vectors([[0., 0., -1.]], [outward])
    points = rotation.apply(raw.reshape(-1, 3)-center)
    directions = rotation.apply(normals.reshape(-1, 3))
    x, y, z = points.T
    valid = (x*x+y*y <= 4) & (directions[:, 2] < -.75)
    regressors = np.c_[x*x, x*y, y*y, x, y, np.ones(len(x))]
    fit, *_ = np.linalg.lstsq(regressors[valid], z[valid], rcond=None)
    curvature = np.array([[fit[0], fit[1]/2], [fit[1]/2, fit[2]]])
    if np.linalg.eigvalsh(curvature).min() <= 0:
        raise RuntimeError('vendor pad patch does not give a convex contact profile')
    origin = np.linalg.solve(curvature, -.5*fit[3:5])
    xy_m = (points[:, :2]-origin)*.001
    keep = valid & (np.abs(xy_m).max(axis=1) <= PAD_WIDTH/2)
    inward = np.zeros_like(directions)
    inward[keep] = -directions[keep]/np.linalg.norm(directions[keep], axis=1, keepdims=True)
    rmse_um = 1000*float(np.sqrt(np.mean((regressors[valid]@fit-z[valid])**2)))
    return curvature*1000, xy_m, inward, keep, rmse_um


class ContactPair:
    def __init__(self, n, gel_e, curvature):
        m, nz = n//2, n//2
        self.m = m
        lower, lower_tets = regular_tets((n, n, nz))
        lower = lower * (WIDTH/H, WIDTH/H, HEIGHT/H)
        gel, gel_tets = regular_tets((m, m, nz))
        gel[:, :2] *= PAD_WIDTH/H
        profile = np.einsum('ni,ij,nj->n', gel[:, :2], curvature, gel[:, :2])
        gel[:, 2] = HEIGHT + profile + (gel[:, 2]/H)*(GEL_THICKNESS-profile)
        self.lower_points, self.gel_points = lower, gel
        self.lower_tets, self.gel_tets = lower_tets, gel_tets
        kl, _, _, _, self.lower_vol = assemble(lower, lower_tets, SAMPLE_E, SAMPLE_NU)
        kg, _, _, _, self.gel_vol = assemble(gel, gel_tets, gel_e, GEL_NU)
        self.k = block_diag((kl, kg), format='csr')
        self.offset = len(lower)
        lower_grid = np.arange(len(lower)).reshape(n+1, n+1, nz+1)
        gel_grid = np.arange(len(gel)).reshape(m+1, m+1, nz+1)
        self.sample_top = lower_grid[n//4:3*n//4+1, n//4:3*n//4+1, -1].ravel()
        self.gel_bottom = gel_grid[:, :, 0].ravel()+self.offset
        self.initial_gap = gel[self.gel_bottom-self.offset, 2]-HEIGHT
        self.base_z = 3*lower_grid[:, :, 0].ravel()+2
        self.backing = (3*(gel_grid[:, :, -1].ravel()+self.offset)[:, None]+np.arange(3)).ravel()
        self.backing_z = self.backing[2::3]
        gauge = [0, 1, 3*lower_grid[-1, 0, 0]+1]
        self.fixed = np.r_[self.base_z, self.backing, gauge]
        self.free = np.setdiff1d(np.arange(self.k.shape[0]), self.fixed)
        count = len(self.sample_top)
        if not np.allclose(lower[self.sample_top, :2], gel[self.gel_bottom-self.offset, :2]):
            raise RuntimeError('contact surface node pairing failed')
        self.c = coo_matrix((np.tile([-1., 1.], count),
                             (np.repeat(np.arange(count), 2),
                              np.c_[3*self.sample_top+2, 3*self.gel_bottom+2].ravel())),
                            shape=(count, self.k.shape[0])).tocsr()
        kff = splu(self.k[self.free][:, self.free].tocsc())
        self.response = kff.solve(self.c[:, self.free].T.toarray())
        compliance = self.c[:, self.free] @ self.response
        compliance = .5*(compliance+compliance.T)
        self.cholesky = cholesky(compliance, lower=True)
        self.free_travel = np.zeros(self.k.shape[0])
        self.free_travel[self.backing_z] = -1.
        self.free_travel[self.free] = kff.solve(-self.k[self.free][:, self.fixed]@self.free_travel[self.fixed])
        self.pixel_axis = np.linspace(-PAD_WIDTH/2, PAD_WIDTH/2, m+1)

    def solve(self, travel):
        u = travel*self.free_travel
        separation = self.initial_gap + self.c@u
        load, _ = nnls(self.cholesky.T, -solve_triangular(self.cholesky, separation, lower=True))
        u[self.free] += self.response@load
        gap = self.initial_gap + self.c@u
        support = self.k@u - self.c.T@load
        return u.reshape(-1, 3), load, gap, support

    def sample_map(self, u, load, travel, xy, directions, keep):
        sensor = u[self.gel_bottom].copy()
        sensor[:, 2] += travel
        sides = (self.pixel_axis, self.pixel_axis)
        field = RegularGridInterpolator(sides, sensor.reshape(self.m+1, self.m+1, 3),
                                        bounds_error=False, fill_value=0)
        nodal_map = np.zeros(len(xy))
        nodal_map[keep] = np.einsum('ij,ij->i', field(xy[keep]), directions[keep])
        density = load/(PAD_WIDTH/self.m)**2
        pressure = RegularGridInterpolator(sides, density.reshape(self.m+1, self.m+1),
                                           bounds_error=False, fill_value=0)
        pressure_map = np.zeros(len(xy))
        pressure_map[keep] = pressure(xy[keep])
        return np.maximum(nodal_map, 0).reshape(240, 240), pressure_map.reshape(240, 240)


def volume_ratio(points, tets, initial, displacement):
    deformed = (points+displacement)[tets]
    volume = np.linalg.det(deformed[:, 1:]-deformed[:, :1])/6
    return float(np.min(volume/initial))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cells', type=int, nargs='+', default=[8, 12, 16])
    parser.add_argument('--gel-E-pa', type=float, default=120_000.)
    args = parser.parse_args()
    if any(n < 4 or n % 4 for n in args.cells) or args.gel_E_pa <= 0:
        parser.error('mesh counts must be multiples of four and gel modulus must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    curvature, xy, directions, keep, profile_rmse = sensor_geometry(args.assets)
    rows, summaries = [], []
    peak = int(np.argmax(TRAVEL_MM))
    for n in args.cells:
        pair = ContactPair(n, args.gel_E_pa, curvature)
        maps, pressures, displacements, mesh_rows = [], [], [], []
        for frame, mm in enumerate(TRAVEL_MM):
            u, load, gap, reactions = pair.solve(mm/1000)
            depth, pressure = pair.sample_map(u, load, mm/1000, xy, directions, keep)
            maps.append(depth)
            pressures.append(pressure)
            displacements.append(u)
            total = float(load.sum())
            top = float(-reactions[pair.backing_z].sum())
            bottom = float(reactions[pair.base_z].sum())
            row = dict(cells=n, frame=frame, platen_travel_mm=mm, contact_force_N=total,
                       backing_reaction_N=top, ground_reaction_N=bottom,
                       max_force_balance_error_N=max(abs(top-total), abs(bottom-total)),
                       maximum_penetration_m=max(0., float(-gap.min())),
                       complementarity_residual_Nm=float(np.max(np.abs(gap*load))),
                       free_residual_N=float(np.max(np.abs(reactions[pair.free]))),
                       load_bearing_nodes=int(np.count_nonzero(load > 1e-8)),
                       max_gel_normal_displacement_m=float(depth.max()),
                       min_sample_volume_ratio=volume_ratio(pair.lower_points, pair.lower_tets,
                                                            pair.lower_vol, u[:pair.offset]),
                       min_gel_volume_ratio=volume_ratio(pair.gel_points, pair.gel_tets,
                                                         pair.gel_vol, u[pair.offset:]))
            mesh_rows.append(row)
            rows.append(row)
        summaries.append(mesh_rows[peak])
        if n == max(args.cells):
            ray_z = HEIGHT+np.einsum('ni,ij,nj->n', xy, curvature, xy)
            np.savez_compressed(args.output/'synchronized_fields.npz', frame=np.arange(len(TRAVEL_MM)),
                                platen_travel_mm=TRAVEL_MM, depth_m=np.stack(maps), pressure_Pa=np.stack(pressures),
                                points_m=np.vstack([pair.lower_points, pair.gel_points]),
                                displacement_m=np.stack(displacements),
                                sample_tets=pair.lower_tets, gel_tets=pair.gel_tets+pair.offset,
                                ray_origin_reference_m=np.c_[xy, ray_z], ray_inward_direction=directions,
                                ray_valid=keep,
                                force_N=np.array([r['contact_force_N'] for r in mesh_rows]))
        print('SHARPA_CONTACT', n, mesh_rows[peak], flush=True)
    with (args.output/'trace.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    peaks = [r['contact_force_N'] for r in sorted(summaries, key=lambda r: r['cells'])]
    fine_pair = sorted(args.cells)[-2:]
    local_convergence = {}
    for mm in (.06, .12, .18):
        reactions_at_mm = [next(r['contact_force_N'] for r in rows
                               if r['cells']==n and r['platen_travel_mm']==mm) for n in fine_pair]
        if len(reactions_at_mm) > 1:
            local_convergence[str(mm)] = abs(reactions_at_mm[-1]-reactions_at_mm[-2])/reactions_at_mm[-1]
    data = dict(geometry_source='https://github.com/sharpa-robotics/sharpa-tacmap',
                geometry_scope='local quadratic approximation to the vendor 4F index-pad central patch',
                profile_fit_rmse_um=profile_rmse, principal_curvatures_per_m=(2*np.linalg.eigvalsh(curvature)).tolist(),
                sample_E_Pa=SAMPLE_E, sample_nu=SAMPLE_NU, assumed_gel_E_Pa=args.gel_E_pa,
                assumed_gel_nu=GEL_NU, assumed_gel_thickness_m=GEL_THICKNESS,
                scope='linear elastic both bodies; vertical frictionless contact on matching surface nodes',
                peak_force_by_mesh_N=peaks,
                fine_mesh_force_change_by_displacement_mm=local_convergence,
                finest_relative_force_change=abs(peaks[-1]-peaks[-2])/peaks[-1] if len(peaks)>1 else None,
                max_force_balance_error_N=max(r['max_force_balance_error_N'] for r in rows),
                maximum_penetration_m=max(r['maximum_penetration_m'] for r in rows),
                max_free_residual_N=max(r['free_residual_N'] for r in rows),
                max_complementarity_residual_Nm=max(r['complementarity_residual_Nm'] for r in rows),
                min_volume_ratio=min(min(r['min_sample_volume_ratio'], r['min_gel_volume_ratio']) for r in rows),
                max_open_gap_force_N=max(r['contact_force_N'] for r in rows if r['platen_travel_mm'] < 0))
    data['numerical_balance_passed'] = (data['max_force_balance_error_N'] < 1e-7
                                       and data['maximum_penetration_m'] < 1e-8
                                       and data['min_volume_ratio'] > 0
                                       and data['max_open_gap_force_N'] < 1e-8)
    (args.output/'validation.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with np.load(args.output/'synchronized_fields.npz') as arrays:
        depth, pressure = arrays['depth_m'][peak], arrays['pressure_Pa'][peak]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, field, title, label in ((axes[0], depth*1e6, 'Gel normal displacement', 'µm'),
                                    (axes[1], pressure*1e-3, 'Interpolated nodal pressure', 'kPa')):
        im = ax.imshow(field, cmap='viridis')
        ax.set(title=title, xlabel='Sharpa column', ylabel='Sharpa row')
        fig.colorbar(im, ax=ax, label=label)
    for n in sorted(args.cells):
        subset = [r for r in rows if r['cells']==n]
        axes[2].plot([r['platen_travel_mm'] for r in subset], [r['contact_force_N'] for r in subset],
                     'o-', label=f'{n} cells across block')
    axes[2].set(title='FEM contact force', xlabel='Backing travel (mm)', ylabel='Normal force (N)')
    axes[2].grid(alpha=.25)
    axes[2].legend(fontsize=7)
    fig.savefig(args.output/'local_contact.png', dpi=165)
    print(json.dumps(data, indent=2), flush=True)


if __name__ == '__main__':
    main()
