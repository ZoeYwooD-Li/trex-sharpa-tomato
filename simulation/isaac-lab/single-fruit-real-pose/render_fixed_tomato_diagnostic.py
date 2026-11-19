#!/usr/bin/env python3
"""Render the fixed-tomato FEM and one-way 4F tactile diagnostic together."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

HERE = Path(__file__).resolve().parent
fem = HERE / "fem_pad_fixed_center_refined"
tactile = HERE / "gel_tacmap_proxy_fixed_center"
states = json.loads((fem / "summary.json").read_text())
geometry = list(csv.DictReader((HERE / "staged_closure_geometry_refined.csv").open()))
with np.load(tactile / "frames.npz") as maps:
    images = maps["tacmap_quantized_proxy"].copy()
    depths = maps["gel_depth_m"].copy()

def load_shape(i):
    with np.load(fem / f"frame_{i:02d}.npz") as data:
        return data["rest_points_m"], data["deformed_points_m"], data["surface_faces"]

fig = plt.figure(figsize=(12.8, 4.8), facecolor="white")
grids = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.8, 1.05], wspace=0.24)
ax3d = fig.add_subplot(grids[0, 0], projection="3d")
axmap = fig.add_subplot(grids[0, 1])
axforce = fig.add_subplot(grids[0, 2])
frames = np.arange(len(states))
index_load = np.array([np.linalg.norm(s["contact_force_on_tomato_N"].get("index", [0, 0, 0]))
                       for s in states])
thumb_load = np.array([np.linalg.norm(s["contact_force_on_tomato_N"].get("thumb", [0, 0, 0]))
                       for s in states])
center = np.array([0.687, -0.265, 0.230])
output = HERE / "fixed_tomato_fem_tacmap_diagnostic.mp4"
with FFMpegWriter(fps=5, metadata={"title": "Fixed tomato FEM and 4F tactile proxy"},
                  bitrate=2500).saving(fig, str(output), dpi=120) as writer:
    for i, state in enumerate(states):
        rest, deformed, faces = load_shape(i)
        displacement_mm = np.linalg.norm(deformed-rest, axis=1)*1000
        face_color = plt.cm.viridis(np.clip(displacement_mm[faces].mean(axis=1)/1.2, 0, 1))
        ax3d.clear()
        surface = Poly3DCollection(deformed[faces], facecolors=face_color,
                                   edgecolors="none", linewidths=0, alpha=0.92)
        ax3d.add_collection3d(surface)
        for side, color in (("thumb", "#2474b5"), ("index", "#c62f39")):
            point = [float(geometry[i][f"{side}_nearest_{axis}_m"]) for axis in "xyz"]
            ax3d.scatter(*point, c=color, s=70, depthshade=False)
        ax3d.set(xlim=(center[0]-.019, center[0]+.019),
                 ylim=(center[1]-.019, center[1]+.019),
                 zlim=(center[2]-.019, center[2]+.019),
                 xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.view_init(elev=23, azim=-67)
        ax3d.set_title(f"Fixed fruit FEM · frame {i:02d}")
        axmap.clear()
        axmap.imshow(images[i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axmap.set_title("4F TacMap-format proxy")
        axmap.set_xticks([]); axmap.set_yticks([])
        axmap.text(0.5, -0.07, f"Gel peak {depths[i].max()*1e6:.0f} µm",
                   transform=axmap.transAxes, ha="center", fontsize=10)
        axforce.clear()
        axforce.plot(frames, thumb_load, color="#2474b5", label="Thumb FEM reaction")
        axforce.plot(frames, index_load, color="#c62f39", label="Index FEM reaction")
        axforce.axvline(i, color="black", alpha=.45, lw=1)
        axforce.set(xlim=(0, len(states)-1), ylim=(0, max(thumb_load.max(), index_load.max())*1.1),
                    xlabel="Kinematic step", ylabel="Reaction magnitude (N)")
        axforce.grid(alpha=.2); axforce.legend(loc="upper left", fontsize=8)
        fig.suptitle("Diagnostic only · fixed fruit · tangent-plane contact · tactile force outside validated range",
                     fontsize=11)
        writer.grab_frame()
plt.close(fig)
print(output)
