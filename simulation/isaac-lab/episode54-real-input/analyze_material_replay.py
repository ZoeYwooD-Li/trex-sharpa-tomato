from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

run=Path(sys.argv[1]).resolve()
out=run/'material_analysis'
out.mkdir(parents=True,exist_ok=False)
with np.load(run/'soft_fruit_trace.npz') as z:
    time_s=np.asarray(z['time_s'],float)
    names=[str(x) for x in z['body_names']]
    metrics=np.asarray(z['per_body_max_and_mean_node_displacement_and_anchor_slip_m'],float)
    node_world=np.asarray(z['nodal_positions_world_m'],float)
    shape_rms=np.asarray(z['rigid_aligned_shape_rms_m'],float)
    centroid=np.asarray(z['centroid_travel_m'],float)
    peak_contact=np.asarray(z['peak_fingertip_contact_force_n'],float)
with np.load(run/'replay_force_trace.npz') as z:
    raw=np.asarray(z['time_peak_finger_force_and_target_position'],float)
body=names.index('tomato_10') if 'tomato_10' in names else 0
m=metrics[:,body,:]
rms=shape_rms[:,body]
travel=centroid[:,body]
nframes=len(time_s)
sensors=['left_thumb','left_index','left_middle','left_ring','left_pinky','right_thumb','right_index','right_middle','right_ring','right_pinky']
# The Isaac replay saves contact forces at 120 Hz and the deformable node trace at 30 Hz.
force_at_frame=raw[np.minimum(np.arange(nframes)*4,len(raw)-1),1:11]
rows=[]
for i,t in enumerate(time_s):
    rows.append([t,m[i,0],m[i,1],m[i,2],rms[i],travel[i],peak_contact[i],*force_at_frame[i].tolist()])
headers=['time_s','max_node_displacement_m','mean_node_displacement_m','attachment_slip_m','rigid_aligned_shape_rms_m','centroid_travel_m','peak_fingertip_contact_sensor_force_n']+[f'{s}_contact_sensor_force_n' for s in sensors]
with (out/'fem_replay_trace_30hz.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(headers);w.writerows(rows)
# Full-rate 120 Hz force trace is retained separately from the 30 Hz FEM node trace.
with (out/'fingertip_contact_force_120hz.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['time_s']+[f'{s}_contact_sensor_force_n' for s in sensors]+['tomato_x_m','tomato_y_m','tomato_z_m']);w.writerows(raw.tolist())

fig,ax=plt.subplots(2,1,figsize=(12,8),sharex=True)
ax[0].plot(time_s,m[:,0]*1000,label='maximum nodal displacement')
ax[0].plot(time_s,m[:,1]*1000,label='mean nodal displacement')
ax[0].plot(time_s,rms*1000,label='rigid-aligned shape RMS')
ax[0].set_ylabel('deformation (mm)');ax[0].legend();ax[0].grid(alpha=.25)
ax[1].plot(time_s,peak_contact,lw=.8,color='#a33')
ax[1].set(xlabel='Episode time (s)',ylabel='contact sensor force (N)',title='PhysX contact-sensor force; not FEM reaction')
ax[1].grid(alpha=.25)
fig.suptitle('Episode 54 | PhysX volume FEM with cherry-tomato material config')
fig.tight_layout();fig.savefig(out/'force_deformation_timeline.png',dpi=160);plt.close(fig)

# Compare undeformed and peak-deformation surfaces in an X-Z projection, removing centroid travel.
xyz0=node_world[0,body]
cent0=xyz0.mean(axis=0)
peak_i=int(np.argmax(m[:,0]))
selected=[0,peak_i,nframes-1]
triangles=ConvexHull(xyz0).simplices
fig,axes=plt.subplots(1,3,figsize=(13,4.5))
for ax,i in zip(axes,selected):
    xyz=node_world[i,body]
    # Remove translation only; the diagnostic is a world-axis projection, not a full strain reconstruction.
    local=xyz-xyz.mean(axis=0)+cent0
    d=np.linalg.norm((xyz-xyz0)-(xyz.mean(axis=0)-cent0),axis=1)*1000
    for tri in triangles:
        pts=local[tri][:,[0,2]]
        ax.plot(np.r_[pts[:,0],pts[0,0]]*1000,np.r_[pts[:,1],pts[0,1]]*1000,color='#adb5bd',lw=.35)
    sc=ax.scatter(local[:,0]*1000,local[:,2]*1000,c=d,s=3,cmap='inferno',vmin=0,vmax=max(0.01,float(np.max(d))))
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_title(f't={time_s[i]:.2f}s | max Δ={m[i,0]*1000:.3f} mm')
    ax.set_xlabel('world X projection (mm)')
axes[0].set_ylabel('world Z projection (mm)')
fig.colorbar(sc,ax=axes.ravel().tolist(),label='node displacement after centroid removal (mm)',shrink=.8)
fig.suptitle('FEM mesh deformation snapshots; 30 mm target material model')
fig.savefig(out/'fem_deformation_snapshots.png',dpi=170,bbox_inches='tight');plt.close(fig)

force_peaks=raw[:,1:11].max(axis=0)
force_counts=(raw[:,1:11]>1e-3).sum(axis=0)
summary={
 'backend':'Isaac Sim 5.1 / PhysX native volume-deformable FEM',
 'material_config':'../tomato_material_config.yaml',
 'material':{'youngs_modulus_Pa':281000.0,'poissons_ratio':0.335,'density_kg_m3':470.0,'derived_total_mass_kg':0.006644},
 'mass_split':{'rigid_6mm_core_kg':0.000425,'deformable_body_kg':0.006219,'total_kg':0.006644},
 'scope':'kinematic replay; FEM participates in the replay; contact forces are PhysX fingertip sensor norms; material is a literature/geometric assumption, not calibrated',
 'body_name':names[body],'node_trace_hz':30,'node_trace_frames':nframes,'force_trace_hz':120,'force_trace_samples':len(raw),
 'peak_max_node_displacement_mm':float(m[:,0].max()*1000),
 'peak_mean_node_displacement_mm':float(m[:,1].max()*1000),
 'peak_rigid_aligned_shape_rms_mm':float(np.nanmax(rms)*1000),
 'peak_attachment_slip_mm':float(m[:,2].max()*1000),
 'peak_centroid_travel_mm':float(np.nanmax(travel)*1000),
 'peak_fingertip_contact_sensor_force_n':float(raw[:,1:11].max()),
 'peak_force_n_by_sensor':{name:float(value) for name,value in zip(sensors,force_peaks)},
 'force_samples_over_1e-3_by_sensor':{name:int(value) for name,value in zip(sensors,force_counts)},
 'notes':['No output force is labeled as calibrated Sharpa Wrench.', 'FEM shape is Isaac/PhysX nodal deformation, not the separate SciPy external solver output.', 'The cherry-tomato material file identifies E and nu as literature references and mass/density as derived/provisional.']
}
(out/'material_replay_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(summary,indent=2,ensure_ascii=False))
