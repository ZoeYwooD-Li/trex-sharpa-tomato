from __future__ import annotations
import csv, json, subprocess, sys
import os
from pathlib import Path
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT = Path(__file__).resolve().parent
DATASET = Path(os.environ["EPISODE54_DATASET"]).expanduser().resolve()
SOURCE_PARQUET = DATASET / 'data/chunk-000/episode_000054.parquet'
SIM_RUN = Path(sys.argv[1])
FRAMES = SIM_RUN / 'captured_dashboard_frames'
FPS = 15
W, H = 640, 400
TITLE, LABEL = 32, 28
ROW = H + LABEL
VIDEO = OUT / 'episode54_real_vs_isaac.mp4'

# Exact LeRobot camera videos belonging to the same source episode as the replay H5.
real_paths = [DATASET / f'videos/chunk-000/observation.images.cam{i}/episode_000054.mp4' for i in range(3)]
frame_paths = sorted(FRAMES.glob('sim_*.png'))
if len(frame_paths) != 476:
    raise RuntimeError(f'Expected 476 Isaac dashboard frames, found {len(frame_paths)}')
if VIDEO.exists():
    VIDEO.unlink()
caps = [cv2.VideoCapture(str(p)) for p in real_paths]
if not all(c.isOpened() for c in caps):
    raise RuntimeError('Could not open all Episode 54 camera videos')

out_w, out_h = W * 3, TITLE + ROW * 2
cmd = ['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','bgr24','-s',f'{out_w}x{out_h}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(VIDEO)]
with (OUT/'ffmpeg_encode.log').open('wb') as log:
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    try:
        for i, sim_path in enumerate(frame_paths):
            sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
            if sim is None or sim.shape[:2] != (1150,1280):
                raise RuntimeError(f'Unexpected simulator dashboard frame: {sim_path}')
            crops = [sim[58:538,20:740], sim[58:358,760:1240], sim[390:690,760:1240]]
            canvas = np.full((out_h,out_w,3),(20,23,29),np.uint8)
            time_s = i/FPS
            cv2.putText(canvas, f'Episode 54 | same-source REAL vs ISAAC | t={time_s:05.2f}s', (12,23), cv2.FONT_HERSHEY_SIMPLEX,.58,(235,239,245),1,cv2.LINE_AA)
            for c,cap in enumerate(caps):
                ok, real = cap.read()
                if not ok: raise RuntimeError(f'Real cam{c} ended before frame {i}')
                # Preserve camera aspect ratio; letterbox the 4:3 wrist streams to the common panel.
                rh,rw=real.shape[:2]
                scale=min(W/rw,H/rh)
                resized=cv2.resize(real,(round(rw*scale),round(rh*scale)),interpolation=cv2.INTER_AREA)
                real_panel=np.full((H,W,3),(12,14,18),np.uint8)
                y=(H-resized.shape[0])//2; x=(W-resized.shape[1])//2
                real_panel[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
                sim_panel=cv2.resize(crops[c],(W,H),interpolation=cv2.INTER_AREA)
                x0=c*W
                cv2.putText(canvas,f'REAL episode54 cam{c}',(x0+12,TITLE+20),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,235,245),1,cv2.LINE_AA)
                cv2.putText(canvas,f'ISAAC cam{c}',(x0+12,TITLE+ROW+20),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,235,245),1,cv2.LINE_AA)
                canvas[TITLE+LABEL:TITLE+ROW,x0:x0+W]=real_panel
                yy=TITLE+ROW+LABEL
                canvas[yy:yy+H,x0:x0+W]=sim_panel
                # Isaac preview is 15 Hz; sample the 30 Hz real stream at the matching even frame.
                if i+1 < len(frame_paths) and not cap.grab():
                    raise RuntimeError(f'Could not advance real cam{c} at frame {i}')
            for x in (W,W*2): cv2.line(canvas,(x,TITLE),(x,out_h-1),(95,105,120),2)
            cv2.line(canvas,(0,TITLE+ROW),(out_w-1,TITLE+ROW),(95,105,120),2)
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
        if enc.wait(): raise RuntimeError('ffmpeg encoding failed; see ffmpeg_encode.log')
    finally:
        for cap in caps: cap.release()
        if enc.poll() is None: enc.kill()

# Load the episode table and establish the state/timestamp contract.
pf=pq.ParquetFile(SOURCE_PARQUET)
tab=pf.read(columns=['observation.state','timestamp','frame_index','source_timestamp_ns','audit.sample_timestamp_ns','audit.receive_timestamp_ns','observation.tactile.left.thumb.wrench','observation.tactile.left.index.wrench','observation.tactile.left.middle.wrench','observation.tactile.left.ring.wrench','observation.tactile.left.pinky.wrench','observation.tactile.right.thumb.wrench','observation.tactile.right.index.wrench','observation.tactile.right.middle.wrench','observation.tactile.right.ring.wrench','observation.tactile.right.pinky.wrench'])
state=np.asarray(tab['observation.state'].combine_chunks().values.to_numpy(zero_copy_only=False)).reshape(-1,116)
time_s=np.asarray(tab['timestamp'].to_numpy(),dtype=np.float64)
source_ns=np.asarray(tab['source_timestamp_ns'].to_numpy(),dtype=np.int64)
frame_index=np.asarray(tab['frame_index'].to_numpy(),dtype=np.int64)
with h5py.File(OUT/'episode54_joint_replay.h5','r') as f:
    replay_pos=np.asarray(f['robot/joint_pos'])
    replay_t=np.asarray(f['robot/timestamps'])
    meta=json.loads(f['metadata'].attrs['extra'])
    raw_names=meta['joint_names']
# LeRobot observation.state positions: left arm 0:7, right arm 14:21, left hand 28:50, right hand 72:94.
state_pos=np.concatenate([state[:,0:7],state[:,14:21],state[:,28:50],state[:,72:94]],axis=1)
# H5 contract preserves arm positions first, then left and right hand positions, exactly as above.
if state_pos.shape != replay_pos.shape: raise RuntimeError(f'Joint shape mismatch {state_pos.shape} vs {replay_pos.shape}')
state_rmse=float(np.sqrt(np.mean((state_pos-replay_pos)**2)))
state_max=float(np.max(np.abs(state_pos-replay_pos)))
if len(state)!=951 or len(replay_pos)!=951: raise RuntimeError('Source and replay sample counts differ')

# Named 58-joint CSV with source-relative time; raw joint positions are radians.
jn=[n.split('/',1)[1] for n in raw_names]
with (OUT/'real_joint_state_episode54.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['time_s']+jn)
    for t,row in zip(time_s,state_pos): w.writerow([f'{t:.9f}']+[f'{v:.9f}' for v in row])

# Preserve the full 120 Hz simulated contact trace as a human-readable CSV.
sim=np.load(OUT/'sim_contact_force_trace.npz')['time_peak_finger_force_and_target_position']
sensor_names=[f'{side}_{finger}_force_n' for side in ('left','right') for finger in ('thumb','index','middle','ring','pinky')]
headers=['time_s']+sensor_names+['tomato_x_m','tomato_y_m','tomato_z_m']
with (OUT/'sim_contact_force_trace.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(headers);w.writerows(sim.tolist())

# Summarize real tactile maps and recorded Wrench channels. The source metadata does not specify force calibration.
fingers=('thumb','index','middle','ring','pinky')
real_wrenches={}
for side in ('left','right'):
  for finger in fingers:
    col=f'observation.tactile.{side}.{finger}.wrench'
    arr=np.asarray(tab[col].combine_chunks().values.to_numpy(zero_copy_only=False)).reshape(-1,6)
    real_wrenches[f'{side}_{finger}']=arr
with (OUT/'real_tactile_wrench_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['sensor','peak_norm_recorded_units','median_norm_recorded_units','frames_norm_gt_0_1','units_note'])
    for name,a in real_wrenches.items():
      norm=np.linalg.norm(a[:,:3],axis=1)
      w.writerow([name,float(norm.max()),float(np.median(norm)),int((norm>0.1).sum()),'recorded force xyz; calibration/units not inferred'])

# Stream timestamp audit: differences between per-topic sample/receive time and the original row time.
sm=np.asarray(tab['audit.sample_timestamp_ns'].combine_chunks().values.to_numpy(zero_copy_only=False)).reshape(-1,34)
rm=np.asarray(tab['audit.receive_timestamp_ns'].combine_chunks().values.to_numpy(zero_copy_only=False)).reshape(-1,34)
# audit field names/indices are stored in the feature metadata.
info=json.loads((DATASET/'meta/info.json').read_text())
keys=info['features']['audit.sample_timestamp_ns']['names']
with (OUT/'timestamp_alignment_summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['stream','sample_offset_median_ms','sample_offset_p95_abs_ms','receive_offset_median_ms','receive_offset_p95_abs_ms'])
    for j,k in enumerate(keys):
      s=(sm[:,j]-source_ns)*1e-6; r=(rm[:,j]-source_ns)*1e-6
      s=s[np.isfinite(s)&(sm[:,j]>0)]; r=r[np.isfinite(r)&(rm[:,j]>0)]
      if len(s) and len(r): w.writerow([k,float(np.median(s)),float(np.percentile(np.abs(s),95)),float(np.median(r)),float(np.percentile(np.abs(r),95))])

# Render selected same-source RGB/depth/tactile keyframe comparisons and force history.
cap=cv2.VideoCapture(str(VIDEO))
key_seconds=[0,5,10,15,20,25,31.6]
thumbs=[]
for s in key_seconds:
    cap.set(cv2.CAP_PROP_POS_MSEC,s*1000); ok,frame=cap.read()
    if not ok: continue
    frame=cv2.resize(frame,(960,444),interpolation=cv2.INTER_AREA)
    cv2.putText(frame,f't={s:.1f}s',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2,cv2.LINE_AA)
    thumbs.append(frame)
cap.release()
rows=[]
for i in range(0,len(thumbs),2):
    pair=thumbs[i:i+2]
    if len(pair)==1: pair.append(np.zeros_like(pair[0]))
    rows.append(np.hstack(pair))
cv2.imwrite(str(OUT/'real_vs_isaac_keyframes.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,92])

# A final-time real-vs-simulator tactile snapshot; distinct outputs/scales, not pixel registration.
real_last={}
for side in ('left','right'):
  for finger in fingers:
    col=f'observation.tactile.{side}.{finger}.deformation'
    one=pq.read_table(SOURCE_PARQUET,columns=[col]).slice(len(state)-1,1).column(0)[0].as_py()
    real_last[f'{side}_{finger}']=np.asarray(one,dtype=np.uint8)
sim_tac=np.load(OUT/'sim_tactile_snapshot.npz')['deformation']
fig,axes=plt.subplots(4,5,figsize=(15,12))
for j,side in enumerate(('left','right')):
  for i,finger in enumerate(fingers):
    ax=axes[j,i];ax.imshow(real_last[f'{side}_{finger}'],cmap='gray',vmin=0,vmax=255);ax.set_title(f'Real {side} {finger} | raw mono8');ax.axis('off')
for j,side in enumerate(('left','right')):
  for i,finger in enumerate(fingers):
    ax=axes[j+2,i];ax.imshow(sim_tac[j,i],cmap='viridis');ax.set_title(f'Isaac {side} {finger} | simulated map');ax.axis('off')
fig.suptitle('Episode 54 final tactile snapshot (different data paths/scales; not pixelwise comparable)',fontsize=15)
fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(OUT/'tactile_final_snapshot_comparison.png',dpi=140);plt.close(fig)

# Force trace plot, preserving its raw scale to expose anomalously high readings.
fig,ax=plt.subplots(figsize=(13,5))
for i,n in enumerate(sensor_names): ax.plot(sim[:,0],sim[:,i+1],lw=.8,label=n.replace('_force_n',''))
ax.set(xlabel='Replay time (s)',ylabel='Isaac/PhysX contact sensor force norm (N)',title='Replay force trace: large spikes indicate unresolved contact/scale artifacts')
ax.grid(True,alpha=.25);ax.legend(ncol=5,fontsize=8);fig.tight_layout();fig.savefig(OUT/'sim_contact_force_trace.png',dpi=160);plt.close(fig)

sim_peak=sim[:,1:11].max(axis=0)
contact_counts=(sim[:,1:11]>1e-3).sum(axis=0)
summary={
 'episode':54,'task':'Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.',
 'source_dataset':str(DATASET),'source_parquet':str(SOURCE_PARQUET),'source_episode_length':len(state),'source_time_s':[float(time_s[0]),float(time_s[-1])],
 'fps':30,'robot_type':'dual_fr3_sharpa','robot_position_joint_count':58,'joint_position_unit':'rad','joint_state_rmse_vs_replay_h5_rad':state_rmse,'joint_state_max_abs_error_rad':state_max,
 'source_timestamp_ns':[int(source_ns[0]),int(source_ns[-1])],'camera_video_paths':[str(x) for x in real_paths],
 'camera_streams':['cam0 640x400 RGB plus uint16 depth 400x640','cam1 640x480 RGB','cam2 640x480 RGB'],'camera_video_fps':30,
 'tactile_streams':'10 fingertips; each raw mono8 deformation map 240x240 and six component recorded Wrench; calibration/units not inferred',
 'replay_h5_samples':len(replay_pos),'isaac_physics_steps':3804,'isaac_physics_dt_s':1/120,'sim_capture_frames':len(frame_paths),'comparison_video_fps':FPS,'comparison_video_duration_s':len(frame_paths)/FPS,
 'sim_force_sensor_names':sensor_names,'sim_force_peak_n_by_sensor':{n:float(v) for n,v in zip(sensor_names,sim_peak)},'sim_force_samples_above_1e-3_by_sensor':{n:int(v) for n,v in zip(sensor_names,contact_counts)},
 'force_trace_note':'PhysX contact-sensor force norms; not FEM tomato reactions or calibrated Sharpa Wrench. Peaks exceed plausible soft-fruit grasp loads and require contact-path audit.',
 'soft_fruit_fem_coupled_in_this_replay':False,'flexible_branch_model_enabled':True,'official_sharpa_tacmap_coupled_in_this_replay':False,
 'sim_tactile_note':'Saved Isaac simulated deformation output has shape [2,5,240,240] at final sampled frame only; no full per-frame local Sharpa FEM/TacMap field in this replay.',
 'wrist_camera_configuration':str(Path(os.environ["EPISODE54_WRIST_JSON"])),
 'wrist_camera_selected_values':{'support_rod_length_each_mm':150,'inward_aim_each_deg':3,'left_support_circumferential_rotation_deg':-5,'left_camera_face_roll_deg':5,'right_output_image_roll_deg':180},
 'mjcf_configuration':str(Path(os.environ["EPISODE54_ALIGNMENT"]) / "camera_assembly_blue_home_palm_left/dual_fr3_sharpa_camera_visuals.xml"),
 'cam0_configuration':'unchanged; side_cam0_episode54_alignment.json',
 'prior_video_mismatch_corrected':'New comparison uses cam0/cam1/cam2 from the exact 121_430 dataset Episode 54. An older video used episode_000000 from another dataset and was excluded.',
}
(OUT/'episode54_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:summary[k] for k in ['source_episode_length','source_time_s','joint_state_rmse_vs_replay_h5_rad','joint_state_max_abs_error_rad','sim_force_peak_n_by_sensor','soft_fruit_fem_coupled_in_this_replay','official_sharpa_tacmap_coupled_in_this_replay']},indent=2))
