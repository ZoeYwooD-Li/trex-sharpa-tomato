#!/usr/bin/env python3
"""Re-run the saved Episode 54 kinematic replay and build a same-source video."""
from __future__ import annotations

import json
import os
import sys
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
PROJECT = Path(os.environ["HANDSON_ROOT"]).expanduser().resolve()
DATASET = Path(os.environ["EPISODE54_DATASET"]).expanduser().resolve()
ALIGN = Path(os.environ["EPISODE54_ALIGNMENT"]).expanduser().resolve()
WRIST_CALIBRATION = Path(os.environ["EPISODE54_WRIST_JSON"]).expanduser().resolve()
calibration = json.loads(WRIST_CALIBRATION.read_text())
expected = {'support_rod_length_each_mm': 150, 'inward_aim_each_deg': 3,
            'left_support_circumferential_rotation_deg': -5, 'left_camera_face_roll_deg': 5,
            'right_output_image_roll_deg': 180}
if any(calibration.get('selected_configuration', {}).get(k) != v for k, v in expected.items()):
    raise RuntimeError('Bundled wrist calibration does not match the user-selected formal configuration')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = OUT / 'reruns' / stamp
run_dir.mkdir(parents=True, exist_ok=False)
frames_dir = run_dir / 'captured_dashboard_frames'
frames_dir.mkdir()
live = run_dir / 'replay_live_view' / 'latest.png'

with socket.socket() as s:
    s.bind(('127.0.0.1', 0))
    live_port = s.getsockname()[1]

command = [
    sys.executable, str(OUT.parent / 'scripts/smoke_isaac_tomato.py'),
    '--enable_cameras', '--replay-h5', str(Path(os.environ["EPISODE54_REPLAY_H5"]).expanduser().resolve()),
    '--steps', '3804', '--render-every', '4', '--replay-live-view',
    '--replay-live-view-port', str(live_port), '--replay-force-limit-n', '0',
    '--flexible-branch', '--home-config', 'configs/sharpa_powder_weighing_real_home.json',
    '--mjcf', str(ALIGN / 'camera_assembly_blue_home_palm_left/dual_fr3_sharpa_camera_visuals.xml'),
    '--wrist-camera-calibration-json', str(WRIST_CALIBRATION),
    '--head-camera-pose-json', str(ALIGN / 'side_cam0_episode54_alignment.json'),
    '--table-top-z-m', '0.161', '--plate-center-xy-m', '0.741', '-0.264',
    '--tomato-group-offset-xy-m', '-0.054', '-0.067', '--basket-first-center-xy-m', '0.63', '0.32',
    '--table-front-x-m', '0.95', '--output', str(run_dir),
]
log_path = run_dir / 'isaac_run.log'
with log_path.open('wb') as log:
    proc = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
    last_mtime = -1
    captured = 0
    while proc.poll() is None:
        try:
            stat = live.stat()
            if stat.st_mtime_ns != last_mtime:
                with Image.open(live) as image:
                    image.load()
                    image.save(frames_dir / f'sim_{captured:06d}.png')
                captured += 1
                last_mtime = stat.st_mtime_ns
        except (FileNotFoundError, OSError):
            pass
        time.sleep(0.001)
    code = proc.wait()
if code:
    raise SystemExit(f'Isaac replay exited with {code}; see {log_path}')

sim_paths = sorted(frames_dir.glob('sim_*.png'))
if len(sim_paths) != 476:
    raise RuntimeError(f'Expected 476 dashboard frames, captured {len(sim_paths)}')
real_paths = [DATASET / f'videos/chunk-000/observation.images.cam{i}/episode_000054.mp4' for i in range(3)]
caps = [cv2.VideoCapture(str(p)) for p in real_paths]
if not all(c.isOpened() for c in caps):
    raise RuntimeError('Could not open all same-source Episode 54 camera videos')

width, image_h, fps, title_h, label_h = 640, 400, 15, 32, 28
row_h, out_w = image_h + label_h, 3 * width
out_h = title_h + 2 * row_h
video = run_dir / 'episode54_real_vs_isaac.mp4'
ffmpeg = ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
          '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
          '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
with (run_dir / 'ffmpeg_encode.log').open('wb') as log:
    enc = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    try:
        for index, sim_path in enumerate(sim_paths):
            sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
            if sim is None or sim.shape[:2] != (1150, 1280):
                raise RuntimeError(f'Unexpected simulation frame: {sim_path}')
            crops = [sim[58:538, 20:740], sim[58:358, 760:1240], sim[390:690, 760:1240]]
            canvas = np.full((out_h, out_w, 3), (20, 23, 29), dtype=np.uint8)
            cv2.putText(canvas, f'Episode 54 | same-source REAL vs ISAAC | t={index/fps:05.2f}s',
                        (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (235,239,245), 1, cv2.LINE_AA)
            for cam, cap in enumerate(caps):
                ok, real = cap.read()
                if not ok:
                    raise RuntimeError(f'Real camera {cam} ended at frame {index}')
                rh, rw = real.shape[:2]
                scale = min(width/rw, image_h/rh)
                resized = cv2.resize(real, (round(rw*scale), round(rh*scale)), interpolation=cv2.INTER_AREA)
                real_panel = np.full((image_h, width, 3), (12,14,18), np.uint8)
                y, x = (image_h-resized.shape[0])//2, (width-resized.shape[1])//2
                real_panel[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
                sim_panel = cv2.resize(crops[cam], (width,image_h), interpolation=cv2.INTER_AREA)
                x0 = cam*width
                cv2.putText(canvas, f'REAL episode54 cam{cam}', (x0+12,title_h+20), cv2.FONT_HERSHEY_SIMPLEX,.52,(230,235,245),1,cv2.LINE_AA)
                cv2.putText(canvas, f'ISAAC cam{cam}', (x0+12,title_h+row_h+20), cv2.FONT_HERSHEY_SIMPLEX,.52,(230,235,245),1,cv2.LINE_AA)
                canvas[title_h+label_h:title_h+row_h,x0:x0+width] = real_panel
                yy = title_h+row_h+label_h
                canvas[yy:yy+image_h,x0:x0+width] = sim_panel
                if index+1 < len(sim_paths) and not cap.grab():
                    raise RuntimeError(f'Could not advance camera {cam}')
            for x in (width,2*width): cv2.line(canvas,(x,title_h),(x,out_h-1),(95,105,120),2)
            cv2.line(canvas,(0,title_h+row_h),(out_w-1,title_h+row_h),(95,105,120),2)
            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
        if enc.wait(): raise RuntimeError('ffmpeg encoding failed')
    finally:
        for cap in caps: cap.release()
        if enc.poll() is None: enc.kill()

manifest = {'command': command, 'isaac_project': str(PROJECT), 'source_dataset': str(DATASET),
            'source_episode': 54, 'simulator_exit_code': code, 'captured_frames': captured,
            'wrist_camera_calibration': str(WRIST_CALIBRATION), 'selected_configuration': expected,
            'video': str(video), 'simulator_log': str(log_path)}
(run_dir/'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(manifest, indent=2, ensure_ascii=False))
