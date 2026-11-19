"""Bounded stem-pinch or stem/fruit-wrap diagnostic; never writes demonstrations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from demos.cherry_tomato_pick.collection import prepare_collection
from demos.cherry_tomato_pick.scene import build, _randomize_recording_config
from demos.cherry_tomato_pick.stem_pinch import validate_config, stem_tilt_degrees, plate_clearance
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/cherry_tomato_pick_stem_pinch.yaml'))
    parser.add_argument('--seed', type=int, help='Use legal collection randomization; otherwise use nominal pose')
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--through-harvest-reach', action='store_true',
                        help='Continue through the left-hand approach; does not certify picking/placing')
    args = parser.parse_args()
    if not np.isfinite(args.duration) or not 0 < args.duration <= 60:
        parser.error('duration must be finite and in (0, 60] simulated seconds')
    if args.report.exists():
        parser.error('report already exists; choose a new path')
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    wrap = bool(cfg['task'].get('bunch_wrap_control'))
    if not cfg['task'].get('stem_pinch_control') and not wrap:
        parser.error('requires stem_pinch_control or bunch_wrap_control')
    if wrap and args.through_harvest_reach and cfg['script'].get('support_lift_only', True):
        parser.error('disable support_lift_only explicitly before requesting harvest reach')
    if args.seed is not None:
        cfg, _, _, _ = prepare_collection(cfg, args.seed, _randomize_recording_config)
    cfg['recording']['enabled'] = False
    cfg['camera']['sensor']['enabled'] = False
    cfg['wrist_cameras']['sensor']['enabled'] = False
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                  config_path=str(args.config.resolve()), effective_config=cfg,
                  passed=False, error='', events=[], samples=[], lift_passed=False,
                  through_harvest_reach=args.through_harvest_reach)
    root = Path(__file__).resolve().parents[3]
    result['source_sha256'] = {
        name: hashlib.sha256((root / 'demos/cherry_tomato_pick' / name).read_bytes()).hexdigest()
        for name in ('scene.py', 'backend.py', 'stem_pinch.py', 'wuji2.py', 'robot.py', 'kinematics.py',
                     'tableware.py', 'bunch_wrap.py', 'tools/check_stem_pinch.py')
    }
    setup = None
    try:
        setup = build(cfg)
        scene = setup.runtime_backend.scenes[0]
        initial_left_tcp = None
        maximum_left_displacement = 0.
        left_parts = scene.robot.left_tableware_meshes
        minimum_left_clearance = float('inf')
        previous = None
        for i in range(round(args.duration / setup.dt)):
            setup.step(i * setup.dt, {})
            state = setup._camera_state
            phase = state['phase']
            if initial_left_tcp is None:
                initial_left_tcp = scene.robot.tcp_poses()[0].copy()
            maximum_left_displacement = max(maximum_left_displacement, float(np.linalg.norm(
                scene.robot.tcp_poses()[0][:3] - initial_left_tcp[:3])))
            left_clearance = min(plate_clearance(
                Rotation.from_quat(scene.robot.visual_pose(p)[1]).apply(vertices)
                + scene.robot.visual_pose(p)[0], scene.tableware) for p, vertices in left_parts)
            minimum_left_clearance = min(minimum_left_clearance, left_clearance)
            changed = phase != previous
            if changed:
                result['events'].append(dict(time_s=i*setup.dt, phase=phase))
                previous = phase
            if changed or i % 20 == 0 or setup.should_stop():
                row = dict(time_s=i*setup.dt, phase=phase,
                           script_time_s=float(state['script_time']),
                           stem_forces_N=scene.right_stem_contact_force_N[scene.support_force_indices].tolist(),
                           support_forces_N=scene.support_contact_forces().tolist(),
                           closure=scene.robot.right_force_closure.tolist(),
                           opposed=scene.wrap_contacts_ready(retention=True) if wrap else scene.stem_pinch_opposed(),
                           ready=bool(scene.support_engaged),
                           support_fruit_forces_N=([float(np.linalg.norm(s.force_matrix.numpy().reshape(-1, 3), axis=1).sum())
                                                   for s in scene.right_support_fruit_sensors] if wrap else []),
                           full_lift_hold_s=state['support_full_lift_hold_s'],
                           maximum_left_tcp_displacement_m=maximum_left_displacement,
                           stem_position_m=scene.stem_grasp_position.tolist(),
                           branch_pose=np.r_[scene.solver.body_pose(scene.branch_index)[0],
                                             scene.solver.body_pose(scene.branch_index)[1]].tolist(),
                           right_tcp=scene.robot.tcp_poses()[1].tolist(),
                           all_contact_forces_N=[float(np.linalg.norm(
                               s.total_force.numpy().reshape(-1, 3), axis=1).sum())
                               for s in scene.right_bunch_sensors],
                           peak_contact_force_N=float(scene.peak_contact_force_N),
                           trial_passed=state['stem_pinch_trial_passed'],
                           measured_lift_m=(0. if state['stem_lift_start'] is None else
                                            float(scene.stem_grasp_position[2] - state['stem_lift_start'][2])),
                           cluster_clearance_m=float(scene.cluster_clearance),
                           stem_tilt_deg=stem_tilt_degrees(scene.solver.body_pose(scene.branch_index)[1],
                                                          cfg['scene']['branch_collision_axis']),
                           left_hand_plate_clearance_m=left_clearance,
                           minimum_left_hand_plate_clearance_m=minimum_left_clearance,
                           fruit_contact_force_N=[float(np.linalg.norm(
                               np.asarray(all_s.force_matrix.numpy()).reshape(2, -1, 3).sum(axis=1)
                               - np.asarray(stem_s.force_matrix.numpy()).reshape(2, -1, 3).sum(axis=1),
                               axis=1).sum())
                               for all_s, stem_s in zip(scene.right_bunch_sensors, scene.right_stem_sensors)],
                           minimum_plate_clearance_m=state['stem_pinch_min_plate_clearance_m'],
                           error=state['stem_pinch_error'])
                if wrap:
                    from demos.cherry_tomato_pick.bunch_wrap import contact_evidence
                    row['support_contacts'] = [
                        dict(force_N=f, position_m=p.tolist(), normal_N=n.tolist())
                        for f, p, n in map(contact_evidence, scene.right_sensors)]
                    pos, quat = scene.solver.body_pose(scene.branch_index)
                    row['bunch_com_m'] = (pos + Rotation.from_quat(quat).apply(
                        scene.branch_com_local)).tolist()
                    row['confirmation_forces_N'] = scene.support_confirmation_forces().tolist()
                result['samples'].append(row)
                if changed or i % 200 == 0:
                    print(json.dumps(row), flush=True)
            if state['stem_pinch_error']:
                result['error'] = state['stem_pinch_error']
                break
            if wrap and state['support_lift_verified'] and not args.through_harvest_reach:
                result['lift_passed'] = result['passed'] = True
                break
            if phase == 'harvest_approach':
                result['lift_passed'] = bool(state['stem_pinch_trial_passed'] and scene.support_engaged
                                        and scene.cluster_clearance >= .005 and not scene.support_lost)
                if not args.through_harvest_reach:
                    result['passed'] = result['lift_passed']
                    break
            if left_clearance < 0.:
                result['error'] = 'left_hand_plate_clearance'
                break
            if args.through_harvest_reach and phase == 'harvest_close':
                result['passed'] = bool(result['lift_passed'] and scene.support_engaged
                                        and minimum_left_clearance >= .005 and not scene.support_lost)
                break
            if setup.should_stop():
                break
        if not result['passed'] and not result['error']:
            result['error'] = ('harvest_reach_not_verified_before_duration' if args.through_harvest_reach
                               else 'full_lift_not_verified_before_duration')
        result['terminal_snapshot'] = dict(
            right_tcp=scene.robot.tcp_poses()[1].tolist(),
            right_command_tcp=scene.right_tcp.tolist(),
            right_hand_q=scene.robot.right.hand_data.qpos.tolist(),
            right_arm_q=scene.robot.arm_q()[1].tolist(),
            branch_pose=np.r_[scene.solver.body_pose(scene.branch_index)[0],
                              scene.solver.body_pose(scene.branch_index)[1]].tolist(),
            nominal_grasp_tcp=scene.robot.right_grasp_tcp.tolist(),
            finger_order=list(scene.robot.tactile_fingers),
            finger_total_force_N=[np.linalg.norm(np.asarray(s.total_force.numpy()).reshape(-1, 3), axis=1).tolist()
                                  for s in scene.right_bunch_sensors],
            finger_bunch_force_N=[np.linalg.norm(np.asarray(s.force_matrix.numpy()).reshape(-1, 3), axis=1).tolist()
                                  for s in scene.right_bunch_sensors],
            pad_stem_contacts=[np.asarray(s.position_matrix.numpy()).reshape(-1, 3).tolist()
                               for s in (scene.right_stem_sensors[j] for j in scene.support_force_indices)],
            pad_stem_normals=[(np.asarray(s.force_matrix.numpy()) -
                              np.asarray(s.force_matrix_friction.numpy())).reshape(-1, 3).tolist()
                              for s in (scene.right_stem_sensors[j] for j in scene.support_force_indices)],
        )
        # Exact contacted shapes distinguish fruit support from table or wrong-fruit pushes.
        contacts = scene.solver.contacts
        count = int(contacts.rigid_contact_count.numpy()[0])
        forces = contacts.force.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        labels = scene.solver.model.shape_label
        result['terminal_snapshot']['active_right_contacts'] = [
            dict(shape0=labels[a], shape1=labels[b], force_N=float(np.linalg.norm(f)))
            for a, b, f in zip(shape0, shape1, forces)
            if a >= 0 and b >= 0 and np.linalg.norm(f) > .01
            and ('right_wuji2_' in labels[a] or 'right_wuji2_' in labels[b])]
        if wrap:
            from demos.cherry_tomato_pick.bunch_wrap import contact_evidence
            result['terminal_snapshot']['support_contacts'] = [
                dict(force_N=f, position_m=p.tolist(), normal_N=n.tolist())
                for f, p, n in map(contact_evidence, scene.right_sensors)]
    except (RuntimeError, ValueError) as exc:
        result['error'] = str(exc)
    finally:
        if setup is not None:
            setup.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('x') as stream:
            json.dump(result, stream, indent=2)
    label = 'HARVEST REACH' if args.through_harvest_reach else 'LIFT'
    print(label, 'PASS' if result['passed'] else 'FAIL', result['error'], args.report, flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
