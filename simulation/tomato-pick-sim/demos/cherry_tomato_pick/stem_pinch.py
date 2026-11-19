"""Opt-in scripted stem-only pinch checks; no attachment or pose teleporting."""
from dataclasses import dataclass

import numpy as np

from .kinematics import minimum_jerk


def validate_config(cfg):
    wrap = bool(cfg['task'].get('bunch_wrap_control', False))
    pinch = bool(cfg['task'].get('stem_pinch_control', False))
    if 'support_trial_ramp_duration_s' in cfg['script']:
        duration = cfg['script']['support_trial_ramp_duration_s']
        if (not wrap or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not np.isfinite(duration) or not 0. < duration < (
                    cfg['script']['support_lift_end_s'] - cfg['script']['support_close_end_s'])):
            raise ValueError('support_trial_ramp_duration_s requires bunch wrap and a duration within the lift')
    if 'support_contact_target_by_finger_N' in cfg['task'] and not wrap:
        raise ValueError('per-finger support force targets require bunch_wrap_control')
    if wrap and pinch:
        raise ValueError('bunch_wrap_control and stem_pinch_control are mutually exclusive')
    if not pinch and not wrap:
        if (cfg['robot'].get('right_reach_hand') or cfg['script'].get('support_preshape_duration_s', 0.)
                or cfg['robot'].get('right_stem_insert_offset_m') is not None):
            raise ValueError('right reach preshape is restricted to scripted stem pinch')
        return
    if wrap and cfg['scene']['support_mode'] != 'stem_and_fruit':
        raise ValueError('bunch_wrap_control requires stem_and_fruit mode')
    if not wrap and cfg['scene']['support_mode'] != 'main_stem':
        raise ValueError('stem_pinch_control requires main_stem mode')
    fingers = cfg['robot']['right_force_fingers']
    if wrap:
        if fingers != ['thumb', 'index', 'middle', 'ring', 'pinky']:
            raise ValueError('bunch wrap requires all five fingers in tactile order')
        from .bunch_wrap import contact_targets
        contact_targets(cfg['task'], fingers)
        names = cfg['scene'].get('support_fruit_names')
        if (not isinstance(names, list) or not names or len(names) != len(set(names))
                or cfg['scene']['target_name'] in names):
            raise ValueError('support_fruit_names must exclude the harvest target and be distinct')
        if not 3 <= cfg['task']['minimum_support_contact_fingers'] <= 5:
            raise ValueError('bunch wrap requires at least three contact fingers')
        hold = cfg['script'].get('support_full_lift_hold_s', 0.)
        if not np.isfinite(hold) or hold <= 0.:
            raise ValueError('support_full_lift_hold_s must be finite and positive')
        if not isinstance(cfg['script'].get('support_lift_only', True), bool):
            raise ValueError('support_lift_only must be boolean')
    elif fingers not in (['thumb', 'index'], ['thumb', 'index', 'middle']):
        raise ValueError('stem pinch requires thumb/index, optionally middle, in that order')
    if cfg['task'].get('minimum_support_wrap_closure', 0.) != 0.:
        raise ValueError('stem pinch must not use the fruit-wrap closure threshold')
    if not wrap and cfg['task']['minimum_support_contact_fingers'] != len(fingers):
        raise ValueError('stem pinch requires all configured contact fingers')
    if cfg['robot'].get('right_stem_preload_joint_offsets_rad'):
        raise ValueError('stem pinch must not combine force control with legacy preload')
    if cfg['script'].get('stem_pinch_open_approach', False):
        hand = cfg['robot']['right_pregrasp_hand']
        pairs = list(zip(hand['joint_order'], hand['positions'], strict=True))
        active = [float(value) for name, value in pairs
                  if name.startswith(tuple(fingers))]
        if len(active) != 4 * len(fingers) or not np.isfinite(active).all() or np.any(np.abs(active) > 1e-8):
            raise ValueError('stem_pinch_open_approach requires all active fingers fully open at startup')
        if cfg['script'].get('support_close_on_early_contact', True):
            raise ValueError('open approach must not turn an early single-finger touch into the grasp pose')
    duration = float(cfg['script'].get('support_preshape_duration_s', 0.))
    if cfg['robot'].get('right_reach_hand') or duration != 0.:
        if (not cfg['robot'].get('right_reach_hand') or not np.isfinite(duration)
                or not 0. < duration < cfg['script']['support_reach_end_s'] - cfg['script']['support_approach_end_s']):
            raise ValueError('preshape requires right_reach_hand and a positive duration shorter than the reach')
    if cfg['robot'].get('right_stem_insert_offset_m') is not None:
        start = float(cfg['script'].get('support_insert_start_s', 0.))
        if (not cfg['robot'].get('right_reach_hand') or not np.isfinite(start)
                or not cfg['script']['support_approach_end_s'] + duration < start < cfg['script']['support_reach_end_s']):
            raise ValueError('stem insertion must follow preshape and precede grasp reach')
    for key in ('stem_pinch_timeout_s', 'stem_pinch_trial_height_m',
                'stem_pinch_trial_hold_s', 'stem_pinch_trial_timeout_s',
                'stem_pinch_max_slip_m'):
        value = float(cfg['script'].get(key, 0.))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    lift = np.asarray(cfg['script']['support_lift_offset_m'], dtype=float)
    if (lift.shape != (3,) or not np.isfinite(lift).all()
            or np.any(lift[:2] != 0.) or not 0 < cfg['script']['stem_pinch_trial_height_m'] < lift[2]):
        raise ValueError('stem pinch trial requires a larger, vertical full lift')
    if not cfg.get('tableware', {}).get('enabled', False):
        raise ValueError('stem pinch requires the configured tableware clearance guard')
    for key in ('minimum_harvest_cluster_clearance_m', 'minimum_left_tableware_clearance_m',
                'harvest_workspace_timeout_s'):
        if key in cfg['task']:
            value = float(cfg['task'][key])
            if not np.isfinite(value) or value <= 0.:
                raise ValueError(f'{key} must be finite and positive')
    if 'maximum_harvest_stem_tilt_deg' in cfg['task']:
        value = float(cfg['task']['maximum_harvest_stem_tilt_deg'])
        if not np.isfinite(value) or not 0. < value <= 90.:
            raise ValueError('maximum_harvest_stem_tilt_deg must be in (0, 90]')


def opposed_stem_tips(tips, center, axis):
    """Each opposing contact position must lie across the stem from the thumb.

    This position check supplements measured pad/stem normal forces. It is not a
    friction-cone or contact-normal force-closure certificate.
    """
    tips, center, axis = map(lambda v: np.asarray(v, dtype=float), (tips, center, axis))
    if (tips.shape not in ((2, 3), (3, 3)) or center.shape != (3,) or axis.shape != (3,)
            or not all(np.isfinite(v).all() for v in (tips, center, axis))
            or np.linalg.norm(axis) < 1e-9):
        return False
    axis = axis / np.linalg.norm(axis)
    delta = tips - center
    radial = delta - np.outer(delta @ axis, axis)
    lengths = np.linalg.norm(radial, axis=1)
    if np.any(lengths < .001) or np.max(np.abs(delta @ axis)) > .030:
        return False
    radial /= lengths[:, None]
    return bool(np.all(radial[1:] @ radial[0] <= -.25))


def trial_script_time(script):
    """Invert the actual nested smoothstep / pose minimum-jerk trajectory."""
    if 'support_trial_ramp_duration_s' in script:
        return script['support_close_end_s'] + script['support_trial_ramp_duration_s']
    fraction = script['stem_pinch_trial_height_m'] / script['support_lift_offset_m'][2]
    low, high = 0., 1.
    for _ in range(50):
        mid = (low + high) / 2.
        if minimum_jerk(mid * mid * (3 - 2 * mid)) < fraction:
            low = mid
        else:
            high = mid
    return script['support_close_end_s'] + high * (
        script['support_lift_end_s'] - script['support_close_end_s'])


@dataclass
class StemPinchTrial:
    closing_s: float = 0.
    trial_s: float = 0.
    stable_s: float = 0.
    passed: bool = False
    error: str = ''

    def update(self, *, dt, closing, lifted, at_trial, ready, lift_m, slip_m, script):
        if self.error:
            return
        if closing and not lifted:
            self.closing_s += dt
            if not ready and self.closing_s >= script['stem_pinch_timeout_s']:
                self.error = 'stem_pinch_grasp_timeout'
        if lifted and (not np.isfinite(slip_m) or slip_m > script['stem_pinch_max_slip_m']):
            self.error = 'stem_pinch_slip'
        if lifted and at_trial and not self.passed:
            self.trial_s += dt
            stable = ready and lift_m >= .8 * script['stem_pinch_trial_height_m']
            self.stable_s = self.stable_s + dt if stable else 0.
            self.passed = self.stable_s + 1e-9 >= script['stem_pinch_trial_hold_s']
            if not self.passed and self.trial_s >= script['stem_pinch_trial_timeout_s']:
                self.error = 'stem_pinch_trial_timeout'


def premature_stem_contact(script_time, reach_end, forces):
    """A touch during the open reach is a failed approach, not a new grasp pose."""
    values = np.asarray(forces, dtype=float)
    return bool(script_time < reach_end and (
        not np.isfinite(values).all() or np.any(values > .3)))


def plate_clearance(vertices, tableware):
    """Conservative pad/distal AABB vs the flat bowl and sloping rim.

    A diagnostic stop, not continuous collision prevention. The envelope may
    reject valid poses close to the rim. It cannot miss a crossing merely
    because the mesh has no vertex inside the circular footprint.
    """
    v = np.asarray(vertices)
    if not np.isfinite(v).all():
        return -np.inf
    center, radius = tableware.plate_center, tableware.plate_radii
    low, high = v[:, :2].min(0) - center, v[:, :2].max(0) - center
    nearest = np.maximum(low, 0.) + np.minimum(high, 0.)
    if np.linalg.norm(nearest / radius) > 1.:
        # The tabletop uses the plate underside as its physical ground, not Z=0.
        tabletop = tableware.floor - tableware.wall
        if tableware.cfg.get('tabletop', {}).get('enabled', False):
            tabletop += .0002  # Match the rendered slab's documented offset.
        return float(v[:, 2].min() - tabletop)
    farthest_radius = np.linalg.norm(np.maximum(np.abs(low), np.abs(high)) / radius)
    rim_fraction = np.clip((farthest_radius - .86) / .14, 0., 1.)
    height = tableware.floor + rim_fraction * tableware.plate_height
    return float(v[:, 2].min() - height)


def stem_tilt_degrees(quaternion, axis):
    """Unsigned angle of the main-stem axis above the horizontal plane."""
    from scipy.spatial.transform import Rotation
    world = Rotation.from_quat(quaternion).apply(axis)
    return float(np.degrees(np.arcsin(np.clip(abs(world[2]) / np.linalg.norm(world), 0., 1.))))


def harvest_workspace_ready(clearance_m, tilt_deg, task):
    """Measured bunch clearance and tilt are independent of commanded lift."""
    return bool(np.isfinite([clearance_m, tilt_deg]).all()
                and clearance_m >= task.get('minimum_harvest_cluster_clearance_m', .005)
                and tilt_deg <= task.get('maximum_harvest_stem_tilt_deg', 90.))
