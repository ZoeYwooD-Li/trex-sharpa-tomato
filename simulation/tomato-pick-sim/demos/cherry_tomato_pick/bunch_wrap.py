"""Opt-in, measured stem/fruit wrap support; never an attachment shortcut."""
import numpy as np

from .kinematics import minimum_jerk


def lift_fraction(time_s, script):
    """Two rest-to-rest segments, with a measured hold at the trial height.

    The caller freezes script time at the junction until the physical trial
    passes. Both velocity and acceleration vanish on either side of that hold.
    """
    start = float(script['support_close_end_s'])
    split = start + float(script['support_trial_ramp_duration_s'])
    end = float(script['support_lift_end_s'])
    height = script['stem_pinch_trial_height_m'] / script['support_lift_offset_m'][2]
    if time_s <= split:
        return height * minimum_jerk(np.clip((time_s - start) / (split - start), 0., 1.))
    return height + (1. - height) * minimum_jerk(np.clip((time_s - split) / (end - split), 0., 1.))


def contact_targets(task, fingers):
    """Optional per-finger force goals; preserve the scalar legacy default.

    Two fingers opposite one thumb need not each request the thumb's force.
    Zero means no active closure force, not removal of collision monitoring.
    """
    mapping = task.get('support_contact_target_by_finger_N')
    if mapping is None:
        values = np.full(len(fingers), float(task.get('support_contact_target_N', 3.)))
    else:
        if not isinstance(mapping, dict) or set(mapping) != set(fingers):
            raise ValueError('support_contact_target_by_finger_N must name every support finger exactly')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in mapping.values()):
            raise ValueError('support_contact_target_by_finger_N values must be numeric forces')
        values = np.asarray([mapping[f] for f in fingers], dtype=float)
    if (not np.isfinite(values).all() or np.any(values < 0.) or values[0] <= 0.
            or np.count_nonzero(values > 0.) < task.get('minimum_support_contact_fingers', 3)):
        raise ValueError('wrap force targets require finite nonnegative forces, a loaded thumb, and enough loaded fingers')
    return values


def closure_rates(force, targets, *, free_rate, gain):
    """Zero-target fingers do not advance freely; obstacles still cause release."""
    force, targets = np.asarray(force), np.asarray(targets)
    return np.where((force < .1) & (targets > 0.),
                    free_rate, np.clip(gain * (targets - force), -.5, .5))


def pose_lock_time(script):
    """Observe settling while parked above the bunch; freeze before descent."""
    return float(script['support_approach_end_s']) + float(script.get('support_preshape_duration_s', 0.))


def closure_feedback(qualified_force, all_contact_force):
    """All obstacles limit closure; only qualified contacts may confirm a grasp."""
    force = np.asarray(all_contact_force, dtype=float)
    if not np.isfinite(force).all():
        raise ValueError('nonfinite wrap contact feedback')
    return np.maximum(qualified_force, force)


def advance_lift_hold(elapsed, dt, *, valid, required):
    """Accumulate only consecutive, post-physics qualified hold samples."""
    elapsed = elapsed + dt if valid else 0.
    return elapsed, bool(valid and elapsed + 1e-9 >= required)


def support_shapes(fruit_shapes, names, target):
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or target in names or any(n not in fruit_shapes for n in names)):
        raise ValueError('support_fruit_names must name distinct available non-harvest fruits')
    return ['branch_collision_000', *(fruit_shapes[n] for n in names)]


def contact_evidence(sensor):
    """Filtered pad+distal resultants; table/other-body totals never qualify."""
    if any(getattr(sensor, name, None) is None for name in
           ('force_matrix', 'force_matrix_friction', 'position_matrix')):
        raise ValueError('bunch wrap requires filtered forces, friction and contact positions')
    force = np.asarray(sensor.force_matrix.numpy(), dtype=float).reshape(-1, 3)
    friction = np.asarray(sensor.force_matrix_friction.numpy(), dtype=float).reshape(-1, 3)
    positions = np.asarray(sensor.position_matrix.numpy(), dtype=float).reshape(-1, 3)
    if (force.shape != friction.shape or force.shape != positions.shape
            or not np.isfinite([force, friction, positions]).all()):
        return 0., np.zeros(3), np.zeros(3)
    weights = np.linalg.norm(force, axis=1)
    magnitude = float(weights.sum())
    if magnitude < 1e-8:
        return 0., np.zeros(3), np.zeros(3)
    return magnitude, np.average(positions, weights=weights, axis=0), (force-friction).sum(0)


def wrap_ready(sensors, fruit_sensors, center, fingers, threshold, minimum_fingers=3,
               *, confirmation_forces=None, fruit_confirmation_forces=None, contact_floor=None):
    """Require opposing contacts plus a fruit contact on a lower supporting finger.

    Normals are forces on the hand: a downward normal at a lower contact means
    that the finger pushes upward on the fruit. This is a necessary geometric
    check, not a friction-cone certificate; the physical trial remains required.
    """
    evidence = [contact_evidence(s) for s in sensors]
    raw = np.asarray([item[0] for item in evidence])
    forces = raw if confirmation_forces is None else np.asarray(confirmation_forces)
    floor = threshold if contact_floor is None else contact_floor
    if (forces.shape != raw.shape or not np.isfinite(forces).all()
            or np.any(forces < 0.)):
        return False
    # Sustained-load confirmation uses the same EMA as the controller, but
    # every qualifying finger must still have a CURRENT measured contact.
    active = [i for i in range(len(evidence)) if forces[i] >= threshold and raw[i] >= floor]
    if 0 not in active or len(active) < minimum_fingers:
        return False
    thumb_pos, thumb_normal = evidence[0][1:]
    thumb_radial = thumb_pos[:2] - np.asarray(center)[:2]
    if np.linalg.norm(thumb_radial) < .001 or np.linalg.norm(thumb_normal) < .1:
        return False
    opposed = False
    supporting = False
    for i in active[1:]:
        _, pos, normal = evidence[i]
        radial = pos[:2] - np.asarray(center)[:2]
        if np.linalg.norm(radial) >= .001 and np.linalg.norm(normal) >= .1:
            opposed |= bool(np.dot(radial, thumb_radial) / (
                np.linalg.norm(radial)*np.linalg.norm(thumb_radial)) <= -.25
                and np.dot(normal, thumb_normal) / (
                    np.linalg.norm(normal)*np.linalg.norm(thumb_normal)) <= -.2)
        if fingers[i] in ('middle', 'ring', 'pinky'):
            force, point, normal = contact_evidence(fruit_sensors[i])
            fruit_force = force if fruit_confirmation_forces is None else fruit_confirmation_forces[i]
            supporting |= bool(force >= floor and np.isfinite(fruit_force) and fruit_force >= threshold
                               and np.linalg.norm(normal) >= .1
                               and point[2] <= center[2] + .010
                               and normal[2] / np.linalg.norm(normal) <= -.15)
    return opposed and supporting
