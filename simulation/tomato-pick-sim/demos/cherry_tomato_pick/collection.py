"""Episode-level sampling and conservative, task-local collection gates.

These geometric gates are not an IK or swept full-robot collision certificate.
Runtime contact/IK guards and the recorded outcome remain necessary.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sample_range(rng, value, name, *, maximum=None):
    bounds = np.asarray(value, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= bounds[1]
            or (maximum is not None and bounds[1] > maximum)):
        raise ValueError(f"{name} requires ordered positive finite [min, max]"
                         + (f" <= {maximum}" if maximum is not None else ""))
    return float(rng.uniform(*bounds))


def sample_appearance_and_goal(cfg, seed):
    """Use independent streams so adding appearance draws does not move objects."""
    randomization = cfg.get("recording", {}).get("randomization", {})
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    visual = randomization.get("visual", {})
    appearance = {"enabled": bool(randomization.get("enabled", False) and visual.get("enabled", False))}
    if appearance["enabled"]:
        appearance.update(
            exposure_gain=sample_range(rng, visual.get("exposure_gain", [1., 1.]),
                                       "visual.exposure_gain"),
            fruit_tint_rgb=[sample_range(rng, bounds, "visual.fruit_tint_rgb", maximum=1.)
                            for bounds in visual.get("fruit_tint_rgb", [[1., 1.]] * 3)],
            tabletop_gain=sample_range(rng, visual.get("tabletop_gain", [1., 1.]),
                                       "visual.tabletop_gain", maximum=1.),
        )
        if len(appearance["fruit_tint_rgb"]) != 3:
            raise ValueError("visual.fruit_tint_rgb requires three RGB ranges")
    cfg["collection_appearance"] = appearance
    goal = cfg.get("recording", {}).get("target_condition", {})
    mode = goal.get("mode", "fixed")
    if mode not in ("fixed", "uniform"):
        raise ValueError("recording.target_condition.mode must be fixed or uniform")
    if not cfg.get("deposit", {}).get("enabled", False):
        if mode != "fixed":
            raise ValueError("target-box sampling requires deposit.enabled")
        return appearance, None
    choices = goal.get("box_indices", [0, 1, 2])
    if (not isinstance(choices, list) or not choices
            or any(type(i) is not int or i not in (0, 1, 2) for i in choices)
            or len(set(choices)) != len(choices)):
        raise ValueError("target_condition.box_indices must be unique box IDs 0, 1, 2")
    index = cfg["deposit"].get("target_box", 1)
    if mode == "uniform":
        index = int(np.random.default_rng(np.random.SeedSequence([seed, 102])).choice(choices))
    if type(index) is not int or index not in (0, 1, 2):
        raise ValueError("deposit.target_box must be 0, 1, or 2")
    cfg["deposit"]["target_box"] = index
    return appearance, {
        "mode": mode, "target_box_id": index,
        "target_box_one_hot": np.eye(3, dtype=int)[index].tolist(),
        "target_box_center_world_m": [*cfg["tableware"]["box_centers_xy"][index],
                                      cfg["tableware"]["floor_top_z_m"]],
        "target_fruit_name": cfg["scene"]["target_name"],
        "instruction": f"Pick {cfg['scene']['target_name']} and place it in box {index}.",
        "policy_input_required": True,
    }


def apply_exposure(rgb, appearance):
    """Shared per-episode RGB exposure approximation, not light power calibration."""
    gain = float(appearance.get("exposure_gain", 1.))
    if gain == 1.:
        return rgb
    return np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * gain), 0, 255).astype(np.uint8)


def fruit_tint(name, cfg):
    if name in ("tomato", "support_tomato") or name.startswith("cluster_"):
        return tuple(cfg.get("collection_appearance", {}).get("fruit_tint_rgb", [1.] * 3))
    return (1., 1., 1.)


def initial_geometry(cfg):
    """Load the authored full bunch footprint, expressed about the target fruit."""
    from .backend import load_optional_visual_assets

    visuals = load_optional_visual_assets(Path(__file__).resolve().parents[2], cfg)
    if not visuals or not any(name.startswith("cluster_") for name in visuals):
        raise ValueError("legal scene generation requires the complete split fruit visual assets")
    # The configured support offset is world-space; footprint vertices must
    # all be body-local before the candidate pose is applied in validation.
    rotation = Rotation.from_quat(
        cfg["scene"].get("cluster_quaternion_xyzw", [0., 0., 0., 1.])
    )
    support = rotation.inv().apply(cfg["scene"]["support_tomato_offset_m"])
    return np.concatenate([
        v.vertices + np.asarray(v.local_position)
        + (support if v.body_name == "support_tomato" else 0.)
        for v in visuals.values()
    ])


def validate_initial_scene(cfg, local_points, limits):
    """Reject rim overhang; correct small floor/rim intersections by lifting all parts.

    Visual vertices plus the two independent fruit collision-sphere bounds are
    checked. The plate's faceted rim is bounded by a conservative radial ramp.
    This is a preflight approximation, not general mesh intersection detection.
    """
    table = cfg["tableware"]
    target = np.asarray(cfg["scene"]["tomato_position"], dtype=float)
    rotation = Rotation.from_quat(cfg["scene"].get("cluster_quaternion_xyzw", [0, 0, 0, 1]))
    points = rotation.apply(local_points) + target
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("scene validation requires finite XYZ bunch geometry and pose")
    radius = float(cfg["scene"]["tomato_radius_m"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("scene validation requires a positive finite tomato radius")
    # Cube corners conservatively contain each independently simulated sphere.
    from itertools import product
    corners = np.array(list(product([-radius, radius], repeat=3)))
    points = np.concatenate([points, corners + target,
                             corners + target + cfg["scene"]["support_tomato_offset_m"]])
    center = np.asarray(table["plate_center_xy"], dtype=float)
    radii = np.asarray(table["plate_radii_m"], dtype=float)
    margin = float(limits.get("plate_margin_m", .001))
    max_lift = float(limits.get("max_height_correction_m", .015))
    max_gap = float(limits.get("max_initial_gap_m", .025))
    if (center.shape != (2,) or radii.shape != (2,)
            or not np.isfinite(np.r_[center, radii, margin, max_lift, max_gap]).all()
            or margin < 0 or np.any(radii <= margin) or max_lift < 0 or max_gap < 0):
        raise ValueError("invalid scene-validation plate dimensions or tolerances")
    if np.max(np.linalg.norm((points[:, :2] - center) / (radii - margin), axis=1)) > 1.:
        return ["plate_overhang"], {}
    radial = np.linalg.norm((points[:, :2] - center) / radii, axis=1) / np.cos(np.pi / 48)
    surface = float(table["floor_top_z_m"]) + np.clip((radial - .86) / .14, 0., 1.) * table["plate_rim_height_m"]
    lift = max(0., float(np.max(surface + .0003 - points[:, 2])))
    if lift > max_lift:
        return ["excessive_initial_penetration"], {}
    if float(np.min(points[:, 2] + lift - surface)) > max_gap:
        return ["bunch_above_plate"], {}
    # Apply a single rigid translation: no independent fruit/anchor relocation.
    for key in ("tomato_position", "stem_anchor"):
        cfg["scene"][key][2] += lift
    target = np.asarray(cfg["scene"]["tomato_position"])
    grasp = target + rotation.apply(cfg["scene"]["stem_grasp_offset_m"])
    offset = np.asarray(cfg["script"]["support_lift_offset_m"])
    checks = {
        "left": [target, target + offset,
                 target + cfg["robot"]["left_pregrasp_offset_m"],
                 np.r_[table["box_centers_xy"][cfg["deposit"]["target_box"]], cfg["deposit"]["drop_center_z_m"]]],
        "right": [grasp, grasp + offset, grasp + cfg["robot"]["right_pregrasp_offset_m"]],
    }
    for side, positions in checks.items():
        lo, hi = (np.asarray(cfg["robot"][f"{side}_workspace_{bound}"]) for bound in ("min", "max"))
        if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(np.r_[lo, hi]).all()
                or np.any(lo >= hi)):
            raise ValueError(f"invalid {side} workspace bounds")
        if np.any(np.asarray(positions) < lo) or np.any(np.asarray(positions) > hi):
            return [f"{side}_workspace_proxy"], {}
    return [], {"height_correction_m": lift, "status": "passed",
                "checks": ["bunch_footprint", "plate_surface_clearance", "workspace_proxies"],
                "full_robot_collision_certified": False}


def prepare_collection(cfg, seed, sample_scene):
    """Bounded rejection sampling; all candidates derive from the episode seed."""
    limits = cfg.get("recording", {}).get("scene_validation", {})
    enabled = bool(limits.get("enabled", False))
    if enabled and (not cfg.get("tableware", {}).get("enabled", False)
                    or not cfg.get("deposit", {}).get("enabled", False)):
        raise ValueError("scene_validation requires tableware and deposit")
    attempts = limits.get("max_attempts", 64) if enabled else 1
    if type(attempts) is not int or not 1 <= attempts <= 10000:
        raise ValueError("scene_validation.max_attempts must be an integer in [1, 10000]")
    points = initial_geometry(cfg) if enabled else None
    rejected = Counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 103]))
    for attempt in range(attempts):
        candidate_seed = seed if attempt == 0 else int(rng.integers(0, 2**32))
        candidate, metadata = sample_scene(cfg, seed=candidate_seed)
        appearance, goal = sample_appearance_and_goal(candidate, seed)
        reasons, validation = validate_initial_scene(candidate, points, limits) if enabled else ([], {"status": "disabled"})
        if reasons:
            rejected.update(reasons)
            continue
        metadata.update(candidate_seed=candidate_seed, visual=appearance)
        # Preserve both the sampled offset and the effective corrected pose.
        metadata["tomato_position_m"] = candidate["scene"]["tomato_position"]
        validation.update(attempts=attempt + 1, rejected_candidates=dict(rejected))
        return candidate, metadata, goal, validation
    raise ValueError(f"no legal initial scene after {attempts} attempts: {dict(rejected)}; "
                     "adjust plate placement or randomization ranges")
