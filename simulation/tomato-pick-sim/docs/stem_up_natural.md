# Stem-up with the historical natural right hand

Select `configs/cherry_tomato_pick_stem_up_natural.yaml` explicitly. Unlike the
historical collection baseline, this preset keeps the bunch **stem-up**. The
right hand retains the original open posture, forward-flexed closed posture,
wrist quaternion and thumb/index distal-first closure. No mirrored meshes,
reversed flexion or inverted hand mount are introduced.

## Calibration and contact feedback

The nominal cluster quaternion is `[0, 1, 0, 0]` (xyzw), with matching stem-up
attachments and axial pull direction. The body-local middle-stem reference
remains `[0.0259, -0.105, -0.0151]` m. Relative to the historical stem-down
calibration, the right grasp-center bias is lowered by 20 mm to
`[-0.009234196, -0.000192398, -0.009820221]` m. This allows fruit-assisted
support around the middle stem; it is not a stem-only pinch.

The left wrist attitude is pre-multiplied by a **world-X rotation of -30 degrees**
relative to the historical baseline. Left open/closed finger joint targets and
the approach offset are unchanged. This moves the closing fingers away from
the overhead main stem while retaining reach for the downward stem-up pull.
Startup IK validation uses the planned raised harvesting height without
translating the stored calibration a second time. A regression checks the
complete nominal 70 mm downward pull, not only a single detachment pose.

Left contact feedback now includes each fingertip patch and its distal link.
Total contact force, including main-stem and neighboring-fruit contacts, drives
force-limited closure and safety checks. Only force against the selected target
tomato qualifies grasp confirmation and retention. Contact on a neighboring
fruit must not be mistaken for a successful target grasp.

Mass, friction, collision geometry, force targets, the **35 N** limit,
right-support contact/closure/hold gates and sequence timing remain unchanged.
The new left sensing/qualification and raised-stage IK checks are shared by the
standalone Wuji2 task runtime; the wrist recalibration is confined to this
explicit preview preset. Other configuration files, OpenPI weights and existing
demonstrations are not changed by this calibration.

The placement target `deposit.drop_center_z_m` is 0.21 m in world coordinates
(held tomato center), 30 mm above the earlier 0.18 m target. The existing
transport interpolation raises the left hand with the fruit before release;
its XY destination, wrist attitude and release checks are unchanged. This
setting does not raise the grasping pose or the right-hand lift height.

## Preview

From the standalone project root, with its NVIDIA GPU runtime available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_stem_up_natural.yaml \
  --frontend rerun --live
```

Restart a running preview to load the changes. Recording is disabled, so
recording-time randomization is also inactive. Future explicitly enabled
recording uses `outputs/stem_up_natural/` with unique episode allocation;
existing data is not overwritten.

## Validation scope

This remains a **nominal-scene development preview**, not a validated randomized
collection preset. Camera-disabled, no-recording physics checks exercise actual
lift, target grasp, detachment, transport and stable placement. These are not
quality-accepted training episodes or measurements of a randomized success rate.
Unit tests separately cover target-only qualification, obstacle force feedback,
unchanged right posture/gates and left IK reachability.

Run-specific results, including unsuccessful calibration candidates, effective
configurations and source hashes, are in
`outputs/left_harvest_fix_20260919/RESULTS.md`. The diagnostic additionally stops
early on force-limit violations or lost support; that wrapper does not change
the normal scripted runner. Its sampled visual-mesh clearance audit covers the
plate/table envelope only, not box walls or continuous collision guarantees.
