# Scripted main-stem pinch

Select `configs/cherry_tomato_pick_stem_pinch.yaml` explicitly. This opt-in
scripted controller is separate from the default deposit preset and trained
OpenPI policy; it does not retrain or modify a checkpoint. Do not mix its
rollouts into accepted training data without checking the complete task.

## Sequence

1. Start with thumb and index fully open. The other three fingers stay tucked.
2. Reach the external pregrasp pose, then form a calibrated two-finger aperture
   during a stationary hold (`support_preshape_duration_s`). The hand is not
   initialized in this curved pose.
3. Lower outside the main-stem end to `right_stem_insert_offset_m`, then move
   along the stem into the aperture. Wrist position checks gate the segment
   boundaries. An early stem touch fails the approach instead of retargeting
   the grasp to whichever finger happened to touch first.
4. Close thumb/index independently using stem-only force feedback. Both
   fingers, opposing contact positions/normals, the position gate and sustained
   contact must agree before lifting.
5. Pause at an actual 10 mm commanded trial lift. Require at least 8 mm of
   measured main-stem rise and 0.4 s stable support before the full lift.

The trial pause inverts both interpolations used by the existing lift path:
smoothstep followed by the pose interpolator's minimum-jerk curve. Inverting
only the first curve commanded about 0.59 mm instead of 10 mm and made the
previous 8 mm measured-rise criterion impossible to satisfy normally.

The optional joint-space initial approach uses an explicit, limit-checked IK
seed to retain the fitted elbow branch. Cartesian insertion and lifting still
use the existing robot adapter. This is not a general motion planner or a
full-arm continuous-collision certificate. Randomized bunch rotations also
rotate the calibrated wrist frame and approach offsets when
`right_grasp_follow_scene_rotation` is enabled.

## Physical evidence and safeguards

Each active finger groups its pad and distal link as one finger. Only the
per-counterpart `force_matrix` against `branch_collision_000` qualifies as
stem contact. Newton's `total_force` is not counterpart-filtered; incidental
fruit contact cannot qualify as a stem grasp. Two-finger mode requires both
fingers; the existing three-finger mode remains supported and requires all
three. A closed command is not evidence of a grasp.

Contact normals and positions must oppose across the stem (radial normalized
dot product at most -0.25). The two-finger preset targets 5 N per finger with
0.1/s/N feedback gain, requires at least 2 N per active finger and at least
3 N thumb/opposing force for 0.25 s. The 30 mm position gate, 8 s grasp timeout,
2 s trial timeout, 8 mm slip limit, support-loss guard, interhand check and
35 N peak-force limit remain. A conservative pad/distal plate-clearance check
also remains enabled. No new weld, kinematic bunch lifting, collision masking,
friction change, or mass change is used.

This is privileged scripted geometry control, including bounded XY recentering,
not learned visual control. The main stem remains an 8 mm-diameter rigid
collision proxy; simulated success would not establish real-stem strength.

## Validation and commands

### Midpoint grasp development preset

`configs/cherry_tomato_pick_stem_midpoint.yaml` targets the main-stem collision
proxy's geometric center, body-local `[0.0259, -0.105, -0.0151]` m. This is a
90 mm shift from the end-grasp target. It uses a separately fitted wrist and
thumb/index aperture, descends from above the middle, and retains the open
startup, measured opposing-contact confirmation, trial lift, slip checks and
35 N peak-contact limit. The per-finger target is 8 N, not a higher
safety-stop threshold.
The vertical lift remains 120 mm and its nominal duration is 5 seconds.

The original end-grasp preset remains available unchanged. **The midpoint
preset failed nominal physical verification:** it reached about 8 mm measured
trial rise, then lost sustained opposing contact. Increasing closure travel
also failed the unchanged peak-force guard. No complete lift or left-hand
approach passed, and no successful training episodes were generated.
The midpoint preset is for development, not randomized batch collection. Measured results
and rejected candidates are in `outputs/stem_pinch_midstem_20260918/RESULTS.md`;
do not infer successful lifting or harvesting from configuration or unit tests.

Before releasing the left hand, the midpoint mode requires at least 60 mm
measured bunch clearance and at most 15 degrees main-stem tilt above horizontal.
Failure to meet the gate within 2 seconds at full lift reports
`harvest_workspace_not_clear`. Commanded wrist elevation alone is insufficient.

The left-hand table guard includes authored palm, proximal and distal meshes,
not only the fingertip colliders. It checks the joint-space midpoint and end of
each changed command, restores the preceding state and latches a stop if the
conservative table/plate envelope is closer than 5 mm. Outside the plate, the
envelope uses the actual tabletop height rather than Z=0. This is a sampled
hand-motion guard, not a full-arm continuous-collision or box-avoidance proof.
Convex-hull extrema cache the same envelope without transforming every visual
triangle each step.

Preview one nominal attempt without recording:

```bash
uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_stem_midpoint.yaml \
  --frontend rerun --live --set recording.enabled=false
```

Run the bounded, camera-free diagnostic using a new report filename:

```bash
uv run --locked python -m demos.cherry_tomato_pick.tools.check_stem_pinch \
  --config configs/cherry_tomato_pick_stem_midpoint.yaml \
  --through-harvest-reach --duration 30 \
  --report outputs/main_stem_midpoint/manual_reach.json
```

`HARVEST REACH PASS` additionally verifies the left approach to the start of
closure. It does not certify detachment or placement. Reports contain the main
stem's actual inclination and the sampled minimum left-hand table/plate gap.

### Earlier end-grasp evidence

Run-specific evidence is stored under `outputs/stem_pinch_policy_20260918/`.
The nominal diagnostic passed: measured stem rise 113.83 mm, terminal opposing
forces 4.110/4.184 N and peak contact 9.719 N. This is one fixed-scene lift,
not a success rate or validated harvesting/deposit sequence. Seed 42 failed
the initial IK preflight; do not treat the preset as ready for randomized
batch collection. Read the run-specific results before collecting data.
Earlier failed calibrations remain under `outputs/stem_pinch_calibration_20260917/`
and `outputs/stem_pinch_calibration_20260918/`.

From the standalone repository, run a bounded diagnostic without cameras or
training-data recording. Use a new report filename each time:

```bash
uv run --locked python -m demos.cherry_tomato_pick.tools.check_stem_pinch \
  --duration 26 --report outputs/main_stem_pinch/manual_lift.json
```

Omitting `--seed` uses the nominal pose. Add `--seed 42` for legal collection
randomization. `LIFT PASS` means the trial plus measured right-hand lift only,
not completed harvesting or deposit. Reports include effective configuration,
source hashes, script/wall-simulation time, contacts, measured rise and failure
reasons. Failed attempts are not accepted demonstrations.

For one visual attempt without recording a training episode:

```bash
uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_stem_pinch.yaml \
  --frontend rerun --live --set recording.enabled=false
```

This preset cannot be used with `tomato-sim eval`: the scripted preshape,
insertion and trial gates are not part of the trained action adapter. Validate
subsequent harvesting/placing and multiple scene seeds before collecting a batch.
