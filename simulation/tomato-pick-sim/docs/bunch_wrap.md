# Scripted stem/fruit wrap experiment

Select `configs/cherry_tomato_pick_bunch_wrap.yaml` explicitly. This is an
opt-in **development preset**, not a validated harvesting or collection preset.
It does not change the default deposit configuration, the stem-only presets,
OpenPI inference, or a trained checkpoint.

## Controller contract

- Target the middle of the main-stem proxy, body-local
  `[0.0259, -0.105, -0.0151]` m. The nominal scene remains stem-up.
- Start with all five fingers open. Limit each finger's closure using **all**
  contact forces, while qualifying support using only filtered contacts.
  Optional `right_reach_hand` / `support_preshape_duration_s` can form
  an aperture while parked above the bunch, not at initialization.
- Optional `task.support_contact_target_by_finger_N` names all five fingers
  in newtons. This allows a thumb force to balance the combined opposing
  fingers instead of assigning every finger the same force. A zero target
  prevents free closure but does not disable sensing or collision guards.
  Without the mapping, the scalar force target retains its previous behavior.
  For a three-finger grasp, `right_grasp_opposing_fingers` must describe the
  active index/middle midpoint, not folded inactive ring/pinky fingers.
- An early single-finger touch must not redefine the final grasp pose. The
  open-approach mode rejects `support_close_on_early_contact: true`. Approach
  collisions remain subject to the independent peak-force/clearance guards;
  finger force feedback does not make an obstructed wrist path valid.
- Retarget the grasp frame during settling/air preshape, then freeze it before
  descent. Execute matching planned joint-space approach waypoints directly;
  do not solve their FK poses again through a potentially different IK branch.
  The one-shot waypoint is not reused for later Cartesian or policy commands.
- Permit contacts on the main stem and the explicit non-harvest fruit allowlist
  `scene.support_fruit_names`. Do not count table contact or the left-hand
  harvest target as support. Pad and distal contacts count as **one finger**.
- Require the thumb and at least two other fingers, opposing contact positions
  and normals, and a lower supporting fruit contact on middle/ring/pinky. The
  lower finger must push upward on the fruit. Contact duration, bilateral force
  and distance checks also apply. A closed command is not grasp evidence.
  Sustained force qualification uses the same 30 ms EMA as force control,
  including a separate fruit-only estimate. Every qualifying finger must
  still have a current measured contact above the retention floor, and contact
  positions/normals remain unfiltered. The independent retention-loss guard
  continues to use raw forces and geometry; stale EMA cannot hide contact loss.
- Pause at a commanded 10 mm trial lift. Require at least 8 mm measured bunch
  rise, 0.4 s sustained support and no excessive slip before continuing.
  Optional `script.support_trial_ramp_duration_s` splits the lift into two
  minimum-jerk segments with zero velocity and acceleration at the pause and
  restart. It must be positive and shorter than the full lift duration.
  The measured trial still gates continuation; this option does not bypass it.
- Command a 120 mm full lift, then require 2 s consecutive post-physics support
  samples, measured rise, at least 60 mm bunch clearance and at most 15 degrees
  stem tilt. Contact loss resets this hold; wrist height alone cannot pass it.
- `script.support_lift_only: true` keeps the left hand parked and ends after
  successful lift/hold verification. This is **not** picking/deposit success.
  Disable it only for a separate full-sequence experiment after validating lift.

The 8 s grasp timeout, 2 s trial timeout, 8 mm slip limit, support-loss guard,
35 N all-contact peak stop, table-clearance and interhand guards remain active.
The force threshold is a simulator guard, not a validated tomato damage limit.
No friction, mass, collision masking or attachment shortcut is introduced.
The grasp gate is a necessary contact-geometry check, not a friction-cone proof.

Internally this mode shares the stem-pinch trial guard; some failure/phase keys
therefore retain the `stem_pinch` prefix. `support_lift_verified` is separate
from complete harvesting and quality acceptance.

## Run one diagnostic

Use the existing project environment and NVIDIA GPU. From the repository root:

```bash
uv run --locked python -m demos.cherry_tomato_pick.tools.check_stem_pinch \
  --config configs/cherry_tomato_pick_bunch_wrap.yaml \
  --duration 28 --report outputs/bunch_wrap_check_01.json
```

Choose a new report filename for each run. This tool disables all cameras and
demonstration recording, writes effective configuration and source hashes, and
returns a nonzero status if lift fails. Its terminal contact-shape report helps
distinguish wrong-fruit pushing from qualified support.
Wrap samples also record per-finger contact positions/normals, bunch center of
mass and filtered confirmation forces, to explain transient support failures.

For one visual preview (Rerun/display required):

```bash
uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_bunch_wrap.yaml \
  --frontend rerun --live --set recording.enabled=false
```

## Validation scope

Unit tests cover counterpart filtering, opposing multi-finger geometry, lower
fruit support, open startup, mode isolation, hold reset and left-hand parking.
They do not establish physical lifting success. The initial nominal physical
test failed: the index finger pushed a fruit outside the central allowlist
before thumb opposition formed, and the peak-force guard stopped the attempt.
The preset still needs physical grasp-pose calibration; do not start batch
collection or training from it. Run-specific evidence and candidate results
are retained in `outputs/bunch_wrap_20260918/RESULTS.md`.

The follow-up controller fix separates obstacle feedback from qualified
support. Previously an excluded fruit could be ignored by the closure
controller even though it correctly did not count as grasp support. Subsequent
headless tests found another failure mode: three touching fingers can still
push from the same side instead of opposing each other. A low peak force or a
three-finger count is therefore not a lifting-success criterion.

Follow-up reports and a contact-debug candidate are recorded in
`outputs/bunch_wrap_fix_20260918/RESULTS.md`. The candidate is not a validated
replacement for collection or inference; consult measured trial/full-lift
results, not merely the absence of `unsafe_contact`.

Three-finger contact-layout, force-allocation and split-lift diagnostics are
retained in `outputs/bunch_wrap_layout_20260918/RESULTS.md`. They demonstrate
physical elevation in the nominal scene, but no continuous full-height hold
pass. The archive's preview candidates are not replacements for collection
presets or policy checkpoints. Increasing a scalar closure command does not
guarantee more supporting force once the authored joint limits clip the path.
