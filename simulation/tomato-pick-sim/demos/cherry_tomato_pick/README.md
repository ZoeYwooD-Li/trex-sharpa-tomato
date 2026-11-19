> This document describes the historical Wuji2 workflows. The standalone
> Sharpa checkout now defaults to `configs/cherry_tomato_pick_sharpa.yaml`;
> see [Sharpa integration](../../docs/sharpa.md) for the active hand model.

# Cherry tomato picking

This demo ports the current cherry-tomato assets and task sequence from
`git@github.com:yxsdev/cherry_tomato_sim.git` revision
`43610761c4f667dc24ac6b5f7d66b71ebf72c1a1` into LycheeSim's Newton runtime.
The selected workflow lays the bunch on the work surface, presses
`tomato_07` with the right arm, and grasps and lifts `tomato_10` with the left
arm. Both sides use the dual-FR3 carrier and the official left/right Wuji Hand
2 Beta2 `with_mount` geometry; there is no proxy gripper. The checked-in hand
assets are pinned to upstream revision
`4c1073d0a3ad1daaf6546d219db751d8448d3888`; see
[`wuji_hand2_beta2_official/PROVENANCE.md`](../assets/robots/wuji_hand2_beta2_official/PROVENANCE.md).

The upstream split contains `stem` and `tomato_00..11`. It identifies four
detachable endpoints: `tomato_07`, `tomato_02`, `tomato_10`, and `tomato_09`.
The current LycheeSim task instantiates one selected harvest fruit and the
opposite pressed fruit as independent dynamic bodies. The stem and remaining
ten fruits stay in the static detailed bunch. This keeps the two manipulated
visual meshes synchronized with their Newton bodies without paying for twelve
independent contact systems.

The purchased tomato model and textures are stored under this demo with the
project owner's authorization. See
[`assets/ASSET_PROVENANCE.md`](assets/ASSET_PROVENANCE.md) for the exact source,
revision, imported files, and licensing boundary.

## Prepare and run

Rebuild the runtime meshes from the checked-in split OBJ source:

```bash
uv run python -m demos.cherry_tomato_pick.tools.import_assets
```

The demo-owned importer prefers the latest split OBJ directory. It applies the
upstream metre scale and `-90°` X assembly rotation,
emits separate centered render meshes for `tomato_00..11` plus the stem,
preserves UVs, and records source hashes in `assets/runtime/metadata.json`.
It also emits `tomato_fruit_basecolor.png`: source red fruit texels are
unchanged, while atlas regions unused by fruit are remapped to red with their
luminance retained. The stem keeps the untouched source atlas. This isolates
fruit material from Newton mesh-attribute reuse when a fruit rotates.
The standalone bilateral demo renders the stem and all twelve fruits as
separate shapes: the target and support visuals follow their dynamic Newton
bodies, while the other ten retain their source offsets on the static branch.
The legacy target/support/cluster outputs remain available to external users.
The older FBX and ZIP inputs remain supported for reproducibility.

The importer preserves the vertices duplicated at texture seams and converts
the source bottom-left UV convention to the runtime's top-left image convention
with `v_runtime = 1 - v_source`. Do not merge these vertices by position: the
single atlas contains both red fruit skin and green stem/leaf regions, so losing
a seam coordinate can visibly paint a tomato green. External replicated scenes
must keep the stem and individual fruits as separate render shapes and bind
`tomato_fruit_basecolor.png` to fruit shapes: object splitting alone did not
prevent rotated fruit from sampling green stem regions in Newton.

Run the scripted task:

```bash
uv run lycheesim run cherry-tomato-pick --headless

uv run lycheesim run cherry-tomato-pick \
  --frontend rerun \
  --live
```

## Plate-to-box harvesting experiment

The separate configuration preserves the earlier support/harvest-only versions:

```bash
uv run --locked lycheesim run cherry-tomato-pick \
  --config configs/cherry_tomato_pick_deposit.yaml --frontend rerun --live
```

This configuration records automatically on every run. HDF5 episodes are written
under `outputs/cherry_tomato_pick_deposit/`; `auto_increment: true` selects the next
unused `episode_XXXXXX.h5` and `overwrite: false` preserves existing recordings.
Robot states and both wrist cameras are sampled at 30 Hz; the central camera
remains at 10 Hz. Existing recording-time position, orientation and lighting
randomization remains enabled, with the episode number folded into the seed.
These randomized attempts are not guaranteed successes; use the collection
quality labels described below before selecting training demonstrations. For a nominal
preview without recording or recording-time randomization, append
`--set recording.enabled=false`. Recorded data and generated caches stay local
under ignored output/cache directories and are not part of the source push.

The deposit preset now starts **stem-up**, rotated 180 degrees about the
authored bunch's long Y axis around the target fruit. In
`configs/cherry_tomato_pick_deposit.yaml`, `scene.cluster_quaternion_xyzw`
is `[0, 1, 0, 0]`. Small per-episode rotations are composed with this nominal
pose, not substituted for it. Fruit visuals, collision bodies, the main-stem
grasp point and compliant attachments share the same pose. The target fruit
keeps its calibrated nominal position; the plate and boxes do not move.

When changing the nominal orientation, also update the world-space
`scene.stem_anchor`, `scene.support_tomato_offset_m`,
`attachment.break_direction_world` and `attachment.rest_quaternion_xyzw`.
Keep the authored **body-local** `stem_grasp_offset_m`,
`branch_collision_center_offset_m` and `attachment.tomato_local_point`
unchanged. The scripted pull already follows the observed stem-to-fruit axis
after lifting. A new orientation still needs physical grasp validation;
geometric scene acceptance is not a successful harvesting label. Older
recordings are unchanged and describe the earlier stem-down distribution.

Stem-up smoke check (2026-09-17, seed 3854939909, 12 simulated seconds):
all three RGB-D streams recorded, with no IK failure or interhand collision
block. However, the right thumb did not establish opposing contact and the
support gate remained closed; no lift/harvest/deposit success was validated.
The attempt was correctly rejected by the quality filter. Retune the right
grasp for this orientation before collecting a large training batch. Local
diagnostic recording: `outputs/stem_up_validation/episode_000001.h5`.

`tableware.py` generates a white shallow plate with configurable semi-axes, three white
open boxes (85 x 85 x 45 mm), and a decorative gold rack from the supplied photos.
These are approximate dimensions, not scanned or manufacturer-calibrated assets.
Each box has five separate convex collision shapes; the dish has a bottom and
48 convex rim sectors. Neither open vessel is represented by a single filled
convex hull. The containers are fixed; the decorative rack is visual-only.
Both the frontend and the three camera renderers receive the same visual meshes.
The tabletop is 4 mm lower in this variant, preserving the previous fruit support
height at the top of the plate floor. Tableware positions are fixed world XYZ
values in metres and are not randomized with the fruit.

To resize the plate, edit `tableware.plate_radii_m` in the deposit
configuration. These are **semi-axes** along world X and Y, in metres; the full
axis lengths are twice these values: `[0.14, 0.21]` gives 280 x 420 mm. For example,
`[0.13, 0.20]` produces a 260 x 400 mm plate. The current bunch/plate layout is
translated 70 mm toward the right-arm base (world -Y). To translate them together,
add the same XY offset to `tableware.plate_center_xy`, `scene.tomato_position`,
and `scene.stem_anchor`; leave the relative `*_offset_m` values unchanged.
For an additional rightward move of `d` metres, subtract `d` from the second
component (Y) of all three parameters. Add it to move left. For example, 5 cm
means subtracting `0.05`, not `5`. Preserve X, Z, and the relative offsets.
The grasp targets are calibrated from the translated fruit/stem positions.
The boxes and robot bases do not move. Resizing or translating may change rim
contacts and IK reachability, so rerun the nominal physics diagnostic after edits.

`tableware.tabletop` adds the photo-inspired wooden desktop below the plate and
boxes. `center_xy` is its world XY center, `size_xy_m: [0.62, 1.00]` gives the
full X/Y dimensions, and `thickness_m: 0.025` gives the board thickness. Dimensions
are estimates from the photo. Wood grain is deterministic colored mesh strips,
not a downloaded texture. Set `enabled: false` to hide the desktop.
The desktop is visual-only and reuses the existing infinite ground collision
plane, so it adds no duplicate coplanar contacts and does not move the props.
Its top is 0.2 mm above `floor_top_z_m - wall_thickness_m` to avoid depth flicker;
that support height must equal `solver.ground_z`. Physical finite table edges
and legs are not modeled. The desktop and grain are included in all three camera
renderers and the frontend, and exported separately as `tabletop*.obj`; they are
not added to the `tomato_boxes.obj` asset.

After actual force-triggered fracture, the left hand translates toward the
selected box while maintaining its grasp attitude. `deposit.target_box` is
zero-based: 0, 1 (middle), or 2. Recording defaults to uniform target selection;
set `recording.target_condition.mode=fixed` to honor `deposit.target_box`.
Only after the measured fruit is over
the usable opening, the wrist reaches its target, and grasp contact remains
confirmed does it open over 0.4 s. The fruit falls under gravity; there is no
fruit-hand weld or pose reset. Transport uses simulator ground truth, not vision.
Failure to reach the release conditions holds the hand closed and fails the
attempt; failure to settle after release also fails it. The existing interhand
guard remains active, but does not certify full-arm or hand-container clearance.

Placement requires the fruit sphere to fit inside the selected box and remain
below the rim with speed below 0.08 m/s for 0.3 s. Intentional release is not
labeled as a dropped-fruit failure in the recorder's `task_state`; its success
field now refers to placement in this variant. `placed_in_box` and `deposit_failed`
are recorded explicitly. Final verification also requires continued right-hand
support and absence of unsafe force, IK failure, or interhand guard failure.
One nominal physics run is not randomized data-collection validation.

### Randomized collection gates and goal conditioning

The deposit configuration enables four task-local collection features. Earlier
harvest-only configurations are not opted into the deposit-specific quality gate.

- `recording.scene_validation`: bounded rejection sampling of the full authored
  bunch footprint and conservative bounds for the two independent fruit
  colliders. Candidates must fit inside the plate with a 1 mm margin. A common
  upward correction of at most 15 mm can clear the plate surface/rim; the
  geometry then settles under physics, rather than independently relocating
  fruit or anchors. The minimum gap is limited to 25 mm. Nominal target, stem
  grasp, approach, lift and selected destination positions must pass configured
  workspace-box proxies. After 64 rejected candidates, construction fails with
  reasons; it never silently accepts an invalid candidate. These checks do not
  certify arm IK, all mesh intersections, or swept robot/container clearance.
  The unchanged runtime IK/contact/interhand guards still apply.
- `recording.randomization.visual`: one exposure gain (0.85–1.15) shared across
  the three RGB outputs for the whole episode, mild fruit-only albedo tint, and
  tabletop color gain. Exposure is an RGB approximation, not calibrated light
  power. Light direction remains shared; tint is applied to camera-rendered
  geometry without recoloring the stem. Geometry, depth, camera intrinsics,
  extrinsics and sampling rates are not randomized by these appearance knobs.
  Nominal tableware mesh/MJCF exports are reused; episode appearance lives in
  render colors and metadata, not thousands of duplicated geometry caches.
- `recording.target_condition`: `uniform` samples among `box_indices: [0, 1, 2]`
  independently of scene rejection and appearance draws. This is uniform
  sampling, not an exact balanced quota. The fruit target stays `tomato_10`.
  HDF5 `metadata.extra.target_condition` contains the target fruit, box ID,
  one-hot vector, box world position and instruction; the same one-hot goal is
  repeated in `robot/extra/target_box_one_hot`. **Supply that goal to the policy**:
  image-only training would otherwise associate indistinguishable scenes with
  contradictory destinations. Simulator object poses are privileged labels,
  not automatically deployable policy inputs.
- `recording.quality_filter`: all captured attempts stay in their original HDF5
  paths. `metadata.extra.quality` is the authoritative `accepted`/`rejected`
  label; an adjacent `episode_XXXXXX.quality.json` provides a readable summary.
  Acceptance requires passed initial checks, physical detachment and settled
  placement, continued lifted right-hand support, no recorded IK/interhand/
  support/deposit failure or excessive peak contact force, finite robot/action
  samples, consistent goal labels, and camera coverage/depth checks. Force peaks
  are latched by the physics scene, not just measured at the 30 Hz record rate.
  Invalid depth pixels remain NaN by design; each camera frame needs at least
  `min_valid_depth_fraction` positive finite pixels. This is a stream-health
  check, not proof that the tomato is visible or perception will succeed.
  Incomplete runs and missing camera streams cannot be accepted. Rejection does
  not delete data. No robot samples means there is no HDF5 episode to label.

The episode stores the seed, candidate seed, rejection counts, height correction,
effective pose/configuration, sampled appearance, calibration, source revision
and source hashes. Disable `recording.randomization.visual.enabled` to turn off
new appearance changes, or set `recording.randomization.enabled=false` to disable
pose/light/appearance sampling. Goal selection and initial validation are
independent switches. Disabling initial validation prevents quality acceptance.
Use separate output directories when collecting concurrently; automatic episode
number selection is intended for sequential runs, not cross-process reservation.

Run without a viewer, with the same automatic recording and randomization:

```bash
uv run --locked lycheesim run cherry-tomato-pick \
  --config configs/cherry_tomato_pick_deposit.yaml --frontend headless
```

Summarize accepted/rejected attempts and accepted coverage by box:

```bash
uv run --locked python -m demos.cherry_tomato_pick.quality \
  outputs/cherry_tomato_pick_deposit --summary
```

Create a training file list containing only explicitly accepted episodes:

```bash
uv run --locked python -m demos.cherry_tomato_pick.quality \
  outputs/cherry_tomato_pick_deposit > outputs/cherry_tomato_pick_deposit/accepted.txt
```

The selector reads HDF5 labels, not sidecars; older unassessed files are excluded.
It does not train a model or modify any episode. Split by whole episodes and
held-out scene/asset conditions, not neighboring frames. First measure pilot
success rates and rejection reasons before scaling up data collection.

### Attempt 100 episodes sequentially

```bash
uv run --locked python -m demos.cherry_tomato_pick.batch --attempts 100
```

This task-local wrapper runs the existing `lycheesim run cherry-tomato-pick` CLI
headlessly in a fresh subprocess for every attempt. Each child still renders and
records all three RGB-D cameras; headless only disables the interactive viewer.
The default config is `configs/cherry_tomato_pick_deposit.yaml`, with its 28 s
simulation duration. The ordinary single-episode command remains unchanged.

The count is **attempts, not successful demonstrations**. A failed harvest,
rejected initial scene, missing episode, or nonzero child exit consumes an attempt
and the batch continues. Seeds advance even when no HDF5 file was written.
A fresh random base seed is selected for each invocation; use `--seed 42` to
reproduce a batch's sampled conditions (not bitwise deterministic GPU physics).
The input YAML is snapshotted before starting, so edits to it during collection
do not change later attempts in that batch.

Every invocation creates a unique `batch_<UTC timestamp>_<suffix>/` directory
under the configured recording output directory, without replacing earlier
batches. Use `--output-dir outputs/my_collection` to select a different parent.
Each batch contains:

- `config.yaml`: the frozen input configuration;
- `episode_000001.h5` through `episode_000100.h5`, when an attempt captures data;
- individual `episode_XXXXXX.log` and recorder-generated quality sidecars;
- `summary.json`: progress, actual seeds/commands, exit codes, failure reasons
  and accepted counts by box, updated after each attempt;
- `accepted.txt`: absolute paths to this batch's accepted demonstrations only,
  also updated after each attempt. Training should use this list: batch
  acceptance additionally requires child exit code zero and no timeout.

Ctrl+C stops the batch and interrupts the active child to allow recorder cleanup;
no remaining attempts are launched. Already completed files are preserved.
There is no automatic resume: another invocation starts a fresh batch.
`--episode-timeout 900` is the default wall-clock limit per attempt, followed by
up to 30 s for graceful shutdown. A hung process is then killed; an incomplete
HDF5 may be unavailable and that attempt is rejected. Increase the wall limit on
slower hardware. Rejected episodes do not make a completed batch return nonzero;
inspect `summary.json`/`accepted.txt`, not just the shell exit code. Ctrl+C returns
130. Storage/controller errors stop the batch instead of silently dropping data.

For a short continuation/cleanup diagnostic (not harvest-success validation):

```bash
uv run --locked python -m demos.cherry_tomato_pick.batch \
  --attempts 2 --duration 0.07 --seed 42 --output-dir outputs/batch_smoke
```

Both attempts should be incomplete/rejected, and the second must still execute.
Omit `--duration` for real collection. A 100-attempt run can take hours and use
substantial disk space; the wrapper never launches this workload merely by
importing the module or requesting `--help`.

Export the reusable visual OBJ files, convex collision pieces and fixed MJCF:

```bash
uv run --locked python -m demos.cherry_tomato_pick.tools.export_tableware
```

The command prints a content-keyed directory under `outputs/tableware_assets`.
It contains `plate.obj`, `tomato_boxes.obj`, individual box/rack visual meshes,
collision OBJ pieces, and `tableware.xml`. Mesh vertices preserve the configured
world layout; import without recentering. The MJCF includes the decorative rack
as a non-colliding mesh. These generated assets are authored locally, with no
external model download or third-party mesh dependency.

GPU diagnostics (recording disabled):

```bash
uv run --locked python -m demos.cherry_tomato_pick.tools.check_tableware_drop
uv run --locked python -m demos.cherry_tomato_pick.tools.check_stem_lift \
  --config configs/cherry_tomato_pick_deposit.yaml --duration 28
```

The first tests independent gravity drops into all three boxes, not robot
harvesting. The second tests the full nominal sequence; add `--with-cameras`
for sensor sampling. Randomized recordings still need their own validation.

## Experimental stem-and-fruit support grasp

The previous carrier sequence remains the default configuration. The separate
`configs/cherry_tomato_pick_main_stem.yaml` now targets the middle of the
existing main stem, approximately 11 cm from the selected harvest fruit.
The right hand approaches fully open, then curls all four fingers opposite
the thumb around the stem and neighboring fruit. The visible stem is not
extended. The earlier end-of-bunch experiment is preserved separately in
`configs/cherry_tomato_pick_end_bundle.yaml`:

```bash
uv run --locked lycheesim run cherry-tomato-pick \
  --config configs/cherry_tomato_pick_main_stem.yaml \
  --frontend rerun --live
```

Closing does not start on incidental early contact. The reached wrist must
be within 4 mm and 0.08 rad of the grasp TCP; otherwise `support_reach_wait`
keeps the hand open. The four-finger curl is a photo-guided approximation,
not joint angles measured from the real robot.
The thumb MCP/IP and index PIP/DIP have deeper nominal flexion. With
`robot.right_distal_first_fingers`, these fingers curl their distal joints
before bringing the base joints inward; measured per-finger contact still
controls and can stop the motion. The other fingers retain linear closure.
The central grasp keeps wrist attitude fixed during its 12 cm lift; the
end-grasp experiment retains its former pitch-following behavior.
The right wrist and pregrasp are on the right-arm-base side of the main stem,
not beyond it toward the left arm. This is a change of approach side, not a
left/right hand-mesh swap.

This is an experimental controller, not a validated successful collection
policy. Nominal physics runs have physically lifted the bunch and detached the
target without triggering the interhand guard, but late left-hand fruit slipping
still fails final retention. Randomized harvesting and full-robot contact safety
are not validated. The left approach is gated on bilateral support contact and
measured whole-bunch clearance; an incomplete attempt must not be labeled a
successful demonstration. The headless task verification intentionally fails
when these conditions are not met.

`script.harvest_track_target_until_contact` updates the left approach translation
from the simulated target fruit through reaching and closure until the first
0.2 N fingertip contact. It preserves the left wrist attitude and then freezes
the reference for grasping/pulling, so it does
not chase contact-induced motion. This is privileged simulator-state tracking,
not camera-based perception. If the left hand stays still, inspect
`support_contact_wait`, `support_lift_wait`, or `support_lost`: the physical
support/lift gate has not passed or retention has failed. There is no automatic
timeout bypass into harvesting.

With `task.harvest_contact_control`, the left thumb/index/middle close
independently using a 30 ms force EMA and a configurable contact target
(`task.harvest_contact_target_N`, 5 N in the experiment). The
`harvest_contact_wait` phase keeps the wrist at the grasp pose until the
configured bilateral grasp hold has been measured; pulling cannot begin just
because the nominal closing time has elapsed. The experimental simulation
budget is 26 s to include these waits and the slower pull. The recorder includes
`left_force_closure` in thumb/index/middle order in addition to actual joint
states. The default carrier configuration does not enable this controller.

After lifting starts, losing bilateral contact for 80 ms latches
`support_lost`: both wrist targets and adaptive right finger closures stop
advancing, while the bunch remains freely simulated. This is failure handling,
not a recovery/regrasp policy. The recorder stores the flag and terminates the
failed episode; the left hand does not continue its scripted harvest.

In this mode the branch and ten non-selected fruits form a dynamic compound
body with individual convex colliders and distributed mass. The support and
target fruits remain separate dynamic bodies with reaction-coupled compliant
attachments. There is no hand-to-bunch weld or commanded branch translation.
This is still a rigid compound approximation, not twelve independently bending
pedicels. The robot hands continue to use the demo's kinematic IK control.

The middle-stem variant approaches from the target's outer side with a tilted
left wrist. The unused ring and pinky fingers are already tucked in the left
pregrasp pose, so they do not sweep through the support thumb during closure.
With `robot.minimum_interhand_distance_m` (5 mm), a CPU MuJoCo
convex-mesh query checks all authored hand links, including palms and proximal
links, at the candidate joint-space midpoint and endpoint. An unsafe command
is rejected and `hand_collision_blocked` latches; the last accepted joint poses
remain in place. The recorder stores this flag and `interhand_distance_m`
(a conservative sampled clearance capped at 50 mm; NaN when disabled), and
verification rejects blocked episodes. If a narrow-phase query returns an
ambiguous zero, disjoint bounding boxes supply a certified separation lower
bound. If rotated/tapered links have overlapping or narrowly separated boxes, a positive projection
gap along a unit convex-mesh face axis can also certify separation. No found
separating axis means the ambiguous zero still blocks motion; this fallback
does not lower the 5 mm threshold or assume that overlapping boxes are safe.
This guard does
not add dynamic hand-hand forces, cover the full FR3 arms/camera housings, or
certify continuous collision avoidance. Other configurations leave it disabled.

`script.harvest_pull_along_stem_axis` aligns the pull with the current attachment
axis when the bunch has been lifted, rather than with the fruit's bending
rotation. The middle-stem pull lasts 2.5 s. The axial break-force threshold,
30 ms EMA, and 80 ms overload duration are unchanged; no timed detachment is
introduced. After fracture, the hand makes a 20 mm smooth axial retreat over
1.5 s using continued physical contact. `task.harvest_retention_contact_target_N`
reduces the per-finger force target from 5 N to 2 N after fracture. When omitted,
the pre-fracture target is retained. The fruit remains dynamic with no fruit-hand
weld. This experimental route has produced physical detachment without triggering
the hand guard in the nominal scene, but late post-detachment slipping remains
unresolved. Full verification still requires final retention and rejects a run
that previously latched success but subsequently dropped the fruit.

The support controller filters per-finger contact using a 30 ms EMA and adjusts
each finger in `robot.right_force_fingers` separately (all five in the middle
grasp, three in the preserved end grasp). The middle-grasp configuration requires
at least three contacting fingers, including the thumb: thumb and strongest
opposing contact must each reach 3 N, and a further finger must reach 0.5 N.
The middle/ring/pinky closure commands must each reach 1.10 before confirmation
(`task.minimum_support_wrap_closure`; zero disables this pose gate). This is a
hand-specific wrap-pose prerequisite, not a replacement for contact sensing.
These EMA conditions must hold together for 250 ms before lifting. This prevents
an early thumb/index touch from starting the lift before the remaining fingers
wrap the bunch. `task.minimum_support_contact_fingers` defaults to two for
existing configurations. The middle-grasp controller uses a free closure rate
of 0.2/s and contact feedback gain of 0.015/(N s), set by
`task.support_free_closure_rate_s` and `task.support_contact_gain_s_N`.
Right contact feedback includes both pad
and distal-link contact against the branch compound and support fruit. A
separate signal measures contact against the stem collider alone. This differs
from the default mode's pad-only support-fruit signal and is identified in
episode metadata. Robot extras also include the branch pose (`xyz,xyzw`),
minimum fruit clearance in metres, and actual adaptive finger closure commands.
For stem-and-fruit mode, sustained support confirmation uses the controller's
30 ms EMA (3 N for 250 ms in this experiment), preventing single-step contact
jitter from repeatedly resetting the hold timer. The independent 80 ms
support-loss guard continues to use raw contact, not the filtered signal.

The experimental recording and Rerun paths are separate from default outputs.
Camera rates are unchanged. Do not use its recordings as successful training
data until whole-bunch lift, target detachment, and continued holding have been
validated across randomized episodes.

For a bounded GPU physics diagnostic with cameras and recording disabled:

```bash
uv run --locked python -m demos.cherry_tomato_pick.tools.check_stem_lift --duration 26
```

To verify only the right-hand lift, add `--verify-lift`. It stops at the first
left-hand approach and checks actual stem height gain, whole-bunch clearance,
support, contact-force limit, and IK status. It does not certify harvesting or
long-term retention. Add `--with-cameras` to retain the configured sensors.
Recording stays disabled in this diagnostic, so it tests the nominal scene,
not recording-time randomization.

`tools/fit_stem_grasp.py` prints approximate nominal closed-hand fitting values.
The middle-grasp fit checks five positions along the open approach plus
open, halfway, and closed configurations at the reached wrist while
keeping the open hand fixed at zero joint positions. The fitter uses
sphere/infinite-cylinder proxies and does not establish dynamic grasp stability.
The optional `tools/preview_stem_grasp.py` uses Matplotlib to render the authored
hand and fruit meshes at the open and nominal closed poses; this is a geometry
preview, not a simulation outcome.

## Collect a multi-rate episode

Enable the task-owned HDF5 recorder with YAML overrides. The headless command
below records the scripted harvest without also creating the generic offline
render cache:

```bash
uv run --locked lycheesim run cherry-tomato-pick \
  --headless \
  --set recording.enabled=true
```

The default `recording.auto_increment=true` advances after the highest existing
episode number: `episode_000001.h5`, `episode_000002.h5`, and so on. Run the same
command for every collection episode; no filename override is needed. Existing
files remain protected. Set `recording.auto_increment=false` to require an exact
unused path, or pass `--set recording.overwrite=true` only when replacement is
intentional. A live Rerun view can be combined with collection:

```bash
uv run --locked lycheesim run cherry-tomato-pick \
  --frontend rerun \
  --live \
  --set recording.enabled=true
```

The output uses the task-owned
`lycheesim.cherry_tomato_pick.multirate/v1` HDF5 schema. Robot state and each
camera stream have independent monotonic timestamps: robot state and both
Gemini 305 wrist cameras run at 30 Hz, while the fixed Gemini 435Le remains at
10 Hz. Images are stored only when a camera produces a new frame; the recorder
does not duplicate the 435Le image onto the 30 Hz robot timeline. It contains:

- 30 Hz robot timestamps, 54 joint positions and finite-difference velocities
  under `robot/`, ordered as left FR3 7, left Wuji2 20, right FR3 7, and right
  Wuji2 20, plus both TCP poses, hand closures, actions, and action timestamps;
- per-camera `timestamps`, `color`, `depth_m`, `world_from_camera`, and
  `valid_depth_pixels` under `cameras/gemini_435le`,
  `cameras/gemini_305_left`, and `cameras/gemini_305_right`;
- static intrinsics, distortion, extrinsics, and stream metadata inside each
  camera's `calibration/` group;
- causal per-robot-sample camera lookup indices and image ages under
  `robot/camera_index/` and `robot/camera_age_s/`;
- TCP/closure control targets, the 10-D task command, tomato poses and
  velocities, fingertip contact forces, attachment state, phase, and task
  outcome flags.

Training code should align a robot timestamp with the newest camera timestamp
that is not later than it. A normal completed run marks the final robot
transition terminated when the task reaches success, drop, or IK failure; an
interrupted or otherwise incomplete run is still finalized and marked
truncated.

The scene also contains a fixed Orbbec Gemini 435Le RGB-D camera on a white
centre pedestal between the two arms, matching the mounting relationship in
the supplied laboratory photographs. Its procedural enclosure follows the
[published 138.5 x 40.5 x 70.0 mm envelope and 95 mm stereo baseline](https://www.orbbec.com/gemini-435le/).
The photo-derived optical pose, look-at target, mount dimensions, 94 x 68 degree
field of view, clipping range, resolution, and sample rate are all owned by the
top-level `camera` section in `configs/cherry_tomato_pick.yaml`. The default
320 x 200 Newton preview publishes aligned color and forward-depth images under
`sensors/gemini_435le/`; replace the YAML pose with a measured hand-eye
calibration before treating the output as real-to-sim extrinsic ground truth.

Two Orbbec Gemini 305 RGB-D cameras are mounted beside the left and right
Wuji2 wrist connections through photo-matched outboard brackets. The optical
origins sit 80 mm outboard of their wrist frames, leaving clearance around the
FR3 flange and link housing. Their procedural
models use the [published 42 x 42 x 23 mm housing, 18 mm stereo baseline,
94 x 68 degree color field of view, and 4 cm minimum operating
range](https://www.orbbec.com/gemini-305/). Both cameras move rigidly with their
respective Wuji2 root frames and publish independent color and forward-depth
images under `sensors/gemini_305_left/` and `sensors/gemini_305_right/`. The
photo-derived `wrist_cameras.mounts` transforms are initial estimates; replace
them with measured camera-to-wrist extrinsics before collecting calibrated
data.

The sequence follows the current upstream behavior design:

1. both arms approach the bunch;
2. the right Wuji2 extends its index finger onto the opposite endpoint and
   keeps the other fingers folded away;
3. the left Wuji2 reaches and forms a thumb/index/middle fruit grasp;
4. the left arm lifts until the pedicel fails;
5. the detached fruit remains physically retained in the closed left hand.

The pedicel has no synthetic line primitive. Before failure, only the purchased
stem mesh is visible. After failure, the target mesh separates from the bunch;
there is no green line or other visual connection left behind.

## Detachment model

Newton does not currently expose the same runtime breakable fixed-joint API as
the source Isaac scene, so the demo uses a task-local six-degree-of-freedom
compliant attachment. Before failure, a spring-damper wrench keeps the source
attachment point and rest orientation on the stem. It is anisotropic: axial
stretch is stiffer than lateral displacement, twist is stiffer than bending,
and all four modes have independent damping. This lets the fruit deflect and
return under incidental side contact without weakening the intended pull
direction. Compression, lateral force, bending, and twist do not cause
fracture. The raw normalized fracture load is:

```text
max(0, axial tensile force) / 2 N
```

The runtime applies a 30 ms exponential moving average to that load and only
fractures after the filtered value remains at or above 1.0 continuously for
80 ms. The left three-finger grasp and right support press start tangent to the
nominal tomato collision sphere; a 1.5 mm per-surface solver margin establishes
contact without authored geometric pre-penetration. Its lower 4,000 N/m contact
stiffness avoids replacing the old overlap with another hard impulse. The
separate 1 mm gap only widens contact detection and does not generate force.

The 2 N tensile threshold comes from the upstream standalone detachment
prototype. Its 0.20 N·m value remains in the configuration for provenance but
torque is no longer part of this first-stage fracture rule. The newer full
Isaac scene contains much larger scene-scale values (128 N and 12.8 N·m);
those are not copied into this SI-scaled 20 g Newton fruit because they do not
reproduce the validated endpoint test.

Failure is armed only after real opposed left-hand contact is held for 0.10 s
and the controller commands a deliberate lift or twist. Detachment is
irreversible. Success additionally requires right-side support, at least 20 mm
of post-fracture target displacement, and 0.10 s of post-fracture in-hand
retention. Pre-fracture elastic motion is excluded from harvest progress and
therefore cannot produce RL carry reward.

## RL design and upstream comparison

Task ID: `LycheeSim-CherryTomatoPick-v0`

The latest upstream tomato contribution is a behavior/waypoint design, not a
tomato-specific RL task. Its `policy/` directory deploys a DP3 checkpoint from
the older lemon-picking workflow. That distinction determines what is safe to
merge:

| Contract | Upstream DP3 path | LycheeSim tomato task |
| --- | --- | --- |
| Learning mode | Offline imitation checkpoint inference | Online PPO environment |
| Task data | Existing lemon demonstrations/checkpoint | Tomato-specific physics, rewards, resets, and terminations |
| Observation | 1024 fused XYZRGB points plus joint history | 61-D actor proprioception, task geometry, attachment state, and tactile force; 76-D privileged critic |
| Action | 29 absolute joint targets: left FR3 7 + left Sharpa 22 | 10 bilateral task-space velocity/yaw/closure commands |
| Right arm | Deterministic scripted motion outside policy action | Policy-controlled support arm |
| Success definition | External lemon workflow checks | Support + grasp + load + irreversible break + retained separation |

The merge therefore imports the new object split, endpoint mapping, hand
profiles, `xyzw` orientations, and press/reach/grasp/lift phase ordering while
retaining LycheeSim's trainable bilateral control contract. Reusing the lemon
checkpoint or silently changing the tomato action space to 29 left-side joint
targets would not be task-equivalent. A future vision-policy adapter can add
the upstream-style dual-camera point cloud as another observation group while
keeping the current physics and success semantics.

The normalized action is:

| Index | Command | Scale |
| --- | --- | --- |
| 0..2 | left harvest TCP linear velocity XYZ | 0.10 m/s |
| 3 | left TCP world-Z yaw rate | 1.5 rad/s |
| 4 | left Wuji2 closure rate | 2.0 s⁻¹ |
| 5..7 | right support TCP linear velocity XYZ | 0.10 m/s |
| 8 | right TCP world-Z yaw rate | 1.5 rad/s |
| 9 | right Wuji2 closure rate | 2.0 s⁻¹ |

Rewards are staged around right-side support, left-hand reach and grasp,
increasing attachment load, detachment, post-fracture separation, retained
success, and safe actions. Episodes terminate on success, a dropped fruit,
invalid kinematics/numerics, or timeout. Training resets randomize bunch XY,
fruit mass, and break force.

Run a framework-neutral smoke rollout:

```bash
uv run lycheesim rl smoke \
  --registration-module demos.cherry_tomato_pick.rl \
  --task LycheeSim-CherryTomatoPick-v0 \
  --config configs/cherry_tomato_pick.yaml \
  --device cuda:0 \
  --num-envs 1 \
  --policy random \
  --steps 20
```

Run one bounded PPO update after the smoke passes:

```bash
uv run --extra rl lycheesim rl train \
  --registration-module demos.cherry_tomato_pick.rl \
  --task LycheeSim-CherryTomatoPick-v0 \
  --config configs/cherry_tomato_pick.yaml \
  --device cuda:0 \
  --num-envs 4 \
  --num-steps-per-env 16 \
  --max-iterations 1
```

The local `LycheeSim-CherryTomatoPick-v0` route still owns one independent
Newton model per environment and remains a correctness-oriented small-batch
implementation. LycheeSim also exposes
`lycheesim.tasks.cherry_tomato_pick.CherryTomatoAttachmentFacade` plus a
content-addressed single-root carrier view and the tracked detailed visual-asset
paths for LitChiMimic's replicated Newton Environment. That view takes nominal
arm/hand servos from [robot_actuators.xml](assets/robot_actuators.xml), with one
position actuator per controlled joint. It preserves the shared carrier's
geometry, passive properties and coupling constraints; the shared MJCF and
standalone demo remain unchanged. The task-owned facade is
device-native fixed-weld fracture state only: it consumes solver-owned
activation and reaction views and never applies a body wrench. Unlike this
standalone demo's separate force-or-torque spring rule,
LitChiMimic decomposes the measured weld reaction against a configured world-
frame break direction. Its flat-bunch task breaks only under upward tensile
load, publishes lateral shear separately, and does not allow shear or wrist
torque to detach the fruit. The visual accessor
lets an external one-world recorder attach the separate stem and twelve fruit
meshes as non-colliding render geometry without replacing analytic training
collision. The source provides explicit attachment anchors for
`tomato_02`, `tomato_07`, `tomato_09`, and `tomato_10`; metadata clearly marks
the other eight all-fruit task anchors as reproducible closest stem/fruit
source-vertex pairs rather than source-authored ground truth. In LitChiMimic,
all twelve fruits are independent physical bodies and balanced episode targets;
LitChiMimic owns its separate observation, reward, reset, terminal, metric, and
native PPO contracts and does not import this demo's RL runtime.
