# Historical forward-flexed collection posture

Select `configs/cherry_tomato_pick_collection_baseline.yaml` explicitly. This
restores the original collection hand and its matching scene/workflow instead
of reusing the negative-flexion development candidates. It is an opt-in preview
preset, not a replacement for the default stem-up deposit or OpenPI policies.

## What is restored

- Start with all 20 right-hand joints at zero (open). No intermediate reverse
  preshape is used. Reach the grasp pose before beginning contact-controlled
  closure.
- Restore the original wrist quaternion and grasp-center bias together with
  the original hand. Do not flip the hand mount or mirror the official mesh.
- Flex the finger joints forward: original closed PIP/DIP values are positive,
  approximately 0.73--1.29 radians. Abduction joints can legitimately be negative;
  they are not flexion joints.
- Wrap with thumb, index, middle, ring and pinky. Keep the original thumb/index
  distal-first sequence, measured-contact controller and support gate, followed
  by lift, left-hand picking and deposit.
- Keep the original 35 N contact-force guard, friction, mass and collision
  geometry. There is no attachment shortcut or change to model weights.

The problematic `radial4_*` development candidates used approximately -0.9 rad
PIP/DIP values. Official joint limits permit some hyperextension; satisfying
those limits and fingertip-position fits does not ensure a natural grasp or
the intended pad-side contact. Do not use those archived candidates as the
natural-hand preset. The new regression tests cover open initialization and
nonnegative flexion throughout contact-closure interpolation.

## Important orientation difference

The original collected scenes were **stem-down**. This preset makes the original
identity cluster quaternion explicit and restores the corresponding anchor,
attachment and grasp settings. It does not claim to solve the stem-up task by
changing only the fingers. The current stem-up default remains unchanged.

## Preview one attempt

From the standalone project root:

```bash
uv run --locked tomato-sim run \
  --config configs/cherry_tomato_pick_collection_baseline.yaml \
  --frontend rerun --live
```

Recording is disabled in the preset, which also disables recording-time scene
randomization. This is a nominal-pose preview. Camera sensors remain available
for visualization. If recording is enabled deliberately in a future validated
experiment, its unique episode allocation goes under `outputs/collection_baseline/`;
the original recordings are not overwritten. Do not start a batch solely on
the basis of a visually plausible posture.

## Provenance and verification

The source is the saved `config.yaml` from the original collection batch
`batch_20260913T060432Z_0mg5ge_s`. That historical batch accepted 48 of 100
attempts. This is historical evidence, not a current success-rate measurement.
The source configuration, accepted-episode metadata, exact configuration
differences and fresh bounded-run reports are archived under
`outputs/natural_grasp_20260918/`. See `RESULTS.md` there for the measured current
runtime outcome. The replay disables cameras/recording and uses the saved
effective scene directly; it does not draw a new scene from the seed.

The preset differs from the saved batch configuration only by disabling
recording for preview, selecting separate output paths and making the implicit
identity cluster quaternion explicit. No existing dataset is edited. Static
hand renders inspect the authored geometry, not physical grasp success.

Fresh 2026-09-18 verification: the nominal preset completed lift, detachment and
deposit in a bounded camera-disabled run, with 19.59 N peak contact force. A
second replay of a historical accepted scene lifted but lost support during
picking. Thus this is a usable posture baseline, not a validated randomized
collection success rate. The full unit suite passed 251 tests.
