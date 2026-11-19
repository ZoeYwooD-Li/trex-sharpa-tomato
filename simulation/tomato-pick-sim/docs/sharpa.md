# Bilateral Sharpa Wave integration

The default CLI and batch preset is `configs/cherry_tomato_pick_sharpa.yaml`.
Run from this checkout:

```bash
uv run --locked tomato-sim run --frontend rerun --live
```

The preset preserves recording, randomization, target conditioning, unique
output allocation and quality filtering. Outputs use
`outputs/cherry_tomato_pick_sharpa/`. Historical Wuji2 presets remain explicit
options and retain their original 20-joint hand contracts.

## Model and control

`demos/cherry_tomato_pick/sharpa.py` loads separate authored left/right Wave
models. Each has 22 joints, including thumb MCP abduction and pinky CMC.
MuJoCo supplies FK/IK; Newton uses the ten elastomer meshes and ten distal
meshes for bilateral contact. Full visible hand hulls remain in the existing
interhand clearance guard. This is not full dynamic palm/proximal contact.

Both hands include a flange/wrist stack and extend along local +Z. The FR3
mount rotation is identity, and target attitudes and wrist-camera extrinsics
are transformed from the historical -Z Wuji2 convention. Finger-centroid
calibration places the grasp targets; these are initial grasp profiles, not
an optimized tomato grasp. Closure feedback addresses joints by the authored
Sharpa finger prefixes and includes all 22 joints per hand.

Robot states now have 58 entries: left FR3 7, left Sharpa 22, right FR3 7,
right Sharpa 22. TCP actions remain 16 entries. HDF5 joint, action and camera
metadata identify Sharpa. Wuji2 policies require embodiment-specific validation
and potentially retraining before use with these hands.

## Sources

Assets are local to `demos/assets/robots/sharpa_wave/`; no runtime import or
asset reference points at another checkout. See that directory's
`PROVENANCE.md`, `LOCAL_ADAPTATION.md`, licenses and `snapshot_sha256.json`.
The authored right model is preserved; a separate derived XML adds the same
29.5 mm wrist stack as the left. The right source's upstream revision is
unknown and is not inferred from the left asset revision.

Initial open/grasp profiles were adapted from the local LycheeSim
`demos/lemon_slice_pick/names.py` snapshot:

- Source checkout revision: `2d75161980a4f2ab6bf9e30dd746aef7046a3d6d`.
- Profile file SHA256: `d1f026a2d98d4c05a47a821a5687a50478223195ae0307d771f86e7b90b37190`.

## Verification (2026-09-20)

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest`: 266 passed.
- `git diff --check`: passed.
- NVIDIA GeForce RTX 5090, Warp 1.16.0, Newton 1.5.0, CUDA driver 13.0.
- A nominal 0.1-second physics/camera startup run completed its steps.
- A 3-second randomized recording smoke (seed 42, dt 0.005, 600 steps)
  completed in 30.33 seconds wall time. No IK failure, support loss or
  interhand collision guard activation occurred. The observed hand clearance
  stayed at the guard's 0.05 m query cutoff (not an exact larger distance).
- Artifact: `outputs/cherry_tomato_pick_sharpa/episode_000001.h5`.
  The HDF5 contains 90 robot states of width 58, 89 actions of width 16,
  30 fixed-camera frames and 90 frames per wrist camera.
- The smoke ended during approach, before grasp contact. Task verification
  returned failure and quality correctly rejected it as incomplete, not
  detached/placed, and right support not ready. It is not training success.

A complete grasp/lift/pick/deposit and its success rate have not been measured.
The short checks establish startup, early motion, RGB-D and recording only.
