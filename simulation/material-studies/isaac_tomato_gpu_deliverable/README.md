# Isaac GPU tomato load/deformation demo

This is an isolated one-tomato + ground scene, created and run with Isaac Sim 5.1 / PhysX volume deformables on the RTX 5090. The original picking scene is untouched. The tomato uses one tetrahedral mesh for both simulation and collision; in this environment that avoids the `collElementIndicesBuffer` GPU tensor-view error seen with separate tetra meshes.

## Run a visible live demo

From the workspace root:

```bash
/path/to/local/anaconda3/envs/RoboDojo/bin/python outputs/isaac_tomato_live_sharedmesh.py \
  --live --force-n 0.25 --live-period-seconds 30 \
  --output outputs/isaac_tomato_live_gpu
```

This launches Isaac Sim, keeps the viewport running, and slowly cycles the gravity-equivalent load from the tomato's weight (0.1962 N) to 0.25 N. Close the Isaac window to stop. The generated stage sets `physxScene:enableGPUDynamics = 1`; no separate GUI toggle is needed. `--force-n` is the commanded equivalent load in this demo, not a measured tomato–ground contact force.

## Reproduce the short headless validation run

```bash
/path/to/local/anaconda3/envs/RoboDojo/bin/python outputs/isaac_tomato_live_sharedmesh.py \
  --headless --output outputs/isaac_tomato_gpu_deliverable
```

The run writes `small_tomato_compression.usda`, `compression_trace.csv`, and `validation.json`.

## Parameters and measured run output

The successful headless run used one physics step of 1 ms, 128 solver position iterations, `E=5 MPa`, `nu=0.42`, elasticity damping `0.005`, contact offset `0.2 mm`, rest offset `0`, tomato radius `17.2 mm`, and mass `20 g`. E, nu, damping, radius and mass are assumptions or inherited project settings; none are calibrated against tomato measurements. `E=5 MPa` is a numerically stable demo setting, not a claimed tomato property.

At 0.2 N commanded gravity-equivalent load for 0.1 s, then release for 0.1 s, the height-reduction proxy peaked at 1.001 mm (2.91%) and returned to 0.0014 mm at the end. The lowest nodal z was −0.0017 mm. This demonstrates that PhysX ran and returned changing soft-body nodes; it does not validate the material or force response.

## Force limitation

The CSV's `measured_contact_force_N` field is intentionally blank and validation JSON records it as null. Isaac Sim 5.1's volume-deformable tensor view in this setup did not expose a usable rigid–deformable contact reaction. The only force-like input is the commanded gravity-equivalent load. Do not use that input as an independent measured force label, and do not infer damage from the 15% screening threshold; no failure law or damage observation is implemented.

A real platen-controlled compression and a verified reaction-force channel remain outstanding. The separate platen experiment is not represented by this deliverable.
