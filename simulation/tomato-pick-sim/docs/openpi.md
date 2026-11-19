# OpenPI policy evaluation

Run a JAX/Orbax checkpoint directly; no `.pt` export is required. Keep OpenPI
and the simulator in separate virtual environments. This adapter implements
the WebSocket/MessagePack NumPy wire contract used by
[OpenPI](https://github.com/Physical-Intelligence/openpi), without installing
`openpi-client` and its NumPy <2 dependency in this NumPy 2 simulator.

## Start the model server

In the OpenPI checkout, with the trained `pi05_lycheesim_tomato_lora` config:

```bash
export OPENPI_DATA_HOME="$PWD/.cache/openpi"
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
  --port=8000 policy:checkpoint \
  --policy.config=pi05_lycheesim_tomato_lora \
  --policy.dir=checkpoints/pi05_lycheesim_tomato_lora/tomato_lora_v1/9999
```

Select the numeric checkpoint directory containing both `params/` and
`assets/`, not `params/` itself. The server loads normalization statistics
from that checkpoint. The upstream script listens on all interfaces and has
no authentication: use only on a trusted/private network, with a firewall or
a localhost-bound server. The evaluator defaults to `ws://127.0.0.1:8000`.
Do not run training concurrently on the same GPU. Disabling JAX preallocation
leaves room for simulation, but is not a hard GPU-memory limit.

## Run one attempt

In this simulator checkout:

```bash
uv sync --locked --extra openpi
uv run --locked --extra openpi tomato-sim eval \
  --checkpoint-label tomato_lora_v1/9999 \
  --execute-steps 4 --duration 28 \
  --set recording.seed=42 --set recording.vary_seed_by_episode=false
```

Use `--frontend rerun --live` to view the rollout, or
`--frontend rerun --out-path outputs/policy_preview.rrd` to save a visualization
(choose a new RRD filename each time). Default is headless; RGB-D sensors still
render. Runner flags (`--config`, `--set`, `--duration`) pass through unchanged.
`--request-timeout 180` bounds each metadata/inference receive; connection
establishment is limited to 10 seconds. First inference may compile for much
longer than later calls. There are no automatic connection retries.

Automatic episode numbering, scene/appearance randomization and quality
filtering remain enabled. By default the target box is sampled by the existing
collection configuration. To fix a goal, use BOTH:

```bash
--set recording.target_condition.mode=fixed --set deposit.target_box=1
```

Prompt indices 0/1/2 mean nearest/middle/farthest respectively, exactly as in
the training converter. For a reproducible scene, specify the seed and disable
`vary_seed_by_episode`; otherwise the episode number contributes to the seed.

## Observation and action contract

- Images: three RGB uint8 HWC `(200, 320, 3)` arrays under `images.base`,
  `images.left_wrist`, `images.right_wrist`. No depth is sent to this policy.
  The base camera runs at 10 Hz, wrists at 30 Hz; use the latest available
  causal frame, as in training. No client resize or normalization is applied.
- State: float32 `(16,)`, measured left XYZ + xyzw, measured right XYZ + xyzw,
  then the two current commanded Wuji closure scalars. These are not the
  per-finger force-controller closures or joint positions.
- Reply: finite `(16, 16)` `actions`, already unnormalized by the server.
  Each action is absolute world-frame TCP poses (meters, xyzw), followed by
  left/right closures (0=open, 1=closed). It is not a velocity or delta.
- By default execute the first 4 targets at 30 Hz simulation time, then
  request a new chunk. `--execute-steps` accepts 1 through 16. Hold targets
  between control ticks; synchronous inference pauses physics. This measures
  simulator closed-loop behavior, not real-time hardware control latency.
- Normalize quaternions and align their signs to the measured TCP, clamp XYZ
  to configured workspaces and closures to [0,1]. Reject zero quaternions,
  nonfinite values and wrong dimensions. Clipping is counted in the report.

The first physics tick holds the initialized targets before the first camera
capture/inference. A control tick records the observed closure *before* the
new action; `robot/action` records the target held over the following robot
sample interval. This preserves the converter's state-to-next-action semantics.

## What still runs in the simulator

The policy replaces scripted reaching, ground-truth tomato retargeting,
pull/twist, retreat and deposit trajectories. The existing absolute TCP IK,
finger contact-force control, collision checks and physical detachment remain.
The existing post-detachment hand-retention blend also remains; it is low-level
hand control, not a substitute transport trajectory. Observations sent to the
model contain no ground-truth fruit poses, contact forces or scripted phase.

Policy mode stops on IK failure, hand collision, excessive recorded peak
contact force, loss of established right-hand support, invalid output or
transport failure. The right-support loss guard becomes active once measured
support is ready and bunch clearance reaches 5 mm. Teleoperation commands
are rejected during evaluation. These checks are simulator task guards, not
real-robot safety certification; workspace clipping does not ensure reachability.

The policy-only peak-force stop threshold defaults to **2000 N** for simulation-only
debugging with `--contact-stop-force-n 2000`. This deliberately permissive setting
is only for observing failed-policy behavior, not for certifying safety or success.
This is not a validated safe force for fruit
or real hardware. Use `--contact-stop-force-n 35` to restore the earlier strict
stop behavior. Values must be finite and positive; the guard cannot be disabled
with infinity. Other stop conditions remain unchanged.

To observe the rest of an unstable simulation, explicitly add
`--disable-contact-stop`. Finite forces above the configured stop threshold no
longer end the rollout. This flag overrides the threshold for early stopping,
not the physics or finger force controllers. NaN/Inf forces, collisions, IK
failure, loss of established support and inference errors still stop the run.
The duration limit and physical completion condition also remain active.
Omit this flag on the next launch to restore force-based early stopping.
This is simulation-only debugging, not a real-robot mode.

Reports identify this mode with `contact_stop_enabled: false` and
`contact_stop_force_N: null`. The measured forces are still recorded and the
35 N quality limit is still checked at finalization, so the process may exit
with verification failure **after** the rollout rather than at first overload.

This stop threshold is separate from `task.unsafe_contact_force_N` (35 N in
the default config). Final scene verification and dataset quality filtering
still enforce that original limit. A rollout may continue beyond 35 N to show
later behavior, but it cannot become quality-accepted merely because the stop
limit was raised. Both limits are recorded in the policy evaluation metadata.
Scripted `run`/`batch` behavior and existing recordings are unchanged.

After physical detachment, predicted left closure <=0.1 enables the placement
observer. Closing the hand again resets the settling interval. Success still
requires the tomato to remain physically inside the
selected box at speed <0.08 m/s for 0.3 s, with safe contact and right support.
No scripted release or transport is called. A safety stop or timeout is not
success. The configured duration is a simulation-time maximum.

## Optional grasp-before-lift mode

Add `--grasp-before-lift --grasp-wait-timeout 8` to `tomato-sim eval` for hybrid
model + grasp-interlock control. It is **off by default**; ordinary model-only
evaluation and scripted collection are unchanged. The timeout is in simulation
seconds, must be finite and positive, and does not replace the overall duration
limit. No training changes or checkpoint conversion are required.

The first right-hand closure command >= `task.minimum_support_grip` (currently
0.72) starts the wait. The model remains responsible for initiating closure;
this mode does not force the hand shut or identify a grasp from an object pose.
During the wait, the right TCP Z target cannot exceed the measured height at
entry. Downward targets lower this ceiling, so repeated tiny upward targets
cannot bypass it. X/Y, orientation, left-arm targets and both hand commands
continue to come from the policy. Finger contact-force control is unchanged.
This constrains TCP targets, not exact physical trajectories or all fingertip
motion: changing wrist orientation can still move fingers vertically.

Every physics tick checks the existing backend `support_engaged` signal. That
requires sufficient commanded grip, grasp proximity, wrapped support fingers,
opposed and multi-finger contact, and the existing sustained-contact interval.
With the default task config, middle/ring/pinky closure must reach 1.10, at
least three fingers must exceed 0.5 N, opposed contact must reach 3 N, and the
contact hold interval is 0.25 s. Closure=1 alone never releases the gate.

On confirmation, discard the cached action prefix. Keep holding the capped
target until the next normal 30 Hz control tick obtains a fresh prediction;
then release the gate. If confirmation disappears before that tick, relock
without resetting the original timeout. A fresh prediction that opens the
support hand below the trigger also stays gated; it cannot open and lift in
the same action. Timeout stops the attempt with
`right grasp confirmation timed out`, retaining a rejected diagnostic recording.
Opening the hand again during the wait also does not reset the deadline.
After the first confirmed release, the existing support-loss guard remains
responsible for detecting a subsequently lost grasp.

This can prevent premature lift but cannot repair an unreachable/misaligned
grasp or guarantee success. It remains compatible with the explicit
`--disable-contact-stop` debugging flag; that flag does not bypass the grasp
gate, its timeout, or any other stop condition.

Reports and HDF5 policy metadata label the mode `hybrid_grasp_before_lift` and
include the gate phase, trigger, timeout, ceiling, constrained-target count and
transition timestamps. Per-frame `robot/extra/right_lift_gate_phase` indices
map to `disabled, approach, waiting, confirmed, released, timed_out` (0–5).
`right_lift_gate_ceiling_m` is zero before a gate ceiling exists; interpret it
together with the phase. Standard `robot/action` records the **constrained,
executed target**, not the model's unclamped prediction. Quality filtering and
success criteria are not relaxed; hybrid results should be reported separately
from model-only results.

## Outputs and interpretation

`outputs/openpi_eval/episode_XXXXXX.h5` contains the regular multirate recording.
Adjacent `.quality.json` and `.evaluation.json` report physical quality,
verification errors, goal, seed, action/inference counts, per-request wall-clock
latency, clipping and server metadata. The checkpoint label is user supplied;
it is not proof of server identity. Actual server metadata is stored separately.
The regular HDF5 source revision/hash metadata remains intact.

Exit code 0 requires both scene verification and quality acceptance. Failed
attempts are retained for diagnosis, including partial recordings. Connection
or scene-construction failures occur before episode setup and may have no
recording. An interrupted attempt is not verified success. Do not infer a
success rate from one attempt or from low training loss; evaluate a held-out,
fixed seed/goal suite for policy comparisons.

## CPU regression tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked --extra openpi pytest
git diff --check
```

These tests verify the transport/action contract and old simulator paths; only
a completed GPU rollout can measure physical task outcomes.
