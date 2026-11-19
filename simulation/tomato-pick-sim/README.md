# Tomato Pick Sim

A standalone extraction of the current LycheeSim tomato-harvesting working
tree: dual Franka FR3 arms, left/right Sharpa Wave hands, a tomato bunch
with compliant breakable pedicels, plate, three receiving boxes, tabletop,
three RGB-D cameras, automatic HDF5 recording, randomized collection and
quality filtering. No original LycheeSim checkout is required at runtime.

The existing `lycheesim` Python namespace and dataset schemas are retained to
avoid changing numerical behavior or breaking dataset readers. This is not
the complete upstream simulator. See [provenance](provenance/README.md).

The default is now `configs/cherry_tomato_pick_sharpa.yaml`. Each hand has
22 joints; recorded bilateral robot positions have 58 values (7 + 22 + 7 + 22).
The TCP action stays at 16 values, with Sharpa closure labels. This changes the
hand embodiment: previous Wuji2 policies and accepted demonstrations do not
certify Sharpa performance. The historical deposit and experimental Wuji2
presets remain available when selected explicitly.

See [Sharpa integration and validation](docs/sharpa.md) for asset provenance,
mounting, contact geometry and current validation limits.

## Installation and daily use

Run all commands from this directory. Use Python 3.11 and `uv`. Full physics
and camera rendering require a compatible NVIDIA GPU/driver; the pinned Torch
build uses CUDA 13.0. Dependencies are installed into this project's own `.venv`.

```bash
uv sync --locked
uv run --locked tomato-sim run --frontend rerun --live
```

This runs **one attempt**, using
[`configs/cherry_tomato_pick_sharpa.yaml`](configs/cherry_tomato_pick_sharpa.yaml).
Automatic recording is enabled by default, with incrementing episode names.
Outputs are local to this project, under `outputs/cherry_tomato_pick_sharpa/`.
Scene and visual randomization, target-box conditioning, and quality checks
remain enabled as configured. A failed attempt is not a successful training
demonstration; inspect its quality report before accepting it.

Once the detached tomato remains inside the selected box below the settling
speed threshold for 0.3 seconds, the scene captures a final RGB-D frame from
each camera and a terminal robot sample, then ends the run and saves the HDF5
and Rerun recording immediately. The configured duration is a maximum, not a
mandatory post-success recording time. The terminal sample can be off the
normal 30 Hz / 10 Hz cadence; use the stored timestamps. Quality checks still
run: stopping does not override rejection criteria. In batch mode the next
attempt then starts. `--attempts 100` means 100 attempts, not 100 accepted demos.

For 100 independent attempts, without opening a viewer:

```bash
uv run --locked tomato-sim batch --attempts 100
```

Each invocation creates a separate batch directory. The batch summary and
`accepted.txt` list the accepted data; rejected attempts remain for
diagnosis. Use `--seed 42` to select a reproducible batch seed. Run
`tomato-sim batch --help` for timeout, duration, configuration and output options.

```bash
uv run --locked tomato-sim quality outputs/cherry_tomato_pick_sharpa/BATCH_DIRECTORY --summary
```

The familiar command also works in this checkout and selects the deposit
configuration by default:

```bash
uv run --locked lycheesim run cherry-tomato-pick --frontend rerun --live
uv run --locked python -m demos.cherry_tomato_pick.batch --attempts 100
```

Disable recording explicitly when only previewing:

```bash
uv run --locked tomato-sim run --frontend rerun --live --set recording.enabled=false
```

## Scripted main-stem grasp

For **stem-up with the original forward-flexed hand**, select the
[stem-up natural-hand preset](docs/stem_up_natural.md). It preserves the right-hand
shape and wrist orientation while recalibrating grasp-center height and the
left-hand harvesting attitude/contact feedback. It is a
separate, recording-disabled preview; see its measured validation limits before
attempting collection.

For the original **forward-flexed five-finger posture**, use the explicit
[historical collection baseline](docs/collection_baseline.md). It restores the
original wrist and hand together, starts open, and disables demonstration
recording for preview. This fallback is **stem-down**, not the current stem-up
experiment, and does not replace the default configuration. Historical accepted
episodes do not certify success with the current runtime.

The opt-in [main-stem pinch](docs/stem_pinch.md) uses staged preshaping,
end-on insertion and measured-contact lift gates. Select its configuration
explicitly; it does not replace the default deposit preset or the trained
OpenPI controller. Check the run-specific validation scope before collection.
The separate `configs/cherry_tomato_pick_stem_midpoint.yaml` development preset
targets the stem center and adds measured bunch-workspace and left-hand
table-clearance gates. Read the linked notes for its current physical-test
status; it is not an accepted-data collection preset.

The separate [stem/fruit wrap experiment](docs/bunch_wrap.md) allows central
fruit support while retaining measured grasp/trial-lift gates. Its default
lift-only diagnostic keeps the left hand parked. It remains a development
preset requiring physical validation, not an accepted-data collection preset.

## Evaluate a trained OpenPI policy

See [OpenPI deployment](docs/openpi.md) for the two-environment launch commands,
absolute action contract, safety stops and evaluation artifacts. Install only
the optional transport dependencies here with `uv sync --locked --extra openpi`.
`tomato-sim eval` runs one model-driven attempt; `run` and `batch` retain their
existing scripted behavior. Do not mix policy rollouts into accepted human or
scripted demonstrations without inspecting their quality and provenance.

## Where to edit

Open `tomato_pick_sim.code-workspace` in VS Code to work on this project alone.

| Location | Responsibility |
| --- | --- |
| `configs/cherry_tomato_pick_sharpa.yaml` | Current picking/deposit sequence, placement, camera and recording parameters |
| `demos/cherry_tomato_pick/scene.py` | Scripted control stages and observation sampling |
| `demos/cherry_tomato_pick/openpi_policy.py`, `openpi_eval.py` | OpenPI transport, absolute action adapter and one-attempt evaluation CLI |
| `demos/cherry_tomato_pick/backend.py` | Contact dynamics, compliant pedicels and detachment |
| `demos/cherry_tomato_pick/sharpa.py`, `wuji2.py`, `kinematics.py` | Hand commands, mount transforms, FK/IK |
| `demos/cherry_tomato_pick/camera.py` | Procedural camera housings and wrist mounts |
| `demos/cherry_tomato_pick/tableware.py` | Procedural plate, three boxes and tabletop |
| `demos/cherry_tomato_pick/recording.py` | HDF5 recording and automatic episode numbering |
| `demos/cherry_tomato_pick/collection.py`, `quality.py`, `batch.py` | Legal scenes, appearance randomization, quality gates and repeated attempts |
| `demos/cherry_tomato_pick/assets/` | Tomato source and runtime mesh/texture assets |
| `demos/assets/robots/` | Carrier, Sharpa and historical Wuji2 MJCF/mesh files |
| `src/lycheesim/` | Required rigid runtime, camera/contact sensors, data and RL interfaces |

The detailed [tomato task notes](demos/cherry_tomato_pick/README.md) are retained
from the source. Their historical configuration examples refer to previous
tomato variants, also included here. Generic upstream commands outside
`run`, `batch`, `quality` and `eval` are not exposed by this focused CLI.

## Scope and limitations

- Robot, camera, tableware, contact, motion and collection parameters are copied
  from the current working tree; extraction does not improve harvesting success.
- Existing tomato RL task code and its supporting interfaces are retained.
  `uv sync --locked --extra rl` installs optional RSL-RL dependencies. Training
  and importing a real-robot policy are **not validated by this extraction**;
  an observation/action adapter and matching calibration/control semantics
  are still required for a particular policy.
- Other task demos, unrelated robot assets, MPM/SPH backends, reconstruction
  pipelines and unrelated sensors are omitted. Only the FBX reader needed to
  rebuild tomato assets is retained from reconstruction utilities.
- Source recordings, checkpoints, `.git` history and the old virtual environment
  were not copied. The original project and its datasets are untouched.
- This is a checkout-based project, not a self-contained asset-bearing wheel.
  Share the whole directory, including `demos/`, `configs/`, `src/`,
  `pyproject.toml`, `uv.lock` and provenance; omit `.venv/`, caches and `outputs/`.
  Review asset permissions before redistribution.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest
```

Disabling plugin autoload isolates these tests from unrelated system ROS pytest
plugins. Unit tests are not a measurement of physical harvesting success.
See [verification](provenance/verification.md) for this extraction's checks.
