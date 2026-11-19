# Extraction verification — 2026-09-14

All checks below were run from this standalone checkout using its freshly
created `.venv`, not the original project's environment. `PYTHONPATH` was
unset for pytest and simulation. Hardware: NVIDIA GeForce RTX 5090, driver
reporting CUDA 13.0; Python 3.11.16, Newton 1.5.0, Warp 1.16.0,
MuJoCo/MuJoCo-Warp 3.11.0 and Torch 2.11.0+cu130.

## Completed checks

- `uv lock` and `uv sync --locked` succeeded. The focused lock resolves 82
  packages including optional RL dependencies; the default environment installs
  60 packages. No original environment directory was copied or symlinked.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --locked pytest -ra`:
  **110 passed in 18.86 seconds** (102 retained tomato/camera tests plus eight
  standalone CLI, discovery, asset and provenance checks).
- Import paths for the runtime and extracted kinematics resolve inside this
  checkout. No tomato Python module imports another demo. Robot and tomato
  assets are ordinary local files, not symlinks to the source repository.
- All 263 copied source-file hashes still matched the original checkout after
  extraction. Original `git status --short` was unchanged; `git diff --check`
  on that source working tree passed.
- CLI help passed for `run`, `batch` and `quality`. The quality summary command
  successfully read the generated batch HDF5 files.

## Bounded GPU physics/recording checks

```bash
uv run --locked tomato-sim batch --attempts 2 --duration 0.15 --seed 42 \
  --output-dir outputs/extraction_smoke --episode-timeout 120
```

Batch directory: `outputs/extraction_smoke/batch_20260914T091338Z_fidmuw69/`.
The two child processes used seeds 42 and 43 and produced separate
`episode_000001.h5` / `episode_000002.h5` files, log files, quality reports,
`summary.json` and an empty `accepted.txt`. Each HDF5 file contains:

| Stream | Samples | Image shape |
| --- | ---: | --- |
| Robot state/action timing | 5 states, 4 action timestamps | n/a |
| Gemini 435Le RGB + depth | 2 | 200 x 320 |
| Gemini 305 left RGB + depth | 5 | 200 x 320 |
| Gemini 305 right RGB + depth | 5 | 200 x 320 |

The records include extraction manifest identity, original commit/dirty state,
and current task-source hashes. Both attempts were correctly rejected as
incomplete/not detached and placed/support not ready; the batch continued after
the first child's nonzero verification exit.

A separate 0.15-second run with the Rerun frontend produced
`outputs/extraction_smoke/preview.rrd` and
`outputs/extraction_smoke/preview/episode_000001.h5`. The final fixed-camera
valid-depth count was 64,000; wrist counts were 63,650 and 55,381. The HDF5 file
was written with automatic recording left at its default enabled setting.

## Not established by these checks

The short simulations end during approach. Their child exit code 1 is the
expected task-verification failure, not a successful harvest. No complete
28-second harvest, 100-attempt success rate, sustained training, live viewer
window interaction or real-robot policy deployment was validated in this
extraction. Existing motion/contact limitations remain unchanged.
