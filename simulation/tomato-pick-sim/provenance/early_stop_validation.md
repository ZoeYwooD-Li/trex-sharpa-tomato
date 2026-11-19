# Successful-deposit early termination — 2026-09-15

The recorder previously stopped accepting samples at its terminal robot frame,
but the shared runner continued physics and frontend recording until the
configured duration and only then verified/saved the episode.

The runner now supports a scene-owned `should_stop` callback. The tomato scene
sets it after the existing deposit criterion is met: the detached tomato stays
inside the selected box with speed below 0.08 m/s continuously for 0.3 s.
The scene renders all three cameras and captures a terminal robot frame before
requesting stop. The runner renders the final frontend snapshot even off its
normal cadence, runs verification, and closes the HDF5 recorder/frontends.
Verification errors are not suppressed. Other scenes retain the duration limit.

The final sensor sample may be off cadence. Its explicit timestamp is the
alignment authority; no camera or robot samples are appended after completion.
Failed and incomplete runs still undergo the existing quality checks.

## Checks

- 114 tests passed in 19.74 s, including early stop off the viewer cadence,
  unchanged duration-based runs, cleanup after verification failure, and
  terminal RGB-D/robot timestamps with no post-terminal samples.
- `git diff --check` passed.
- One bounded headless GPU attempt, on RTX 5090, maximum simulation duration
  28 s, wall timeout 600 s, seed 3854939909:

```bash
uv run --locked tomato-sim batch --attempts 1 --seed 3854939909 \
  --duration 28 --episode-timeout 600 \
  --output-dir outputs/early_stop_validation_20260915
```

Artifact directory:
`outputs/early_stop_validation_20260915/batch_20260915T061056Z_wllhwdtm/`.

Result: **accepted**, no rejection reasons, child exit code 0. The first and
only recorded `placed_in_box=true` sample is the final sample at **19.245 s**.
The runner exited at that same time instead of 28 s. Physics-loop wall time
was 221.76 s; serialization followed during cleanup.

All final RGB-D timestamps equal the robot timestamp, 19.245 s:

| Stream | Samples |
| --- | ---: |
| Robot | 579 |
| Gemini 435Le | 194 |
| Gemini 305 left | 579 |
| Gemini 305 right | 579 |

The final transition has `terminated=true`, `truncated=false`. Existing
successful-demonstration selection accepted it and wrote it to `accepted.txt`.
This one regression is not a new 100-attempt success-rate measurement. No
training or policy deployment was performed.
