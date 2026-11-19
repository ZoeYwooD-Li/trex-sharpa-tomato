#!/usr/bin/env python3
"""Run paired Isaac pick-and-place rollouts for touch/no-touch action chunks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TOMATO_PROJECT = PROJECT.parent / "tomato-pick-sim-local"
DEFAULT_DATASET = Path("/path/to/local/company-remote-2/real/harvest-tomato/data/data-lerobot/sharpa")


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total == 0:
        return None
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * ((rate * (1 - rate) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-touch", type=Path, default=PROJECT / "outputs/sharpa_chunks/no_touch/policy.pt")
    parser.add_argument("--with-touch", type=Path, default=PROJECT / "outputs/sharpa_chunks/with_touch/policy.pt")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/isaac_tomato/ablation")
    parser.add_argument("--episodes", type=int, default=10, help="Paired randomized initial conditions")
    parser.add_argument("--start-seed", type=int, default=170)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed task_evaluation.json rollouts and continue missing paired episodes")
    parser.add_argument("--allow-smoke-checkpoints", action="store_true",
                        help="Permit pilot checkpoints with fewer than 10 training episodes")
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0:
        parser.error("--episodes and --seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    model_dirs = {"no_touch": args.no_touch.expanduser().resolve().parent,
                  "with_touch": args.with_touch.expanduser().resolve().parent}
    checkpoints = {"no_touch": args.no_touch.expanduser().resolve(),
                   "with_touch": args.with_touch.expanduser().resolve()}
    summaries = {}
    for name, path in checkpoints.items():
        summary_path = model_dirs[name] / "summary.json"
        if not path.is_file() or not summary_path.is_file():
            parser.error(f"Missing trained checkpoint or summary for {name}: {path}")
        summaries[name] = json.loads(summary_path.read_text())
        if not args.allow_smoke_checkpoints and len(summaries[name].get("train_episodes", [])) < 10:
            parser.error(f"{name} checkpoint is only a smoke test; train on the full dataset or pass --allow-smoke-checkpoints")
    for key in ("dataset", "train_episodes", "validation_episodes", "train_windows",
                "validation_windows", "chunk", "sample_stride", "rgb_size", "seed"):
        if summaries["no_touch"].get(key) != summaries["with_touch"].get(key):
            parser.error(f"Touch/no-touch training configurations differ in {key}")
    if summaries["no_touch"].get("mode") != "rgb-state" or summaries["with_touch"].get("mode") != "rgb-state-touch":
        parser.error("Expected rgb-state and rgb-state-touch checkpoints")

    mjcf = PROJECT / "outputs/isaac_tomato/dual_fr3_sharpa.xml"
    build_command = [sys.executable, str(PROJECT / "scripts/build_dual_fr3_sharpa_mjcf.py"),
                     "--source", str(TOMATO_PROJECT), "--output", str(mjcf)]
    subprocess.run(build_command, check=True, cwd=PROJECT)

    rows = []
    for episode_offset in range(args.episodes):
        seed = args.start_seed + episode_offset
        for condition in ("no_touch", "with_touch"):
            rollout_dir = args.output / condition / f"seed_{seed}"
            rollout_dir.mkdir(parents=True, exist_ok=True)
            result_path = rollout_dir / "task_evaluation.json"
            if args.resume and result_path.is_file():
                row = json.loads(result_path.read_text())
                row["condition"] = condition
                row["rollout_dir"] = str(rollout_dir)
                position = row.get("final_tomato_position_m", [])
                row["simulation_valid"] = bool(
                    row.get("tomato_in_workspace")
                    and len(position) == 3
                    and all(math.isfinite(float(value)) for value in position)
                    and math.isfinite(float(row.get("tomato_displacement_m", math.inf)))
                    and float(row.get("tomato_displacement_m", math.inf)) <= 0.70
                )
                rows.append(row)
                print(f"{condition:10s} seed={seed}: reused completed rollout", flush=True)
                continue
            command = [sys.executable, str(PROJECT / "scripts/smoke_isaac_tomato.py"),
                       "--mjcf", str(mjcf), "--output", str(rollout_dir),
                       "--policy-schema", str(args.dataset),
                       "--action-chunk-ckpt", str(checkpoints[condition]),
                       "--policy-apply", "--task-eval", "--eval-seed", str(seed),
                       "--episode-seconds", str(args.seconds), "--render-every", "16"]
            log_path = rollout_dir / "rollout.log"
            with log_path.open("w") as log:
                proc = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0 or not result_path.is_file():
                raise RuntimeError(f"Isaac rollout failed for {condition}, seed {seed}; see {log_path}")
            row = json.loads(result_path.read_text())
            row["condition"] = condition
            row["rollout_dir"] = str(rollout_dir)
            position = row.get("final_tomato_position_m", [])
            row["simulation_valid"] = bool(
                row.get("tomato_in_workspace")
                and len(position) == 3
                and all(math.isfinite(float(value)) for value in position)
                and math.isfinite(float(row.get("tomato_displacement_m", math.inf)))
                and float(row.get("tomato_displacement_m", math.inf)) <= 0.70
            )
            rows.append(row)
            print(f"{condition:10s} seed={seed}: harvest={row['harvest_success']} "
                  f"basket={row['basket_success']} pick_place={row['pick_and_place_success']} "
                  f"peak_force={row['peak_fingertip_force_n']:.3f} N", flush=True)

    with (args.output / "episodes.csv").open("w", newline="") as stream:
        fields = ("condition", "seed", "simulation_valid", "tomato_in_workspace",
                  "harvest_success", "basket_success", "pick_and_place_success",
                  "tomato_displacement_m", "peak_fingertip_force_n", "rollout_dir")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)

    aggregates = {}
    for condition in ("no_touch", "with_touch"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        total = len(condition_rows)
        valid_rows = [row for row in condition_rows if row.get("simulation_valid")]
        aggregates[condition] = {}
        for metric in ("harvest_success", "basket_success", "pick_and_place_success"):
            count = sum(bool(row[metric]) for row in condition_rows)
            aggregates[condition][metric] = {
                "successes": count, "episodes": total,
                "rate": count / total if total else None,
                "wilson_95_ci": _wilson(count, total),
            }
            valid_successes = sum(bool(row[metric]) for row in valid_rows)
            aggregates[condition][f"{metric}_valid_simulations"] = {
                "successes": valid_successes,
                "valid_episodes": len(valid_rows),
                "invalid_episodes": total - len(valid_rows),
                "rate": valid_successes / len(valid_rows) if valid_rows else None,
                "wilson_95_ci": _wilson(valid_successes, len(valid_rows)),
            }
        aggregates[condition]["simulation_validity"] = {
            "valid_episodes": len(valid_rows),
            "invalid_episodes": total - len(valid_rows),
            "valid_seeds": [row["seed"] for row in valid_rows],
        }
        aggregates[condition]["mean_peak_fingertip_force_n"] = sum(
            row["peak_fingertip_force_n"] for row in condition_rows) / total
    paired = {}
    for metric in ("harvest_success", "basket_success", "pick_and_place_success"):
        touch_rows = {row["seed"]: row for row in rows if row["condition"] == "with_touch"}
        no_touch_rows = {row["seed"]: row for row in rows if row["condition"] == "no_touch"}
        seeds = sorted(seed for seed in set(touch_rows) & set(no_touch_rows)
                       if touch_rows[seed].get("simulation_valid")
                       and no_touch_rows[seed].get("simulation_valid"))
        touch_by_seed = {seed: bool(touch_rows[seed][metric]) for seed in seeds}
        no_touch_by_seed = {seed: bool(no_touch_rows[seed][metric]) for seed in seeds}
        paired[metric] = {
            "seeds": seeds,
            "with_touch_only_successes": sum(touch_by_seed[s] and not no_touch_by_seed[s] for s in seeds),
            "no_touch_only_successes": sum(no_touch_by_seed[s] and not touch_by_seed[s] for s in seeds),
        }
    summary = {
        "task": "dual FR3 + Sharpa cherry tomato detach and place in first basket",
        "simulator": "Isaac Sim / PhysX",
        "paired_seeds": list(range(args.start_seed, args.start_seed + args.episodes)),
        "episode_seconds": args.seconds,
        "initial_condition_randomization": "tomato x/y position +/- 12 mm; both arms/hands reset to the same first frame of the real LeRobot dataset; same seed paired across modes",
        "success_definition": rows[0]["success_definition"],
        "conditions": aggregates,
        "paired_outcomes": paired,
        "interpretation": (
            "Success rates over all scheduled episodes are shown, but only episodes with the tomato remaining "
            "inside the workcell and finite are valid for task inference. Numerical divergence invalidates "
            "the affected episode; valid-only rates are reported separately."
        ),
        "metrics": "harvest, basket, and combined task success plus peak fingertip force",
        "touch_sensor_calibrated_to_hardware": False,
        "episodes_csv": "episodes.csv",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "episodes.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
