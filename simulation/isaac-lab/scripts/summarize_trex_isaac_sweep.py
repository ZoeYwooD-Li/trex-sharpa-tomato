#!/usr/bin/env python3
"""Summarize paired T-Rex checkpoint / tactile ablation Isaac rollouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [
    ("ckpt1250_highcam", PROJECT / "outputs/isaac_tomato/trex_sweep_30s_ckpt1250_highcam/trex_eval_seed_0017"),
    ("ckpt2500_highcam", PROJECT / "outputs/isaac_tomato/trex_sweep_30s_ckpt2500/trex_eval_seed_0017"),
    ("ckpt3750_highcam", PROJECT / "outputs/isaac_tomato/trex_sweep_30s_ckpt3750/trex_eval_seed_0017"),
    ("ckpt5000_highcam", PROJECT / "outputs/isaac_tomato/trex_sweep_30s_ckpt5000/trex_eval_seed_0017"),
    ("ckpt6250_tactile_highcam", PROJECT / "outputs/isaac_tomato/trex_sweep_30s_ckpt6250_highcam/trex_eval_seed_0017"),
    ("ckpt6250_no_tactile", PROJECT / "outputs/isaac_tomato/trex_sweep_30s_ckpt6250_no_tactile/trex_eval_seed_0017"),
]


def summarize(label: str, path: Path) -> dict:
    task = json.loads((path / "task_evaluation.json").read_text())
    report = json.loads((path / "trex_closed_loop_report.json").read_text())["trex_closed_loop"]
    calls = report.get("replanning_calls", [])
    latencies = sorted(float(row["latency_ms"]) for row in calls)
    camera_values = [float(value) for row in calls
                     for value in row.get("camera_nonblack_fraction", {}).values()]
    return {
        "condition": label,
        "path": str(path.resolve()),
        "seed": task.get("seed"),
        "episode_seconds": task.get("episode_seconds"),
        "replanning_calls": len(calls),
        "ik_failures": sum(bool(row.get("ik_failed")) for row in calls),
        "finite_outputs": bool(report.get("finite_outputs")),
        "mean_inference_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "p95_inference_latency_ms": latencies[min(len(latencies) - 1, int(.95 * len(latencies)))] if latencies else None,
        "minimum_camera_nonblack_fraction": min(camera_values) if camera_values else None,
        "tomato_displacement_m": task.get("tomato_displacement_m"),
        "harvest_success": task.get("harvest_success"),
        "basket_success": task.get("basket_success"),
        "pick_and_place_success": task.get("pick_and_place_success"),
        "peak_fingertip_force_n": task.get("peak_fingertip_force_n"),
        "all_f6_zero": all(value == 0 for row in report.get("replanning_calls", [])
                            for value in row.get("f6_history_range", [])),
        "video": str((path / "rollout.mp4").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs/isaac_tomato/trex_sweep_summary")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    rows, missing = [], []
    for label, path in DEFAULT_RUNS:
        if not (path / "task_evaluation.json").exists():
            missing.append(str(path))
            continue
        rows.append(summarize(label, path))
    if missing and not args.allow_missing:
        raise SystemExit("Missing completed runs:\n" + "\n".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    if rows:
        with (args.output / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        columns = ("condition", "episode_seconds", "replanning_calls", "ik_failures",
                   "tomato_displacement_m", "peak_fingertip_force_n", "harvest_success",
                   "basket_success", "pick_and_place_success")
        lines = ["# T-Rex Isaac checkpoint comparison", "",
                 "All rows use seed 17 and the same initial scene. `all_f6_zero` indicates the Isaac run had no fingertip contact; tactile input ablation therefore cannot establish the value of tactile feedback until contact occurs.", "",
                 "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
        lines += ["| " + " | ".join(str(row.get(key)) for key in columns) + " |" for row in rows]
        if missing:
            lines += ["", "Incomplete runs:", *[f"- `{item}`" for item in missing]]
        (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} completed run summaries to {args.output}")
    if missing:
        print("Still missing:\n" + "\n".join(missing))


if __name__ == "__main__":
    main()
