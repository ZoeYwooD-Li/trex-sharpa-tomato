"""Sequential, bounded collection using the existing single-episode CLI."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time

import yaml

from .quality import QUALITY_SCHEMA


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/cherry_tomato_pick_sharpa.yaml"


def _run_episode(command, log_path, timeout_s):
    """Isolate GPU lifetime; gracefully finalize the child on timeout/Ctrl+C."""
    with log_path.open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "TQDM_DISABLE": "1"}, start_new_session=True,
            )
        except OSError as exc:
            log.write(f"Failed to launch episode: {exc}\n")
            return None, "launch_error"
        try:
            return process.wait(timeout=timeout_s), None
        except subprocess.TimeoutExpired:
            reason = "wall_timeout"
        except KeyboardInterrupt:
            reason = "interrupted"
        # SIGINT lets runner.finally close the HDF5 recorder. A hung driver may
        # not respond; bound cleanup as well and never leave a child running.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                reason = "interrupted"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return process.returncode, reason


def _episode_result(path, returncode, process_reason):
    """Read authoritative HDF5 quality, never infer success from file existence."""
    import h5py

    reasons = [] if process_reason is None else [process_reason]
    if returncode != 0:
        reasons.append("nonzero_exit")
    quality = {}
    if not path.is_file():
        reasons.append("episode_not_written")
    else:
        try:
            with h5py.File(path, "r") as archive:
                metadata = json.loads(archive["metadata"].attrs["extra"])
                quality = metadata.get("quality", {})
                if not isinstance(quality, dict):
                    raise ValueError("quality must be an object")
                if quality.get("schema") != QUALITY_SCHEMA:
                    reasons.append("unassessed_episode")
                elif quality.get("accepted") is not True:
                    failures = quality.get("reasons") or ["quality_rejected"]
                    if not isinstance(failures, list) or any(not isinstance(r, str) for r in failures):
                        raise ValueError("quality reasons must be a list of strings")
                    reasons.extend(failures)
        except (OSError, KeyError, ValueError, TypeError, AttributeError):
            quality = {}
            reasons.append("unreadable_episode_metadata")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "target_box_id": quality.get("target_box_id"),
        "returncode": returncode,
    }


def _write_summary(directory, summary):
    results = summary["attempts"]
    accepted = [r for r in results if r["accepted"]]
    summary["attempted"] = len(results)
    summary["accepted"] = len(accepted)
    summary["rejected"] = len(results) - len(accepted)
    summary["rejection_reasons"] = dict(Counter(reason for r in results for reason in r["reasons"]))
    summary["accepted_by_box"] = dict(Counter(str(r["target_box_id"]) for r in accepted))
    # A private directory per batch means these are owned reports, not user files.
    for name, content in (
        ("summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"),
        ("accepted.txt", "".join(str(directory / r["episode"]) + "\n" for r in accepted)),
    ):
        temporary = directory / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(directory / name)


def run_batch(*, config=DEFAULT_CONFIG, attempts=100, output_dir=None,
              seed=None, duration=None, episode_timeout_s=900.):
    if type(attempts) is not int or attempts <= 0:
        raise ValueError("attempts must be a positive integer")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(episode_timeout_s) or episode_timeout_s <= 0:
        raise ValueError("episode timeout must be finite and positive")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    config = Path(config).expanduser().resolve()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config must contain a YAML mapping")
    for section in ("recording", "tableware", "deposit", "camera", "wrist_cameras"):
        if not isinstance(cfg.get(section, {}), dict):
            raise ValueError(f"config.{section} must be a mapping")
    recording = cfg.get("recording", {})
    for section in ("quality_filter", "scene_validation"):
        if not isinstance(recording.get(section, {}), dict):
            raise ValueError(f"recording.{section} must be a mapping")
    for section in ("camera", "wrist_cameras"):
        if not isinstance(cfg.get(section, {}).get("sensor", {}), dict):
            raise ValueError(f"{section}.sensor must be a mapping")
    if not all(cfg.get(key, {}).get("enabled", False) for key in ("tableware", "deposit")):
        raise ValueError("batch collection requires tableware and deposit enabled")
    if not all(recording.get(key, {}).get("enabled", False) for key in ("quality_filter", "scene_validation")):
        raise ValueError("batch collection requires quality_filter and scene_validation enabled")
    if not all(cfg.get(key, {}).get("sensor", {}).get("enabled", False) for key in ("camera", "wrist_cameras")):
        raise ValueError("batch collection requires all three camera sensors enabled")
    if output_dir is None:
        if not isinstance(recording.get("output_path"), str) or not recording["output_path"]:
            raise ValueError("recording.output_path is required without --output-dir")
        output_dir = Path(recording["output_path"]).parent
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
    parent = Path(output_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    # New batch -> new random base seed by default. --seed gives repeatable
    # sampling; attempt index advances even if a rejected scene writes no HDF5.
    base_seed = secrets.randbits(32) if seed is None else seed
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(tempfile.mkdtemp(prefix=f"batch_{stamp}_", dir=parent))
    snapshot = directory / "config.yaml"
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    summary = {
        "schema": "lycheesim.cherry_tomato_pick.batch/v1", "status": "running",
        "requested_attempts": attempts, "base_seed": base_seed,
        "config": str(config), "config_snapshot": str(snapshot), "started_utc": stamp, "attempts": [],
        "episode_timeout_s": episode_timeout_s, "duration_override_s": duration,
    }
    _write_summary(directory, summary)
    print(f"Batch directory: {directory}\nAttempts: {attempts}; base seed: {base_seed}", flush=True)
    try:
        for index in range(attempts):
            number, episode_seed = index + 1, base_seed + index
            path = directory / f"episode_{number:06d}.h5"
            log = path.with_suffix(".log")
            command = [
                sys.executable, "-m", "lycheesim.cli", "run", "cherry-tomato-pick",
                "--config", str(snapshot), "--frontend", "headless",
                "--set", "recording.enabled=true",
                "--set", "recording.overwrite=false",
                "--set", "recording.auto_increment=false",
                "--set", "recording.vary_seed_by_episode=false",
                "--set", f"recording.seed={episode_seed}",
                "--set", f"recording.output_path={json.dumps(str(path))}",
            ]
            if duration is not None:
                command.extend(["--duration", str(duration)])
            print(f"[{number}/{attempts}] seed={episode_seed} -> {path.name}", flush=True)
            started = time.monotonic()
            returncode, process_reason = _run_episode(command, log, episode_timeout_s)
            result = _episode_result(path, returncode, process_reason)
            result.update(attempt=number, seed=episode_seed, episode=path.name,
                          log=log.name, command=command, wall_seconds=time.monotonic() - started)
            summary["attempts"].append(result)
            _write_summary(directory, summary)
            label = "accepted" if result["accepted"] else "rejected"
            print(f"[{number}/{attempts}] {label}; reasons={result['reasons']}; "
                  f"accepted so far={summary['accepted']}", flush=True)
            if process_reason == "interrupted":
                summary["status"] = "interrupted"
                break
        else:
            summary["status"] = "completed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception:
        summary["status"] = "controller_error"
        raise
    finally:
        _write_summary(directory, summary)
    print(f"Batch {summary['status']}: {summary['attempted']}/{attempts} attempts, "
          f"{summary['accepted']} accepted, {summary['rejected']} rejected.\n"
          f"Summary: {directory / 'summary.json'}\n"
          f"Training list: {directory / 'accepted.txt'}", flush=True)
    return directory, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Attempt N tomato-pick episodes, continuing after failed attempts.")
    parser.add_argument("--attempts", type=int, default=100)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Parent directory; a unique batch subdirectory is created")
    parser.add_argument("--seed", type=int, help="Reproducible base seed (default: a fresh random seed per batch)")
    parser.add_argument("--duration", type=float, help="Simulation seconds per attempt; omit for config duration")
    parser.add_argument("--episode-timeout", type=float, default=900., help="Wall seconds before interrupting one attempt")
    args = parser.parse_args(argv)
    try:
        _, summary = run_batch(config=args.config, attempts=args.attempts, output_dir=args.output_dir,
                               seed=args.seed, duration=args.duration, episode_timeout_s=args.episode_timeout)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    # A completed batch need not have 100 successful demonstrations.
    return 130 if summary["status"] == "interrupted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
