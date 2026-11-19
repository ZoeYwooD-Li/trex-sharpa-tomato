"""Shared demo entrypoint: argparse → frontend list → step loop → broadcast.

A demo's `run.py` is a three-line wrapper around `main`:

    from lycheesim.engine.runner import main
    from demos.<name>.scene import build

    if __name__ == "__main__":
        main(build)

`build(cfg)` returns a `DemoSetup` (see below) that bundles the solver, the
per-step driver, and the per-frame snapshot push. The runner stays
solver-agnostic.
"""
from __future__ import annotations

import argparse
import inspect
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import warp as wp
import yaml
from loguru import logger
from tqdm import tqdm

from .frontend import Frontend


@dataclass
class DemoSetup:
    """The runner only needs these handles from a demo."""

    dt: float
    duration: float
    every: int = 1
    step: Callable[[float, dict], None] = field(default=lambda t, inputs: None)
    register: Callable[[Frontend], None] = field(default=lambda fe: None)
    snapshot: Callable[[Frontend, float], None] = field(default=lambda fe, t: None)
    verify: Callable[[], None] = field(default=lambda: None)
    close: Callable[[], None] = field(default=lambda: None)
    warmup_seconds: float = 0.0
    warmup_step: Callable[[], None] | None = None
    runtime_backend: Any | None = None
    should_stop: Callable[[], bool] = field(default=lambda: False)


def _parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_config: Path | None = None,
    prog: str | None = None,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument(
        "--config",
        required=default_config is None,
        default=default_config,
        type=Path,
        help=(
            "scene YAML (defaults to the demo's installed checkout config)"
            if default_config is not None
            else "scene YAML"
        ),
    )
    p.add_argument("--frontend", default=None,
                   help="comma-separated list: rerun,recorder,headless. "
                        "Default is derived from --no-render / --record / --live.")
    p.add_argument("--no-render", "--headless", dest="no_render", action="store_true")
    p.add_argument("--out-path", default=None, help="rerun .rrd output (or --live)")
    p.add_argument("--live", action="store_true", help="rerun: spawn a viewer")
    p.add_argument("--record", default=None, type=Path,
                   help="also write a .npz cache for the offline render layer")
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="override YAML value, e.g. --set control.mode=teleop")
    p.add_argument("--realtime", action="store_true",
                   help="pace the simulation loop to wall-clock time; useful for live teleop")
    p.add_argument("--wait-for-input", action="store_true",
                   help="after registering frontends, wait until a frontend publishes input")
    p.add_argument("--wait-for-input-timeout", type=float, default=None,
                   help="seconds to wait for --wait-for-input before starting anyway")
    return p.parse_args(list(argv) if argv is not None else None)


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _set_nested(cfg: dict, key: str, value: Any) -> None:
    parts = [p for p in key.split(".") if p]
    if not parts:
        raise ValueError("--set requires a non-empty key")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"--set {key}: {part!r} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.duration is not None:
        cfg["sim"]["duration"] = args.duration
    if args.out_path is not None:
        cfg.setdefault("render", {})["out_path"] = args.out_path
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), _parse_scalar(value.strip()))


from .frontend.selection import select_frontends as _select_frontends


def _wait_for_frontend_input(
    fes: list[Frontend],
    *,
    timeout: float | None,
) -> None:
    """Block until a reverse-channel frontend produces an input message."""
    logger.info("waiting for frontend input{}", "" if timeout is None else f" (timeout={timeout:.1f}s)")
    start = time.monotonic()
    while True:
        for fe in fes:
            inputs = fe.read_inputs(0.0)
            if inputs and not inputs.get("__stop__"):
                logger.info("frontend input received; starting simulation")
                return
        if timeout is not None and time.monotonic() - start >= timeout:
            logger.warning("wait-for-input timed out; starting simulation")
            return
        time.sleep(0.05)


def main(
    build_fn: Callable[[dict], DemoSetup],
    demo_name: str | None = None,
    *,
    argv: Sequence[str] | None = None,
    default_config: Path | None = None,
    prog: str | None = None,
) -> None:
    args = _parse_args(argv, default_config=default_config, prog=prog)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"config must contain a YAML mapping: {args.config}")
    _apply_overrides(cfg, args)

    wp.init()
    # Prefer the two-argument builder contract when the callable accepts it.
    # Inspect before calling: catching TypeError around the build itself hides
    # genuine scene-construction bugs and can build an expensive scene twice.
    signature = inspect.signature(build_fn)
    try:
        signature.bind(cfg, args.config)
    except TypeError:
        setup = build_fn(cfg)
    else:
        setup = build_fn(cfg, args.config)
    dn = demo_name or args.config.stem

    fes = _select_frontends(cfg, args, dn, setup.dt, setup.every)
    for fe in fes:
        setup.register(fe)

    if args.wait_for_input:
        _wait_for_frontend_input(fes, timeout=args.wait_for_input_timeout)

    if setup.warmup_seconds > 0 and setup.warmup_step is not None:
        warmup_steps = int(setup.warmup_seconds / setup.dt)
        if warmup_steps > 0:
            logger.info("warmup: {} steps", warmup_steps)
            for _ in range(warmup_steps):
                setup.warmup_step()

    n_steps = int(setup.duration / setup.dt)
    logger.info("running {} steps (dt={:.5f}, every={})", n_steps, setup.dt, setup.every)

    inputs: dict[str, Any] = {}
    t_start = time.time()
    realtime_start = time.monotonic()
    steps_ran = 0
    interrupted = False
    try:
        for s in tqdm(range(n_steps), desc=dn, unit="step"):
            t = s * setup.dt
            inputs.clear()
            for fe in fes:
                inputs.update(fe.read_inputs(t))
            if inputs.get("__stop__"):
                logger.info("stop requested by frontend at t={:.3f}s", t)
                break
            setup.step(t, inputs)
            steps_ran = s + 1
            stop_requested = setup.should_stop()
            if s % setup.every == 0 or stop_requested:
                for fe in fes:
                    setup.snapshot(fe, t)
                    fe.frame(t)
            if stop_requested:
                logger.info("scene completed at t={:.3f}s; finalizing now", t)
                break
            if args.realtime:
                target_wall_t = realtime_start + (s + 1) * setup.dt
                sleep_s = target_wall_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("interrupted by user")
    finally:
        wp.synchronize()
        elapsed = time.time() - t_start
        logger.info("done: {:.2f}s wall, {:.2f}ms/step",
                    elapsed, 1000 * elapsed / max(steps_ran, 1))
        try:
            if not interrupted:
                setup.verify()
        finally:
            try:
                setup.close()
            finally:
                for fe in fes:
                    fe.close()
