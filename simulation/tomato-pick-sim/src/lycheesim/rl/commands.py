"""Implement the shared RL smoke, train, and checkpoint-evaluation commands."""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, TextIO

import torch

from lycheesim.engine.frontend import RerunFrontend
from lycheesim.rl import (
    RLEnvironment,
    RLExecutor,
    RLTaskContext,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.float().mean().item()),
    }


def _task_options(values: list[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name:
            raise ValueError(f"--task-option must be KEY=VALUE, got {value!r}")
        options[name] = raw
    return options


def smoke(args: Any, *, stdout: TextIO | None = None) -> int:
    """Run deterministic or random actions through a registered task."""

    stdout = stdout or sys.stdout
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    options = _task_options(args.task_option)
    os.environ["LYCHEESIM_RL_DEVICE"] = args.device
    os.environ["LYCHEESIM_NUM_ENVS"] = str(args.num_envs)
    os.environ["LYCHEESIM_VISUALIZE"] = "0"
    if args.config is not None:
        os.environ["LYCHEESIM_TASK_CONFIG"] = str(args.config.resolve())
    importlib.import_module(args.registration_module)

    context = RLTaskContext(
        training=False,
        device=args.device,
        num_envs=args.num_envs,
        config_path=args.config.resolve() if args.config is not None else None,
        profile=args.profile,
        visualize=False,
        seed=args.seed,
        options=options,
    )
    cfg = load_env_cfg(args.task, context=context)
    if args.decimation is not None:
        if args.decimation <= 0:
            raise ValueError("--decimation must be positive")
        cfg.decimation = args.decimation
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s

    torch.manual_seed(args.seed)
    env = RLEnvironment(cfg, device=args.device)
    executor = RLExecutor(env)
    rewards: list[float] = []
    terminal_count = 0
    timeout_count = 0
    episode_logs: list[dict[str, float]] = []
    try:
        executor.reset(seed=args.seed)
        observations = executor.observations
        torch_device = torch.device(env.device)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        rollout_started = time.perf_counter()
        for _ in range(args.steps):
            if args.policy == "zero":
                action = torch.zeros(
                    (env.num_envs, env.action_manager.total_action_dim),
                    device=env.device,
                )
            elif args.policy == "random":
                action = 2.0 * torch.rand(
                    (env.num_envs, env.action_manager.total_action_dim),
                    device=env.device,
                ) - 1.0
            else:
                action = torch.zeros(
                    (env.num_envs, env.action_manager.total_action_dim),
                    device=env.device,
                )
                if action.shape[1] < 2:
                    raise ValueError("down policy requires an action dimension >= 2")
                action[:, 1] = -1.0
            result = executor.step(action)
            observations = result.observations
            reward = result.reward
            terminated = result.terminated
            truncated = result.truncated
            extras = result.extras
            rewards.extend(reward.detach().cpu().tolist())
            terminal_count += int(terminated.sum().item())
            timeout_count += int(truncated.sum().item())
            if extras.get("log"):
                episode_logs.append(dict(extras["log"]))

        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        rollout_seconds = time.perf_counter() - rollout_started
        backend = getattr(env.runtime, "backend", None)

        report = {
            "task": args.task,
            "profile": args.profile,
            "policy": args.policy,
            "steps": args.steps,
            "num_envs": env.num_envs,
            "physics_dt": env.physics_dt,
            "step_dt": env.step_dt,
            "reward_sum": float(sum(rewards)),
            "reward_mean": float(sum(rewards) / max(len(rewards), 1)),
            "terminations": terminal_count,
            "timeouts": timeout_count,
            "rollout_seconds": rollout_seconds,
            "environment_steps_per_second": (
                args.steps * env.num_envs / rollout_seconds
            ),
            "runtime": {
                "type": type(env.runtime).__name__,
                "backend": type(backend).__name__ if backend is not None else None,
                "physics_device": str(getattr(backend, "device", env.device)),
            },
            "episode_logs": episode_logs,
            "observations": {
                name: _tensor_summary(value)
                for name, value in observations.items()
                if isinstance(value, torch.Tensor)
            },
        }
        signals = getattr(env.runtime, "signals", None)
        if callable(signals):
            report["signals"] = {
                name: _tensor_summary(value)
                for name, value in signals().items()
            }
        print(json.dumps(report, indent=2, sort_keys=True), file=stdout)
    finally:
        executor.close()
    return 0


def _positive_override(name: str, value: int | None) -> int | None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def train(args: Any, *, stdout: TextIO | None = None) -> int:
    """Train a registered task with the optional RSL-RL backend."""

    from lycheesim.rl.rsl_rl import RslRlVecEnvWrapper

    stdout = stdout or sys.stdout
    _positive_override("--max-iterations", args.max_iterations)
    _positive_override("--num-steps-per-env", args.num_steps_per_env)
    _positive_override("--save-interval", args.save_interval)
    _positive_override("--decimation", args.decimation)
    _positive_override("--num-envs", args.num_envs)
    options = _task_options(args.task_option)
    os.environ["LYCHEESIM_RL_DEVICE"] = args.device
    os.environ["LYCHEESIM_NUM_ENVS"] = str(args.num_envs)
    os.environ["LYCHEESIM_VISUALIZE"] = "0"
    if args.config is not None:
        os.environ["LYCHEESIM_TASK_CONFIG"] = str(args.config.resolve())
    importlib.import_module(args.registration_module)

    context = RLTaskContext(
        training=True,
        device=args.device,
        num_envs=args.num_envs,
        config_path=args.config.resolve() if args.config is not None else None,
        profile=args.profile,
        visualize=False,
        options=options,
    )
    env_cfg = load_env_cfg(args.task, context=context)
    rl_cfg = load_rl_cfg(args.task, context=context)
    if args.decimation is not None:
        env_cfg.decimation = args.decimation
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    if args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        rl_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        rl_cfg.save_interval = args.save_interval

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{rl_cfg.run_name}" if rl_cfg.run_name else ""
    log_dir = args.log_root / rl_cfg.experiment_name / f"{timestamp}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=False)

    env = RLEnvironment(env_cfg, device=args.device)
    wrapped: Any | None = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
        if wrapped.num_envs != args.num_envs:
            raise RuntimeError(
                f"Requested {args.num_envs} environments, but the task built "
                f"{wrapped.num_envs}"
            )
        backend = getattr(wrapped.unwrapped.runtime, "backend", None)
        physics_device = getattr(backend, "device", "scene-default")
        print(
            "Task execution: "
            f"task={args.task} runtime={type(wrapped.unwrapped.runtime).__name__} "
            f"backend={type(backend).__name__} profile={args.profile} "
            f"envs={wrapped.num_envs} policy_device={wrapped.device} "
            f"physics_device={physics_device}",
            file=stdout,
        )
        rollout_samples = rl_cfg.num_steps_per_env * wrapped.num_envs
        minimum_samples = max(2, int(rl_cfg.algorithm.num_mini_batches))
        if rollout_samples < minimum_samples:
            raise ValueError(
                f"PPO requires at least {minimum_samples} rollout samples to "
                "normalize advantages and fill every mini-batch; increase "
                "--num-steps-per-env"
            )
        runner_cls = load_runner_cls(args.task)
        if runner_cls is None:
            raise ValueError(f"Task {args.task!r} does not define an RSL-RL runner")
        runner = runner_cls(
            wrapped,
            rl_cfg.to_dict(),
            log_dir=str(log_dir),
            device=args.device,
        )
        if args.resume_checkpoint is not None:
            runner.load(str(args.resume_checkpoint), map_location=args.device)
            print(f"Resumed from {args.resume_checkpoint}", file=stdout)
        runner.learn(
            num_learning_iterations=rl_cfg.max_iterations,
            init_at_random_ep_len=args.init_at_random_episode_length,
        )
        runner.save(str(log_dir / "model_final.pt"))
        print(f"Training complete: {log_dir}", file=stdout)
    finally:
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()
    return 0


def _float(value: Any) -> float:
    return float(torch.as_tensor(value).detach().cpu().reshape(-1)[0].item())


def _visual_backend(env: RLEnvironment) -> Any:
    backend = getattr(env.runtime, "backend", None)
    required = ("register_frontend", "snapshot_frontend")
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise TypeError(
            "Task runtime does not expose checkpoint visualization hooks: "
            + ", ".join(missing)
        )
    return backend


def evaluate(args: Any, *, stdout: TextIO | None = None) -> int:
    """Replay an RSL-RL checkpoint in a Rerun frontend."""

    from lycheesim.rl.rsl_rl import RslRlVecEnvWrapper

    stdout = stdout or sys.stdout
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    options = _task_options(args.task_option)
    os.environ["LYCHEESIM_NUM_ENVS"] = "1"
    os.environ["LYCHEESIM_RL_DEVICE"] = args.device
    os.environ["LYCHEESIM_VISUALIZE"] = "1"
    if args.config is not None:
        os.environ["LYCHEESIM_TASK_CONFIG"] = str(args.config.resolve())
    importlib.import_module(args.registration_module)

    context = RLTaskContext(
        training=False,
        device=args.device,
        num_envs=1,
        config_path=args.config.resolve() if args.config is not None else None,
        profile=args.profile,
        visualize=True,
        seed=args.seed,
        options=options,
    )
    env_cfg = load_env_cfg(args.task, context=context)
    rl_cfg = load_rl_cfg(args.task, context=context)
    env_cfg.auto_reset = False
    torch.manual_seed(args.seed)
    env = RLEnvironment(env_cfg, device=args.device)
    wrapped: Any | None = None
    frontend: RerunFrontend | None = None
    backend: Any | None = None
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task)
        if runner_cls is None:
            raise ValueError(f"Task {args.task!r} does not define an RSL-RL runner")
        runner = runner_cls(
            wrapped,
            rl_cfg.to_dict(),
            log_dir=None,
            device=args.device,
        )
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

        observations, _ = wrapped.reset()
        backend = _visual_backend(wrapped.unwrapped)
        frontend = RerunFrontend(
            out_path=str(args.out.resolve()) if args.out is not None else None,
            fps=max(1, round(1.0 / wrapped.unwrapped.step_dt)),
            live=args.live,
            view_up=getattr(backend, "view_up", "y"),
            eye_position=[0.72, 0.30, 0.72],
            look_target=[0.50, 0.08, 0.50],
        )
        backend.register_frontend(frontend)
        backend.snapshot_frontend(frontend, 0.0)
        frontend.frame(0.0)

        reward_total = 0.0
        actions: list[torch.Tensor] = []
        terminal_step: int | None = None
        termination: dict[str, bool] = {}
        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                action = policy(observations)
            actions.append(action.detach().cpu())
            observations, reward, dones, _ = wrapped.step(action)
            reward_value = _float(reward)
            reward_total += reward_value

            time_s = step * wrapped.unwrapped.step_dt
            backend.snapshot_frontend(frontend, time_s)
            frontend.frame(time_s)
            print(f"step={step:03d} reward={reward_value:+.3f}", file=stdout)

            if bool(dones.any().item()):
                terminal_step = step
                termination = {
                    name: bool(values[0].item())
                    for name, values in (
                        wrapped.unwrapped.termination_manager.term_dones.items()
                    )
                }
                break

        action_tensor = torch.cat(actions, dim=0)
        report = {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "steps": terminal_step or args.steps,
            "reward_total": reward_total,
            "action_mean": action_tensor.mean(dim=0).tolist(),
            "action_min": action_tensor.min(dim=0).values.tolist(),
            "action_max": action_tensor.max(dim=0).values.tolist(),
            "termination": termination,
            "recording": str(args.out.resolve()) if args.out is not None else "live",
        }
        print(json.dumps(report, indent=2, sort_keys=True), file=stdout)
    finally:
        if backend is not None:
            verify = getattr(backend, "verify", None)
            if callable(verify):
                verify()
        if frontend is not None:
            frontend.close()
        if wrapped is not None:
            wrapped.close()
        else:
            env.close()
    return 0
