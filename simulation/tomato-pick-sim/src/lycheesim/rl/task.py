"""Task definitions and registry for the framework-neutral RL layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .environment import RLEnvironmentCfg


@dataclass(frozen=True, kw_only=True)
class RLTaskContext:
    """Runtime choices supplied by a launcher, not embedded in a task module."""

    training: bool = True
    device: str = "cpu"
    num_envs: int = 1
    config_path: Path | None = None
    profile: str = "full"
    visualize: bool = False
    seed: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)


EnvironmentCfgFactory = Callable[[RLTaskContext], RLEnvironmentCfg]
TrainerCfgFactory = Callable[[RLTaskContext], Any]


@dataclass(frozen=True)
class RLTask:
    """Everything the shared RL layer needs to instantiate one task.

    A demo normally publishes one instance of this class. Physics backends stay
    in the demo; rollout, policy execution, registration, and training adapters
    stay in :mod:`lycheesim.rl`.
    """

    task_id: str
    environment_cfg: EnvironmentCfgFactory
    trainer_cfg: TrainerCfgFactory | None = None
    runner_cls: type | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id cannot be empty")


_REGISTRY: dict[str, RLTask] = {}


def register_task(task: RLTask) -> None:
    if task.task_id in _REGISTRY:
        raise ValueError(f"Task {task.task_id!r} is already registered")
    _REGISTRY[task.task_id] = task


def list_tasks() -> list[str]:
    return sorted(_REGISTRY)


def get_task(task_name: str) -> RLTask:
    try:
        return _REGISTRY[task_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task {task_name!r}. Registered tasks: {list_tasks()}"
        ) from exc


def load_env_cfg(
    task_name: str,
    *,
    play: bool = False,
    context: RLTaskContext | None = None,
) -> RLEnvironmentCfg:
    """Build a fresh environment configuration for a registered task."""

    context = context or RLTaskContext(training=not play)
    if play and context.training:
        context = replace(context, training=False)
    return deepcopy(get_task(task_name).environment_cfg(context))


def load_rl_cfg(
    task_name: str,
    *,
    context: RLTaskContext | None = None,
) -> Any:
    """Build the optional trainer configuration for a registered task."""

    task = get_task(task_name)
    if task.trainer_cfg is None:
        raise ValueError(f"Task {task_name!r} does not define a training backend")
    return deepcopy(task.trainer_cfg(context or RLTaskContext()))


def load_runner_cls(task_name: str) -> type | None:
    return get_task(task_name).runner_cls


def register_lycheesim_task(
    task_id: str,
    env_cfg: RLEnvironmentCfg,
    play_env_cfg: RLEnvironmentCfg,
    rl_cfg: Any,
    runner_cls: type | None = None,
) -> None:
    """Register an existing static task while it migrates to :class:`RLTask`."""

    def environment_factory(context: RLTaskContext) -> RLEnvironmentCfg:
        return deepcopy(env_cfg if context.training else play_env_cfg)

    def trainer_factory(_: RLTaskContext) -> Any:
        return deepcopy(rl_cfg)

    register_task(
        RLTask(
            task_id=task_id,
            environment_cfg=environment_factory,
            trainer_cfg=trainer_factory,
            runner_cls=runner_cls,
        )
    )


def clear_registry() -> None:
    """Clear registrations; intended for isolated tests."""

    _REGISTRY.clear()


__all__ = [
    "RLTask",
    "RLTaskContext",
    "clear_registry",
    "get_task",
    "list_tasks",
    "load_env_cfg",
    "load_rl_cfg",
    "load_runner_cls",
    "register_lycheesim_task",
    "register_task",
]
