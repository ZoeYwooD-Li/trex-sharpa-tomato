"""Optional RL command generators (targets, goals, desired velocities)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .base import batched_tensor

CommandFn = Callable[..., Any]


@dataclass(kw_only=True)
class CommandTermCfg:
    func: CommandFn
    params: dict[str, Any] = field(default_factory=dict)
    resampling_time_range_s: tuple[float, float] = (1.0, 1.0)


class CommandManager:
    def __init__(self, cfg: dict[str, CommandTermCfg], env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)
        self.commands: dict[str, torch.Tensor] = {}
        self._time_left = {
            name: torch.zeros(env.num_envs, device=env.device) for name in cfg
        }

    def _sample_duration(self, name: str, count: int) -> torch.Tensor:
        low, high = self.cfg[name].resampling_time_range_s
        if low <= 0.0 or high < low:
            raise ValueError(
                f"Invalid resampling_time_range_s for command {name!r}: {(low, high)}"
            )
        if high == low:
            return torch.full((count,), float(low), device=self.env.device)
        return torch.empty(count, device=self.env.device).uniform_(float(low), float(high))

    def _resample(self, name: str, env_ids: torch.Tensor) -> None:
        term_cfg = self.cfg[name]
        value = batched_tensor(
            term_cfg.func(self.env, env_ids=env_ids, **term_cfg.params),
            num_envs=len(env_ids),
            device=self.env.device,
        )
        current = self.commands.get(name)
        if current is None:
            current = torch.zeros(
                (self.env.num_envs, *value.shape[1:]), device=self.env.device
            )
            self.commands[name] = current
        if tuple(current.shape[1:]) != tuple(value.shape[1:]):
            raise ValueError(f"Command {name!r} changed shape across resamples")
        current[env_ids] = value
        self._time_left[name][env_ids] = self._sample_duration(name, len(env_ids))

    def compute(self, *, dt: float) -> None:
        for name in self.active_terms:
            self._time_left[name] -= float(dt)
            ids = (self._time_left[name] <= 0.0).nonzero(as_tuple=False).squeeze(-1)
            if len(ids) > 0:
                self._resample(name, ids)

    def get_command(self, name: str) -> torch.Tensor:
        return self.commands[name]

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        for name in self.active_terms:
            self._resample(name, env_ids)
        return {}


class NullCommandManager:
    active_terms: list[str] = []

    def compute(self, *, dt: float) -> None:
        del dt

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        del env_ids
        return {}

    def get_command(self, name: str) -> torch.Tensor:
        raise KeyError(f"No command terms are configured; requested {name!r}")
