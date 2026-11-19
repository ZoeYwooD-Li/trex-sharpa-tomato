"""Weighted RL reward terms with episodic accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .base import batched_scalar

RewardFn = Callable[..., Any]


@dataclass(kw_only=True)
class RewardTermCfg:
    func: RewardFn
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


class RewardManager:
    def __init__(
        self,
        cfg: dict[str, RewardTermCfg],
        env: Any,
        *,
        scale_by_dt: bool = True,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.scale_by_dt = bool(scale_by_dt)
        self.active_terms = list(cfg)
        self.term_values = {
            name: torch.zeros(env.num_envs, device=env.device) for name in cfg
        }
        self.episode_sums = {
            name: torch.zeros(env.num_envs, device=env.device) for name in cfg
        }

    def compute(self, *, dt: float) -> torch.Tensor:
        reward = torch.zeros(self.env.num_envs, device=self.env.device)
        time_scale = float(dt) if self.scale_by_dt else 1.0
        for name, term_cfg in self.cfg.items():
            raw = batched_scalar(
                term_cfg.func(self.env, **term_cfg.params),
                num_envs=self.env.num_envs,
                device=self.env.device,
            )
            weighted = raw * float(term_cfg.weight) * time_scale
            self.term_values[name] = weighted
            self.episode_sums[name] += weighted
            reward += weighted
        return reward

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        logs: dict[str, float] = {}
        if len(env_ids) > 0:
            for name, total in self.episode_sums.items():
                logs[f"Episode/Reward/{name}"] = float(total[env_ids].mean().item())
                total[env_ids] = 0.0
                func = self.cfg[name].func
                reset_fn = getattr(func, "reset", None)
                if callable(reset_fn):
                    reset_fn(self.env, env_ids)
        return logs
