"""RL per-step and per-substep scalar metric accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .base import batched_scalar

MetricFn = Callable[..., Any]


@dataclass(kw_only=True)
class MetricsTermCfg:
    func: MetricFn
    params: dict[str, Any] = field(default_factory=dict)
    per_substep: bool = False


class MetricsManager:
    def __init__(self, cfg: dict[str, MetricsTermCfg], env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)
        self.sums = {
            name: torch.zeros(env.num_envs, device=env.device) for name in cfg
        }
        self.counts = {
            name: torch.zeros(env.num_envs, device=env.device) for name in cfg
        }

    def _compute_filtered(self, *, per_substep: bool) -> None:
        for name, term_cfg in self.cfg.items():
            if term_cfg.per_substep != per_substep:
                continue
            value = batched_scalar(
                term_cfg.func(self.env, **term_cfg.params),
                num_envs=self.env.num_envs,
                device=self.env.device,
            )
            self.sums[name] += value
            self.counts[name] += 1.0

    def compute_substep(self) -> None:
        self._compute_filtered(per_substep=True)

    def compute(self) -> None:
        self._compute_filtered(per_substep=False)

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        logs: dict[str, float] = {}
        for name in self.active_terms:
            denominator = torch.clamp(self.counts[name][env_ids], min=1.0)
            value = self.sums[name][env_ids] / denominator
            if len(env_ids) > 0:
                logs[f"Episode/Metric/{name}"] = float(value.mean().item())
            self.sums[name][env_ids] = 0.0
            self.counts[name][env_ids] = 0.0
        return logs


class NullMetricsManager:
    active_terms: list[str] = []

    def compute_substep(self) -> None:
        return None

    def compute(self) -> None:
        return None

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        del env_ids
        return {}
