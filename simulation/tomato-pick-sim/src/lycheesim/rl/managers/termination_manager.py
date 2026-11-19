"""RL termination and time-limit terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .base import batched_scalar

TerminationFn = Callable[..., Any]


@dataclass(kw_only=True)
class TerminationTermCfg:
    func: TerminationFn
    time_out: bool = False
    params: dict[str, Any] = field(default_factory=dict)


class TerminationManager:
    def __init__(self, cfg: dict[str, TerminationTermCfg], env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)
        self.terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.time_outs = torch.zeros_like(self.terminated)
        self.term_dones = {
            name: torch.zeros_like(self.terminated) for name in self.active_terms
        }
        self.term_counts = {
            name: torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
            for name in self.active_terms
        }

    def compute(self) -> torch.Tensor:
        self.terminated.zero_()
        self.time_outs.zero_()
        for name, term_cfg in self.cfg.items():
            value = batched_scalar(
                term_cfg.func(self.env, **term_cfg.params),
                num_envs=self.env.num_envs,
                device=self.env.device,
                dtype=torch.bool,
            ).to(torch.bool)
            self.term_dones[name] = value
            self.term_counts[name] += value.to(torch.long)
            if term_cfg.time_out:
                self.time_outs |= value
            else:
                self.terminated |= value
        return self.terminated | self.time_outs

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        logs: dict[str, float] = {}
        if len(env_ids) > 0:
            for name, counts in self.term_counts.items():
                logs[f"Episode/Termination/{name}"] = float(
                    counts[env_ids].to(torch.float32).mean().item()
                )
                counts[env_ids] = 0
        return logs
