"""Optional RL curriculum callbacks evaluated at episode reset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

CurriculumFn = Callable[..., Any]


@dataclass(kw_only=True)
class CurriculumTermCfg:
    func: CurriculumFn
    params: dict[str, Any] = field(default_factory=dict)


class CurriculumManager:
    def __init__(self, cfg: dict[str, CurriculumTermCfg], env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)
        self._latest: dict[str, Any] = {}

    def compute(self, *, env_ids: torch.Tensor) -> None:
        for name, term_cfg in self.cfg.items():
            self._latest[name] = term_cfg.func(
                self.env, env_ids=env_ids, **term_cfg.params
            )

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        del env_ids
        logs: dict[str, float] = {}
        for name, value in self._latest.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                logs[f"Curriculum/{name}"] = float(value.float().mean().item())
            elif isinstance(value, (int, float)):
                logs[f"Curriculum/{name}"] = float(value)
        return logs


class NullCurriculumManager:
    active_terms: list[str] = []

    def compute(self, *, env_ids: torch.Tensor) -> None:
        del env_ids

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        del env_ids
        return {}
