"""RL startup, reset, step, and interval events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

EventMode = Literal["startup", "reset", "step", "interval"]
EventFn = Callable[..., None]


@dataclass(kw_only=True)
class EventTermCfg:
    func: EventFn
    mode: EventMode
    params: dict[str, Any] = field(default_factory=dict)
    interval_steps: int | None = None


class EventManager:
    def __init__(self, cfg: dict[str, EventTermCfg], env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)
        self.available_modes = {term.mode for term in cfg.values()}
        for name, term in cfg.items():
            if term.mode == "interval" and (
                term.interval_steps is None or term.interval_steps <= 0
            ):
                raise ValueError(
                    f"Interval event {name!r} requires interval_steps > 0"
                )

    def apply(
        self,
        *,
        mode: EventMode,
        env_ids: torch.Tensor | None = None,
        global_env_step_count: int | None = None,
    ) -> None:
        for term_cfg in self.cfg.values():
            if term_cfg.mode != mode:
                continue
            if mode == "interval":
                assert term_cfg.interval_steps is not None
                if global_env_step_count is None:
                    raise ValueError("interval events require global_env_step_count")
                if global_env_step_count % term_cfg.interval_steps != 0:
                    continue
            term_cfg.func(self.env, env_ids=env_ids, **term_cfg.params)

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        return {}
