"""Lightweight RL rollout hook manager.

A recorder callback receives ``event`` plus optional ``env_ids``. This keeps the
hot loop free of file I/O unless a recorder is explicitly configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

RecorderFn = Callable[..., None]


@dataclass(kw_only=True)
class RecorderTermCfg:
    func: RecorderFn
    params: dict[str, Any] = field(default_factory=dict)


class RecorderManager:
    def __init__(self, cfg: dict[str, RecorderTermCfg], env: Any) -> None:
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)

    def _record(self, event: str, env_ids: torch.Tensor | None = None) -> None:
        for term_cfg in self.cfg.values():
            term_cfg.func(
                self.env,
                event=event,
                env_ids=env_ids,
                **term_cfg.params,
            )

    def record_pre_step(self) -> None:
        self._record("pre_step")

    def record_post_step(self) -> None:
        self._record("post_step")

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        self._record("pre_reset", env_ids)

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        self._record("post_reset", env_ids)

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        del env_ids
        return {}

    def close(self) -> None:
        self._record("close")


class NullRecorderManager:
    active_terms: list[str] = []

    def record_pre_step(self) -> None:
        return None

    def record_post_step(self) -> None:
        return None

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def record_post_reset(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        del env_ids
        return {}

    def close(self) -> None:
        return None
