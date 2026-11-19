"""RL action slicing, scaling, clipping, and application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .base import parameter_tensor

ActionApplyFn = Callable[..., None]


@dataclass(kw_only=True)
class ActionTermCfg:
    func: ActionApplyFn
    action_dim: int
    scale: float | Sequence[float] | torch.Tensor = 1.0
    offset: float | Sequence[float] | torch.Tensor = 0.0
    clip: tuple[float, float] | None = (-1.0, 1.0)
    params: dict[str, Any] = field(default_factory=dict)


class ActionManager:
    def __init__(self, cfg: dict[str, ActionTermCfg], env: Any) -> None:
        if not cfg:
            raise ValueError("At least one action term is required")
        self.cfg = cfg
        self.env = env
        self.active_terms = list(cfg)
        self.action_term_dim = [int(cfg[name].action_dim) for name in self.active_terms]
        if any(dim <= 0 for dim in self.action_term_dim):
            raise ValueError("Every action term must have action_dim > 0")
        self.total_action_dim = sum(self.action_term_dim)

        self._slices: dict[str, slice] = {}
        cursor = 0
        for name, dim in zip(self.active_terms, self.action_term_dim, strict=True):
            self._slices[name] = slice(cursor, cursor + dim)
            cursor += dim

        shape = (env.num_envs, self.total_action_dim)
        self.action = torch.zeros(shape, device=env.device)
        self.prev_action = torch.zeros_like(self.action)
        self.processed_actions: dict[str, torch.Tensor] = {
            name: torch.zeros((env.num_envs, cfg[name].action_dim), device=env.device)
            for name in self.active_terms
        }

    def process_action(self, action: torch.Tensor) -> None:
        action = torch.as_tensor(action, device=self.env.device, dtype=torch.float32)
        if action.ndim == 1 and self.env.num_envs == 1:
            action = action.unsqueeze(0)
        expected = (self.env.num_envs, self.total_action_dim)
        if tuple(action.shape) != expected:
            raise ValueError(f"Expected actions with shape {expected}, got {tuple(action.shape)}")

        self.prev_action.copy_(self.action)
        self.action.copy_(action)
        for name in self.active_terms:
            term_cfg = self.cfg[name]
            value = action[:, self._slices[name]]
            if term_cfg.clip is not None:
                value = torch.clamp(value, float(term_cfg.clip[0]), float(term_cfg.clip[1]))
            scale = parameter_tensor(
                term_cfg.scale, device=self.env.device, width=term_cfg.action_dim
            )
            offset = parameter_tensor(
                term_cfg.offset, device=self.env.device, width=term_cfg.action_dim
            )
            # Keep the preallocated mutable buffer. RSL-RL collects rollouts
            # under torch.inference_mode(); rebinding to the expression result
            # would make a later ordinary reset unable to clear the tensor.
            self.processed_actions[name].copy_(value * scale + offset)

    def apply_action(self) -> None:
        for name in self.active_terms:
            term_cfg = self.cfg[name]
            term_cfg.func(
                self.env,
                self.processed_actions[name],
                **term_cfg.params,
            )

    def get_term(self, name: str) -> torch.Tensor:
        return self.processed_actions[name]

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        self.action[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0
        for value in self.processed_actions.values():
            value[env_ids] = 0.0
        return {}
