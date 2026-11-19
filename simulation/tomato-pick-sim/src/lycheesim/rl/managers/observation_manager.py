"""Composable RL observation terms and actor/critic groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .base import batched_tensor, parameter_tensor

ObservationFn = Callable[..., Any]
NoiseFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(kw_only=True)
class ObservationTermCfg:
    func: ObservationFn
    params: dict[str, Any] = field(default_factory=dict)
    scale: float | Sequence[float] | torch.Tensor = 1.0
    clip: tuple[float, float] | None = None
    noise: NoiseFn | None = None
    history_length: int = 1
    flatten_history_dim: bool = True


@dataclass(kw_only=True)
class ObservationGroupCfg:
    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)
    concatenate_terms: bool = True
    preserve_single_term_shape: bool = False
    enable_corruption: bool = False


class ObservationManager:
    def __init__(self, cfg: dict[str, ObservationGroupCfg], env: Any) -> None:
        if not cfg:
            raise ValueError("At least one observation group is required")
        self.cfg = cfg
        self.env = env
        self.active_terms = {name: list(group.terms) for name, group in cfg.items()}
        self.group_obs_concatenate = {
            name: group.concatenate_terms for name, group in cfg.items()
        }
        self.group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = {}
        self._history: dict[tuple[str, str], torch.Tensor] = {}
        self._history_needs_fill: dict[tuple[str, str], torch.Tensor] = {}

        for group_name, group in cfg.items():
            if not group.terms:
                raise ValueError(f"Observation group {group_name!r} has no terms")
            if group.preserve_single_term_shape and len(group.terms) != 1:
                raise ValueError(
                    f"{group_name}: preserve_single_term_shape requires exactly one term"
                )
            for term_name, term in group.terms.items():
                if term.history_length <= 0:
                    raise ValueError(
                        f"{group_name}.{term_name}: history_length must be positive"
                    )
                self._history_needs_fill[(group_name, term_name)] = torch.ones(
                    env.num_envs, dtype=torch.bool, device=env.device
                )

    def _compute_term(
        self,
        group_name: str,
        term_name: str,
        *,
        update_history: bool,
    ) -> torch.Tensor:
        group_cfg = self.cfg[group_name]
        term_cfg = group_cfg.terms[term_name]
        value = batched_tensor(
            term_cfg.func(self.env, **term_cfg.params),
            num_envs=self.env.num_envs,
            device=self.env.device,
        )

        flat_width = int(value[0].numel())
        scale = parameter_tensor(term_cfg.scale, device=self.env.device, width=flat_width)
        value = (value.reshape(self.env.num_envs, flat_width) * scale).reshape_as(value)
        if term_cfg.clip is not None:
            value = torch.clamp(value, float(term_cfg.clip[0]), float(term_cfg.clip[1]))
        if group_cfg.enable_corruption and term_cfg.noise is not None:
            value = term_cfg.noise(value)

        if term_cfg.history_length == 1:
            return value

        key = (group_name, term_name)
        expected_shape = (self.env.num_envs, term_cfg.history_length, *value.shape[1:])
        history = self._history.get(key)
        if history is None or tuple(history.shape) != expected_shape:
            history = value.unsqueeze(1).expand(expected_shape).clone()
            self._history[key] = history
            self._history_needs_fill[key].fill_(False)
        elif update_history:
            history.copy_(torch.roll(history, shifts=-1, dims=1))
            history[:, -1] = value
            fill_ids = self._history_needs_fill[key]
            if fill_ids.any():
                history[fill_ids] = value[fill_ids].unsqueeze(1)
                self._history_needs_fill[key][fill_ids] = False

        if term_cfg.flatten_history_dim:
            return history.flatten(start_dim=1)
        return history

    def compute(self, *, update_history: bool = True) -> dict[str, Any]:
        observations: dict[str, Any] = {}
        for group_name, group_cfg in self.cfg.items():
            term_values = {
                term_name: self._compute_term(
                    group_name, term_name, update_history=update_history
                )
                for term_name in group_cfg.terms
            }
            if group_cfg.preserve_single_term_shape:
                group_value = next(iter(term_values.values()))
                observations[group_name] = group_value
                self.group_obs_dim[group_name] = tuple(group_value.shape[1:])
            elif group_cfg.concatenate_terms:
                flattened = [
                    value.reshape(self.env.num_envs, -1) for value in term_values.values()
                ]
                group_value = torch.cat(flattened, dim=-1)
                observations[group_name] = group_value
                self.group_obs_dim[group_name] = tuple(group_value.shape[1:])
            else:
                observations[group_name] = term_values
                self.group_obs_dim[group_name] = [
                    tuple(value.shape[1:]) for value in term_values.values()
                ]
        return observations

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        for flag in self._history_needs_fill.values():
            flag[env_ids] = True
        return {}
