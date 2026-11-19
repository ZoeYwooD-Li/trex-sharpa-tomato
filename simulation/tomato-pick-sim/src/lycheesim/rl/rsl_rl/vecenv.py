"""Adapter from :class:`RLEnvironment` to RSL-RL's VecEnv interface."""

from __future__ import annotations

from typing import Any

import torch

from ..environment import RLEnvironment, RLEnvironmentCfg
from ..spaces import Box, batch_space

try:  # Optional dependency: keep core environment imports usable without RSL-RL.
    from rsl_rl.env import VecEnv as _VecEnv
except ImportError:  # pragma: no cover - exercised only when optional dep is absent.
    _VecEnv = object  # type: ignore[assignment,misc]


class RslRlVecEnvWrapper(_VecEnv):  # type: ignore[misc]
    def __init__(
        self,
        env: RLEnvironment,
        clip_actions: float | None = None,
    ) -> None:
        try:
            from tensordict import TensorDict
        except ImportError as exc:
            raise ImportError(
                "RslRlVecEnvWrapper requires the optional dependencies "
                "'rsl-rl-lib' and 'tensordict'."
            ) from exc
        try:
            import rsl_rl  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RslRlVecEnvWrapper requires the optional package 'rsl-rl-lib'."
            ) from exc

        self._TensorDict = TensorDict
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim
        self._modify_action_space()

        # RSL-RL does not own the initial environment reset.
        self.env.reset()

    @property
    def cfg(self) -> RLEnvironmentCfg:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> RLEnvironment:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def _to_tensordict(self, observations: dict[str, Any]) -> Any:
        return self._TensorDict(observations, batch_size=[self.num_envs])

    def get_observations(self) -> Any:
        obs_dict = self.unwrapped.observation_manager.compute(update_history=False)
        return self._to_tensordict(obs_dict)

    def reset(self) -> tuple[Any, dict[str, Any]]:
        obs_dict, extras = self.env.reset()
        return self._to_tensordict(obs_dict), extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._to_tensordict(obs_dict), rewards, dones, extras

    def close(self) -> None:
        self.env.close()

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return
        self.unwrapped.single_action_space = Box(
            shape=(self.num_actions,),
            low=-float(self.clip_actions),
            high=float(self.clip_actions),
        )
        self.unwrapped.action_space = batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )
