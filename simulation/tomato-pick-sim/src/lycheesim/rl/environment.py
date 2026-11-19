"""Composable RL environment for LycheeSim runtimes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .managers import (
    ActionManager,
    ActionTermCfg,
    CommandManager,
    CommandTermCfg,
    CurriculumManager,
    CurriculumTermCfg,
    EventManager,
    EventTermCfg,
    MetricsManager,
    MetricsTermCfg,
    NullCommandManager,
    NullCurriculumManager,
    NullMetricsManager,
    NullRecorderManager,
    ObservationGroupCfg,
    ObservationManager,
    RecorderManager,
    RecorderTermCfg,
    RewardManager,
    RewardTermCfg,
    TerminationManager,
    TerminationTermCfg,
)

from . import types
from .runtime import RuntimeCfg, SimulationRuntime
from .spaces import Box, DictSpace, batch_space


@dataclass(kw_only=True)
class RLEnvironmentCfg:
    """Configuration for a composable RL environment.

    The configuration mirrors the useful boundary of MJLab's environment API,
    but replaces MuJoCo-specific ``scene`` and ``sim`` objects with a LycheeSim
    :class:`RuntimeCfg`.
    """

    runtime: RuntimeCfg
    decimation: int = 1
    observations: dict[str, ObservationGroupCfg] = field(default_factory=dict)
    actions: dict[str, ActionTermCfg] = field(default_factory=dict)
    events: dict[str, EventTermCfg] = field(default_factory=dict)
    seed: int | None = None

    episode_length_s: float = 0.0
    rewards: dict[str, RewardTermCfg] = field(default_factory=dict)
    terminations: dict[str, TerminationTermCfg] = field(default_factory=dict)
    commands: dict[str, CommandTermCfg] = field(default_factory=dict)
    curriculum: dict[str, CurriculumTermCfg] = field(default_factory=dict)
    metrics: dict[str, MetricsTermCfg] = field(default_factory=dict)
    recorders: dict[str, RecorderTermCfg] = field(default_factory=dict)

    is_finite_horizon: bool = False
    auto_reset: bool = True
    scale_rewards_by_dt: bool = True

    def __post_init__(self) -> None:
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        if self.episode_length_s < 0.0:
            raise ValueError("episode_length_s cannot be negative")


class RLEnvironment:
    """Vector-environment facade over a stateful LycheeSim runtime."""

    is_vector_env = True
    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(
        self,
        cfg: RLEnvironmentCfg,
        device: str | None = None,
        render_mode: str | None = None,
        **_: Any,
    ) -> None:
        self.cfg = cfg
        self.render_mode = render_mode
        self.runtime: SimulationRuntime = cfg.runtime.build()
        runtime_device = torch.device(self.runtime.device)
        self._device = torch.device(device) if device is not None else runtime_device
        if self._device != runtime_device:
            raise ValueError(
                "The RL device must match runtime.device in this first implementation: "
                f"requested={self._device}, runtime={runtime_device}. Add an explicit "
                "device-transfer layer before separating them."
            )
        self._num_envs = int(self.runtime.num_envs)
        self._sim_step_counter = 0
        self.common_step_counter = 0
        self.extras: dict[str, Any] = {"log": {}}
        self.obs_buf: types.VecEnvObs = {}

        if cfg.seed is not None:
            cfg.seed = self.seed(cfg.seed)

        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.reset_terminated = torch.zeros_like(self.reset_buf)
        self.reset_time_outs = torch.zeros_like(self.reset_buf)
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)
        self._manual_reset_pending = torch.zeros_like(self.reset_buf)

        self.load_managers()
        self.single_action_space = Box(
            shape=(self.action_manager.total_action_dim,),
            low=float("-inf"),
            high=float("inf"),
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = DictSpace()
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
            self.runtime.refresh()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def physics_dt(self) -> float:
        return float(self.runtime.physics_dt)

    @property
    def step_dt(self) -> float:
        return self.physics_dt * self.cfg.decimation

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def max_episode_length_s(self) -> float:
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        if self.max_episode_length_s <= 0.0:
            return torch.iinfo(torch.int64).max
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def unwrapped(self) -> "RLEnvironment":
        return self

    def load_managers(self) -> None:
        # The ordering is intentional: randomization and commands may be consumed
        # by observations/actions; reward/termination are downstream of both.
        self.event_manager = EventManager(self.cfg.events, self)
        self.command_manager = (
            CommandManager(self.cfg.commands, self)
            if self.cfg.commands
            else NullCommandManager()
        )
        self.action_manager = ActionManager(self.cfg.actions, self)
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        self.reward_manager = RewardManager(
            self.cfg.rewards,
            self,
            scale_by_dt=self.cfg.scale_rewards_by_dt,
        )
        self.curriculum_manager = (
            CurriculumManager(self.cfg.curriculum, self)
            if self.cfg.curriculum
            else NullCurriculumManager()
        )
        self.metrics_manager = (
            MetricsManager(self.cfg.metrics, self)
            if self.cfg.metrics
            else NullMetricsManager()
        )
        self.recorder_manager = (
            RecorderManager(self.cfg.recorders, self)
            if self.cfg.recorders
            else NullRecorderManager()
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[types.VecEnvObs, dict[str, Any]]:
        del options
        if seed is not None:
            self.seed(seed)
        if env_ids is None:
            env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.extras = {"log": {}}
        self._reset_idx(env_ids)
        self.runtime.refresh()
        self.command_manager.compute(dt=0.0)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        self._configure_observation_spaces()
        self.recorder_manager.record_post_reset(env_ids)
        self._manual_reset_pending[env_ids] = False
        return self.obs_buf, self.extras

    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        if not self.cfg.auto_reset and self._manual_reset_pending.any():
            pending = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
            raise RuntimeError(
                f"Environments {pending.cpu().tolist()} must be reset before step()"
            )

        self.extras = {"log": {}}
        self.recorder_manager.record_pre_step()
        self.action_manager.process_action(action.to(self.device))

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.runtime.step()
            self.metrics_manager.compute_substep()

        # Derived task signals must represent the post-decimation state before RL
        # terms are evaluated.
        self.runtime.refresh()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        timeout_mask = self.termination_manager.time_outs
        if self.cfg.is_finite_horizon:
            self.reset_terminated = self.termination_manager.terminated | timeout_mask
            self.reset_time_outs = torch.zeros_like(timeout_mask)
        else:
            self.reset_terminated = self.termination_manager.terminated.clone()
            self.reset_time_outs = timeout_mask.clone()

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.metrics_manager.compute()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            if self.cfg.auto_reset:
                self.recorder_manager.record_pre_reset(reset_env_ids)
                self._reset_idx(reset_env_ids)
                self.runtime.refresh()
                self.recorder_manager.record_post_reset(reset_env_ids)
            else:
                self._manual_reset_pending[reset_env_ids] = True

        self.command_manager.compute(dt=self.step_dt)
        if "step" in self.event_manager.available_modes:
            self.event_manager.apply(mode="step")
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="interval",
                global_env_step_count=self.common_step_counter,
            )
        if {"step", "interval"} & self.event_manager.available_modes:
            self.runtime.refresh()

        self.obs_buf = self.observation_manager.compute(update_history=True)
        self.recorder_manager.record_post_step()
        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.curriculum_manager.compute(env_ids=env_ids)
        self.runtime.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                global_env_step_count=self.common_step_counter,
            )

        # Keep reset order explicit because histories and previous actions are
        # stateful and must be cleared before the next observation.
        for manager in (
            self.observation_manager,
            self.action_manager,
            self.reward_manager,
            self.metrics_manager,
            self.curriculum_manager,
            self.command_manager,
            self.event_manager,
            self.termination_manager,
            self.recorder_manager,
        ):
            info = manager.reset(env_ids)
            if info:
                self.extras["log"].update(info)
        self.episode_length_buf[env_ids] = 0
        self._manual_reset_pending[env_ids] = False

    def _configure_observation_spaces(self) -> None:
        single = DictSpace()
        for group_name, group_value in self.obs_buf.items():
            if isinstance(group_value, torch.Tensor):
                single.spaces[group_name] = Box(shape=tuple(group_value.shape[1:]))
            else:
                nested = DictSpace()
                for term_name, term_value in group_value.items():
                    nested.spaces[term_name] = Box(shape=tuple(term_value.shape[1:]))
                single.spaces[group_name] = nested
        self.single_observation_space = single
        self.observation_space = batch_space(single, self.num_envs)

    def render(self) -> Any:
        return self.runtime.render(self.render_mode)

    def close(self) -> None:
        self.recorder_manager.close()
        self.runtime.close()

    def seed(self, seed: int = -1) -> int:
        if seed == -1:
            seed = int(np.random.randint(0, 10_000))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_fn = getattr(self.runtime, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)
        return seed

    def state_dict(self) -> dict[str, Any]:
        runtime_state_fn = getattr(self.runtime, "state_dict", None)
        runtime_state = runtime_state_fn() if callable(runtime_state_fn) else {}
        return {
            "common_step_counter": self.common_step_counter,
            "sim_step_counter": self._sim_step_counter,
            "episode_length_buf": self.episode_length_buf.detach().cpu(),
            "runtime": runtime_state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.common_step_counter = int(state.get("common_step_counter", 0))
        self._sim_step_counter = int(state.get("sim_step_counter", 0))
        if "episode_length_buf" in state:
            self.episode_length_buf.copy_(
                torch.as_tensor(state["episode_length_buf"], device=self.device)
            )
        runtime_load = getattr(self.runtime, "load_state_dict", None)
        if callable(runtime_load) and "runtime" in state:
            runtime_load(state["runtime"])
        self.runtime.refresh()
