"""Framework-neutral policy execution for LycheeSim RL environments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from .environment import RLEnvironment
from .types import VecEnvObs

if TYPE_CHECKING:
    from .task import RLTaskContext


@runtime_checkable
class Policy(Protocol):
    """Minimal policy contract understood by :class:`RLExecutor`."""

    def act(self, observation: Any) -> torch.Tensor: ...

    def reset(self, done: torch.Tensor | None = None) -> None: ...


PolicyLike = Policy | Callable[[Any], torch.Tensor]


def _lookup(value: Any, path: str) -> Any:
    for part in path.split("/"):
        if not part:
            continue
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Unknown output path {path!r}")
        value = value[part]
    return value


@dataclass(frozen=True)
class RLStep:
    """Complete result of one policy-rate environment step."""

    observations: VecEnvObs
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    extras: Mapping[str, Any]
    signals: Mapping[str, torch.Tensor]
    policy_observation: Any

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    def output(self, name: str) -> Any:
        """Read a standard output or a slash-delimited nested value.

        Supported roots are ``observations``, ``signals``, ``extras``,
        ``action``, ``reward``, ``terminated``, ``truncated``, ``done``, and
        ``policy_observation``. For example: ``observations/critic`` or
        ``signals/cut_coverage``.
        """

        root, _, path = name.partition("/")
        values = {
            "observations": self.observations,
            "observation": self.observations,
            "signals": self.signals,
            "extras": self.extras,
            "action": self.action,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "policy_observation": self.policy_observation,
        }
        if root not in values:
            raise KeyError(f"Unknown RL output {root!r}; available={sorted(values)}")
        value = values[root]
        return _lookup(value, path) if path else value

    def select(self, *names: str) -> dict[str, Any]:
        return {name: self.output(name) for name in names}


@dataclass(frozen=True)
class Rollout:
    """Selected outputs collected by :meth:`RLExecutor.run`."""

    steps: tuple[Mapping[str, Any], ...]
    final_observations: VecEnvObs


class RLExecutor:
    """Own an environment and expose one uniform policy/action/output API."""

    def __init__(
        self,
        environment: RLEnvironment,
        *,
        policy_observation: str = "actor",
    ) -> None:
        self.environment = environment
        self.policy_observation = policy_observation
        self._observations: VecEnvObs | None = None
        self._last_step: RLStep | None = None

    @classmethod
    def from_task(
        cls,
        task_name: str,
        context: "RLTaskContext",
        *,
        policy_observation: str = "actor",
        render_mode: str | None = None,
    ) -> "RLExecutor":
        """Build an executor directly from a registered task and context."""

        from .task import load_env_cfg

        environment = RLEnvironment(
            load_env_cfg(task_name, context=context),
            device=context.device,
            render_mode=render_mode,
        )
        return cls(environment, policy_observation=policy_observation)

    @property
    def observations(self) -> VecEnvObs:
        if self._observations is None:
            raise RuntimeError("reset() must be called before reading observations")
        return self._observations

    @property
    def last_step(self) -> RLStep:
        if self._last_step is None:
            raise RuntimeError("step() must be called before reading the last step")
        return self._last_step

    def observe(self, group: str | None = None) -> Any:
        """Return one observation group, or every group when ``group`` is empty."""

        if group is None:
            return self.observations
        return _lookup(self.observations, group)

    def reset(
        self,
        *,
        seed: int | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> Any:
        self._observations, _ = self.environment.reset(seed=seed, env_ids=env_ids)
        self._last_step = None
        return self.observe(self.policy_observation)

    def step(self, action: Any) -> RLStep:
        action_tensor = torch.as_tensor(
            action,
            dtype=torch.float32,
            device=self.environment.device,
        )
        if action_tensor.ndim == 1 and self.environment.num_envs == 1:
            action_tensor = action_tensor.unsqueeze(0)
        observations, reward, terminated, truncated, extras = self.environment.step(
            action_tensor
        )
        self._observations = observations
        signals_fn = getattr(self.environment.runtime, "signals", None)
        signals = dict(signals_fn()) if callable(signals_fn) else {}
        self._last_step = RLStep(
            observations=observations,
            action=action_tensor,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            extras=extras,
            signals=signals,
            policy_observation=self.observe(self.policy_observation),
        )
        return self._last_step

    def output(self, name: str) -> Any:
        """Read one output from the latest step."""

        return self.last_step.output(name)

    @staticmethod
    def _act(policy: PolicyLike, observation: Any) -> torch.Tensor:
        act = getattr(policy, "act", None)
        return act(observation) if callable(act) else policy(observation)

    def run(
        self,
        policy: PolicyLike,
        *,
        steps: int,
        outputs: Sequence[str] = ("reward", "done"),
        stop_on_done: bool = False,
        callback: Callable[[RLStep], None] | None = None,
    ) -> Rollout:
        """Execute a policy without coupling it to a training library."""

        if steps <= 0:
            raise ValueError("steps must be positive")
        if self._observations is None:
            self.reset()

        records: list[Mapping[str, Any]] = []
        for _ in range(steps):
            with torch.inference_mode():
                action = self._act(policy, self.observe(self.policy_observation))
            result = self.step(action)
            records.append(result.select(*outputs))
            if callback is not None:
                callback(result)
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy) and bool(result.done.any().item()):
                reset_policy(result.done)
            if stop_on_done and bool(result.done.any().item()):
                break
        return Rollout(tuple(records), self.observations)

    def close(self) -> None:
        self.environment.close()

    def __enter__(self) -> "RLExecutor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


__all__ = ["Policy", "PolicyLike", "RLExecutor", "RLStep", "Rollout"]
