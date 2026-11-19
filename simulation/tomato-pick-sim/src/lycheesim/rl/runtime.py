"""Runtime boundary between LycheeSim physics and RL.

The existing demo runner owns callbacks. RL instead needs a stateful object whose
simulation state survives across calls and can be reset per episode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SimulationRuntime(Protocol):
    """Minimal simulator contract consumed by :class:`RLEnvironment`.

    ``physics_dt`` is the duration of one repeatable runtime tick exposed to the
    RL layer. A runtime tick may itself execute internal solver substeps. For the
    a deformable scene may expose one frame (``frame_dt``) while retaining its
    solver-specific substep loop inside the runtime.
    """

    num_envs: int
    device: str | torch.device
    physics_dt: float

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset all or selected environments to an episode-start state."""

    def step(self) -> None:
        """Advance one runtime tick using targets already written by actions."""

    def refresh(self) -> None:
        """Refresh observations/sensors derived from the current physics state."""

    def render(self, mode: str | None = None) -> Any:
        """Render the current state, or return ``None`` when unsupported."""

    def close(self) -> None:
        """Release simulator and renderer resources."""

    def state_dict(self) -> Mapping[str, Any]:
        """Return checkpointable runtime state when supported."""

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""


RuntimeFactory = Callable[..., SimulationRuntime]


@dataclass(kw_only=True)
class RuntimeCfg:
    """Serializable-ish recipe for constructing a simulation runtime."""

    factory: RuntimeFactory
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self) -> SimulationRuntime:
        runtime = self.factory(**self.kwargs)
        required = ("num_envs", "device", "physics_dt", "reset", "step", "refresh")
        missing = [name for name in required if not hasattr(runtime, name)]
        if missing:
            raise TypeError(
                f"Runtime {type(runtime).__name__} is missing required members: {missing}"
            )
        if int(runtime.num_envs) <= 0:
            raise ValueError("runtime.num_envs must be positive")
        if float(runtime.physics_dt) <= 0.0:
            raise ValueError("runtime.physics_dt must be positive")
        return runtime


@runtime_checkable
class SignalBackend(Protocol):
    """Small backend contract for stateful simulations exposed as signals."""

    physics_dt: float
    num_envs: int

    def step(self) -> None: ...

    def read_signals(self) -> Mapping[str, Any]: ...


SignalBackendFactory = Callable[[], SignalBackend]


@dataclass(frozen=True, kw_only=True)
class SignalCommandCfg:
    """Map one named RL command to a backend setter."""

    backend_method: str
    shape: tuple[int, ...]
    initial: float | Sequence[float] = 0.0

    def __post_init__(self) -> None:
        if not self.backend_method:
            raise ValueError("backend_method cannot be empty")
        if any(size <= 0 for size in self.shape):
            raise ValueError("command shape dimensions must be positive")


class SignalRuntime:
    """Reusable runtime for backends driven by commands and named signals.

    It handles device/host conversion, batching, selected-environment reset,
    monotonically clipped delta signals, rendering, and checkpoint state. A
    task supplies only command mappings and the signal names it requires.
    """

    def __init__(
        self,
        *,
        backend_factory: SignalBackendFactory,
        device: str = "cpu",
        commands: Mapping[str, SignalCommandCfg] | None = None,
        required_signals: Sequence[str] = (),
        deltas: Mapping[str, str] | None = None,
        rebuild_on_reset: bool = True,
        reuse_initial_state_on_first_reset: bool = False,
    ) -> None:
        self.backend_factory = backend_factory
        self.device = torch.device(device)
        self.command_cfg = dict(commands or {})
        self.required_signals = frozenset(required_signals)
        self.delta_sources = dict(deltas or {})
        self.rebuild_on_reset = bool(rebuild_on_reset)
        self._initial_reset_pending = bool(reuse_initial_state_on_first_reset)
        self._backend = backend_factory()
        self._install_backend(initial=True)
        self._commands = {
            name: self._initial_command(cfg)
            for name, cfg in self.command_cfg.items()
        }
        self._signals: dict[str, torch.Tensor] = {}
        self._previous: dict[str, torch.Tensor] = {}
        self._send_all_commands()
        self.refresh()

    @property
    def backend(self) -> SignalBackend:
        return self._backend

    def _install_backend(self, *, initial: bool) -> None:
        num_envs = int(getattr(self._backend, "num_envs", 1))
        if num_envs <= 0:
            raise ValueError("signal backend num_envs must be positive")
        if not initial and num_envs != self.num_envs:
            raise RuntimeError(
                "Signal backend changed num_envs during reset: "
                f"{self.num_envs} -> {num_envs}"
            )
        self.num_envs = num_envs
        backend_device = getattr(self._backend, "device", None)
        if backend_device is not None and torch.device(str(backend_device)) != self.device:
            raise ValueError(
                "Torch and backend devices must match: "
                f"torch={self.device}, backend={backend_device}"
            )
        self.physics_dt = float(self._backend.physics_dt)
        if self.physics_dt <= 0.0:
            raise ValueError("signal backend physics_dt must be positive")

    def _initial_command(self, cfg: SignalCommandCfg) -> torch.Tensor:
        value = torch.as_tensor(cfg.initial, dtype=torch.float32, device=self.device)
        target_shape = (self.num_envs, *cfg.shape)
        if value.ndim == 0:
            return torch.full(target_shape, float(value.item()), device=self.device)
        try:
            return torch.broadcast_to(value, target_shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                f"Initial command cannot broadcast to {target_shape}: {tuple(value.shape)}"
            ) from exc

    def _validate_env_ids(self, env_ids: torch.Tensor | None) -> torch.Tensor | None:
        if env_ids is None:
            return None
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() and bool(((ids < 0) | (ids >= self.num_envs)).any().item()):
            raise ValueError(f"Environment ids must be in [0, {self.num_envs})")
        return ids

    def _send_command(self, name: str) -> None:
        cfg = self.command_cfg[name]
        setter = getattr(self._backend, cfg.backend_method, None)
        if not callable(setter):
            raise TypeError(
                f"Signal backend {type(self._backend).__name__} has no callable "
                f"{cfg.backend_method!r} for command {name!r}"
            )
        value = self._commands[name]
        if bool(getattr(self._backend, "accepts_torch", False)):
            setter(value)
            return
        host = value.detach().cpu().numpy().astype(np.float32, copy=False)
        setter(host[0] if self.num_envs == 1 else host)

    def _send_all_commands(self) -> None:
        for name in self.command_cfg:
            self._send_command(name)

    def set_command(self, name: str, value: Any) -> None:
        try:
            target = self._commands[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown command {name!r}; available={sorted(self._commands)}"
            ) from exc
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.shape == target.shape[1:] and self.num_envs == 1:
            tensor = tensor.unsqueeze(0)
        if tuple(tensor.shape) != tuple(target.shape):
            raise ValueError(
                f"Expected command {name!r} shape {tuple(target.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        target.copy_(tensor)
        self._send_command(name)

    def command(self, name: str) -> torch.Tensor:
        try:
            return self._commands[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown command {name!r}; available={sorted(self._commands)}"
            ) from exc

    def step(self) -> None:
        self._initial_reset_pending = False
        self._send_all_commands()
        self._backend.step()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = self._validate_env_ids(env_ids)
        if self._initial_reset_pending:
            self._initial_reset_pending = False
        else:
            reset_fn = getattr(self._backend, "reset", None)
            can_reset = callable(reset_fn) and (
                not self.rebuild_on_reset
                or bool(getattr(self._backend, "supports_snapshot_reset", False))
            )
            if can_reset:
                reset_fn(ids)
            else:
                if ids is not None and self.num_envs > 1 and ids.numel() < self.num_envs:
                    raise RuntimeError(
                        "A batched backend must implement selected reset or snapshot reset"
                    )
                close_fn = getattr(self._backend, "close", None)
                if callable(close_fn):
                    close_fn()
                self._backend = self.backend_factory()
                self._install_backend(initial=False)

        for name, cfg in self.command_cfg.items():
            initial = self._initial_command(cfg)
            if ids is None:
                self._commands[name].copy_(initial)
            else:
                self._commands[name][ids] = initial[ids]
        self._send_all_commands()
        self._signals.clear()
        for value in self._previous.values():
            if ids is None:
                value.zero_()
            else:
                value[ids] = 0.0

    @staticmethod
    def _as_batched_signal(
        value: Any, *, device: torch.device, num_envs: int
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 0:
            return tensor.expand(num_envs)
        if tensor.shape[0] == num_envs:
            return tensor
        if num_envs == 1:
            return tensor.unsqueeze(0)
        raise ValueError(
            f"Signal leading dimension must be {num_envs}, got {tuple(tensor.shape)}"
        )

    def refresh(self) -> None:
        raw = dict(self._backend.read_signals())
        missing = sorted(self.required_signals - raw.keys())
        if missing:
            raise KeyError(
                f"Signal backend read_signals() is missing required keys: {missing}"
            )
        signals = {
            name: self._as_batched_signal(
                value, device=self.device, num_envs=self.num_envs
            )
            for name, value in raw.items()
        }
        for name, value in self._commands.items():
            signals.setdefault(name, value.clone())
        for output_name, source_name in self.delta_sources.items():
            if source_name not in signals:
                raise KeyError(
                    f"Delta signal {output_name!r} references missing {source_name!r}"
                )
            current = signals[source_name].reshape(self.num_envs)
            previous = self._previous.get(source_name)
            if previous is None:
                previous = current.clone()
            signals[output_name] = torch.clamp(current - previous, min=0.0)
            self._previous[source_name] = current.clone()
        self._signals = signals

    def signal(self, name: str) -> torch.Tensor:
        try:
            return self._signals[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown signal {name!r}; available={sorted(self._signals)}"
            ) from exc

    def signals(self) -> Mapping[str, torch.Tensor]:
        return self._signals

    def render(self, mode: str | None = None) -> Any:
        render_fn = getattr(self._backend, "render", None)
        return render_fn(mode) if callable(render_fn) else None

    def close(self) -> None:
        close_fn = getattr(self._backend, "close", None)
        if callable(close_fn):
            close_fn()

    def seed(self, seed: int) -> None:
        seed_fn = getattr(self._backend, "seed", None)
        if callable(seed_fn):
            seed_fn(seed)

    def state_dict(self) -> Mapping[str, Any]:
        state_fn = getattr(self._backend, "state_dict", None)
        return {
            "commands": {name: value.detach().cpu() for name, value in self._commands.items()},
            "previous": {name: value.detach().cpu() for name, value in self._previous.items()},
            "backend": state_fn() if callable(state_fn) else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("backend") is not None:
            load_fn = getattr(self._backend, "load_state_dict", None)
            if callable(load_fn):
                load_fn(state["backend"])
        for name, value in state.get("commands", {}).items():
            self.set_command(name, value)
        for name, value in state.get("previous", {}).items():
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            self._previous[name] = tensor.clone()
        self.refresh()
