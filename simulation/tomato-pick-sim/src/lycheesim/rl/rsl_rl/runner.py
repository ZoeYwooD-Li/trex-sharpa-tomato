"""Optional RSL-RL runner extensions for environment checkpoint state."""

from __future__ import annotations

import warnings
from typing import Any

import torch

try:  # Optional dependency.
    from rsl_rl.runners import OnPolicyRunner as _OnPolicyRunner
except ImportError:  # pragma: no cover
    _OnPolicyRunner = object  # type: ignore[assignment,misc]


class LycheeSimOnPolicyRunner(_OnPolicyRunner):  # type: ignore[misc]
    """On-policy runner that stores manager/runtime counters in checkpoint infos."""

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        if _OnPolicyRunner is object:
            raise ImportError(
                "LycheeSimOnPolicyRunner requires the optional package 'rsl-rl-lib'."
            )
        # RSL-RL model constructors do not accept absent optional features as
        # explicit None values in all releases.
        for key in ("actor", "critic"):
            model_cfg = train_cfg.get(key)
            if not isinstance(model_cfg, dict):
                continue
            for optional in ("cnn_cfg", "distribution_cfg"):
                if model_cfg.get(optional) is None:
                    model_cfg.pop(optional, None)
            if model_cfg.get("rnn_type") is None:
                for optional in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                    model_cfg.pop(optional, None)
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        env_state = self.env.unwrapped.state_dict()
        merged = {**(infos or {}), "lycheesim_env_state": env_state}
        try:
            super().save(path, infos=merged)
        except TypeError:
            # Compatibility fallback for RSL-RL releases with positional infos.
            super().save(path, merged)

    def load(self, path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        infos = super().load(path, *args, **kwargs)
        if infos and "lycheesim_env_state" in infos:
            env_state = infos["lycheesim_env_state"]
            saved_episode_lengths = env_state.get("episode_length_buf")
            saved_num_envs = (
                int(torch.as_tensor(saved_episode_lengths).numel())
                if saved_episode_lengths is not None
                else self.env.num_envs
            )
            if saved_num_envs == self.env.num_envs:
                self.env.unwrapped.load_state_dict(env_state)
            else:
                # Evaluation commonly loads a many-environment training
                # checkpoint into one visual environment and immediately
                # resets it.  Model, optimizer and normalization state remain
                # valid; only the simulator counters/runtime snapshot have an
                # incompatible leading dimension.
                warnings.warn(
                    "Skipping LycheeSim environment state from checkpoint: "
                    f"saved num_envs={saved_num_envs}, current "
                    f"num_envs={self.env.num_envs}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return infos
