"""RL task registration for the cherry-tomato harvesting demo."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from lycheesim.rl import (
    ActionTermCfg,
    LycheeSimOnPolicyRunner,
    MetricsTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RLEnvironmentCfg,
    RLTask,
    RLTaskContext,
    RewardTermCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RuntimeCfg,
    SignalCommandCfg,
    SignalRuntime,
    TerminationTermCfg,
    list_tasks,
    register_task,
)

from . import mdp
from .backend import CherryTomatoPickBackend


TASK_ID = "LycheeSim-CherryTomatoPick-v0"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPOSITORY_ROOT / "configs" / "cherry_tomato_pick.yaml"
_REQUIRED_SIGNALS = (
    "left_arm_q",
    "right_arm_q",
    "left_tcp_pose",
    "right_tcp_pose",
    "hand_closure",
    "tomato_to_left_hand",
    "support_to_right_hand",
    "tomato_pos",
    "tomato_quat",
    "tomato_lin_vel",
    "tomato_ang_vel",
    "attachment_error",
    "attachment_load",
    "attachment_force_N",
    "attachment_torque_Nm",
    "break_force_N",
    "break_torque_Nm",
    "detached",
    "support_ready",
    "grasped",
    "harvest_distance",
    "left_contact_force_N",
    "right_contact_force_N",
    "contact_force_N",
    "ik_failed",
    "dropped",
    "success",
)


def load_cherry_tomato_pick_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = _DEFAULT_CONFIG if path is None else Path(path).expanduser().resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("cherry_tomato_pick config must contain a mapping")
    return value


def _apply_context_options(cfg: dict[str, Any], options: dict[str, Any]) -> None:
    supported = {
        "break_force_N": ("attachment", "break_force_N", float),
        "break_torque_Nm": ("attachment", "break_torque_Nm", float),
        "position_randomization_m": ("rl", "randomization", "position_xy_m", float),
    }
    for name, raw in options.items():
        if name not in supported:
            raise ValueError(
                f"unknown cherry-tomato task option {name!r}; choose from {sorted(supported)}"
            )
        path = supported[name]
        converter = path[-1]
        target = cfg
        for key in path[:-2]:
            target = target[key]
        target[path[-2]] = converter(raw)


def _build_backend(
    cfg: dict[str, Any],
    *,
    num_envs: int,
    device: str,
    training: bool,
    seed: int,
    config_path: Path | None,
) -> CherryTomatoPickBackend:
    return CherryTomatoPickBackend(
        cfg,
        num_envs=num_envs,
        device=device,
        randomize=training,
        enable_visual_assets=False,
        seed=seed,
        config_path=config_path,
    )


def make_cherry_tomato_pick_env_cfg(
    backend_factory,
    *,
    device: str,
    training: bool,
    cfg: dict[str, Any],
) -> RLEnvironmentCfg:
    runtime = RuntimeCfg(
        factory=SignalRuntime,
        kwargs={
            "backend_factory": backend_factory,
            "device": device,
            "commands": {
                "dual_arm_command": SignalCommandCfg(
                    backend_method="set_robot_command", shape=(10,)
                )
            },
            "required_signals": _REQUIRED_SIGNALS,
            "deltas": {
                "attachment_load_delta": "attachment_load",
                "detached_delta": "detached",
                "harvest_distance_delta": "harvest_distance",
            },
            "rebuild_on_reset": False,
            "reuse_initial_state_on_first_reset": True,
        },
    )
    actor_terms = {
        "arm_joint_position": ObservationTermCfg(func=mdp.arm_joint_position),
        "tcp_orientation": ObservationTermCfg(func=mdp.tcp_orientation),
        "hand_closure": ObservationTermCfg(func=mdp.hand_closure),
        "task_geometry": ObservationTermCfg(
            func=mdp.task_geometry, scale=10.0, clip=(-5.0, 5.0)
        ),
        "tomato_velocity": ObservationTermCfg(
            func=mdp.tomato_velocity, scale=(2.0, 2.0, 2.0, 0.1, 0.1, 0.1), clip=(-5.0, 5.0)
        ),
        "attachment": ObservationTermCfg(func=mdp.attachment_state),
        "contact": ObservationTermCfg(func=mdp.contact_force, scale=0.05, clip=(0.0, 5.0)),
        "last_action": ObservationTermCfg(func=mdp.last_action),
    }
    task_cfg = cfg["task"]
    return RLEnvironmentCfg(
        runtime=runtime,
        decimation=int(cfg["rl"]["decimation"]),
        episode_length_s=float(cfg["rl"]["episode_length_s"]),
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(
                terms={
                    **actor_terms,
                    "privileged_physics": ObservationTermCfg(func=mdp.privileged_physics),
                },
                concatenate_terms=True,
            ),
        },
        actions={
            "dual_arm_command": ActionTermCfg(
                func=mdp.set_dual_arm_command,
                action_dim=10,
                scale=(
                    0.10,
                    0.10,
                    0.10,
                    1.5,
                    2.0,
                    0.10,
                    0.10,
                    0.10,
                    1.5,
                    2.0,
                ),
                clip=(-1.0, 1.0),
            )
        },
        rewards={
            "reach": RewardTermCfg(func=mdp.reach_reward, weight=0.4),
            "support": RewardTermCfg(func=mdp.support_reward, weight=0.6),
            "grasp": RewardTermCfg(func=mdp.grasp_reward, weight=1.0),
            "stem_load_progress": RewardTermCfg(
                func=mdp.attachment_load_progress, weight=2.0
            ),
            "detach": RewardTermCfg(func=mdp.detachment_progress, weight=8.0),
            "carry_progress": RewardTermCfg(func=mdp.harvest_progress, weight=30.0),
            "success": RewardTermCfg(func=mdp.success_bonus, weight=20.0),
            "drop": RewardTermCfg(func=mdp.dropped_penalty, weight=-12.0),
            "unsafe_contact": RewardTermCfg(
                func=mdp.unsafe_contact,
                weight=-2.0,
                params={"safe_force_N": float(task_cfg["unsafe_contact_force_N"])},
            ),
            "ik_failure": RewardTermCfg(func=mdp.ik_failure_penalty, weight=-2.0),
            "action_l2": RewardTermCfg(func=mdp.action_l2, weight=-0.002),
            "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "success": TerminationTermCfg(func=mdp.success),
            "dropped": TerminationTermCfg(func=mdp.dropped),
            "invalid_state": TerminationTermCfg(func=mdp.workspace_violation),
        },
        metrics={
            "detached": MetricsTermCfg(func=mdp.detached_metric),
            "support_ready": MetricsTermCfg(func=mdp.support_metric),
            "grasped": MetricsTermCfg(func=mdp.grasped_metric),
            "harvest_distance_m": MetricsTermCfg(func=mdp.harvest_distance_metric),
            "attachment_load": MetricsTermCfg(func=mdp.attachment_load_metric),
        },
        seed=42,
        auto_reset=True,
        scale_rewards_by_dt=False,
    )


def make_cherry_tomato_pick_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=42,
        num_steps_per_env=48,
        max_iterations=3000,
        save_interval=100,
        experiment_name="cherry_tomato_pick",
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-4,
            entropy_coef=0.004,
            gamma=0.99,
            lam=0.95,
        ),
    )


def _environment_cfg(context: RLTaskContext) -> RLEnvironmentCfg:
    config_path = context.config_path or _DEFAULT_CONFIG
    cfg = load_cherry_tomato_pick_config(config_path)
    _apply_context_options(cfg, dict(context.options))
    seed = 42 if context.seed is None else int(context.seed)
    backend_factory = partial(
        _build_backend,
        deepcopy(cfg),
        num_envs=context.num_envs,
        device=context.device,
        training=context.training,
        seed=seed,
        config_path=Path(config_path),
    )
    env_cfg = make_cherry_tomato_pick_env_cfg(
        backend_factory, device=context.device, training=context.training, cfg=cfg
    )
    env_cfg.seed = seed
    return env_cfg


TASK = RLTask(
    task_id=TASK_ID,
    environment_cfg=_environment_cfg,
    trainer_cfg=lambda _context: make_cherry_tomato_pick_ppo_cfg(),
    runner_cls=LycheeSimOnPolicyRunner,
)


def register() -> None:
    if TASK_ID not in list_tasks():
        register_task(TASK)


register()


__all__ = [
    "TASK",
    "TASK_ID",
    "load_cherry_tomato_pick_config",
    "make_cherry_tomato_pick_env_cfg",
    "make_cherry_tomato_pick_ppo_cfg",
    "register",
]
