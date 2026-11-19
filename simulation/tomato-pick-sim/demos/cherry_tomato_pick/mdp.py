"""MDP terms for task-space dual-FR3 tomato harvesting."""

from __future__ import annotations

from typing import Any

import torch


def signal(env: Any, name: str) -> torch.Tensor:
    return env.runtime.signal(name)


def set_dual_arm_command(env: Any, action: torch.Tensor) -> None:
    env.runtime.set_command("dual_arm_command", action)


def arm_joint_position(env: Any) -> torch.Tensor:
    return torch.cat((signal(env, "left_arm_q"), signal(env, "right_arm_q")), dim=-1)


def tcp_orientation(env: Any) -> torch.Tensor:
    return torch.cat(
        (signal(env, "left_tcp_pose")[..., 3:], signal(env, "right_tcp_pose")[..., 3:]),
        dim=-1,
    )


def hand_closure(env: Any) -> torch.Tensor:
    return signal(env, "hand_closure")


def task_geometry(env: Any) -> torch.Tensor:
    return torch.cat(
        (signal(env, "tomato_to_left_hand"), signal(env, "support_to_right_hand")),
        dim=-1,
    )


def tomato_velocity(env: Any) -> torch.Tensor:
    return torch.cat(
        (signal(env, "tomato_lin_vel"), signal(env, "tomato_ang_vel")), dim=-1
    )


def attachment_state(env: Any) -> torch.Tensor:
    return torch.cat(
        (
            signal(env, "attachment_load"),
            signal(env, "detached"),
            signal(env, "support_ready"),
            signal(env, "grasped"),
            signal(env, "harvest_distance"),
        ),
        dim=-1,
    )


def contact_force(env: Any) -> torch.Tensor:
    return signal(env, "contact_force_N")


def last_action(env: Any) -> torch.Tensor:
    return env.action_manager.action


def privileged_physics(env: Any) -> torch.Tensor:
    return torch.cat(
        (
            signal(env, "tomato_pos"),
            signal(env, "tomato_quat"),
            signal(env, "attachment_error"),
            signal(env, "attachment_force_N"),
            signal(env, "attachment_torque_Nm"),
            signal(env, "break_force_N"),
            signal(env, "break_torque_Nm"),
            signal(env, "ik_failed"),
        ),
        dim=-1,
    )


def reach_reward(env: Any, std: float = 0.08) -> torch.Tensor:
    distance = torch.linalg.vector_norm(signal(env, "tomato_to_left_hand"), dim=-1)
    return torch.exp(-torch.square(distance / float(std)))


def support_reward(env: Any, std: float = 0.08) -> torch.Tensor:
    distance = torch.linalg.vector_norm(signal(env, "support_to_right_hand"), dim=-1)
    geometric = torch.exp(-torch.square(distance / float(std)))
    return geometric + signal(env, "support_ready").reshape(env.num_envs)


def grasp_reward(env: Any) -> torch.Tensor:
    return signal(env, "grasped").reshape(env.num_envs)


def attachment_load_progress(env: Any) -> torch.Tensor:
    return signal(env, "attachment_load_delta").reshape(env.num_envs)


def detachment_progress(env: Any) -> torch.Tensor:
    return signal(env, "detached_delta").reshape(env.num_envs)


def harvest_progress(env: Any) -> torch.Tensor:
    return signal(env, "harvest_distance_delta").reshape(env.num_envs)


def success_bonus(env: Any) -> torch.Tensor:
    return signal(env, "success").reshape(env.num_envs)


def unsafe_contact(env: Any, safe_force_N: float = 35.0) -> torch.Tensor:
    excess = torch.clamp(
        signal(env, "contact_force_N").amax(dim=-1) - float(safe_force_N), min=0.0
    )
    return torch.square(excess / max(float(safe_force_N), 1.0e-6))


def dropped_penalty(env: Any) -> torch.Tensor:
    return signal(env, "dropped").reshape(env.num_envs)


def ik_failure_penalty(env: Any) -> torch.Tensor:
    return signal(env, "ik_failed").reshape(env.num_envs)


def action_l2(env: Any) -> torch.Tensor:
    return torch.square(env.action_manager.action).sum(dim=-1)


def action_rate_l2(env: Any) -> torch.Tensor:
    delta = env.action_manager.action - env.action_manager.prev_action
    return torch.square(delta).sum(dim=-1)


def time_out(env: Any) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length


def success(env: Any) -> torch.Tensor:
    return signal(env, "success").reshape(env.num_envs) > 0.5


def dropped(env: Any) -> torch.Tensor:
    return signal(env, "dropped").reshape(env.num_envs) > 0.5


def workspace_violation(env: Any) -> torch.Tensor:
    left = signal(env, "left_tcp_pose")
    right = signal(env, "right_tcp_pose")
    return ~(torch.isfinite(left).all(dim=-1) & torch.isfinite(right).all(dim=-1))


def detached_metric(env: Any) -> torch.Tensor:
    return signal(env, "detached").reshape(env.num_envs)


def support_metric(env: Any) -> torch.Tensor:
    return signal(env, "support_ready").reshape(env.num_envs)


def grasped_metric(env: Any) -> torch.Tensor:
    return signal(env, "grasped").reshape(env.num_envs)


def harvest_distance_metric(env: Any) -> torch.Tensor:
    return signal(env, "harvest_distance").reshape(env.num_envs)


def attachment_load_metric(env: Any) -> torch.Tensor:
    return signal(env, "attachment_load").reshape(env.num_envs)


__all__ = [name for name in globals() if not name.startswith("_")]
