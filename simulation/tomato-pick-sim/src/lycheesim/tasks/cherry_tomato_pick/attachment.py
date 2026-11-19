"""Fixed-weld fracture state for the cherry-tomato harvesting task.

The simulator owns the weld and publishes its reaction wrench. This module
keeps only task-adjacent physical state: directional tensile load, irreversible
fracture, and persistent device-native views. It does not synthesize springs or
apply body wrenches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class CherryTomatoAttachmentViews:
    """Persistent batch-first tensors owned by the attachment facade."""

    attachment_error: torch.Tensor
    attachment_force: torch.Tensor
    attachment_torque: torch.Tensor
    attachment_load: torch.Tensor
    attachment_shear_load: torch.Tensor
    break_force: torch.Tensor
    detached: torch.Tensor
    harvest_armed: torch.Tensor


def _quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[:, :3]
    scalar = quaternion[:, 3:4]
    cross = 2.0 * torch.linalg.cross(xyz, vector)
    return vector + scalar * cross + torch.linalg.cross(xyz, cross)


class CherryTomatoAttachmentFacade:
    """Vectorized directional fracture state for solver-owned fixed welds."""

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        config: Mapping[str, float | list[float]],
        *,
        constraint_active: torch.Tensor,
        constraint_wrench: torch.Tensor,
    ) -> None:
        if int(num_envs) <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._closed = False
        self._constraint_active = constraint_active
        self._constraint_wrench = constraint_wrench
        if (
            tuple(constraint_active.shape) != (self.num_envs,)
            or tuple(constraint_wrench.shape) != (self.num_envs, 6)
            or constraint_active.device != self.device
            or constraint_wrench.device != self.device
        ):
            raise ValueError("fixed constraint runtime views are invalid")

        self._local_point = self._constant_vector(
            config["tomato_local_point"], 3, "tomato_local_point"
        )
        self._break_direction = self._constant_vector(
            config["break_direction_world"], 3, "break_direction_world"
        )
        break_direction_norm = torch.linalg.vector_norm(
            self._break_direction, dim=-1, keepdim=True
        )
        if torch.any(break_direction_norm <= 0.0):
            raise ValueError("break_direction_world must be non-zero")
        self._break_direction /= break_direction_norm
        self._break_load_scale = float(config.get("break_load_scale", 1.0))
        if self._break_load_scale <= 0.0:
            raise ValueError("break_load_scale must be positive")

        shape = (self.num_envs,)
        self._anchor = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._break_force = torch.ones(
            shape, dtype=torch.float32, device=self.device
        )
        self._detached = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._harvest_armed = torch.zeros_like(self._detached)
        self._attachment_error = torch.zeros_like(self._anchor)
        self._attachment_force = torch.zeros_like(self._break_force)
        self._attachment_torque = torch.zeros_like(self._break_force)
        self._attachment_load = torch.zeros_like(self._break_force)
        self._attachment_shear_load = torch.zeros_like(self._break_force)
        self._views = CherryTomatoAttachmentViews(
            attachment_error=self._attachment_error,
            attachment_force=self._attachment_force.unsqueeze(-1),
            attachment_torque=self._attachment_torque.unsqueeze(-1),
            attachment_load=self._attachment_load.unsqueeze(-1),
            attachment_shear_load=self._attachment_shear_load.unsqueeze(-1),
            break_force=self._break_force.unsqueeze(-1),
            detached=self._detached.unsqueeze(-1),
            harvest_armed=self._harvest_armed.unsqueeze(-1),
        )

    @property
    def views(self) -> CherryTomatoAttachmentViews:
        self._require_open()
        return self._views

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        anchor: torch.Tensor,
        break_force: torch.Tensor,
    ) -> None:
        self._require_open()
        ids = self._env_ids(env_ids)
        count = int(ids.shape[0])
        anchor = self._reset_value(anchor, (count, 3), "anchor")
        break_force = self._reset_value(
            break_force, (count,), "break_force"
        )
        if torch.any(break_force <= 0.0):
            raise ValueError("break_force must be positive")

        self._anchor[ids] = anchor
        self._break_force[ids] = break_force
        self._clear_measurements(ids)
        self._detached[ids] = False
        self._harvest_armed[ids] = False
        self._constraint_active[ids] = True
        self._constraint_wrench[ids] = 0.0

    def reset_detached(self, env_ids: torch.Tensor) -> None:
        """Reset selected rows as free fruit with their weld disabled."""

        self._require_open()
        ids = self._env_ids(env_ids)
        self._clear_measurements(ids)
        self._detached[ids] = True
        self._harvest_armed[ids] = False
        self._constraint_active[ids] = False
        self._constraint_wrench[ids] = 0.0

    def update(
        self,
        *,
        target_position: torch.Tensor,
        target_rotation_xyzw: torch.Tensor,
        harvest_armed: torch.Tensor,
    ) -> CherryTomatoAttachmentViews:
        """Consume the latest weld reaction and update fracture state."""

        self._require_open()
        target_position = self._batch_vector(
            target_position, 3, "target_position"
        )
        target_rotation = self._normalized_quaternion(
            target_rotation_xyzw, "target_rotation_xyzw"
        )
        armed = torch.as_tensor(
            harvest_armed, dtype=torch.bool, device=self.device
        ).reshape(-1)
        if tuple(armed.shape) != (self.num_envs,):
            raise ValueError(
                "harvest_armed must have shape "
                f"{(self.num_envs,)}, got {tuple(armed.shape)}"
            )
        self._harvest_armed[:] = armed

        attachment_point = target_position + _quat_rotate(
            target_rotation, self._local_point
        )
        error = self._anchor - attachment_point
        force = self._constraint_wrench[:, :3]
        torque = self._constraint_wrench[:, 3:]
        force_magnitude = torch.linalg.vector_norm(force, dim=-1)
        torque_magnitude = torch.linalg.vector_norm(torque, dim=-1)
        axial_force = torch.sum(force * self._break_direction, dim=-1)
        tensile_force = torch.clamp(-axial_force, min=0.0)
        shear_force = torch.linalg.vector_norm(
            force - axial_force.unsqueeze(-1) * self._break_direction,
            dim=-1,
        )
        tensile_load = (
            self._break_load_scale * tensile_force / self._break_force
        )
        shear_load = (
            self._break_load_scale * shear_force / self._break_force
        )
        active_before_update = ~self._detached
        self._detached |= (
            active_before_update & armed & (tensile_load >= 1.0)
        )
        self._constraint_active[:] = ~self._detached

        self._attachment_error[:] = error
        self._attachment_force[:] = torch.where(
            active_before_update,
            force_magnitude,
            torch.zeros_like(force_magnitude),
        )
        self._attachment_torque[:] = torch.where(
            active_before_update,
            torque_magnitude,
            torch.zeros_like(torque_magnitude),
        )
        self._attachment_load[:] = torch.where(
            active_before_update,
            tensile_load,
            torch.zeros_like(tensile_load),
        )
        self._attachment_shear_load[:] = torch.where(
            active_before_update,
            shear_load,
            torch.zeros_like(shear_load),
        )
        return self._views

    def close(self) -> None:
        self._closed = True

    def _clear_measurements(self, ids: torch.Tensor) -> None:
        self._attachment_error[ids] = 0.0
        self._attachment_force[ids] = 0.0
        self._attachment_torque[ids] = 0.0
        self._attachment_load[ids] = 0.0
        self._attachment_shear_load[ids] = 0.0

    def _env_ids(self, value: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(value, dtype=torch.long, device=self.device)
        if ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional tensor")
        return ids

    def _reset_value(
        self, value: torch.Tensor, expected: tuple[int, ...], name: str
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        )
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _constant_vector(self, value, width: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        if tuple(tensor.shape) != (1, width):
            raise ValueError(f"{name} must contain {width} values")
        return tensor.expand(self.num_envs, -1).clone()

    def _batch_vector(
        self, value: torch.Tensor, width: int, name: str
    ) -> torch.Tensor:
        tensor = torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        )
        expected = (self.num_envs, width)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
            )
        return tensor

    def _normalized_quaternion(
        self, value: torch.Tensor, name: str
    ) -> torch.Tensor:
        quaternion = self._batch_vector(value, 4, name)
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        return quaternion / torch.clamp_min(norm, 1.0e-8)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("cherry-tomato attachment facade is closed")


__all__ = ["CherryTomatoAttachmentFacade", "CherryTomatoAttachmentViews"]
