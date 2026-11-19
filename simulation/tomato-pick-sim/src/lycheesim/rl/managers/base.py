"""Shared RL manager utilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

import torch


def to_tensor(
    value: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def batched_tensor(
    value: Any,
    *,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    scalar_as_column: bool = True,
) -> torch.Tensor:
    tensor = to_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.expand(num_envs)
    if tensor.shape[0] != num_envs:
        if num_envs == 1:
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(
                f"Expected first dimension num_envs={num_envs}, got {tuple(tensor.shape)}"
            )
    if scalar_as_column and tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor


def batched_scalar(
    value: Any,
    *,
    num_envs: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = batched_tensor(
        value,
        num_envs=num_envs,
        device=device,
        dtype=dtype,
        scalar_as_column=False,
    )
    if tensor.ndim > 1:
        if tensor.numel() != num_envs:
            raise ValueError(
                f"Expected one scalar per environment, got shape {tuple(tensor.shape)}"
            )
        tensor = tensor.reshape(num_envs)
    return tensor


def parameter_tensor(
    value: float | int | Sequence[float] | torch.Tensor,
    *,
    device: torch.device | str,
    width: int,
) -> torch.Tensor:
    tensor = to_tensor(value, device=device).flatten()
    if tensor.numel() == 1:
        tensor = tensor.expand(width)
    if tensor.numel() != width:
        raise ValueError(f"Expected parameter width {width}, got {tensor.numel()}")
    return tensor


def config_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: config_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {key: config_to_dict(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return tuple(config_to_dict(val) for val in value)
    if isinstance(value, list):
        return [config_to_dict(val) for val in value]
    return value


def strip_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_none(val) for key, val in value.items() if val is not None}
    if isinstance(value, list):
        return [strip_none(val) for val in value]
    if isinstance(value, tuple):
        return tuple(strip_none(val) for val in value)
    return value
