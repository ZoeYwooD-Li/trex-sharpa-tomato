"""Type aliases for vectorized RL environments."""

from __future__ import annotations

from typing import Any, TypeAlias

import torch

VecEnvObs: TypeAlias = dict[str, torch.Tensor | dict[str, torch.Tensor]]
VecEnvStepReturn: TypeAlias = tuple[
    VecEnvObs,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]
