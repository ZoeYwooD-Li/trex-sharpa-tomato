"""Tiny RL space descriptors used without requiring Gymnasium.

RSL-RL mainly needs dimensions. Projects that already depend on Gymnasium may
replace these descriptors with gymnasium.spaces equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Box:
    shape: tuple[int, ...]
    low: float = float("-inf")
    high: float = float("inf")
    dtype: Any = float


@dataclass
class DictSpace:
    spaces: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.spaces[key]


def batch_space(space: Any, num_envs: int) -> Any:
    if isinstance(space, Box):
        return Box(
            shape=(int(num_envs), *space.shape),
            low=space.low,
            high=space.high,
            dtype=space.dtype,
        )
    if isinstance(space, DictSpace):
        return DictSpace(
            {name: batch_space(value, num_envs) for name, value in space.spaces.items()}
        )
    raise TypeError(f"Unsupported space type: {type(space)!r}")
