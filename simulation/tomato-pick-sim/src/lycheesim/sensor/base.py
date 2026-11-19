"""Common sensor contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorSpec:
    """Stable metadata shared by all simulation sensor families."""

    sensor_type: str
    adapter: str
    input_fields: tuple[str, ...]
    output_fields: tuple[str, ...]
    differentiable: bool


class SensorBase(ABC):
    """Base class for sensors with a normalized metadata contract."""

    @property
    @abstractmethod
    def spec(self) -> SensorSpec:
        """Describe the adapter's native inputs and normalized outputs."""


__all__ = [
    "SensorBase",
    "SensorSpec",
]
