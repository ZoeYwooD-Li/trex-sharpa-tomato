"""Solver-independent camera sampling interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from ..base import SensorBase


@dataclass(frozen=True)
class CameraSensorInput:
    state: Any
    refit_geometry: bool = True


class CameraSensorBase(SensorBase):
    @abstractmethod
    def update(self, state: Any, *, refit_bvh: bool = True):
        """Render one observation from a solver-native state."""

    def sample(self, sample: CameraSensorInput):
        return self.update(sample.state, refit_bvh=sample.refit_geometry)


__all__ = ["CameraSensorBase", "CameraSensorInput"]
