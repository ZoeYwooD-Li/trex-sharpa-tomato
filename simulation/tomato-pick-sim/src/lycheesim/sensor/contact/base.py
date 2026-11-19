"""Common explicit-contact sensor interface."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from ..base import SensorBase


@dataclass(frozen=True)
class ContactSensorInput:
    state: Any
    contacts: Any
    solver: Any | None = None


class ContactSensorBase(SensorBase):
    @abstractmethod
    def update(
        self,
        state: Any,
        contacts: Any,
        *,
        solver: Any | None = None,
    ) -> None:
        """Consume an explicit contact manifold and resolved forces."""

    def sample(self, sample: ContactSensorInput) -> None:
        self.update(sample.state, sample.contacts, solver=sample.solver)


__all__ = ["ContactSensorBase", "ContactSensorInput"]
