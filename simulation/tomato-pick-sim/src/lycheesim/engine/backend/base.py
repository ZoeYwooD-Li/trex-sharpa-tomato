"""Shared contract for physics solvers used by demo scene adapters.

The runner does not depend on this protocol directly.  A demo owns the solver
and exposes runner callbacks through :class:`lycheesim.engine.runner.DemoSetup`.
The protocol exists to give backend implementations one unambiguous time-step
contract despite their different internal Newton/Warp substep schemes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PhysicsSolver(Protocol):
    """Minimal public contract implemented by shared physics backends.

    ``dt`` is the amount of simulated time advanced by one call to ``step``.
    A solver may divide that interval into any number of internal substeps.
    State accessors and control methods remain solver-family specific.
    """

    dt: float

    def step(self) -> None:
        """Advance the simulation by exactly ``dt`` seconds."""


__all__ = ["PhysicsSolver"]
