"""No-op frontend for batch sim and perf tests."""
from __future__ import annotations

from .base import Frontend


class HeadlessFrontend(Frontend):
    def __init__(self) -> None:
        self.frame_idx = 0

    def frame(self, t: float) -> None:
        self.frame_idx += 1
