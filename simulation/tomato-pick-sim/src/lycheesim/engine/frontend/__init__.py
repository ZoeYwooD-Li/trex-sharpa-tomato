"""Display and recording sinks for the standalone tomato scene."""
from .base import Frontend
from .headless import HeadlessFrontend
from .recorder import RecorderFrontend
from .rerun import RerunFrontend

__all__ = ["Frontend", "HeadlessFrontend", "RecorderFrontend", "RerunFrontend"]
