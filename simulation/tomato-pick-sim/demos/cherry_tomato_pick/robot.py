"""Shared robot contracts for the cherry-tomato picking demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .kinematics import minimum_jerk


@dataclass(frozen=True)
class RobotCollisionPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    side: str
    finger: str | None
    source_id: int


def interpolate_pose(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
    """Minimum-jerk interpolation of XYZ and an ``xyzw`` quaternion."""

    # The polynomial can round slightly above 1 near the endpoint. Slerp
    # rejects that value even though the input time was already clamped.
    alpha = float(np.clip(minimum_jerk(float(np.clip(amount, 0.0, 1.0))), 0.0, 1.0))
    result = np.empty(7, dtype=np.float64)
    result[:3] = start[:3] + alpha * (end[:3] - start[:3])
    rotations = Rotation.from_quat(np.stack((start[3:], end[3:])))
    result[3:] = Slerp((0.0, 1.0), rotations)(alpha).as_quat()
    return result


__all__ = ["RobotCollisionPart", "interpolate_pose"]
