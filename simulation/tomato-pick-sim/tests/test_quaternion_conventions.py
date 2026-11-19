from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lycheesim.sensor.camera.newton import NewtonCameraSensor


class _LookAtCamera(NewtonCameraSensor):
    def __init__(self) -> None:
        self.world_count = 1
        self.config = SimpleNamespace(camera_count=1)
        self.captured_positions: np.ndarray | None = None
        self.captured_quaternions: np.ndarray | None = None

    def set_camera_poses(self, positions, quaternions_xyzw) -> None:
        self.captured_positions = np.asarray(positions)
        self.captured_quaternions = np.asarray(quaternions_xyzw)


def test_newton_camera_look_at_emits_xyzw() -> None:
    camera = _LookAtCamera()
    camera.set_look_at(
        positions=(0.0, 0.0, 1.0),
        targets=(0.0, 0.0, 0.0),
        up_vectors=(0.0, 1.0, 0.0),
    )

    np.testing.assert_allclose(
        camera.captured_quaternions,
        np.asarray([[[0.0, 0.0, 0.0, 1.0]]]),
        atol=1.0e-7,
    )
