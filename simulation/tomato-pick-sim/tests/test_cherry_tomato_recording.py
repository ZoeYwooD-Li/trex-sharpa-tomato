from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from demos.cherry_tomato_pick.recording import (
    DATASET_SCHEMA,
    CherryTomatoEpisodeRecorder,
    next_episode_path,
    world_from_pose,
)
from lycheesim.training import CameraSpec, EpisodeFrame


def _frame(time_s: float, value: int) -> EpisodeFrame:
    return EpisodeFrame(
        time=time_s,
        images={},
        joint_pos=np.full(4, value, dtype=np.float32),
        joint_vel=np.full(4, value * 0.5, dtype=np.float32),
        ee_pose=np.zeros((2, 7), dtype=np.float32),
        gripper=np.zeros(2, dtype=np.float32),
        extra={
            "phase_index": np.asarray((value,), dtype=np.int16),
        },
    )


def test_cherry_tomato_episode_recorder_writes_multirate_hdf5(
    tmp_path: Path,
) -> None:
    output = tmp_path / "episode.h5"
    recorder = CherryTomatoEpisodeRecorder(
        output,
        seed=7,
        object_ids=("tomato_10",),
        cameras={
            "camera": CameraSpec(
                image_size=(2, 3),
                intrinsics=np.eye(3),
            )
        },
        metadata={"joint_names": ["a", "b", "c", "d"]},
    )

    for time_s, value in ((0.0, 0), (0.1, 1)):
        recorder.append_camera(
            "camera",
            time=time_s,
            color=np.full((2, 3, 3), value, dtype=np.uint8),
            depth_m=np.full((2, 3), value, dtype=np.float32),
            world_from_camera=np.eye(4),
            valid_depth_pixels=6,
        )
    recorder.append_robot(_frame(0.0, 0))
    recorder.append_robot(
        _frame(0.035, 1),
        transition_action=np.asarray((0.2, 0.3), dtype=np.float32),
    )
    recorder.append_robot(
        _frame(0.065, 2),
        transition_action=np.asarray((0.4, 0.5), dtype=np.float32),
    )
    assert recorder.close() == output
    assert recorder.close() == output

    with h5py.File(output, "r") as episode:
        assert episode.attrs["schema"] == DATASET_SCHEMA
        metadata = json.loads(episode["metadata"].attrs["extra"])
        assert metadata["robot_frame_count"] == 3
        assert metadata["camera_frame_counts"] == {"camera": 2}
        assert metadata["truncated"] is True
        assert episode["robot/timestamps"].shape == (3,)
        assert episode["robot/joint_pos"].shape == (3, 4)
        assert episode["robot/action"].shape == (2, 2)
        np.testing.assert_allclose(
            episode["robot/action_timestamps"][:], (0.0, 0.035)
        )
        assert episode["robot/truncated"][:].tolist() == [False, True]
        assert episode["robot/camera_index/camera"][:].tolist() == [0, 0, 0]
        np.testing.assert_allclose(
            episode["robot/camera_age_s/camera"][:], (0.0, 0.035, 0.065)
        )
        assert episode["cameras/camera/timestamps"].shape == (2,)
        assert episode["cameras/camera/color"].shape == (2, 2, 3, 3)
        assert episode["cameras/camera/depth_m"].shape == (2, 2, 3)
        assert episode["cameras/camera/world_from_camera"].shape == (2, 4, 4)

    incremented = CherryTomatoEpisodeRecorder(
        output,
        seed=8,
        object_ids=(),
        cameras={},
        metadata={},
    )
    assert incremented.output_path == tmp_path / "episode_000001.h5"

    with pytest.raises(FileExistsError, match="episode already exists"):
        CherryTomatoEpisodeRecorder(
            output,
            seed=7,
            object_ids=(),
            cameras={},
            metadata={},
            auto_increment=False,
        )


def test_next_episode_path_advances_existing_numbered_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "episode_000001.h5"
    second = tmp_path / "episode_000002.h5"
    first.touch()
    second.touch()

    assert next_episode_path(first) == tmp_path / "episode_000003.h5"


def test_next_episode_path_does_not_fill_numbering_gaps(tmp_path: Path) -> None:
    requested = tmp_path / "episode_000001.h5"
    (tmp_path / "episode_000002.h5").touch()

    assert next_episode_path(requested) == tmp_path / "episode_000003.h5"
    recorder = CherryTomatoEpisodeRecorder(
        requested,
        seed=9,
        object_ids=(),
        cameras={},
        metadata={},
    )
    assert recorder.output_path == tmp_path / "episode_000003.h5"


def test_world_from_pose_uses_xyzw_quaternions() -> None:
    transform = world_from_pose(
        np.asarray((1.0, 2.0, 3.0)),
        np.asarray((0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))),
    )
    np.testing.assert_allclose(transform[:3, 3], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(
        transform[:3, :3] @ np.asarray((1.0, 0.0, 0.0)),
        (0.0, 1.0, 0.0),
        atol=1.0e-7,
    )
