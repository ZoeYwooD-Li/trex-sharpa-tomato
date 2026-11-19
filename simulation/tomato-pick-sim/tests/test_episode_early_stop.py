from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from lycheesim.engine import runner
from lycheesim.training import CameraSpec, EpisodeFrame
from demos.cherry_tomato_pick.recording import CherryTomatoEpisodeRecorder


@pytest.mark.parametrize("stop_at,verify_fails", [(2, False), (None, False), (2, True)])
def test_runner_final_frame_and_cleanup(monkeypatch, tmp_path, stop_at, verify_fails):
    events = []
    times = []
    config = tmp_path / "scene.yaml"
    config.write_text("sim: {}\n")
    frontend = SimpleNamespace(
        read_inputs=lambda t: {}, frame=lambda t: events.append(("frame", t)),
        close=lambda: events.append("frontend_close"),
    )
    monkeypatch.setattr(runner.wp, "init", lambda: None)
    monkeypatch.setattr(runner.wp, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "_select_frontends", lambda *args: [frontend])

    def verify():
        events.append("verify")
        if verify_fails:
            raise RuntimeError("quality failure is not hidden by early stop")

    setup = runner.DemoSetup(
        dt=.01, duration=.05, every=4,
        step=lambda t, inputs: times.append(t),
        snapshot=lambda fe, t: events.append(("snapshot", t)),
        should_stop=lambda: stop_at is not None and len(times) >= stop_at,
        verify=verify, close=lambda: events.append("recorder_close"),
    )
    if verify_fails:
        with pytest.raises(RuntimeError, match="quality failure"):
            runner.main(lambda cfg: setup, argv=["--config", str(config)])
    else:
        runner.main(lambda cfg: setup, argv=["--config", str(config)])
    assert len(times) == (5 if stop_at is None else stop_at)
    if stop_at is not None:
        # The terminal step is off the viewer cadence, but is rendered once.
        assert events.count(("frame", times[-1])) == 1
        assert events.count(("snapshot", times[-1])) == 1
    assert events[-3:] == ["verify", "recorder_close", "frontend_close"]


def test_terminal_robot_and_all_camera_frames_are_last(tmp_path):
    names = ("gemini_435le", "gemini_305_left", "gemini_305_right")
    recorder = CherryTomatoEpisodeRecorder(
        tmp_path / "episode.h5", seed=42, object_ids=(),
        cameras={n: CameraSpec(image_size=(2, 3)) for n in names}, metadata={},
    )

    def camera(name, time):
        return recorder.append_camera(
            name, time=time, color=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_m=np.ones((2, 3), dtype=np.float32),
            world_from_camera=np.eye(4), valid_depth_pixels=6,
        )

    def frame(time, placed=False):
        return EpisodeFrame(time=time, joint_pos=np.zeros(54),
                            extra={"placed_in_box": np.array([placed])})

    for name in names:
        assert camera(name, 0.)
    recorder.append_robot(frame(0.))
    # Success between regular 30 Hz sample instants, not at configured duration.
    terminal_time = .025
    for name in names:
        assert camera(name, terminal_time)
    recorder.append_robot(frame(terminal_time, True),
                          transition_action=np.zeros(16), terminated=True)
    assert recorder.capture_finished
    for name in names:
        assert not camera(name, .1)
    assert not recorder.append_robot(frame(.1), transition_action=np.zeros(16))
    path = recorder.close()
    with h5py.File(path) as archive:
        np.testing.assert_allclose(archive["robot/timestamps"][:], [0., terminal_time])
        assert archive["robot/terminated"][:].tolist() == [True]
        assert archive["robot/truncated"][:].tolist() == [False]
        assert archive["robot/extra/placed_in_box"][-1, 0]
        for name in names:
            assert archive[f"cameras/{name}/timestamps"][-1] == terminal_time
            assert archive[f"robot/camera_age_s/{name}"][-1] == 0.
