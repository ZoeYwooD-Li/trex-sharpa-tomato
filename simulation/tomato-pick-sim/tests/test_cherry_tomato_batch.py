import json
from pathlib import Path
import subprocess

import h5py
import pytest

from demos.cherry_tomato_pick import batch
from demos.cherry_tomato_pick.quality import QUALITY_SCHEMA


def write_quality(path, accepted=True):
    with h5py.File(path, "w") as archive:
        archive.create_group("metadata").attrs["extra"] = json.dumps({
            "quality": {"schema": QUALITY_SCHEMA, "accepted": accepted,
                        "target_box_id": 1, "reasons": [] if accepted else ["support_lost"]},
        })


def output_path(command):
    value = next(arg for arg in command if arg.startswith("recording.output_path="))
    return Path(json.loads(value.split("=", 1)[1]))


def test_exactly_100_attempts_continue_after_failure_and_advance_seed(tmp_path, monkeypatch):
    commands = []

    def run(command, log, timeout):
        commands.append(command)
        log.touch()
        path = output_path(command)
        index = len(commands)
        if index == 1:  # Initial generation error; no HDF5 was produced.
            return 1, None
        write_quality(path, accepted=index != 2)
        return (1 if index == 3 else 0), None  # HDF5 accepted, but final verification failed.

    monkeypatch.setattr(batch, "_run_episode", run)
    directory, summary = batch.run_batch(output_dir=tmp_path, seed=42)
    assert len(commands) == summary["attempted"] == 100
    assert summary["status"] == "completed"
    assert summary["accepted"] == 97
    assert summary["rejected"] == 3
    assert [r["seed"] for r in summary["attempts"]] == list(range(42, 142))
    for number, command in enumerate(commands, 1):
        assert output_path(command).name == f"episode_{number:06d}.h5"
        assert f"recording.seed={41 + number}" in command
        assert "recording.auto_increment=false" in command
        assert "recording.overwrite=false" in command
        assert command[command.index("--config") + 1] == str(directory / "config.yaml")
    assert len((directory / "accepted.txt").read_text().splitlines()) == 97
    assert json.loads((directory / "summary.json").read_text())["accepted"] == 97


def test_each_batch_is_unique_and_preserves_previous_files(tmp_path, monkeypatch):
    def run(command, log, timeout):
        write_quality(output_path(command))
        return 0, None

    monkeypatch.setattr(batch, "_run_episode", run)
    monkeypatch.setattr(batch.secrets, "randbits", lambda n: 12345)
    first, a = batch.run_batch(attempts=1, output_dir=tmp_path)
    contents = (first / "summary.json").read_bytes()
    second, b = batch.run_batch(attempts=1, output_dir=tmp_path)
    assert first != second
    assert a["base_seed"] == b["base_seed"] == 12345
    assert (first / "summary.json").read_bytes() == contents


def test_interrupt_stops_instead_of_launching_remaining_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "_run_episode", lambda *args: (-2, "interrupted"))
    directory, summary = batch.run_batch(attempts=100, output_dir=tmp_path, seed=1)
    assert summary["status"] == "interrupted"
    assert summary["attempted"] == 1
    assert summary["accepted"] == 0
    assert "interrupted" in summary["attempts"][0]["reasons"]
    assert (directory / "accepted.txt").read_text() == ""


@pytest.mark.parametrize("kwargs", [{"attempts": 0}, {"seed": -1}, {"duration": float("nan")},
                                   {"episode_timeout_s": 0}])
def test_invalid_arguments_create_no_batch(tmp_path, kwargs):
    with pytest.raises(ValueError):
        batch.run_batch(output_dir=tmp_path, **kwargs)
    assert not list(tmp_path.iterdir())


def test_invalid_collection_config_fails_before_attempts(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("deposit: {enabled: false}\n")
    with pytest.raises(ValueError, match="requires tableware and deposit"):
        batch.run_batch(config=config, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [config]


def test_corrupt_or_unassessed_metadata_cannot_be_accepted(tmp_path):
    path = tmp_path / "episode.h5"
    for quality in ([], {}, {"schema": QUALITY_SCHEMA, "accepted": False, "reasons": 42}):
        with h5py.File(path, "w") as archive:
            archive.create_group("metadata").attrs["extra"] = json.dumps({"quality": quality})
        result = batch._episode_result(path, 0, None)
        assert not result["accepted"]


def test_wall_timeout_rejects_even_an_accepted_hdf5(tmp_path):
    path = tmp_path / "episode.h5"
    write_quality(path)
    result = batch._episode_result(path, 0, "wall_timeout")
    assert not result["accepted"]
    assert result["reasons"] == ["wall_timeout"]


def test_timeout_interrupts_child_and_waits_for_cleanup(tmp_path, monkeypatch):
    signals = []

    class Child:
        pid = 123
        returncode = None
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("test", timeout)
            self.returncode = 130
            return self.returncode

    child = Child()
    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(batch.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert batch._run_episode(["test"], tmp_path / "episode.log", 1) == (130, "wall_timeout")
    assert signals == [(123, batch.signal.SIGINT)]


def test_cli_default_is_100_attempts(monkeypatch):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return Path("unused"), {"status": "completed"}

    monkeypatch.setattr(batch, "run_batch", run)
    assert batch.main([]) == 0
    assert captured["attempts"] == 100


def test_ctrl_c_during_timeout_cleanup_stops_the_batch(tmp_path, monkeypatch):
    signals = []

    class Child:
        pid = 123
        returncode = -9
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("test", timeout)
            if self.calls == 2:
                raise KeyboardInterrupt
            return self.returncode

    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(batch.os, "killpg", lambda pid, sig: signals.append(sig))
    assert batch._run_episode(["test"], tmp_path / "episode.log", 1) == (-9, "interrupted")
    assert signals == [batch.signal.SIGINT, batch.signal.SIGKILL]
