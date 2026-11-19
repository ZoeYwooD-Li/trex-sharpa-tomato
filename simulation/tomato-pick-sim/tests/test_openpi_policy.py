"""CPU contract tests; these do not establish physical harvesting success."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from demos.cherry_tomato_pick.openpi_policy import (
    PROMPTS, PolicyController, absolute_action, decode_numpy, encode_numpy, make_observation,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "configs/cherry_tomato_pick_deposit.yaml").read_text())


def example():
    pose = np.array([.4, 0, .2, 0, 0, 0, 1.])
    scene = SimpleNamespace(robot=SimpleNamespace(tcp_poses=lambda: (pose.copy(), pose.copy())),
                            left_grip=.2, right_grip=.8)
    image = np.zeros((200, 320, 3), dtype=np.uint8)
    cameras = {"camera_color": image, "wrist_camera_color": {"left": image + 1, "right": image + 2}}
    return scene, cameras


@pytest.mark.parametrize("box", range(3))
def test_observation_matches_training(box):
    scene, cameras = example()
    obs = make_observation(scene, cameras, box)
    assert obs["prompt"] == PROMPTS[box]
    assert obs["state"].dtype == np.float32
    np.testing.assert_allclose(obs["state"][-2:], [.2, .8])
    np.testing.assert_equal(obs["state"][:7], scene.robot.tcp_poses()[0].astype(np.float32))
    assert set(obs) == {"images", "state", "prompt"}  # No ground-truth object poses.
    assert obs["images"]["left_wrist"][0, 0, 0] == 1


def test_action_is_absolute_normalized_clipped_not_a_delta(cfg):
    scene, cameras = example()
    measured = make_observation(scene, cameras, 0)["state"]
    action = measured.copy()
    action[0], action[7], action[14], action[15] = .5, 99, -1, 2
    action[3:7] *= -2
    output, count = absolute_action(action, measured, cfg["robot"])
    assert output[0] == .5  # Not measured[0] + .5.
    assert output[7] == cfg["robot"]["right_workspace_max"][0]
    np.testing.assert_equal(output[3:7], [0, 0, 0, 1])
    np.testing.assert_equal(output[14:], [0, 1])
    assert count == 3


@pytest.mark.parametrize("bad", [np.zeros(16), np.full(16, np.nan), np.zeros(32)])
def test_reject_invalid_actions(cfg, bad):
    with pytest.raises(ValueError):
        absolute_action(bad, np.ones(16), cfg["robot"])


def test_receding_horizon_and_zero_velocity(cfg):
    scene, cameras = example()
    observation = make_observation(scene, cameras, 0)
    chunk = np.tile(observation["state"], (16, 1))
    chunk[:, 0] = .3 + np.arange(16) * .001
    calls, commands, targets = [], [], []
    client = SimpleNamespace(infer=lambda obs: (calls.append(obs) or {"actions": chunk}),
                             metadata={}, url="ws://test")
    backend = SimpleNamespace(set_robot_command=lambda a: commands.append(a),
                              set_robot_states=lambda l, r, c: targets.append((l, r, c)))
    policy = PolicyController(client, execute_steps=4)
    policy.configure(cfg)
    times = []
    for i in range(100):
        t = .005 * i
        if policy.due(t, .005):
            times.append(t)
            policy.apply(t, .005, scene, cameras, backend)
    assert len(times) == 15
    assert len(calls) == 4
    np.testing.assert_allclose([x[0][0, 0] for x in targets[:5]], [.3, .301, .302, .303, .3])
    assert all(not command.any() for command in commands)
    assert policy.summary()["action_steps"] == 15


@pytest.mark.parametrize("shape", [(16,), (1, 16, 16), (8, 16), (16, 32)])
def test_reject_wrong_chunk_shape(cfg, shape):
    policy = PolicyController(SimpleNamespace(infer=lambda obs: {"actions": np.ones(shape)}))
    policy.configure(cfg)
    scene, cameras = example()
    with pytest.raises(ValueError, match="actions"):
        policy.apply(0, .005, scene, cameras, None)


def test_reject_wrong_image():
    scene, cameras = example()
    cameras["camera_color"] = np.zeros((3, 200, 320), dtype=np.uint8)
    with pytest.raises(ValueError, match="RGB"):
        make_observation(scene, cameras, 0)


def test_wire_protocol_roundtrip():
    msgpack = pytest.importorskip("msgpack")
    data = {"image": np.zeros((200, 320, 3), dtype=np.uint8),
            "actions": np.arange(256, dtype=np.float32).reshape(16, 16), "value": np.float32(1.)}
    result = msgpack.unpackb(msgpack.packb(data, default=encode_numpy), object_hook=decode_numpy)
    for key in data:
        np.testing.assert_equal(result[key], data[key])
    with pytest.raises(ValueError):
        decode_numpy({b"__ndarray__": True, b"dtype": "O", b"data": b"", b"shape": (0,)})


def test_policy_step_has_no_scripted_motion():
    # Keep the explicit branch boundary: no oracle trajectory or deposit command.
    import ast
    source = ast.parse((ROOT / "demos/cherry_tomato_pick/scene.py").read_text())
    step = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "step")
    branch = next(n for n in step.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "policy is not None")
    calls = [ast.unparse(n.func) for node in branch.body for n in ast.walk(node) if isinstance(n, ast.Call)]
    assert "_scripted_state" not in calls
    assert "deposit.command" not in calls
    assert "stem_script_time" not in calls


def test_config_rejects_overwrite(cfg):
    cfg["recording"]["overwrite"] = True
    with pytest.raises(ValueError, match="overwrite"):
        PolicyController(None).configure(cfg)


def test_relaxed_contact_stop_preserves_quality_limit(cfg):
    policy = PolicyController(SimpleNamespace(metadata={}, url="ws://test"))
    policy.configure(cfg)
    assert not policy.contact_requires_stop(122.295364)
    assert not policy.contact_requires_stop(579.486694)
    assert not policy.contact_requires_stop(2000.)
    assert policy.contact_requires_stop(2000.01)
    assert policy.contact_requires_stop(np.nan)
    assert policy.contact_requires_stop(np.inf)
    assert cfg["task"]["unsafe_contact_force_N"] == 35.
    assert policy.summary()["quality_contact_force_N"] == 35.
    assert policy.summary()["contact_stop_force_N"] == 2000.
    assert PolicyController(None, contact_stop_force_n=150.).contact_requires_stop(579.)
    assert PolicyController(None, contact_stop_force_n=35.).contact_requires_stop(122.)


@pytest.mark.parametrize("limit", [0., -1., np.nan, np.inf])
def test_invalid_contact_stop_limit(limit):
    with pytest.raises(ValueError, match="finite and positive"):
        PolicyController(None, contact_stop_force_n=limit)


def test_disabled_contact_stop_keeps_numerical_guard_and_quality(cfg):
    policy = PolicyController(SimpleNamespace(metadata={}, url="ws://test"), disable_contact_stop=True)
    policy.configure(cfg)
    for force in [35., 2000., 2026.07788, 1e10]:
        assert not policy.contact_requires_stop(force)
    for force in [np.nan, np.inf, -np.inf]:
        assert policy.contact_requires_stop(force)
    report = policy.summary()
    assert report["contact_stop_enabled"] is False
    assert report["contact_stop_force_N"] is None
    assert report["quality_contact_force_N"] == 35.
    assert cfg["task"]["unsafe_contact_force_N"] == 35.


def test_transport_metadata_infer_and_text_error(monkeypatch):
    msgpack = pytest.importorskip("msgpack")
    client_module = pytest.importorskip("websockets.sync.client")
    from demos.cherry_tomato_pick.openpi_policy import OpenPIClient
    messages = iter([msgpack.packb({"model": "test"}),
                     msgpack.packb({"actions": np.ones((16, 16))}, default=encode_numpy), "test server error"])
    sent, closed = [], []
    socket = SimpleNamespace(recv=lambda timeout: next(messages), send=sent.append,
                             close=lambda: closed.append(True))
    monkeypatch.setattr(client_module, "connect", lambda *a, **kw: socket)
    client = OpenPIClient()
    assert client.metadata == {"model": "test"}
    assert client.infer({"state": np.ones(16)})["actions"].shape == (16, 16)
    with pytest.raises(RuntimeError, match="test server error"):
        client.infer({})
    assert closed == [True]


def test_transport_timeout_closes(monkeypatch):
    msgpack = pytest.importorskip("msgpack")
    client_module = pytest.importorskip("websockets.sync.client")
    from demos.cherry_tomato_pick.openpi_policy import OpenPIClient
    closed = []
    def receive(timeout):
        raise TimeoutError("deadline reached")
    socket = SimpleNamespace(recv=receive, close=lambda: closed.append(True))
    monkeypatch.setattr(client_module, "connect", lambda *a, **kw: socket)
    with pytest.raises(RuntimeError, match="deadline reached"):
        OpenPIClient(timeout=.01)
    assert closed == [True]


def test_placement_needs_release_correct_box_and_settling():
    from demos.cherry_tomato_pick.tableware import DepositMotion
    # Isolate the unchanged physical observer from all scripted trajectories.
    deposit = DepositMotion.__new__(DepositMotion)
    deposit.failed = False
    deposit.placed = False
    deposit.release_time = None
    deposit.index = 1
    deposit.settle_s = 0.
    deposit.tableware = SimpleNamespace(inside=lambda p, r, index: p[0] == index)
    for _ in range(100):
        deposit.observe([1, 0, 0], [0, 0, 0], .01, .005)
    assert not deposit.placed  # No release.
    deposit.release_time = 1.
    for _ in range(100):
        deposit.observe([0, 0, 0], [0, 0, 0], .01, .005)
    assert not deposit.placed  # Wrong box.
    for _ in range(100):
        deposit.observe([1, 0, 0], [.09, 0, 0], .01, .005)
    assert not deposit.placed  # Not settled.
    for _ in range(61):
        deposit.observe([1, 0, 0], [0, 0, 0], .01, .005)
    assert deposit.placed


def test_reclosing_resets_policy_release_observer():
    deposit = SimpleNamespace(release_time=None, settle_s=0., placed=False)
    PolicyController.update_release(deposit, 1., True, 0.)
    assert deposit.release_time == 1.
    deposit.settle_s = .29
    PolicyController.update_release(deposit, 2., True, .8)
    assert deposit.release_time is None
    assert deposit.settle_s == 0.
    assert not deposit.placed
    PolicyController.update_release(deposit, 3., False, 0.)
    assert deposit.release_time is None


@pytest.mark.parametrize("quality_accepted", [True, False])
@pytest.mark.parametrize("disable_contact_stop", [True, False])
@pytest.mark.parametrize("grasp_before_lift", [True, False])
def test_eval_exit_requires_saved_quality_accepted_recording(cfg, tmp_path, monkeypatch, quality_accepted,
                                                          disable_contact_stop, grasp_before_lift):
    import json
    from demos.cherry_tomato_pick import openpi_eval, scene
    from lycheesim.engine import runner
    closed = []
    client = SimpleNamespace(metadata={}, url="ws://test", close=lambda: closed.append(True))
    monkeypatch.setattr(openpi_eval, "OpenPIClient", lambda *args: client)
    recorder = SimpleNamespace(metadata={}, output_path=tmp_path / "episode.h5", seed=42,
                               quality_report={"accepted": quality_accepted})

    def build(cfg, path, *, policy):
        policy.configure(cfg)
        return SimpleNamespace(_episode_recorder=recorder, step=lambda *args: None,
                               verify=lambda: None, close=lambda: recorder.output_path.touch())

    def run(builder, **kwargs):
        setup = builder(cfg, None)
        setup.step(0, {})
        try:
            setup.verify()
        finally:
            setup.close()

    monkeypatch.setattr(scene, "build", build)
    monkeypatch.setattr(runner, "main", run)
    args = ["--disable-contact-stop"] if disable_contact_stop else []
    if grasp_before_lift:
        args += ["--grasp-before-lift", "--grasp-wait-timeout", "6"]
    assert openpi_eval.main(args) == (0 if quality_accepted else 1)
    report = json.loads(recorder.output_path.with_suffix(".evaluation.json").read_text())
    assert report["accepted"] == quality_accepted
    assert report["recording_saved"]
    assert report["contact_stop_enabled"] is not disable_contact_stop
    assert report["grasp_lift_gate"]["enabled"] is grasp_before_lift
    if grasp_before_lift:
        assert report["grasp_lift_gate"]["timeout_s"] == 6.
    assert closed == [True]
