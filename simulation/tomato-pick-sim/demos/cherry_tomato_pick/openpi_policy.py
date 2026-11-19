"""OpenPI transport and absolute TCP control; no model dependencies in the simulator.

Wire format follows Physical-Intelligence/openpi's msgpack_numpy protocol.
This is a small independent implementation, not a vendored openpi-client.
"""
from __future__ import annotations

from collections import deque
import time

import numpy as np

from .grasp_lift_gate import GraspLiftGate

PROMPTS = (
    "Pick the ripe tomato and place it in the nearest box.",
    "Pick the ripe tomato and place it in the middle box.",
    "Pick the ripe tomato and place it in the farthest box.",
)


def encode_numpy(value):
    if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic) and value.dtype.kind in "biuf":
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"unsupported wire value: {type(value).__name__}")


def decode_numpy(value):
    if b"__ndarray__" in value or b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError(f"unsupported wire dtype: {dtype}")
        if b"__ndarray__" in value:
            return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
        return dtype.type(value[b"data"])
    return value


class OpenPIClient:
    """One bounded connection, one in-flight inference, no hidden retry loop."""

    def __init__(self, url="ws://127.0.0.1:8000", timeout=180.0):
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be finite and positive")
        try:
            import msgpack
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("Install inference transport: uv sync --locked --extra openpi") from exc
        self.url, self.timeout, self.codec = url, timeout, msgpack
        self.socket = None
        try:
            self.socket = connect(url, compression=None, max_size=16 * 1024 * 1024,
                                  open_timeout=min(timeout, 10), close_timeout=2,
                                  ping_interval=None, proxy=None)
            self.metadata = self._receive()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"OpenPI connection failed ({url}): {exc}") from exc

    def _receive(self):
        frame = self.socket.recv(timeout=self.timeout)
        if isinstance(frame, str):
            raise RuntimeError(f"OpenPI server error: {frame[:2000]}")
        value = self.codec.unpackb(frame, object_hook=decode_numpy)
        if not isinstance(value, dict):
            raise ValueError("OpenPI response must be a dictionary")
        return value

    def infer(self, observation):
        try:
            self.socket.send(self.codec.packb(observation, default=encode_numpy))
            return self._receive()
        except Exception as exc:
            self.close()
            raise RuntimeError(f"OpenPI inference failed: {exc}") from exc

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None


def make_observation(scene, camera_state, box_index):
    if box_index not in range(3):
        raise ValueError("target box must be 0, 1 or 2")
    images = {"base": camera_state["camera_color"],
              "left_wrist": camera_state["wrist_camera_color"]["left"],
              "right_wrist": camera_state["wrist_camera_color"]["right"]}
    for name, image in images.items():
        if image is None or image.dtype != np.uint8 or image.shape != (200, 320, 3):
            raise ValueError(f"{name} must be RGB uint8 with training shape (200, 320, 3)")
    left, right = scene.robot.tcp_poses()
    state = np.concatenate((left, right, [scene.left_grip, scene.right_grip])).astype(np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("measured state must be finite with shape (16,)")
    return {"images": images, "state": state, "prompt": PROMPTS[box_index]}


def absolute_action(action, measured_state, robot_cfg):
    """Normalize xyzw, align signs, clamp world XYZ/closures; never add deltas."""
    result = np.asarray(action, dtype=np.float64).copy()
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError("action must be finite with shape (16,)")
    clipped = 0
    for side, offset in (("left", 0), ("right", 7)):
        xyz = result[offset:offset + 3]
        bounded = np.clip(xyz, robot_cfg[f"{side}_workspace_min"], robot_cfg[f"{side}_workspace_max"])
        clipped += int(np.count_nonzero(xyz != bounded))
        result[offset:offset + 3] = bounded
        quat = result[offset + 3:offset + 7]
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError(f"{side} action quaternion has invalid norm")
        quat /= norm
        if np.dot(quat, measured_state[offset + 3:offset + 7]) < 0:
            quat *= -1
    closures = np.clip(result[14:], 0., 1.)
    clipped += int(np.count_nonzero(result[14:] != closures))
    result[14:] = closures
    return result, clipped


class PolicyController:
    """30 Hz simulator-time execution, replan after a short action prefix."""

    def __init__(self, client, execute_steps=4, checkpoint_label="unspecified", contact_stop_force_n=2000.,
                 *, disable_contact_stop=False, grasp_before_lift=False, grasp_wait_timeout_s=8.):
        if not 1 <= execute_steps <= 16:
            raise ValueError("execute_steps must be between 1 and the trained horizon 16")
        if not np.isfinite(contact_stop_force_n) or contact_stop_force_n <= 0:
            raise ValueError("contact stop force must be finite and positive")
        self.contact_stop_force_n = float(contact_stop_force_n)
        self.contact_stop_enabled = not disable_contact_stop
        self.grasp_gate = GraspLiftGate(grasp_before_lift, grasp_wait_timeout_s)
        self.client, self.execute_steps = client, execute_steps
        self.checkpoint_label = checkpoint_label
        self.pending = deque()
        self.next_time = 0.
        self.action_steps = 0
        self.latencies = []
        self.clipped_components = 0
        self.error = None
        self.support_engaged = False
        self.last_sim_time = 0.
        self.verified = False

    def configure(self, cfg):
        self.robot_cfg = cfg["robot"]
        self.quality_force_n = float(cfg["task"]["unsafe_contact_force_N"])
        self.grasp_gate.configure(float(cfg["task"]["minimum_support_grip"]))
        self.box_index = int(cfg["deposit"]["target_box"])
        if not cfg["recording"]["enabled"] or not cfg["deposit"]["enabled"]:
            raise ValueError("policy evaluation requires recording and deposit enabled")
        for side in ("left", "right"):
            lo, hi = (np.asarray(self.robot_cfg[f"{side}_workspace_{key}"]) for key in ("min", "max"))
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
                raise ValueError(f"invalid {side} workspace")
        if not np.isclose(cfg["recording"]["robot_sample_hz"], 30):
            raise ValueError("trained policy requires 30 Hz robot sampling")
        if (not np.isfinite(cfg["sim"]["duration"]) or cfg["sim"]["duration"] <= 0
                or cfg["sim"]["dt"] > 1 / 30):
            raise ValueError("evaluation needs positive duration and physics dt <= 1/30 s")
        if cfg["recording"].get("overwrite", False):
            raise ValueError("policy evaluation must not overwrite existing episodes")
        for section, rate in (("camera", 10), ("wrist_cameras", 30)):
            sensor = cfg[section]["sensor"]
            if not sensor.get("enabled", False) or not np.isclose(sensor.get("sample_hz", rate), rate):
                raise ValueError(f"policy requires {section} enabled at {rate} Hz")
        duration = cfg["script"]["retention_tighten_duration_s"]
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("retention_tighten_duration_s must be positive")

    def due(self, time_s, dt):
        return time_s + .5 * dt >= self.next_time

    def contact_requires_stop(self, peak_force_n):
        # Debugging may ignore finite overloads, never numerical divergence.
        return (not np.isfinite(peak_force_n)
                or (self.contact_stop_enabled and peak_force_n > self.contact_stop_force_n))

    def update_grasp_wait(self, time_s, scene):
        if not self.grasp_gate.enabled:
            return
        # The backend refreshes this every physics tick. It already requires
        # opposed/multi-finger contact, sufficient wrap, proximity and hold time.
        if self.grasp_gate.observe(time_s, bool(scene.support_engaged)):
            self.pending.clear()
        if self.grasp_gate.phase == "timed_out":
            self.error = (f"right grasp confirmation timed out after {self.grasp_gate.timeout_s:g}s; "
                          "support was not ready for release of the lift gate")

    @staticmethod
    def update_release(deposit, time_s, detached, closure):
        # Unlike the demonstration script, a policy can close the hand again.
        # Require an open hand throughout settling, not just once in the past.
        if not detached or closure > .1:
            deposit.release_time = None
            deposit.settle_s = 0.
            deposit.placed = False
        elif deposit.release_time is None:
            deposit.release_time = float(time_s)

    def apply(self, time_s, dt, scene, camera_state, backend):
        self.update_grasp_wait(time_s, scene)
        if self.error:
            raise RuntimeError(self.error)
        observation = make_observation(scene, camera_state, self.box_index)
        fresh_inference = False
        if not self.pending:
            start = time.monotonic()
            reply = self.client.infer(observation)
            self.latencies.append(time.monotonic() - start)
            actions = np.asarray(reply.get("actions"), dtype=np.float64)
            if actions.shape != (16, 16) or not np.isfinite(actions).all():
                raise ValueError(f"expected finite OpenPI actions (16, 16), got {actions.shape}")
            for offset in (3, 10):
                norms = np.linalg.norm(actions[:, offset:offset + 4], axis=1)
                if not np.isfinite(norms).all() or np.any(norms < 1e-6):
                    raise ValueError("OpenPI chunk contains an invalid quaternion")
            self.pending.extend(actions[:self.execute_steps].copy())
            fresh_inference = True
        action, clipped = absolute_action(self.pending.popleft(), observation["state"], self.robot_cfg)
        action = self.grasp_gate.constrain(action, observation["state"][9], time_s,
                                           fresh_inference=fresh_inference)
        backend.set_robot_command(np.zeros(10, dtype=np.float32))
        backend.set_robot_states(action[None, :7], action[None, 7:14], action[None, 14:])
        self.clipped_components += clipped
        self.action_steps += 1
        while self.next_time <= time_s + .5 * dt:
            self.next_time += 1 / 30

    def summary(self):
        return {"controller": "openpi_absolute_tcp", "checkpoint_label": self.checkpoint_label,
                "control_mode": "hybrid_grasp_before_lift" if self.grasp_gate.enabled else "model_only",
                "grasp_lift_gate": self.grasp_gate.summary(),
                "server_metadata": self.client.metadata, "endpoint": self.client.url,
                "prompt": PROMPTS[self.box_index], "target_box": self.box_index,
                "action_hz": 30, "execute_steps": self.execute_steps,
                "contact_stop_enabled": self.contact_stop_enabled,
                "contact_stop_force_N": self.contact_stop_force_n if self.contact_stop_enabled else None,
                "quality_contact_force_N": self.quality_force_n,
                "action_steps": self.action_steps, "inference_calls": len(self.latencies),
                "inference_latency_s": self.latencies, "clipped_components": self.clipped_components,
                "sim_time_s": self.last_sim_time, "verified_success": self.verified,
                "error": self.error}
