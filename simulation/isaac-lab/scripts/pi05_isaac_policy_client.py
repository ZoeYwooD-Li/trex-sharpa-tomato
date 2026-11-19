"""Synchronous, NumPy-safe client for the local OpenPI websocket policy server."""

from __future__ import annotations

import time

import msgpack
import numpy as np
from websockets.sync.client import connect


def _pack_numpy(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported observation dtype: {value.dtype}")
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"Cannot MessagePack {type(value).__name__}")


def _unpack_numpy(value):
    if b"__ndarray__" in value:
        return np.frombuffer(value[b"data"], dtype=np.dtype(value[b"dtype"])).reshape(value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def packb(value) -> bytes:
    return msgpack.packb(value, default=_pack_numpy, use_bin_type=True)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=_unpack_numpy, raw=False, strict_map_key=False)


class PI05IsaacClient:
    """Persistent connection that validates the checkpoint contract once."""

    def __init__(self, url: str, *, timeout_s: float = 180.0):
        self.url = url
        self.timeout_s = timeout_s
        self._socket = connect(url, compression=None, max_size=None, open_timeout=10)
        metadata_frame = self._socket.recv(timeout=10)
        if isinstance(metadata_frame, str):
            self._socket.close()
            raise RuntimeError(f"PI05 server metadata was not MessagePack: {metadata_frame}")
        self.metadata = unpackb(metadata_frame)
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        expected = {
            "action_dim": 58,
            "action_horizon": 50,
            "action_hz": 30,
            "action_order": ["left_arm_7", "left_hand_22", "right_arm_7", "right_hand_22"],
            "camera_mapping": {"base": "cam0", "left_wrist": "cam2", "right_wrist": "cam1"},
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ValueError(f"PI05 server metadata mismatch for {key}: "
                                 f"{self.metadata.get(key)!r} != {value!r}")

    def infer(self, *, base: np.ndarray, left_wrist: np.ndarray, right_wrist: np.ndarray,
              state: np.ndarray, prompt: str) -> tuple[np.ndarray, float, dict]:
        images = (base, left_wrist, right_wrist)
        if any(image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 for image in images):
            raise ValueError("PI05 camera observations must be uint8 HWC RGB images")
        if np.asarray(state).shape != (58,) or not np.isfinite(state).all():
            raise ValueError("PI05 state must be a finite measured 58D joint vector")
        observation = {
            "observation/image": np.ascontiguousarray(base),
            "observation/left_wrist_image": np.ascontiguousarray(left_wrist),
            "observation/right_wrist_image": np.ascontiguousarray(right_wrist),
            "observation/state": np.asarray(state, dtype=np.float32),
            "prompt": str(prompt),
        }
        start = time.perf_counter()
        self._socket.send(packb(observation))
        response = self._socket.recv(timeout=self.timeout_s)
        latency_s = time.perf_counter() - start
        if isinstance(response, str):
            raise RuntimeError(f"PI05 inference server error:\n{response}")
        result = unpackb(response)
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (50, 58) or not np.isfinite(actions).all():
            raise ValueError(f"PI05 returned invalid action chunk {actions.shape}")
        return actions, latency_s, result.get("server_timing", {})

    def close(self) -> None:
        self._socket.close()
