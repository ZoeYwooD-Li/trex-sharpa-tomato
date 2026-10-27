"""Run one synthetic T-Rex inference request; never connects to a robot."""

from __future__ import annotations

import argparse
import io
import pickle

import numpy as np
import zmq
from PIL import Image


def png() -> bytes:
    image = Image.new("RGB", (384, 288), "black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def probe(host: str, port: int, timeout_ms: int) -> tuple[int, int]:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(f"tcp://{host}:{port}")
    picture = png()
    request = {
        "mode": "slow_and_fast",
        "image_head": picture,
        "image_wrist_right": picture,
        "image_wrist_left": picture,
        "task_description": "Pick a cherry tomato.",
        "tactile_f6": np.zeros((16, 10, 6), dtype=np.float32),
        "tactile_deform": np.zeros((10, 240, 240), dtype=np.uint8),
    }
    try:
        socket.send(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(socket.recv())
    finally:
        socket.close()
        context.term()
    if response.get("status") != "success":
        raise RuntimeError(f"Inference failed: {response.get('message', response)}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (16, 62) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid action output: shape={actions.shape}, finite={np.isfinite(actions).all()}")
    print(f"Inference OK: actions={actions.shape}, chunk_id={response['chunk_id']}, "
          f"latency_ms={response['latency_ms']:.1f}")
    return actions.shape


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--timeout-ms", type=int, default=180000)
    args = parser.parse_args()
    probe(args.host, args.port, args.timeout_ms)
