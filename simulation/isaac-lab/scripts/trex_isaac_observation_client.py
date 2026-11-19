#!/usr/bin/env python3
"""Send a saved Isaac RGB/tactile observation to a local T-Rex server."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import zmq
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5687)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=("slow_and_fast", "fast"), default="slow_and_fast")
    parser.add_argument("--timeout-ms", type=int, default=180000)
    args = parser.parse_args()

    root = args.input_dir.expanduser().resolve()
    tactile_path = root / "trex_isaac_tactile_input.npz"
    with np.load(tactile_path) as data:
        f6 = np.asarray(data["tactile_f6"], dtype=np.float32)
        deform = np.asarray(data["tactile_deform"], dtype=np.uint8)
    if f6.shape != (16, 10, 6) or deform.shape != (10, 240, 240):
        raise ValueError(f"Unexpected Isaac tactile contract: f6={f6.shape}, deform={deform.shape}")
    if not np.isfinite(f6).all():
        raise ValueError("Isaac force history contains non-finite values")

    image_paths = {
        "image_head": root / "trex_isaac_head.png",
        "image_wrist_right": root / "trex_isaac_wrist_right.png",
        "image_wrist_left": root / "trex_isaac_wrist_left.png",
    }
    for path in image_paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing Isaac camera image: {path}")
    with Image.open(image_paths["image_head"]) as image:
        head_shape = [image.height, image.width, 3]
    with Image.open(image_paths["image_wrist_right"]) as image:
        right_shape = [image.height, image.width, 3]
    with Image.open(image_paths["image_wrist_left"]) as image:
        left_shape = [image.height, image.width, 3]
    payload = {key: path.read_bytes() for key, path in image_paths.items()}
    payload.update({
        "mode": args.mode,
        "task_description": args.task,
        "tactile_f6": f6,
        "tactile_deform": deform,
    })

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.connect(f"tcp://{args.host}:{args.port}")
    try:
        socket.send(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        response = pickle.loads(socket.recv())
    finally:
        socket.close()
        context.term()

    if response.get("status") != "success":
        raise RuntimeError(f"T-Rex returned error: {response.get('message', response)}")
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    if actions.shape != (16, 62) or not np.isfinite(actions).all():
        raise RuntimeError(f"Invalid T-Rex action chunk: {actions.shape}")
    np.save(root / "trex_isaac_action_chunk.npy", actions)
    report = {
        "server": f"{args.host}:{args.port}",
        "task_description": args.task,
        "status": response["status"],
        "mode": response.get("mode"),
        "chunk_id": int(response["chunk_id"]),
        "latency_ms": float(response["latency_ms"]),
        "action_shape": list(actions.shape),
        "all_finite": bool(np.isfinite(actions).all()),
        "action_abs_max": float(np.abs(actions).max()),
        "head_rgb_shape": head_shape,
        "wrist_right_rgb_shape": right_shape,
        "wrist_left_rgb_shape": left_shape,
        "tactile_deform_shape": list(deform.shape),
        "tactile_f6_history_shape": list(f6.shape),
        "action_applied": False,
        "action_contract": "delta-base link8 xyz+rot6d and absolute hand targets; no Isaac IK conversion yet",
    }
    (root / "trex_isaac_policy_result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
