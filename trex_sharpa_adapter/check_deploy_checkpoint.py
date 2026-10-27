"""Validate a full-trained or merged-LoRA Sharpa checkpoint before inference.

This is a file/contract check; the GPU model load and live sensors need separate
validation. It does not deserialize model.pt or change any checkpoint file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check(ckpt: Path) -> dict:
    ckpt = ckpt.resolve(strict=True)
    required = (
        "model.pt", "config.json", "training_args.json", "stats_data.json",
        "adapter_metadata.json", "processor/tokenizer.json",
        "processor/preprocessor_config.json", "robot/left.urdf", "robot/right.urdf",
    )
    for relative in required:
        require((ckpt / relative).is_file(), f"Missing {relative}")
    size = (ckpt / "model.pt").stat().st_size
    require(size > 1_000_000_000, "model.pt is too small for a full checkpoint")

    args = read_json(ckpt / "training_args.json")
    metadata = read_json(ckpt / "adapter_metadata.json")
    config = read_json(ckpt / "config.json")
    stats = read_json(ckpt / "stats_data.json")

    for key, expected in {
        "action_dim": 62, "action_chunk": 16,
        "use_robot_state": 0, "use_tactile_deform": 1,
        "use_tactile_vec": 1, "use_tactile_vqvae": 1,
    }.items():
        require(args.get(key) == expected, f"training_args.{key}: expected {expected}, got {args.get(key)}")
    rank = args.get("lora_rank")
    require(type(rank) is int and rank >= 0, "training_args.lora_rank must be a nonnegative integer")
    origin_file = ckpt / "lora_origin.json"
    if rank == 0:
        require(not origin_file.exists(), "Full-training checkpoint has unexpected lora_origin.json")
        kind = "full_finetune"
    else:
        require(origin_file.is_file(),
                "LoRA adapter is not a deployable checkpoint; merge it with the matching midtrain model first")
        origin = read_json(origin_file)
        require(origin.get("rank") == rank, "Merged LoRA rank disagrees with training_args.json")
        require(origin.get("alpha") == args.get("lora_alpha"),
                "Merged LoRA alpha disagrees with training_args.json")
        require(all(isinstance(origin.get(key), str) and origin[key]
                    for key in ("base_checkpoint", "adapter_checkpoint")),
                "Merged LoRA provenance is incomplete")
        kind = "merged_lora"
    require(config.get("model_type") == "qwen3_vl", "Expected Qwen3-VL checkpoint")
    require(metadata.get("action_dim") == 62 and metadata.get("action_chunk") == 16,
            "Action dimensions disagree with adapter metadata")
    require(metadata.get("cameras") == {
        "head": "cam0", "wrist_right": "cam1", "wrist_left": "cam2",
    }, "Camera identities differ from training")
    require(len(metadata.get("hand_order", [])) == 22, "Expected 22 joints per hand")
    for side in ("left", "right"):
        model = metadata.get("robot_models", {}).get(side, {})
        require(model.get("tip_link") == f"{side}_fr3v2_link8", f"{side} arm must target link8")
        urdf = ckpt / model.get("urdf", "")
        require(urdf.is_file(), f"Missing {side} arm URDF")
        digest = hashlib.sha256(urdf.read_bytes()).hexdigest()
        require(digest == model.get("sha256"), f"{side} arm URDF hash mismatch")

    require("rlbench" in stats, "Expected training statistics key 'rlbench'")
    block = stats["rlbench"]
    require(len(block["action"]["mask"]) == 62, "Action normalization mask is not 62D")
    for key in ("q01", "q99"):
        values = block["action"][key]
        require(len(values) == 16 and all(len(row) == 62 for row in values),
                f"Action {key} statistics are not [16,62]")
    for key in ("mask", "q01", "q99"):
        require(len(block["tactile_f6"][key]) == 60, f"Tactile F6 {key} is not 10x6")

    return {
        "checkpoint": str(ckpt), "model_bytes": size,
        "checkpoint_kind": kind, "lora_rank": rank,
        "action_shape": [16, 62], "tactile_f6_shape": [10, 6],
        "cameras": metadata["cameras"], "arm_tip": "link8",
        "processor": str(ckpt / "processor"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint), ensure_ascii=False, indent=2))
