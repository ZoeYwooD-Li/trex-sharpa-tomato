"""Export a LoRA run as a standard T-Rex checkpoint for scripts/test.py."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from sharpa_lora import merge_lora_weights


def merge(base: Path, adapter: Path, output: Path) -> None:
    base, adapter, output = base.resolve(), adapter.resolve(), output.resolve()
    if output in (base, adapter) or output.exists():
        raise FileExistsError("Output must be a new directory distinct from both inputs")
    base_file = base / "model.pt" if base.is_dir() else base
    config = json.loads((adapter / "lora_config.json").read_text(encoding="utf-8"))
    base_sd = torch.load(base_file, map_location="cpu", weights_only=False)
    if "state_dict" in base_sd:
        base_sd = base_sd["state_dict"]
    lora_sd = torch.load(adapter / "lora.pt", map_location="cpu", weights_only=True)
    merged = merge_lora_weights(base_sd, lora_sd, config)

    output.mkdir(parents=True)
    torch.save(merged, output / "model.pt")
    for name in ("config.json", "training_args.json", "stats_data.json",
                 "adapter_metadata.json"):
        src = adapter / name
        if not src.exists() and base.is_dir():
            src = base / name
        if src.exists():
            shutil.copy2(src, output / name)
    for name in ("processor", "robot"):
        src = adapter / name
        if not src.exists() and base.is_dir():
            src = base / name
        if src.exists():
            shutil.copytree(src, output / name)
    (output / "lora_origin.json").write_text(json.dumps({
        "base_checkpoint": str(base_file), "adapter_checkpoint": str(adapter),
        "rank": config["rank"], "alpha": config["alpha"]
    }, indent=2), encoding="utf-8")
    print("Merged standard T-Rex checkpoint:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.base_checkpoint, args.adapter_checkpoint, args.output)
