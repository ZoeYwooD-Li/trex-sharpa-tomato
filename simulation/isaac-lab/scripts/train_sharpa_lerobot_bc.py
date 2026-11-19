#!/usr/bin/env python3
"""Train a small Sharpa behavior-cloning baseline from LeRobot Parquet demos.

Modes are comparable: both consume the same 116-D robot state and predict the
same 58-D action; tactile-force additionally consumes pooled tactile maps and
the ten 6-axis fingertip wrench measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[1]


def pool_tactile(images: np.ndarray, grid: int) -> np.ndarray:
    batch, sides, fingers, height, width = images.shape
    if height % grid or width % grid:
        raise ValueError(f"Tactile images {height}x{width} cannot be pooled to {grid}x{grid}")
    h, w = height // grid, width // grid
    pooled = images.reshape(batch, sides, fingers, grid, h, grid, w).mean(axis=(4, 6))
    return np.ascontiguousarray(pooled.reshape(batch, -1) / 255.0, dtype=np.float32)


class Policy(nn.Module):
    def __init__(self, input_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, action_size),
        )

    def forward(self, value):
        return self.network(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Sharpa LeRobot root, containing A/* and/or B/* datasets")
    parser.add_argument("--mode", choices=("tactile-force", "state"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--stride", type=int, default=4,
                        help="Use every Nth frame to keep training memory manageable")
    parser.add_argument("--tactile-grid", type=int, default=8,
                        help="Spatial grid per sensor; 8 pools each 240x240 map to 8x8")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.stride, args.tactile_grid) < 1:
        parser.error("epochs, batch-size, stride, and tactile-grid must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sys_path = str(PROJECT / "handson")
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from handson.data.sharpa_lerobot_bridge import discover_datasets, iter_episode_batches

    dataset_root = args.dataset.expanduser().resolve()
    datasets = discover_datasets(dataset_root)
    episodes: list[tuple[np.ndarray, np.ndarray, str]] = []
    for dataset in datasets:
        for episode in dataset.episodes:
            episode_index = int(episode["episode_index"])
            xs, ys = [], []
            offset = 0
            for batch in iter_episode_batches(dataset, episode_index, batch_size=8):
                take = (np.arange(len(batch.action)) + offset) % args.stride == 0
                offset += len(batch.action)
                if not np.any(take):
                    continue
                state = batch.state[take]
                if args.mode == "state":
                    features = state
                else:
                    tactile = pool_tactile(batch.deformation[take], args.tactile_grid)
                    wrench = batch.wrench[take].reshape(int(take.sum()), -1)
                    features = np.concatenate((state, tactile, wrench), axis=1)
                actions = batch.action[take]
                # Use the same rows for both ablation arms. Tactile maps are
                # uint8, so only wrench values can add non-finite samples.
                valid = (np.isfinite(state).all(axis=1)
                         & np.isfinite(actions).all(axis=1)
                         & np.isfinite(batch.wrench[take]).all(axis=(1, 2, 3)))
                if valid.any():
                    xs.append(features[valid])
                    ys.append(actions[valid])
            if xs:
                episodes.append((np.concatenate(xs), np.concatenate(ys), f"{dataset.root.name}:{episode_index}"))
            print(f"loaded {dataset.root.name} episode {episode_index}", flush=True)
    if not episodes:
        parser.error("No finite state/action samples found in the dataset")

    order = np.random.permutation(len(episodes)).tolist()
    validation_ids = set(order[:max(1, round(0.1 * len(order)))]) if len(order) > 1 else set()
    train_eps = [episodes[i] for i in range(len(episodes)) if i not in validation_ids]
    val_eps = [episodes[i] for i in range(len(episodes)) if i in validation_ids]
    if not train_eps:
        train_eps, val_eps = episodes, []
    x_train = np.concatenate([e[0] for e in train_eps])
    y_train = np.concatenate([e[1] for e in train_eps])
    x_val = np.concatenate([e[0] for e in val_eps]) if val_eps else x_train
    y_val = np.concatenate([e[1] for e in val_eps]) if val_eps else y_train
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0).clip(1e-5)
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0).clip(1e-5)
    xn = lambda x: np.clip((x - x_mean) / x_std, -10, 10).astype(np.float32)
    yn = lambda y: ((y - y_mean) / y_std).astype(np.float32)
    x_train_n, y_train_n, x_val_n, y_val_n = xn(x_train), yn(y_train), xn(x_val), yn(y_val)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n)),
                        batch_size=args.batch_size, shuffle=True)
    model = Policy(x_train_n.shape[1], y_train_n.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    val_x, val_y = torch.from_numpy(x_val_n).to(device), torch.from_numpy(y_val_n).to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_mse", "val_mse"))
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(bx), by)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_x), val_y).cpu())
            train_loss = float(np.mean(losses))
            writer.writerow((epoch, train_loss, val_loss))
            stream.flush()
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(), "input_size": int(x_train.shape[1]),
                    "action_size": int(y_train.shape[1]), "mode": args.mode,
                    "x_mean": torch.from_numpy(x_mean.astype(np.float32)),
                    "x_std": torch.from_numpy(x_std.astype(np.float32)),
                    "y_mean": torch.from_numpy(y_mean.astype(np.float32)),
                    "y_std": torch.from_numpy(y_std.astype(np.float32)),
                }, args.output / "policy.pt")
            print(f"epoch {epoch:03d}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
    summary = {
        "kind": "one-step behavior-cloning baseline", "mode": args.mode,
        "input": (["116-D robot state"] if args.mode == "state" else
                  ["116-D robot state", f"10 tactile maps pooled to {args.tactile_grid}x{args.tactile_grid}",
                   "10 six-axis fingertip wrench readings"]),
        "target": "58-D recorded joint action", "episodes": len(episodes),
        "train_samples": int(len(x_train)), "validation_samples": int(len(x_val)),
        "train_episodes": [e[2] for e in train_eps],
        "validation_episodes": [e[2] for e in val_eps],
        "dataset": str(dataset_root), "seed": args.seed,
        "input_dim": int(x_train.shape[1]), "action_dim": int(y_train.shape[1]),
        "frame_stride": args.stride, "device": device_name, "best_validation_mse": best_val,
        "rgb_used": False, "checkpoint": "policy.pt", "calibrated_to_hardware": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
