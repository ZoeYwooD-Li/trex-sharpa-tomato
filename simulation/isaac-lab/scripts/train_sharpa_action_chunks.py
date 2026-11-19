#!/usr/bin/env python3
"""Train a compact ACT-style RGB/state/tactile action-chunk policy on Sharpa LeRobot data.

This is a controlled first model for the touch ablation, not the original ACT
CVAE or T-REx. Both modes instantiate the same network; the no-touch mode masks
the tactile token while retaining the same RGB, state, actions, and split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "handson"))

from handson.data.sharpa_lerobot_bridge import discover_datasets, iter_episode_batches  # noqa: E402
from scripts.train_sharpa_lerobot_bc import pool_tactile  # noqa: E402


@dataclass
class Episode:
    name: str
    rgb: np.ndarray          # [samples, 3 cameras, 3 channels, H, W] uint8
    state: np.ndarray        # [samples, 116]
    touch: np.ndarray        # [samples, 10*grid*grid + 60]
    indices: np.ndarray      # absolute frame indices for samples
    actions: np.ndarray      # [all episode frames, 58]


def decode_video(path: Path, wanted: np.ndarray, size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open camera video: {path}")
    frames = np.empty((len(wanted), 3, size, size), dtype=np.uint8)
    cursor = 0
    try:
        for frame_index in range(int(wanted[-1]) + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Video ends before frame {frame_index}: {path}")
            if frame_index == wanted[cursor]:
                rgb = cv2.cvtColor(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB)
                frames[cursor] = np.transpose(rgb, (2, 0, 1))
                cursor += 1
                if cursor == len(wanted):
                    break
    finally:
        capture.release()
    if cursor != len(wanted):
        raise ValueError(f"Decoded {cursor}/{len(wanted)} requested frames from {path}")
    return frames


def load_episode(dataset, episode_index: int, stride: int, grid: int,
                 rgb_size: int, chunk: int) -> Episode:
    actions, states, touches, indices = [], [], [], []
    offset = 0
    for batch in iter_episode_batches(dataset, episode_index, batch_size=8):
        n = len(batch.action)
        if not np.array_equal(batch.frame_index, np.arange(offset, offset + n)):
            raise ValueError(f"Nonconsecutive frame indices in {dataset.root}, episode {episode_index}")
        actions.append(batch.action)
        take = (batch.frame_index % stride) == 0
        if take.any():
            states.append(batch.state[take])
            tactile = pool_tactile(batch.deformation[take], grid)
            wrench = batch.wrench[take].reshape(int(take.sum()), -1)
            touches.append(np.concatenate((tactile, wrench), axis=1))
            indices.append(batch.frame_index[take])
        offset += n

    all_actions = np.concatenate(actions, axis=0)
    state = np.concatenate(states, axis=0)
    touch = np.concatenate(touches, axis=0)
    index = np.concatenate(indices, axis=0)
    if len(all_actions) != dataset.episodes[episode_index]["length"]:
        raise ValueError(f"Parquet length mismatch in {dataset.root}, episode {episode_index}")
    action_valid = np.isfinite(all_actions).all(axis=1)
    valid = np.isfinite(state).all(axis=1) & np.isfinite(touch).all(axis=1)
    for i, start in enumerate(index):
        valid[i] &= action_valid[start:min(start + chunk, len(all_actions))].all()
    if not valid.any():
        raise ValueError(f"No finite samples in {dataset.root}, episode {episode_index}")
    state, touch, index = state[valid], touch[valid], index[valid]

    cameras = [decode_video(dataset.video_file(episode_index, camera), index, rgb_size)
               for camera in ("cam0", "cam1", "cam2")]
    rgb = np.stack(cameras, axis=1)
    return Episode(f"{dataset.root.name}:{episode_index}", rgb, state, touch,
                   index, all_actions)


class Windows(Dataset):
    def __init__(self, episodes: list[Episode], chunk: int,
                 state_mean: np.ndarray, state_std: np.ndarray,
                 touch_mean: np.ndarray, touch_std: np.ndarray,
                 action_mean: np.ndarray, action_std: np.ndarray):
        self.episodes = episodes
        self.chunk = chunk
        self.state_mean, self.state_std = state_mean, state_std
        self.touch_mean, self.touch_std = touch_mean, touch_std
        self.action_mean, self.action_std = action_mean, action_std
        self.index = [(ei, i) for ei, ep in enumerate(episodes) for i in range(len(ep.indices))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        episode_index, sample_index = self.index[item]
        ep = self.episodes[episode_index]
        start = int(ep.indices[sample_index])
        end = min(start + self.chunk, len(ep.actions))
        target = np.zeros((self.chunk, 58), dtype=np.float32)
        target[:end - start] = (ep.actions[start:end] - self.action_mean) / self.action_std
        mask = np.zeros(self.chunk, dtype=np.float32)
        mask[:end - start] = 1.0
        state = np.clip((ep.state[sample_index] - self.state_mean) / self.state_std, -10, 10)
        touch = np.clip((ep.touch[sample_index] - self.touch_mean) / self.touch_std, -10, 10)
        return (ep.rgb[sample_index], state.astype(np.float32), touch.astype(np.float32),
                target, mask)


class RGBEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, width),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, cameras, channels, height, width = images.shape
        flat = images.reshape(batch * cameras, channels, height, width).float() / 255.0
        return self.network(flat).reshape(batch, cameras, -1)


class ActionChunkPolicy(nn.Module):
    def __init__(self, touch_size: int, chunk: int, width: int = 256):
        super().__init__()
        self.rgb = RGBEncoder(width)
        self.state = nn.Sequential(nn.Linear(116, width), nn.GELU(), nn.Linear(width, width))
        self.touch = nn.Sequential(nn.Linear(touch_size, width), nn.GELU(), nn.Linear(width, width))
        self.camera_embedding = nn.Parameter(torch.randn(3, width) * 0.02)
        self.query = nn.Parameter(torch.randn(chunk, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, nhead=8, dim_feedforward=width * 2,
                                           dropout=0.1, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.output = nn.Linear(width, 58)

    def forward(self, rgb: torch.Tensor, state: torch.Tensor,
                touch: torch.Tensor, use_touch: bool) -> torch.Tensor:
        camera = self.rgb(rgb) + self.camera_embedding[None]
        state_token = self.state(state)[:, None]
        if use_touch:
            touch_token = self.touch(touch)[:, None]
        else:
            touch_token = torch.zeros_like(state_token)
        query = self.query[None].expand(len(rgb), -1, -1)
        tokens = torch.cat((camera, state_token, touch_token, query), dim=1)
        return self.output(self.transformer(tokens)[:, -len(self.query):])


def masked_l1(prediction: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    error = (prediction - target).abs().mean(dim=-1)
    return (error * mask).sum() / mask.sum().clamp(min=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=("rgb-state", "rgb-state-touch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--tactile-grid", type=int, default=8)
    parser.add_argument("--rgb-size", type=int, default=96)
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="First N episodes for a data/training smoke test; 0 uses all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.sample_stride, args.chunk,
           args.tactile_grid, args.rgb_size) < 1 or args.max_episodes < 0:
        parser.error("Training sizes must be positive and --max-episodes nonnegative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    datasets = discover_datasets(args.dataset.expanduser().resolve())
    episodes = []
    for dataset in datasets:
        for metadata in dataset.episodes:
            if args.max_episodes and len(episodes) >= args.max_episodes:
                break
            ep = load_episode(dataset, int(metadata["episode_index"]), args.sample_stride,
                              args.tactile_grid, args.rgb_size, args.chunk)
            episodes.append(ep)
            print(f"loaded {ep.name}: {len(ep.indices)} windows", flush=True)
        if args.max_episodes and len(episodes) >= args.max_episodes:
            break
    if not episodes:
        parser.error("No usable episodes")

    order = np.random.permutation(len(episodes))
    val_ids = set(order[:max(1, round(len(order) * 0.1))]) if len(episodes) > 1 else set()
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_ids]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_ids] or train_eps
    states = np.concatenate([ep.state for ep in train_eps])
    touches = np.concatenate([ep.touch for ep in train_eps])
    actions = np.concatenate([ep.actions for ep in train_eps])
    actions = actions[np.isfinite(actions).all(axis=1)]
    mean_std = lambda x: (x.mean(axis=0).astype(np.float32),
                          x.std(axis=0).clip(1e-5).astype(np.float32))
    state_mean, state_std = mean_std(states)
    touch_mean, touch_std = mean_std(touches)
    action_mean, action_std = mean_std(actions)
    stats = (state_mean, state_std, touch_mean, touch_std, action_mean, action_std)
    train = Windows(train_eps, args.chunk, *stats)
    val = Windows(val_eps, args.chunk, *stats)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(device_name)
    model = ActionChunkPolicy(train_eps[0].touch.shape[1], args.chunk).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    use_touch = args.mode == "rgb-state-touch"
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_l1", "validation_l1"))
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_total, train_weight = 0.0, 0.0
            for rgb, state, touch, target, mask in train_loader:
                rgb, state, touch, target, mask = (x.to(device, non_blocking=True)
                                                   for x in (rgb, state, touch, target, mask))
                optimizer.zero_grad(set_to_none=True)
                loss = masked_l1(model(rgb, state, touch, use_touch), target, mask)
                loss.backward()
                optimizer.step()
                weight = float(mask.sum().item())
                train_total += float(loss.item()) * weight
                train_weight += weight
            model.eval()
            val_total, val_weight = 0.0, 0.0
            with torch.no_grad():
                for rgb, state, touch, target, mask in val_loader:
                    rgb, state, touch, target, mask = (x.to(device, non_blocking=True)
                                                       for x in (rgb, state, touch, target, mask))
                    loss = masked_l1(model(rgb, state, touch, use_touch), target, mask)
                    weight = float(mask.sum().item())
                    val_total += float(loss.item()) * weight
                    val_weight += weight
            train_l1, val_l1 = train_total / train_weight, val_total / val_weight
            writer.writerow((epoch, train_l1, val_l1))
            stream.flush()
            print(f"epoch {epoch}/{args.epochs}: train_l1={train_l1:.6f} val_l1={val_l1:.6f}", flush=True)
            if val_l1 < best:
                best = val_l1
                torch.save({"model_state_dict": model.state_dict(), "mode": args.mode,
                            "chunk": args.chunk, "sample_stride": args.sample_stride,
                            "rgb_size": args.rgb_size,
                            "tactile_grid": args.tactile_grid,
                            "state_names": list(datasets[0].state_names),
                            "action_names": list(datasets[0].action_names),
                            "state_mean": torch.from_numpy(state_mean),
                            "state_std": torch.from_numpy(state_std),
                            "touch_mean": torch.from_numpy(touch_mean),
                            "touch_std": torch.from_numpy(touch_std),
                            "action_mean": torch.from_numpy(action_mean),
                            "action_std": torch.from_numpy(action_std)},
                           args.output / "policy.pt")
    summary = {
        "kind": "compact multimodal action-chunk Transformer (ACT-style)",
        "mode": args.mode, "dataset": str(args.dataset.expanduser().resolve()),
        "train_episodes": [ep.name for ep in train_eps],
        "validation_episodes": [ep.name for ep in val_eps],
        "train_windows": len(train), "validation_windows": len(val),
        "rgb_cameras": ["cam0", "cam1", "cam2"], "state_dim": 116,
        "tactile_dim": int(train_eps[0].touch.shape[1]), "action_dim": 58,
        "chunk": args.chunk, "sample_stride": args.sample_stride,
        "rgb_size": args.rgb_size, "tactile_grid": args.tactile_grid,
        "seed": args.seed, "epochs": args.epochs, "device": device_name,
        "best_validation_l1": best, "checkpoint": "policy.pt",
        "closed_loop_success_measured": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
