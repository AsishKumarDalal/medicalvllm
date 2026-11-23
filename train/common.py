"""Shared training utilities: config, seed, device, optimizer, loop, saves.

Deliberately plain PyTorch + YAML (no Lightning / W&B). The structure mirrors
minillm's trainers: hourly atomic saves, JSON metrics, a stop file, SIGTERM-safe
final save, and optional `--steps` / `--hours` caps.
"""

import json
import logging
import math
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- config
def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


# -------------------------------------------------------------------- seeding
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------- saves
def atomic_torch_save(state: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


class Metrics:
    """Append-only JSONL metrics with a mirrored ``metrics.json``."""

    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.dir / "metrics.jsonl"
        self.json = self.dir / "metrics.json"
        self.rows = []

    def log(self, **row):
        row.setdefault("time", round(time.time(), 3))
        self.rows.append(row)
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.json.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")


# --------------------------------------------------------------- optimization
def build_lr_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup then cosine decay to ``min_lr_ratio`` of peak."""
    warmup = max(0, int(warmup_steps))
    total = max(warmup + 1, int(total_steps))
    floor = float(min_lr_ratio)

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


class StopSignal:
    """Handles ``--stop`` file and SIGTERM for clean final saves."""

    def __init__(self, stop_file: str | Path | None = None):
        self.stop_file = Path(stop_file) if stop_file else None
        self.triggered = False
        if self.stop_file and self.stop_file.exists():
            self.stop_file.unlink()
        signal.signal(signal.SIGTERM, self._handle)
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, *_):
        self.triggered = True

    def check(self) -> bool:
        if self.triggered:
            return True
        if self.stop_file and self.stop_file.exists():
            self.triggered = True
        return self.triggered


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def clip_grad_norm(model, max_norm: float) -> float:
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm).item()
