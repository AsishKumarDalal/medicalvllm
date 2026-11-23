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
from dataclasses import asdict, dataclass
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


# ------------------------------------------------------------- run state
@dataclass
class RunState:
    """Counters that survive restarts and cross-phase hand-off.

    ``step`` is GLOBAL: phase 2 continues from the step reached in phase 1.
    """

    phase: int = 1
    step: int = 0
    examples_seen: int = 0
    total_text_tokens: int = 0
    total_images_seen: int = 0
    epoch: float = 0.0
    resumed_from: str = "fresh"

    def update(
        self,
        n_examples: int,
        n_text_tokens: int,
        n_images: int,
        epoch_size: int = 0,
    ):
        self.examples_seen += int(n_examples)
        self.total_text_tokens += int(n_text_tokens)
        self.total_images_seen += int(n_images)
        if epoch_size and epoch_size > 0:
            self.epoch = self.examples_seen / epoch_size

    def as_dict(self) -> dict:
        return asdict(self)


def restore_state(ckpt: dict, state: RunState) -> RunState:
    """Overwrite ``state`` with counters stored in a checkpoint."""
    for key in (
        "step", "examples_seen", "total_text_tokens", "total_images_seen",
    ):
        setattr(state, key, int(ckpt.get(key, getattr(state, key))))
    state.epoch = float(ckpt.get("epoch", state.epoch))
    state.resumed_from = str(ckpt.get("resumed_from", "checkpoint"))
    return state


# ------------------------------------------------- named optimizer state
def optimizer_state_named(model, optimizer) -> dict:
    """Serialize optimizer state keyed by parameter NAME, not tensor object.

    Tensor objects change identity between model instances, so PyTorch's
    default optimizer state_dict cannot be reloaded into a fresh model. Names
    are stable, and let phase 1's connector-only optimizer hand its state to
    phase 2's connector+LLM optimizer (LLM params simply start fresh).
    """
    name_by_id = {id(p): n for n, p in model.named_parameters()}
    groups = []
    for g in optimizer.param_groups:
        groups.append({
            "params": [name_by_id.get(id(p)) for p in g["params"]],
            "lr": g.get("lr"),
            "weight_decay": g.get("weight_decay"),
            "betas": g.get("betas"),
            "eps": g.get("eps"),
        })
    state = {}
    for p, s in optimizer.state.items():
        name = name_by_id.get(id(p))
        if name is None:
            continue
        entry = {}
        for k, v in s.items():
            entry[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v
        state[name] = entry
    return {"param_groups": groups, "state": state}


def load_optimizer_state_named(model, optimizer, saved: dict | None) -> int:
    """Restore matching parameters. Returns how many params were restored."""
    if not saved:
        return 0
    name_to_param = dict(model.named_parameters())
    id_to_name = {id(p): n for n, p in name_to_param.items()}
    restored = 0
    for name, s in saved.get("state", {}).items():
        p = name_to_param.get(name)
        if p is None:
            continue
        entry = {}
        for k, v in s.items():
            entry[k] = v.to(p.device).clone() if torch.is_tensor(v) else v
        optimizer.state[p] = entry
        restored += 1

    # Restore a group's LR only if its parameter names still match. Phase 2
    # adds an LLM group, so restoring by index would overwrite its fresh LR.
    for group, saved_group in zip(
        optimizer.param_groups, saved.get("param_groups", [])
    ):
        current_names = {
            id_to_name.get(id(p)) for p in group.get("params", [])
        }
        saved_names = set(saved_group.get("params") or [])
        if (
            current_names
            and current_names == saved_names
            and saved_group.get("lr") is not None
        ):
            group["lr"] = saved_group["lr"]
    return restored


def fast_forward_scheduler(scheduler, steps: int):
    """Advance the LR schedule without calling ``step()``.

    Calling ``step()`` before any ``optimizer.step()`` triggers a PyTorch
    warning; setting ``last_epoch`` and recomputing the lambdas is equivalent
    and silent.
    """
    steps = int(steps)
    if steps <= 0:
        return
    scheduler.last_epoch = steps
    try:
        lrs = scheduler.get_lr()
    except Exception:
        return
    scheduler._last_lr = lrs
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group["lr"] = lr


# ------------------------------------------------------------- checkpoints
def build_checkpoint(
    model,
    state: RunState,
    optimizer,
    scheduler,
    stream_stats: dict,
    llm_cfg: dict,
    vision_cfg: dict,
    connector_cfg: dict,
) -> dict:
    """One checkpoint: weights + optimizer + all counters + configs."""
    return {
        "format": "medicalvllm-v2",
        **state.as_dict(),
        "llm": model.llm.state_dict(),
        "connector": model.connector.state_dict(),
        "optimizer": optimizer_state_named(model, optimizer),
        "scheduler": scheduler.state_dict(),
        "vision": {
            "encoder": model.vision.encoder_name,
            "encoder_backend": model.vision.backend,
            "image_size": model.vision.image_size,
            "num_visual_tokens": model.num_visual_tokens,
        },
        "config": {
            "llm": llm_cfg,
            "vision": vision_cfg,
            "connector": connector_cfg,
        },
        "stream_stats": stream_stats,
    }


def latest_local_checkpoint(directory: str | Path, patterns: tuple[str, ...]):
    """Most recently modified file matching any pattern, or ``None``."""
    directory = Path(directory)
    files = []
    for pat in patterns:
        files.extend(f for f in directory.glob(pat) if f.is_file())
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


# ------------------------------------------------------------------ graphs
def save_plots(out_dir: str | Path):
    """Write loss and LR graphs from metrics.json (best effort)."""
    out_dir = Path(out_dir)
    path = out_dir / "metrics.json"
    if not path.is_file():
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = json.loads(path.read_text(encoding="utf-8"))
        plotted = [r for r in rows if "step" in r and "loss" in r]
        if plotted:
            xs = [r["step"] for r in plotted]
            ys = [r["loss"] for r in plotted]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(xs, ys, marker="o", markersize=2, linewidth=1)
            ax.set_xlabel("global step")
            ax.set_ylabel("cross-entropy loss")
            ax.set_title("Training loss")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "loss_vs_step.png", dpi=120)
            plt.close(fig)

        lr_rows = [
            r for r in rows
            if "step" in r
            and any(k in r for k in ("lr", "lr_connector", "lr_llm"))
        ]
        if lr_rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, label in (
                ("lr", "LR"),
                ("lr_connector", "LR connector"),
                ("lr_llm", "LR LLM"),
            ):
                vals = [(r["step"], r[key]) for r in lr_rows if key in r]
                if vals:
                    ax.plot([v[0] for v in vals], [v[1] for v in vals], label=label)
            ax.set_xlabel("global step")
            ax.set_ylabel("learning rate")
            ax.set_title("Learning rate schedule")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "lr_vs_step.png", dpi=120)
            plt.close(fig)
    except Exception as exc:
        logger.warning("could not write plots (%s): %s",
                       type(exc).__name__, exc)
