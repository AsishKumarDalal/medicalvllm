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

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

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


def select_devices(name: str = "auto", num_gpus: int = 2):
    """Return ``(device, gpu_ids)`` — like minillm's ``select_device``.

    ``gpu_ids`` has more than one entry only when the caller wants multiple
    GPUs and CUDA actually exposes them; trainers then wrap with
    ``nn.DataParallel``.
    """
    requested = str(name).lower().strip()
    if requested == "cpu":
        return torch.device("cpu"), []
    if not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU")
        return torch.device("cpu"), []
    if requested.startswith("cuda:") and ":" in requested:
        try:
            idx = int(requested.split(":", 1)[1])
        except ValueError:
            idx = 0
        if idx >= torch.cuda.device_count():
            logger.warning("cuda:%d unavailable; using cuda:0", idx)
            idx = 0
        return torch.device(f"cuda:{idx}"), [idx]
    n = max(1, min(int(num_gpus), torch.cuda.device_count()))
    return torch.device("cuda:0"), list(range(n))


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


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
    elapsed_seconds: float = 0.0
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
    state.elapsed_seconds = float(
        ckpt.get("elapsed_seconds", state.elapsed_seconds)
    )
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


def shutdown_streaming_http():
    """Gracefully stop fsspec's background IO loop.

    ``datasets`` streaming downloads parquet shards through an fsspec
    aiohttp session that lives on a *daemon* thread (``fsspecIO``). If Python
    starts shutting down while a range request is in flight, the socket is
    closed under it and huggingface_hub's retry loop spams "Bad file
    descriptor" for ~30 s. Cancelling pending tasks and stopping the loop
    first removes the race entirely.
    """
    try:
        import asyncio
        import fsspec.asyn as asyn

        loop = asyn.loop[0]
        thread = asyn.iothread[0]
        if loop is None:
            return

        async def _cancel_pending():
            tasks = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        fut = asyncio.run_coroutine_threadsafe(_cancel_pending(), loop)
        fut.result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        asyn.loop[0] = None
        asyn.iothread[0] = None
    except Exception as exc:
        logger.debug("streaming HTTP shutdown: %s", exc)

    # Even after the loop stops, huggingface_hub may still have a range
    # request in flight on a httpx connection. If interpreter teardown closes
    # that socket first, its retry loop fires "Bad file descriptor" and hangs
    # exit for ~30 s. A short drain lets those requests finish. Verified with
    # 5/5 clean exits on a minimal datasets-only reproducer.
    time.sleep(2.0)


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


# ---------------------------------------------------------- tqdm bars (minillm)
class LibraryNoiseFilter(logging.Filter):
    """Drop per-request library chatter that survives logger level changes.

    Some bundled libraries attach their own handler and reset their logger
    level on import, so setting WARNING alone is not enough. This filter is
    installed on the root handler AND on the noisy loggers' own handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (
            msg.startswith("HTTP Request:")
            or msg.startswith("Resolving data files:")
            or msg.startswith("Warning: You are sending unauthenticated")
        )


class TqdmLoggingHandler(logging.StreamHandler):
    """Print log lines ABOVE an active tqdm bar instead of through it."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if tqdm is not None:
                tqdm.write(message, file=self.stream)
            else:
                self.stream.write(message + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> None:
    """Same logging style as minillm: timestamped lines above the tqdm bar.

    Also silences the libraries that log every HTTP range request / backend
    probe at INFO, otherwise the training log is drowned in "HTTP Request:
    GET ..." lines and "Resolving data files" bars.
    """
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    noise_filter = LibraryNoiseFilter()
    handler.addFilter(noise_filter)
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for name in (
        "httpx", "httpx2", "httpcore", "httpcore2",
        "urllib3", "requests", "datasets", "huggingface_hub",
        "huggingface_hub.utils._http", "fsspec", "numexpr",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        lg.addFilter(noise_filter)
        for h in lg.handlers:
            h.addFilter(noise_filter)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Xet opens background Rust threads for parquet range requests; at process
    # exit those threads hit closed file descriptors and retry for ~30 s.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from datasets import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


def format_compact(value: float) -> str:
    """Format a large number as 1.2k / 3.4M / 5.6B for a narrow tqdm bar."""
    value = float(value)
    if not math.isfinite(value):
        return "nan"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}k"
    return f"{value:.1f}"


def style_progress_bar(pbar) -> None:
    """Make the notebook bar green and keep all stats visible beside it."""
    try:
        desc_widget, bar_widget, stats_widget = pbar.container.children
        pbar.container.layout.width = "100%"
        desc_widget.layout.flex = "0 0 auto"
        desc_widget.layout.width = "auto"
        desc_widget.layout.overflow = "visible"
        bar_widget.layout.flex = "0 0 12%"
        bar_widget.layout.width = "12%"
        bar_widget.layout.min_width = "60px"
        stats_widget.layout.flex = "1 1 auto"
        stats_widget.layout.width = "auto"
        stats_widget.layout.min_width = "max-content"
        stats_widget.layout.overflow = "visible"
        bar_widget.bar_style = "success"
    except Exception:
        # Terminal tqdm has no widget container; colour="green" handles it.
        pass


def build_progress_bar(
    phase: int,
    max_steps,
    max_hours,
    state: RunState,
    epoch_size: int = 0,
):
    """minillm-style tqdm bar with three mutually exclusive progress modes.

    Priority (first match wins):

    1. ``--steps`` given   → percent of the requested optimizer steps
    2. ``--hours`` given   → percent of the wall-clock budget
    3. neither given       → percent of dataset rows consumed
                             (``examples_seen / data.epoch_size``)

    ``initial`` and row progress come from the restored ``state`` so a restart
    never resets the bar to zero.
    """
    if tqdm is None:
        return None

    if max_steps:
        mode, total, initial = "steps", float(max_steps), float(state.step)
        unit, desc = "step", f"phase{phase}"
    elif max_hours:
        mode = "hours"
        total = float(max_hours) * 3600.0
        initial = min(total, max(0.0, state.elapsed_seconds))
        unit, desc = "s", f"phase{phase} (time budget)"
    else:
        mode = "rows"
        total = float(epoch_size) if epoch_size and epoch_size > 0 else None
        # Wrap at one dataset pass so the bar fills once per epoch.
        initial = float(state.examples_seen % epoch_size) if total else 0.0
        unit, desc = "row", f"phase{phase} (rows)"

    pbar = tqdm(
        total=total,
        initial=initial,
        desc=desc,
        unit=unit,
        # Fixed width: dynamic_ncols in a Kaggle output cell is detected as a
        # tiny terminal, which clips the right-hand stats.
        ncols=150,
        postfix="",
        colour="green",
        bar_format="{desc}: {percentage:3.0f}%|{bar}|{postfix}",
    )
    pbar._mv_mode = mode
    pbar._mv_epoch_size = int(epoch_size)
    style_progress_bar(pbar)
    return pbar


def progress_update(
    pbar,
    n_rows: int = 0,
    elapsed_seconds: float = 0.0,
    examples_seen: int = 0,
):
    """Advance the bar according to its active mode."""
    if pbar is None:
        return
    mode = getattr(pbar, "_mv_mode", "steps")
    if mode == "hours":
        pbar.update(max(0.0, elapsed_seconds))
    elif mode == "rows":
        epoch_size = getattr(pbar, "_mv_epoch_size", 0)
        target = int(examples_seen % epoch_size) if epoch_size else int(n_rows)
        delta = target - int(pbar.n)
        if delta > 0:
            pbar.update(delta)
        elif delta < 0:            # new epoch started
            pbar.n = 0
            pbar.update(target)
    else:
        pbar.update(1)


def show_sample_image(pixel, out_dir, step: int):
    """Un-normalize a sample tensor, save it and display it in a notebook."""
    from PIL import Image as PILImage

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    x = pixel.detach().cpu().float()
    while x.dim() > 3:
        x = x[0]
    undone = torch.zeros_like(x)
    for c in range(3):
        undone[c] = x[c] * std[c] + mean[c]
    arr = (undone.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")

    sample_dir = Path(out_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    path = sample_dir / f"sample_step{step:07d}.png"
    PILImage.fromarray(arr).save(path)

    try:  # Kaggle / Jupyter: actually show the image
        from IPython.display import display
        display(PILImage.open(path))
    except Exception:
        pass
    return path


@torch.no_grad()
def log_sample_generation(
    model,
    sample_iter,
    device,
    step: int,
    pbar=None,
    out_dir: str | Path | None = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_k: int | None = None,
    eos_id: int = 50256,
):
    """Generate from ONE held-out dataset image and print the result.

    Used every ``sample_interval`` steps so you can *see* what the model
    learned, not just watch the loss. Mirrors minillm's ``_log_sample``.
    """
    from data.tokenizer import EOT_TOKEN_ID, get_encoder

    enc = get_encoder()
    try:
        ex = next(sample_iter)
    except StopIteration:
        return None

    labels = ex["labels"]
    n_prompt = sum(1 for t in labels if t == -100)
    if n_prompt == 0 or n_prompt >= len(ex["input_ids"]):
        return None
    prompt_ids = ex["input_ids"][:n_prompt]
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pixel = ex["pixel_values"]
    pixel = pixel.unsqueeze(0).to(device) if pixel is not None else None

    was_training = model.training
    model.eval()
    try:
        out = model.generate(
            ids,
            pixel_values=pixel,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )
        prompt = enc.decode(prompt_ids)
        generated = enc.decode(out[0, ids.size(1):].tolist())
        ref_ids = [
            t for t in ex["labels"] if t != -100 and t != EOT_TOKEN_ID
        ]
        ref = enc.decode(ref_ids)
    except Exception as exc:
        logger.warning("sample generation failed at step %d (%s): %s",
                       step, type(exc).__name__, exc)
        return None
    finally:
        model.train(was_training)

    image_path = None
    if out_dir is not None and pixel is not None:
        try:
            image_path = show_sample_image(pixel, out_dir, step)
        except Exception as exc:
            logger.warning("could not show sample image (%s): %s",
                           type(exc).__name__, exc)

    msg = (
        f"sample @ step {step:,} | "
        f"IMAGE: {image_path} | "
        f"PROMPT: {' '.join(prompt.split())[:220]} | "
        f"GENERATED: {' '.join(generated.split())[:220]} | "
        f"REFERENCE: {' '.join(ref.split())[:120]}"
    )
    if pbar is not None:
        pbar.write(msg)
    else:
        logger.info("%s", msg)
    return msg


def postfix(
    state: RunState,
    loss: float,
    grad_norm: float = 0.0,
    extra: str = "",
) -> str:
    """Full minillm-style live status string (all stats, no truncation)."""
    ppl = math.exp(min(loss, 50.0)) if math.isfinite(loss) else float("nan")
    return (
        f"#{state.step:,} | loss {loss:.3f} | ppl {ppl:,.1f} | "
        f"{format_compact(state.total_text_tokens)} tok | "
        f"{format_compact(state.total_images_seen)} img | "
        f"ep {state.epoch:.2f} | gn {grad_norm:.2f}{extra}"
    )
