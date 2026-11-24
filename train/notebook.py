"""Show training samples *inside* the notebook.

A trainer started with ``python -m train.vqa_sft`` is a plain process: it has
no Jupyter display hook, so ``IPython.display`` there does nothing. There are
two reliable ways to get the sample PNGs into the notebook, plus a best-effort
fallback for ``!python -m ...`` cells.

1. Run the trainer **in the kernel** (the cell stays busy while it trains)::

       from train.notebook import run_training
       run_training("vqa_sft", ["--qwen", "--sample", 10])

   Every sample is displayed while the cell executes, so it appears live.
   ``%run -m train.vqa_sft --qwen --sample 10`` works the same way, as long as
   the repo root is importable (``cd`` into it first).

2. Keep the trainer in a terminal and run a **blocking watcher** cell::

       from train.notebook import watch_samples
       watch_samples("/kaggle/temp/medicalvllm/outs/vqa")

   The cell polls ``samples/sample_step*.png`` and displays every new file.
   It must stay busy: Jupyter only renders ``display()`` calls made while a
   cell is executing.

3. ``!python -m train.vqa_sft ...`` in a cell: the trainer queues a display
   request to the notebook kernel (``display_image``). The kernel is busy
   running the training cell, so the request is processed when that cell ends
   — the samples show up together at the end, not live. Use option 1 or 2 for
   live updates.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _shell():
    """Return the running IPython shell, or None outside a kernel."""
    try:
        from IPython import get_ipython

        return get_ipython()
    except Exception:
        return None


def _in_notebook() -> bool:
    shell = _shell()
    return shell is not None and getattr(shell, "kernel", None) is not None


# ---------------------------------------------------------------------------
# 1. Run the trainer in the kernel
# ---------------------------------------------------------------------------

_STAGE_ALIASES = {
    "align": "align",
    "stage1": "align",
    "stage_1": "align",
    "phase1": "align",
    "vqa": "vqa_sft",
    "sft": "vqa_sft",
    "vqa_sft": "vqa_sft",
    "vqa-sft": "vqa_sft",
    "stage2": "vqa_sft",
    "stage_2": "vqa_sft",
    "phase2": "vqa_sft",
}


def run_training(stage: str = "vqa_sft", argv: list[str] | None = None):
    """Run ``train.<stage>`` in this kernel so samples render inline.

    ``stage`` is ``"align"`` or ``"vqa_sft"`` (aliases such as ``"stage2"`` are
    accepted). ``argv`` is the flag list you would pass on the command line,
    e.g. ``["--qwen", "--sample", 10]``. ``--detach`` is rejected: a detached
    process cannot display into the notebook.
    """
    import importlib

    key = str(stage).strip().lower().replace(" ", "_")
    module_name = _STAGE_ALIASES.get(key)
    if module_name is None:
        raise ValueError(
            f"unknown stage {stage!r}; use 'align' or 'vqa_sft'"
        )
    args = [str(a) for a in (argv or [])]
    if any(a == "--detach" for a in args):
        raise ValueError(
            "--detach cannot display in the notebook; run it in a terminal "
            "and use watch_samples() instead"
        )

    if not _in_notebook():
        print(
            "note: not running inside a Jupyter kernel; samples are written "
            "to disk but cannot be displayed here"
        )
    module = importlib.import_module(f"train.{module_name}")

    # The trainer installs its own SIGINT/SIGTERM handlers for a clean stop.
    # Restore IPython's handlers when it returns, so Ctrl-C still interrupts
    # cells normally afterwards.
    import signal

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
        except Exception:
            pass
    try:
        return module.main(args)
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 2. Watch a training run from a (blocking) notebook cell
# ---------------------------------------------------------------------------

def _step_of(path: Path) -> int:
    digits = "".join(c for c in path.stem if c.isdigit())
    return int(digits) if digits else 0


def _latest_sample(sample_dir: Path) -> Path | None:
    files = list(sample_dir.glob("sample_step*.png"))
    if not files:
        latest = sample_dir.parent / "sample_latest.png"
        return latest if latest.exists() else None
    return max(files, key=lambda p: (_step_of(p), p.stat().st_mtime))


def _display_sample(path: Path) -> None:
    from IPython.display import Image, Markdown, display

    display(Markdown(f"**sample @ step {_step_of(path):,}**"))
    display(Image(filename=str(path)))


def watch_samples(
    out_dir: str | Path,
    interval: float = 5.0,
    clear: bool = True,
    show_existing: bool = True,
    timeout: float | None = None,
):
    """Display new ``sample_step*.png`` files until interrupted.

    Run this in a notebook cell **while training runs in a terminal**. The cell
    blocks (that is what makes Jupyter render the images live); interrupt the
    cell to stop watching.

    Args:
        out_dir: the trainer's ``out_dir``, e.g.
            ``/kaggle/temp/medicalvllm/outs/vqa``.
        interval: seconds between directory polls.
        clear: replace the previous sample instead of appending (keeps the
            output cell small during a long run).
        show_existing: display the newest sample that already exists.
        timeout: stop after this many seconds (``None`` = until interrupted).
    """
    if not _in_notebook():
        print("watch_samples() only works inside a notebook kernel")
        return

    from IPython.display import Markdown, clear_output, display

    sample_dir = Path(out_dir) / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if not show_existing:
        seen = {p.name for p in sample_dir.glob("sample_step*.png")}

    deadline = time.time() + timeout if timeout else None
    waiting_shown = False
    try:
        while True:
            files = sorted(
                sample_dir.glob("sample_step*.png"),
                key=lambda p: (_step_of(p), p.stat().st_mtime),
            )
            new = [p for p in files if p.name not in seen]
            if new:
                for p in new:
                    seen.add(p.name)
                # With clear=True only the newest of a burst stays visible.
                for p in new if not clear else new[-1:]:
                    try:
                        if clear:
                            clear_output(wait=True)
                        _display_sample(p)
                    except Exception as exc:   # partial file while saving
                        logger.debug("sample display failed (%s): %s",
                                     type(exc).__name__, exc)
                waiting_shown = False
            elif not waiting_shown and not seen:
                if clear:
                    clear_output(wait=True)
                display(Markdown(
                    f"waiting for samples in `{sample_dir}` … "
                    f"(every {interval:g}s)"
                ))
                waiting_shown = True

            if deadline is not None and time.time() >= deadline:
                return
            time.sleep(max(0.2, float(interval)))
    except KeyboardInterrupt:
        print("stopped watching samples")
        return


def show_latest(out_dir: str | Path) -> None:
    """Display the newest saved sample once (after training, for example)."""
    if not _in_notebook():
        print("show_latest() only works inside a notebook kernel")
        return
    path = _latest_sample(Path(out_dir) / "samples")
    if path is None:
        print(f"no samples found under {Path(out_dir) / 'samples'}")
        return
    _display_sample(path)


# ---------------------------------------------------------------------------
# 3. Best-effort display from a subprocess (`!python -m ...`)
# ---------------------------------------------------------------------------

_pending: "queue.Queue[tuple[str, str]]" = queue.Queue(maxsize=1)
_worker_lock = threading.Lock()
_worker_started = False


def _open_kernel_client():
    """Connect to the newest local Jupyter kernel, or return None.

    Deliberately does **not** call ``wait_for_ready``: when the kernel is busy
    executing the cell that launched the trainer it never answers, and the old
    code treated that as "no notebook" and dropped the sample. Sending the
    request while busy is fine — the kernel queues it.
    """
    try:
        from jupyter_client import BlockingKernelClient
    except Exception:
        return None

    runtime = Path(
        os.environ.get("JUPYTER_RUNTIME_DIR")
        or (Path.home() / ".local" / "share" / "jupyter" / "runtime")
    )
    files = sorted(runtime.glob("kernel-*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    kc = BlockingKernelClient(connection_file=str(files[-1]))
    kc.load_connection_file()
    kc.start_channels()
    return kc


def _kernel_display_worker() -> None:
    while True:
        path, caption = _pending.get()
        kc = _open_kernel_client()
        if kc is None:
            continue
        try:
            code = (
                "from IPython.display import display, Image as _SampleImage\n"
                f"display(_SampleImage(filename={json.dumps(path)}))\n"
                f"print({json.dumps(caption)})\n"
            )
            kc.execute(code, store_history=False, allow_stdin=False,
                       stop_on_error=False)
            # Bounded wait so the request is flushed before the socket closes.
            # A busy kernel simply runs it when the current cell finishes.
            try:
                kc.get_shell_msg(timeout=5)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("notebook display request failed (%s): %s",
                         type(exc).__name__, exc)
        finally:
            try:
                kc.stop_channels()
            except Exception:
                pass


def _queue_kernel_display(path: str, caption: str) -> None:
    global _worker_started
    with _worker_lock:
        if not _worker_started:
            threading.Thread(
                target=_kernel_display_worker,
                name="nb-sample-display",
                daemon=True,
            ).start()
            _worker_started = True
    try:
        _pending.put_nowait((path, caption))
    except queue.Full:
        # Keep only the newest sample; the older one is already on disk.
        try:
            _pending.get_nowait()
        except queue.Empty:
            pass
        try:
            _pending.put_nowait((path, caption))
        except queue.Full:
            pass


def display_image(image_path: str | Path, caption: str | None = None) -> None:
    """Display a sample PNG in the notebook, from either process type."""
    path = str(image_path)
    if _in_notebook():
        from IPython.display import Image, display

        display(Image(filename=path))
        if caption:
            print(caption)
    else:
        _queue_kernel_display(path, caption or "")
