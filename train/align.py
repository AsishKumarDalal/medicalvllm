"""Stage 1: connector-only alignment on ROCOv2 radiology captions.

Flags::

    --hf      resume from / upload to HuggingFace (medical_vllm/phase1/)
    --no-hf   never touch HuggingFace (local files only)
    --new     random LLM weights, ignore every checkpoint, start at step 0

Run::

    python -m train.align --steps 200
    python -m train.align --hours 4 --hf --detach
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from .common import (
    Metrics,
    RunState,
    StopSignal,
    atomic_torch_save,
    build_checkpoint,
    build_lr_lambda,
    clip_grad_norm,
    fast_forward_scheduler,
    get_device,
    latest_local_checkpoint,
    load_optimizer_state_named,
    load_yaml,
    move_batch,
    resolve_path,
    restore_state,
    save_plots,
    set_seed,
)
from .hf_sync import download_latest, upload_checkpoint, upload_outs
from data.captions import CaptionStream
from data.collate import collate_fn
from model.medical_vllm import MedicalVLLM

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
PHASE = 1


def resolve_llm_checkpoint(cfg: dict, base: Path) -> Path:
    """Prefer a local SFT checkpoint; fall back to downloading from HF."""
    local = resolve_path(str(cfg.get("llm_checkpoint", "")), base)
    if local.is_file():
        logger.info("using local SFT checkpoint %s", local)
        return local
    fb = cfg.get("hf_fallback") or {}
    if fb.get("repo"):
        from huggingface_hub import hf_hub_download

        cached = Path(hf_hub_download(
            repo_id=fb["repo"],
            filename=fb.get("filename", "sft_model.pt"),
        ))
        logger.info("using HF SFT checkpoint %s", cached)
        return cached
    raise FileNotFoundError(f"no SFT checkpoint at {local} and no hf_fallback")


def build_model(cfg, base, ckpt, device, load_weights=True):
    """Build the model. ``ckpt`` may be a plain SFT or a full v2 checkpoint."""
    state = torch.load(ckpt, map_location="cpu", weights_only=True) if ckpt else {}
    if "model_config" in state:                 # plain minillm SFT
        llm_cfg = state["model_config"]
    elif "config" in state:                     # our v2 checkpoint
        llm_cfg = state["config"]["llm"]
    else:                                        # --new
        llm_cfg = load_yaml(base / "configs" / "model.yaml")
    vision_cfg = load_yaml(base / "configs" / "vision.yaml")

    model = MedicalVLLM(llm_cfg, vision_cfg).to(device)
    if load_weights and state:
        # Plain SFT has no "connector" key; MedicalVLLM.load skips it safely.
        model.load(ckpt, load_connector=True, strict_llm=True)
    model.freeze_llm()            # Stage 1: target language stays fixed
    model.vision.set_frozen(True)
    model.connector.train()
    return model, llm_cfg, vision_cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "align.yaml"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop", default=None)
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--hf", action="store_true",
                    help="resume from and upload to HuggingFace")
    ap.add_argument("--no-hf", action="store_true",
                    help="never read or write HuggingFace")
    ap.add_argument("--new", action="store_true",
                    help="random LLM weights; ignore all checkpoints; step 0")
    args = ap.parse_args(argv)

    if args.detach:
        _detach()
        return

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    base = Path(args.config).resolve().parents[1]
    cfg = load_yaml(args.config)
    device = get_device(args.device)
    stop = StopSignal(args.stop or (base / ".stop_align"))

    # -- CLI flags override config. No upload unless --hf is given.
    use_hf = bool(args.hf) and not args.no_hf
    hf_cfg = dict(cfg.get("hf") or {})
    hf_cfg["enabled"] = use_hf
    if use_hf:
        logger.info("HF sync ENABLED (--hf)")
    else:
        logger.info("HF sync DISABLED (no --hf / --no-hf)")

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "checkpoints/align"), base)
    out_dir = resolve_path(cfg.get("out_dir", "outs/align"), base)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- resume source
    resume = None
    if not args.new:
        if use_hf:
            resume = download_latest(hf_cfg, PHASE, base / ".hf_cache")
        if resume is None:
            resume = latest_local_checkpoint(
                ckpt_dir,
                ("phase_1_latest.pt", "medicalvllm_align_*.pt"),
            )
        if resume is None:
            resume = resolve_llm_checkpoint(cfg, base)  # fresh from SFT
            logger.info("no previous phase-1 checkpoint; starting from SFT")
    if args.new:
        logger.info("--new: random LLM weights, starting at step 0")

    model, llm_cfg, vision_cfg = build_model(
        cfg, base, resume, device, load_weights=not args.new
    )
    counts = model.param_counts()
    logger.info("params (M): %s", {k: round(v / 1e6, 2) for k, v in counts.items()})

    # ---------------------------------------------------------------- data
    dcfg = cfg["data"]
    stream = CaptionStream(
        dataset=dcfg.get("dataset", "eltorio/ROCOv2-radiology"),
        budget=int(dcfg.get("max_text_tokens", 112)),
        image_size=int(vision_cfg.get("image_size", 224)),
        train=True,
        shuffle_buffer=int(dcfg.get("shuffle_buffer", 500)),
        seed=args.seed,
    )
    data_iter = iter(stream)
    epoch_size = int(dcfg.get("epoch_size", 130_000))

    # ---------------------------------------------------------- optimization
    batch_size = int(cfg.get("batch_size", 16))
    accum = int(cfg.get("grad_accum_steps", 2))
    max_steps = args.steps or cfg.get("steps")
    max_hours = args.hours or cfg.get("hours")
    optimizer = torch.optim.AdamW(
        model.connector.parameters(),
        lr=float(cfg.get("learning_rate", 1e-4)),
        weight_decay=0.01,
    )
    total = max_steps or 10_000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(int(cfg.get("warmup_steps", 200)), total)
    )

    state = RunState(phase=PHASE)
    if resume and not args.new:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        restore_state(saved, state)
        n = load_optimizer_state_named(model, optimizer, saved.get("optimizer"))
        fast_forward_scheduler(scheduler, state.step)
        state.resumed_from = str(resume)
        logger.info(
            "resumed phase 1: step=%d epoch=%.3f text_tokens=%d images=%d "
            "(optimizer params restored: %d)",
            state.step, state.epoch, state.total_text_tokens,
            state.total_images_seen, n,
        )

    metrics = Metrics(out_dir)
    t0 = time.time()
    last_save = t0
    interval = float(cfg.get("checkpoint_interval_minutes", 60)) * 60
    running, nloss, micro, halted = 0.0, 0, 0, False

    def save(tag):
        ckpt = build_checkpoint(
            model, state, optimizer, scheduler,
            stream.stats, llm_cfg, vision_cfg, model.connector_cfg,
        )
        step_path = ckpt_dir / f"phase1_step{state.step:07d}.pt"
        atomic_torch_save(ckpt, step_path)
        atomic_torch_save(ckpt, ckpt_dir / "phase_1_latest.pt")
        save_plots(out_dir)
        metrics.log(event="checkpoint", step=state.step,
                    text_tokens=state.total_text_tokens,
                    images=state.total_images_seen)
        logger.info("saved phase-1 checkpoint at step %d", state.step)
        if use_hf:
            upload_checkpoint(hf_cfg, PHASE, step_path, state.step)
            upload_outs(hf_cfg, PHASE, out_dir)

    logger.info("Stage 1 start: step=%d steps=%s hours=%s batch=%s accum=%s",
                state.step, max_steps, max_hours, batch_size, accum)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while True:
            if stop.check():
                halted = True
                break
            if max_steps and state.step >= max_steps:
                break
            if max_hours and (time.time() - t0) / 3600 >= max_hours:
                halted = True
                break

            batch = _next_batch(data_iter, batch_size, collate_fn)
            batch = move_batch(batch, device)
            out = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = out[1] if isinstance(out, tuple) else out
            (loss / accum).backward()
            running += loss.item()
            nloss += 1
            micro += 1

            n_tokens = int((batch["labels"] != -100).sum())
            n_imgs = (
                0 if batch["pixel_values"] is None
                else batch["pixel_values"].size(0)
            )
            state.update(batch_size, n_tokens, n_imgs, epoch_size)

            if micro % accum == 0:
                gn = clip_grad_norm(model, float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.step += 1
                avg = running / max(1, nloss)
                running, nloss = 0.0, 0
                if state.step % 10 == 0:
                    logger.info(
                        "step %d loss %.4f lr %.2e gn %.3f | tok %d img %d",
                        state.step, avg, scheduler.get_last_lr()[0], gn,
                        state.total_text_tokens, state.total_images_seen,
                    )
                    metrics.log(step=state.step, loss=round(avg, 5),
                                lr=scheduler.get_last_lr()[0],
                                grad_norm=round(gn, 4),
                                total_text_tokens=state.total_text_tokens,
                                total_images_seen=state.total_images_seen,
                                epoch=round(state.epoch, 4))

            now = time.time()
            if now - last_save >= interval:
                save(f"step{state.step}")
                last_save = now
    except KeyboardInterrupt:
        halted = True
    finally:
        save("final")
        metrics.log(event="end", step=state.step, halted=halted,
                    hours=round((time.time() - t0) / 3600, 3),
                    total_text_tokens=state.total_text_tokens,
                    total_images_seen=state.total_images_seen)
        save_plots(out_dir)
        if use_hf:
            upload_outs(hf_cfg, PHASE, out_dir)
        logger.info("Stage 1 done at step %d; stats=%s", state.step, stream.stats)
    return state.step


def _next_batch(data_iter, batch_size, collate):
    items = []
    while len(items) < batch_size:
        try:
            items.append(next(data_iter))
        except StopIteration:
            raise RuntimeError("data stream exhausted unexpectedly")
    return collate(items)


def _detach():
    import subprocess
    import sys

    log = ROOT / "outs" / "align_detached.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "train.align"] + [
        a for a in sys.argv[1:] if a != "--detach"
    ]
    with log.open("w") as f:
        subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"detached; log: {log}")


if __name__ == "__main__":
    main()
