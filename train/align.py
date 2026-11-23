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
    build_progress_bar,
    build_checkpoint,
    build_lr_lambda,
    clip_grad_norm,
    fast_forward_scheduler,
    get_device,
    latest_local_checkpoint,
    select_devices,
    load_optimizer_state_named,
    load_yaml,
    log_sample_generation,
    move_batch,
    postfix,
    progress_update,
    resolve_path,
    restore_state,
    save_plots,
    set_seed,
    setup_logging,
    shutdown_streaming_http,
    unwrap_model,
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
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="max GPUs for DataParallel (default: config num_gpus=2)")
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

    setup_logging()   # minillm-style: log lines appear above the tqdm bar
    set_seed(args.seed)
    base = Path(args.config).resolve().parents[1]
    cfg = load_yaml(args.config)
    device, gpu_ids = select_devices(
        args.device,
        args.num_gpus if args.num_gpus is not None else int(cfg.get("num_gpus", 2)),
    )
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
    sample_every = int(cfg.get("sample_interval", 500))
    sample_stream = None
    sample_iter = None
    if sample_every > 0:
        sample_stream = CaptionStream(
            dataset=dcfg.get("dataset", "eltorio/ROCOv2-radiology"),
            budget=int(dcfg.get("max_text_tokens", 112)),
            image_size=int(vision_cfg.get("image_size", 224)),
            train=False, shuffle_buffer=0, seed=args.seed + 12345,
        )
        sample_iter = iter(sample_stream)

    # ---------------------------------------------------------- optimization
    batch_size = int(cfg.get("batch_size", 16))
    accum = int(cfg.get("grad_accum_steps", 2))
    max_steps = args.steps or cfg.get("steps")
    max_hours = args.hours or cfg.get("hours")
    # Bar mode is chosen from CLI flags ONLY. Config hours/steps are stop
    # limits; supplying no flag means "rows" mode, per the documented rule.
    bar_steps = args.steps if args.steps is not None else None
    bar_hours = args.hours if args.hours is not None else None
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
        # A plain minillm SFT checkpoint is a WEIGHTS-ONLY starting point: its
        # own `step` belongs to the old SFT run, so start our counters at 0.
        is_ours = saved.get("format") in {"medicalvllm-v1", "medicalvllm-v2"}
        if is_ours:
            restore_state(saved, state)
            n = load_optimizer_state_named(
                model, optimizer, saved.get("optimizer")
            )
            fast_forward_scheduler(scheduler, state.step)
            state.resumed_from = str(resume)
        else:
            n = 0
        if is_ours:
            logger.info(
                "resumed phase 1: step=%d epoch=%.3f text_tokens=%d images=%d "
                "(optimizer params restored: %d)",
                state.step, state.epoch, state.total_text_tokens,
                state.total_images_seen, n,
            )
        else:
            logger.info("starting phase 1 from SFT weights at global step 0")
    if state.examples_seen and epoch_size:
        state.epoch = state.examples_seen / epoch_size  # keep bar/state in sync

    # DataParallel uses every visible GPU; keep an unwrapped handle for
    # state_dict, sampling and config access (avoids "module." key prefixes).
    base_model = model
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        logger.info("using nn.DataParallel across %d GPUs: %s",
                    len(gpu_ids), gpu_ids)
    else:
        logger.info("using device %s", device)

    metrics = Metrics(out_dir)
    t0 = time.time()
    last_save = t0
    interval = float(cfg.get("checkpoint_interval_minutes", 60)) * 60
    # HF push cadence — same as local saves unless overridden in config.
    hf_interval = float(
        cfg.get("hf_upload_interval_minutes", interval / 60)
    ) * 60
    last_hf_upload = t0
    log_first = int(cfg.get("log_first_steps", 10))
    log_every = int(cfg.get("log_interval", 100))
    running, nloss, micro, halted = 0.0, 0, 0, False
    gn = 0.0

    def save(tag):
        state.elapsed_seconds = time.time() - t0
        ckpt = build_checkpoint(
            base_model, state, optimizer, scheduler,
            stream.stats, llm_cfg, vision_cfg, base_model.connector_cfg,
        )
        step_path = ckpt_dir / f"phase1_step{state.step:07d}.pt"
        atomic_torch_save(ckpt, step_path)
        atomic_torch_save(ckpt, ckpt_dir / "phase_1_latest.pt")
        save_plots(out_dir)
        metrics.log(event="checkpoint", step=state.step,
                    text_tokens=state.total_text_tokens,
                    images=state.total_images_seen,
                    elapsed_seconds=round(state.elapsed_seconds, 1))
        logger.info("saved phase-1 checkpoint at step %d", state.step)
        return step_path

    def upload_to_hf(step_path):
        if not use_hf:
            return
        save_plots(out_dir)   # regenerate graphs before uploading
        ok_ckpt = upload_checkpoint(hf_cfg, PHASE, step_path, state.step)
        ok_outs = upload_outs(hf_cfg, PHASE, out_dir)
        logger.info(
            "HF sync phase %d: checkpoint=%s graphs/metrics=%s at step %d",
            PHASE, "OK" if ok_ckpt else "FAILED",
            "OK" if ok_outs else "FAILED", state.step,
        )

    logger.info("Stage 1 start: step=%d steps=%s hours=%s batch=%s accum=%s",
                state.step, max_steps, max_hours, batch_size, accum)
    pbar = build_progress_bar(PHASE, bar_steps, bar_hours, state, epoch_size)
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
            if loss.dim() > 0:      # DataParallel gathers scalar losses into a vector
                loss = loss.mean()
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

                state.elapsed_seconds = time.time() - t0
                progress_update(
                    pbar,
                    n_rows=batch_size,
                    elapsed_seconds=state.elapsed_seconds,
                    examples_seen=state.examples_seen,
                )
                if pbar is not None:
                    pbar.set_postfix_str(postfix(state, avg, gn))
                if state.step <= log_first or state.step % log_every == 0:
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

                if sample_every and state.step % sample_every == 0:
                    log_sample_generation(
                        base_model, sample_iter, device, state.step, pbar,
                        out_dir=out_dir,
                        max_new_tokens=int(cfg.get("sample_max_new_tokens", 32)),
                        temperature=float(cfg.get("sample_temperature", 0.0)),
                        top_k=cfg.get("sample_top_k") or None,
                    )

            now = time.time()
            if now - last_save >= interval:
                step_path = save(f"step{state.step}")
                last_save = now
                if now - last_hf_upload >= hf_interval:
                    upload_to_hf(step_path)
                    last_hf_upload = now
    except KeyboardInterrupt:
        halted = True
    finally:
        if pbar is not None:
            pbar.close()
        # Release streaming HTTP connections before interpreter shutdown.
        stream.close()
        if sample_stream is not None:
            sample_stream.close()
        shutdown_streaming_http()
        final_path = save("final")
        metrics.log(event="end", step=state.step, halted=halted,
                    hours=round((time.time() - t0) / 3600, 3),
                    total_text_tokens=state.total_text_tokens,
                    total_images_seen=state.total_images_seen)
        if use_hf:
            upload_to_hf(final_path)
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
