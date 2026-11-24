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
import math
import time
from pathlib import Path

import torch

from .common import (
    QWEN_MODEL,
    Metrics,
    RunState,
    StopSignal,
    assert_finite_parameters,
    atomic_torch_save,
    build_progress_bar,
    build_checkpoint,
    build_lr_lambda,
    checkpoint_llm_backend,
    clip_grad_norm,
    fast_forward_scheduler,
    finalize_llm_config,
    get_device,
    latest_local_checkpoint,
    make_qwen_llm_config,
    nonfinite_parameter_names,
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
from .hf_sync import download_latest, remote_phase_dir, upload_checkpoint, upload_outs
from data.captions import CaptionStream
from data.collate import collate_fn
from data.tokenizer import set_tokenizer_backend
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


def build_model(
    cfg, base, ckpt, device, load_weights=True,
    use_qwen=False, qwen_model=QWEN_MODEL,
):
    """Build the model. ``ckpt`` may be a plain SFT or a full v2 checkpoint.

    With ``use_qwen`` the GPT-2/minillm LLM is replaced by Qwen3-0.6B. Weights
    from a GPT-2 checkpoint are never loaded into it; a fresh pretrained Qwen
    decoder and a fresh connector are used instead.
    """
    state = torch.load(ckpt, map_location="cpu", weights_only=True) if ckpt else {}
    saved_llm_cfg = None
    if "model_config" in state:                 # plain minillm SFT
        saved_llm_cfg = state["model_config"]
    elif "config" in state:                     # our v2 checkpoint
        saved_llm_cfg = state["config"]["llm"]

    backend = "qwen" if use_qwen else "gpt2"
    saved_backend = checkpoint_llm_backend(ckpt) or "gpt2"
    can_load = load_weights and bool(state) and saved_backend == backend

    if backend == "qwen":
        qwen_cfg = dict(cfg.get("qwen") or {})
        if saved_backend == "qwen" and saved_llm_cfg:
            llm_cfg = dict(saved_llm_cfg)
            # The checkpoint records architecture + token ids, but the CURRENT
            # YAML/CLI decides precision and memory settings. An old fp32
            # checkpoint must not silently disable fp16 / checkpointing.
            for key in ("context_length", "dtype", "gradient_checkpointing"):
                if key in qwen_cfg:
                    llm_cfg[key] = qwen_cfg[key]
            llm_cfg["model_name"] = qwen_model
        else:
            llm_cfg = make_qwen_llm_config(
                model_name=qwen_model,
                context_length=int(qwen_cfg.get("context_length", 1024)),
                dtype=str(qwen_cfg.get("dtype", "float32")),
                gradient_checkpointing=bool(
                    qwen_cfg.get("gradient_checkpointing", False)
                ),
            )
        llm_cfg = finalize_llm_config(llm_cfg)
        # The connector is cast to the LLM dtype and is the only trainable
        # module here. fp16 AdamW moments underflow (eps=1e-8 -> 0), so one
        # step turns the connector into NaN. Train in fp32.
        requested = str(llm_cfg.get("dtype", "float32")).lower()
        if requested not in {"float32", "fp32"}:
            logger.warning(
                "Qwen alignment in %s would poison the connector with NaN "
                "(fp16/bf16 AdamW moments underflow); using float32 instead",
                requested,
            )
            llm_cfg = dict(llm_cfg)
            llm_cfg["dtype"] = "float32"
    else:
        llm_cfg = dict(saved_llm_cfg) if saved_llm_cfg else load_yaml(
            base / "configs" / "model.yaml"
        )

    vision_cfg = load_yaml(base / "configs" / "vision.yaml")

    model = MedicalVLLM(llm_cfg, vision_cfg).to(device)
    if can_load:
        # Plain SFT has no "connector" key; MedicalVLLM.load skips it safely.
        model.load(ckpt, load_connector=True, strict_llm=True)
    elif load_weights and state and saved_backend != backend:
        logger.warning(
            "ignoring %s LLM weights in %s (backend mismatch: %s != %s)",
            saved_backend, ckpt, saved_backend, backend,
        )
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
    ap.add_argument("--qwen", action="store_true",
                    help=f"use Qwen3 instead of the GPT-2/minillm LLM (default {QWEN_MODEL})")
    ap.add_argument("--qwen-model", default=None,
                    help=f"HuggingFace Qwen model id (default {QWEN_MODEL})")
    ap.add_argument("--qween", dest="qwen", action="store_true",
                    help=argparse.SUPPRESS)  # common misspelling
    ap.add_argument("--save", dest="save_minutes", type=float, default=None,
                    help="save a checkpoint every N minutes (always local; also uploaded when --hf is given)")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                    help="micro-batch size (overrides config; effective batch = batch-size × grad_accum_steps)")
    ap.add_argument("--batch", dest="batch_size", type=int,
                    default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16", "auto"],
                    default=None, help="LLM dtype for --qwen (overrides config qwen.dtype; training is auto-promoted to float32)")
    ap.add_argument("--sample", type=int, default=None,
                    help="show one generation every N optimizer steps (0 disables; overrides config)")
    ap.add_argument("--smaple", dest="sample", type=int,
                    default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.detach:
        _detach()
        return

    setup_logging()   # minillm-style: log lines appear above the tqdm bar
    set_seed(args.seed)
    base = Path(args.config).resolve().parents[1]
    cfg = load_yaml(args.config)

    # CLI overrides for cadence (config values are the fallbacks).
    if args.save_minutes is not None:
        cfg["checkpoint_interval_minutes"] = float(args.save_minutes)
        logger.info(
            "checkpoint interval: %g minute(s) (local save; uploaded too with --hf)",
            args.save_minutes,
        )
    if args.sample is not None:
        cfg["sample_interval"] = int(args.sample)
        logger.info("sample interval: %d optimizer step(s)", args.sample)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
        logger.info("batch size: %d", args.batch_size)
    if args.dtype is not None:
        cfg.setdefault("qwen", {})["dtype"] = args.dtype
        logger.info("Qwen dtype: %s", args.dtype)

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

    # Qwen checkpoints use a separate remote layout:
    # <repo>/medical_vllm/qwen/align and .../sft. Apply before the resume download below.
    if args.qwen and cfg.get("hf_qwen"):
        hf_cfg.update(dict(cfg["hf_qwen"]))
        logger.info(
            "HF Qwen layout: repo=%s dirs=%s / %s",
            hf_cfg.get("repo"),
            remote_phase_dir(hf_cfg, 1), remote_phase_dir(hf_cfg, 2),
        )

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
        if resume is None and not args.qwen:
            resume = resolve_llm_checkpoint(cfg, base)  # fresh from SFT
            logger.info("no previous phase-1 checkpoint; starting from SFT")
        if resume is None and args.qwen:
            logger.info("no previous phase-1 checkpoint; starting from pretrained Qwen")
    if args.new:
        logger.info("--new: random LLM weights, starting at step 0")

    # --------------------------------------------------------- LLM backend
    qwen_cfg = dict(cfg.get("qwen") or {})
    qwen_model = args.qwen_model or str(qwen_cfg.get("model", QWEN_MODEL))
    use_qwen = bool(args.qwen)
    if resume:
        resume_backend = checkpoint_llm_backend(resume)
        if resume_backend == "qwen":
            use_qwen = True
            logger.info("checkpoint %s uses Qwen; switching tokenizer/LLM backend",
                        resume)
        elif use_qwen:
            logger.warning(
                "ignoring %s checkpoint %s: --qwen cannot reuse GPT-2 weights",
                resume_backend, resume,
            )
            resume = None
    set_tokenizer_backend("qwen" if use_qwen else "gpt2", model_name=qwen_model)
    if use_qwen:
        logger.info("LLM backend: Qwen3 (%s)", qwen_model)

    model, llm_cfg, vision_cfg = build_model(
        cfg, base, resume, device, load_weights=not args.new,
        use_qwen=use_qwen, qwen_model=qwen_model,
    )
    counts = model.param_counts()
    logger.info("params (M): %s", {k: round(v / 1e6, 2) for k, v in counts.items()})
    logger.info(
        "LLM dtype=%s grad_checkpointing=%s",
        model.llm.tok_emb.weight.dtype,
        getattr(getattr(model.llm, "hf", None), "is_gradient_checkpointing", False),
    )
    assert_finite_parameters(model, source=str(resume or "fresh Qwen model"))

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
        # Gather outputs on the LAST GPU: with DataParallel the optimizer and
        # the gathered logits otherwise all sit on GPU 0, which OOMs first.
        model = torch.nn.DataParallel(
            model, device_ids=gpu_ids, output_device=gpu_ids[-1]
        )
        logger.info("using nn.DataParallel across %d GPUs: %s",
                    len(gpu_ids), gpu_ids)
    else:
        logger.info("using device %s", device)

    metrics = Metrics(out_dir)
    t0 = time.time()
    last_save = t0
    interval = float(cfg.get("checkpoint_interval_minutes", 60)) * 60
    log_first = int(cfg.get("log_first_steps", 10))
    log_every = int(cfg.get("log_interval", 100))
    running, nloss, micro, halted = 0.0, 0, 0, False
    gn = 0.0
    nan_skips = 0
    fatal = False

    def save(tag):
        bad = nonfinite_parameter_names(base_model)
        if bad:
            logger.error(
                "refusing to save %s checkpoint: %d non-finite parameter "
                "tensor(s), e.g. %s", tag, len(bad), ", ".join(bad[:3]),
            )
            return None
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
        if not use_hf or step_path is None:
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
    if pbar is not None:
        # No optimizer step measured yet: show nan, not a fake 0.000 loss.
        pbar.set_postfix_str(postfix(state, float("nan"), 0.0))
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
            if not bool(loss.isfinite()):
                # Never backprop a NaN/Inf: it would poison every parameter
                # and every AdamW moment. Skip the batch instead.
                nan_skips += 1
                logger.warning(
                    "non-finite loss at step %d (micro %d), skipping batch "
                    "(consecutive skips: %d)", state.step, micro, nan_skips,
                )
                if nan_skips >= 20:
                    fatal = True
                    raise RuntimeError(
                        "20 consecutive non-finite losses; the weights are "
                        "probably already NaN. Restore a clean checkpoint or "
                        "rerun with --new."
                    )
                optimizer.zero_grad(set_to_none=True)
                continue
            nan_skips = 0
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
                avg = running / nloss if nloss else float("nan")
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
                metrics.log(step=state.step,
                            loss=round(avg, 5) if math.isfinite(avg) else None,
                            lr=scheduler.get_last_lr()[0],
                            grad_norm=round(gn, 4),
                            total_text_tokens=state.total_text_tokens,
                            total_images_seen=state.total_images_seen,
                            epoch=round(state.epoch, 4))

                if sample_every and state.step % sample_every == 0:
                    log_sample_generation(
                        base_model, sample_iter, device, state.step, pbar,
                        out_dir=out_dir,
                        max_new_tokens=int(cfg.get("sample_max_new_tokens", 200)),
                        max_text_tokens=int(cfg.get("sample_max_tokens", 200)),
                        temperature=float(cfg.get("sample_temperature", 0.0)),
                        top_k=cfg.get("sample_top_k") or None,
                    )

            now = time.time()
            if now - last_save >= interval:
                step_path = save(f"step{state.step}")
                last_save = now
                upload_to_hf(step_path)   # no-op unless --hf was given
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
        halted = halted or fatal
        if fatal:
            logger.error(
                "aborted on non-finite loss; NOT saving a final checkpoint "
                "(the last good checkpoint is untouched)"
            )
            final_path = None
        else:
            final_path = save("final")
        metrics.log(event="end", step=state.step, halted=halted,
                    hours=round((time.time() - t0) / 3600, 3),
                    total_text_tokens=state.total_text_tokens,
                    total_images_seen=state.total_images_seen)
        if use_hf and final_path is not None:
            upload_to_hf(final_path)
        for i in gpu_ids:
            try:
                logger.info(
                    "GPU %d peak allocated %.2f GB", i,
                    torch.cuda.max_memory_allocated(i) / 1e9,
                )
            except Exception:
                pass
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
