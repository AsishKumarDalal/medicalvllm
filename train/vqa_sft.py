"""Stage 2: medical VQA SFT (connector + LLM train, vision frozen).

Flags::

    --hf      resume from / upload to HuggingFace (medical_vllm/phase2/)
    --no-hf   never touch HuggingFace (local files only)
    --new     random LLM weights; ignore all checkpoints; start at step 0

When phase 2 starts from a phase-1 checkpoint it silently inherits the GLOBAL
step, total text tokens, images seen and optimizer state; this hand-off is
logged once at INFO level. Run::

    python -m train.vqa_sft --steps 100
    python -m train.vqa_sft --hours 8 --hf --detach
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
    build_optimizer,
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
from data.collate import collate_fn
from data.medical_vqa import BatchMixSampler, SOURCES, VQAStream, build_vqa_streams
from data.tokenizer import set_tokenizer_backend
from model.medical_vllm import MedicalVLLM

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
PHASE = 2


def build_model(
    cfg, base, device, llm_cfg=None,
    use_qwen=False, qwen_model=QWEN_MODEL,
):
    """Build phase 2 from the checkpoint given by ``init_from`` (or --new)."""
    state = torch.load(cfg["_init_from"], map_location="cpu", weights_only=True) \
        if cfg.get("_init_from") else {}
    saved_llm_cfg = None
    if "config" in state:
        saved_llm_cfg = state["config"]["llm"]
    elif "model_config" in state:
        saved_llm_cfg = state["model_config"]

    backend = "qwen" if use_qwen else "gpt2"
    saved_backend = checkpoint_llm_backend(cfg.get("_init_from")) or "gpt2"
    can_load = bool(cfg.get("_init_from")) and saved_backend == backend

    if llm_cfg is None:
        if backend == "qwen":
            qwen_cfg = dict(cfg.get("qwen") or {})
            if saved_backend == "qwen" and saved_llm_cfg:
                llm_cfg = dict(saved_llm_cfg)
                # Architecture/token ids come from the checkpoint, but precision
                # and VRAM settings come from the CURRENT YAML/CLI (an old fp32
                # checkpoint must not silently disable fp16 / checkpointing).
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
        elif saved_llm_cfg:
            llm_cfg = dict(saved_llm_cfg)
        else:
            llm_cfg = load_yaml(base / "configs" / "model.yaml")

    if backend == "qwen":
        # AdamW stores moments in the parameter dtype. In fp16 the default
        # eps=1e-8 underflows to 0, tiny squared grads flush to 0, and AdamW
        # divides 0/0: every weight becomes NaN in one step. Full Qwen SFT
        # therefore always trains in fp32; gradient checkpointing keeps it
        # inside a 16 GB T4. (Mixed precision saves nothing here because a
        # correct fp32 master + fp32 moments costs the same as fp32.)
        requested = str(llm_cfg.get("dtype", "float32")).lower()
        if requested not in {"float32", "fp32"}:
            logger.warning(
                "Qwen training in %s would poison the weights with NaN "
                "(fp16/bf16 AdamW moments underflow); using float32 instead",
                requested,
            )
            llm_cfg = dict(llm_cfg)
            llm_cfg["dtype"] = "float32"

    if "config" in state:
        vision_cfg = state["config"]["vision"]
        connector_cfg = state["config"].get("connector") or vision_cfg
    else:
        vision_cfg = load_yaml(base / "configs" / "vision.yaml")
        connector_cfg = vision_cfg

    model = MedicalVLLM(llm_cfg, vision_cfg, connector_cfg).to(device)
    if can_load:
        model.load(cfg["_init_from"], load_connector=True, strict_llm=True)
    elif cfg.get("_init_from") and saved_backend != backend:
        logger.warning(
            "ignoring %s LLM weights in %s (backend mismatch: %s != %s)",
            saved_backend, cfg["_init_from"], saved_backend, backend,
        )
    if bool(cfg.get("freeze_llm", False)):
        model.freeze_llm()
    else:
        model.unfreeze_llm()
    model.vision.set_frozen(True)
    model.unfreeze_connector()
    model.connector.train()
    return model, llm_cfg, vision_cfg, connector_cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "vqa_sft.yaml"))
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

    setup_logging()   # minillm-style bar + library-noise filter (same as align)
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
    stop = StopSignal(args.stop or (base / ".stop_vqa"))

    use_hf = bool(args.hf) and not args.no_hf
    hf_cfg = dict(cfg.get("hf") or {})
    hf_cfg["enabled"] = use_hf
    logger.info("HF sync %s", "ENABLED (--hf)" if use_hf else "DISABLED")

    # Qwen: <repo>/medical_vllm/qwen/align and .../sft (set before download).
    if args.qwen and cfg.get("hf_qwen"):
        hf_cfg.update(dict(cfg["hf_qwen"]))
        logger.info(
            "HF Qwen layout: repo=%s dirs=%s / %s",
            hf_cfg.get("repo"),
            remote_phase_dir(hf_cfg, 1), remote_phase_dir(hf_cfg, 2),
        )

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "checkpoints/vqa"), base)
    out_dir = resolve_path(cfg.get("out_dir", "outs/vqa"), base)
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
                ("phase_2_latest.pt", "medicalvllm_vqa_*.pt"),
            )
        if resume is None:
            init = resolve_path(str(cfg.get("init_from", "")), base)
            if init.is_file():
                resume = init
                logger.info("no phase-2 checkpoint; continuing from phase 1: %s",
                            init)
            else:
                logger.info("no checkpoint found; starting phase 2 from scratch")
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

    cfg["_init_from"] = None if args.new else str(resume) if resume else None

    model, llm_cfg, vision_cfg, connector_cfg = build_model(
        cfg, base, device, use_qwen=use_qwen, qwen_model=qwen_model
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
    budget = int(cfg["data"].get("max_seq_budget", 112))
    streams = build_vqa_streams(cfg["data"], budget=budget)
    weights = dict(cfg["data"].get("mix", {}))
    batch_size = int(cfg.get("batch_size", 8))
    accum = int(cfg.get("grad_accum_steps", 2))
    sampler = BatchMixSampler(streams, weights, batch_size, seed=args.seed)
    epoch_size = int(cfg["data"].get("epoch_size", 200_000))
    sample_every = int(cfg.get("sample_interval", 500))
    sample_stream = None
    sample_iter = None
    if sample_every > 0:
        # One fixed held-out image source so samples are comparable over time.
        sample_stream = VQAStream(
            "vqa_rad", SOURCES["vqa_rad"][0], budget=budget,
            shuffle_buffer=0, train=False, seed=args.seed + 12345,
        )
        sample_iter = iter(sample_stream)

    # ---------------------------------------------------------- optimization
    groups = [
        {"params": list(model.connector.parameters()),
         "lr": float(cfg.get("lr_connector", 5e-5))},
        {"params": list(model.llm.parameters()),
         "lr": float(cfg.get("lr_llm", 1e-5))},
    ]
    optimizer = build_optimizer(
        groups, str(cfg.get("optimizer", "adamw")), weight_decay=0.01
    )
    logger.info("optimizer: %s", type(optimizer).__name__)

    max_steps = args.steps or cfg.get("steps")
    max_hours = args.hours or cfg.get("hours")
    # Bar mode is chosen from CLI flags ONLY. Config hours/steps are stop
    # limits; supplying no flag means "rows" mode, per the documented rule.
    bar_steps = args.steps if args.steps is not None else None
    bar_hours = args.hours if args.hours is not None else None
    total = max_steps or 10_000
    warmup = int(cfg.get("warmup_steps", 100))
    lr_fn = build_lr_lambda(warmup, total)
    # One schedule for all groups: each group's base lr already encodes the
    # connector:LLM ratio (5e-5 : 1e-5). The old extra *0.2 on the LLM group
    # silently turned lr_llm=1e-5 into a 2e-6 peak.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    state = RunState(phase=PHASE)
    if resume and not args.new:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        is_ours = saved.get("format") in {"medicalvllm-v1", "medicalvllm-v2"}
        n = 0
        if is_ours:
            restore_state(saved, state)
            n = load_optimizer_state_named(
                model, optimizer, saved.get("optimizer")
            )
            fast_forward_scheduler(scheduler, state.step)
            state.resumed_from = str(resume)
        if int(saved.get("phase", 1)) == 1:
            # One-time, low-key hand-off notice for phase 1 -> phase 2.
            logger.info(
                "phase-2 hand-off from phase 1 (global step continues): "
                "step=%d text_tokens=%d images=%d optimizer_params=%d",
                state.step, state.total_text_tokens,
                state.total_images_seen, n,
            )
        else:
            logger.info(
                "resumed phase 2: step=%d text_tokens=%d images=%d "
                "optimizer_params=%d",
                state.step, state.total_text_tokens,
                state.total_images_seen, n,
            )
    if state.examples_seen and epoch_size:
        state.epoch = state.examples_seen / epoch_size  # keep bar/state in sync

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
    eval_cfg = cfg.get("eval") or {}
    eval_every = int(eval_cfg.get("every_steps", 0))
    dev_samples = int(eval_cfg.get("dev_samples", 100))
    dev_sampler = BatchMixSampler(
        build_vqa_streams(cfg["data"], budget=budget),
        {k: v for k, v in weights.items() if k != "text_replay"},
        batch_size=1, seed=args.seed + 1,
    )
    dev_iter = iter(dev_sampler)

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
            {k: v.stats for k, v in streams.items()},
            llm_cfg, vision_cfg, connector_cfg,
        )
        step_path = ckpt_dir / f"phase2_step{state.step:07d}.pt"
        atomic_torch_save(ckpt, step_path)
        atomic_torch_save(ckpt, ckpt_dir / "phase_2_latest.pt")
        save_plots(out_dir)
        metrics.log(event="checkpoint", step=state.step,
                    text_tokens=state.total_text_tokens,
                    images=state.total_images_seen,
                    elapsed_seconds=round(state.elapsed_seconds, 1))
        logger.info("saved phase-2 checkpoint at step %d", state.step)
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

    logger.info("Stage 2 start: step=%d steps=%s hours=%s batch=%s accum=%s",
                state.step, max_steps, max_hours, batch_size, accum)
    pbar = build_progress_bar(PHASE, bar_steps, bar_hours, state, epoch_size)
    if pbar is not None:
        # "loss nan" is honest: no optimizer step has been measured yet. The
        # old initial "loss 0.000" looked like a real zero loss.
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

            source, examples = next(sampler)
            batch = move_batch(collate_fn(examples), device)
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
            state.update(len(examples), n_tokens, n_imgs, epoch_size)

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
                    n_rows=len(examples),
                    elapsed_seconds=state.elapsed_seconds,
                    examples_seen=state.examples_seen,
                )
                if pbar is not None:
                    pbar.set_postfix_str(postfix(state, avg, gn))
                if state.step <= log_first or state.step % log_every == 0:
                    logger.info(
                        "step %d loss %.4f lr_c %.2e lr_l %.2e | tok %d img %d",
                        state.step, avg,
                        optimizer.param_groups[0]["lr"],
                        optimizer.param_groups[1]["lr"],
                        state.total_text_tokens, state.total_images_seen,
                    )
                metrics.log(
                    step=state.step,
                    loss=round(avg, 5) if math.isfinite(avg) else None,
                    source=source,
                    lr_connector=optimizer.param_groups[0]["lr"],
                    lr_llm=optimizer.param_groups[1]["lr"],
                    grad_norm=round(gn, 4),
                    total_text_tokens=state.total_text_tokens,
                    total_images_seen=state.total_images_seen,
                    epoch=round(state.epoch, 4),
                )

                if sample_every and state.step % sample_every == 0:
                    log_sample_generation(
                        base_model, sample_iter, device, state.step, pbar,
                        out_dir=out_dir,
                        max_new_tokens=int(cfg.get("sample_max_new_tokens", 200)),
                        max_text_tokens=int(cfg.get("sample_max_tokens", 200)),
                        temperature=float(cfg.get("sample_temperature", 0.0)),
                        top_k=cfg.get("sample_top_k") or None,
                    )

            if eval_every and state.step and state.step % eval_every == 0:
                score = _quick_eval(base_model, dev_iter, dev_samples, device)
                metrics.log(step=state.step, eval_loss=round(score, 5))
                logger.info("eval loss %.4f", score)
                model.train()

            now = time.time()
            if now - last_save >= interval:
                step_path = save(f"step{state.step}")
                last_save = now
                upload_to_hf(step_path)   # no-op unless --hf was given
    except (KeyboardInterrupt, StopIteration):
        halted = True
    finally:
        if pbar is not None:
            pbar.close()
        sampler.close()
        dev_sampler.close()
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
        logger.info("Stage 2 done at step %d; stats=%s", state.step,
                    {k: v.stats for k, v in streams.items()})
    return state.step


@torch.no_grad()
def _quick_eval(model, dev_iter, n: int, device) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for _ in range(n):
        try:
            _, examples = next(dev_iter)
        except StopIteration:
            break
        batch = move_batch(collate_fn(examples), device)
        out = model(batch["input_ids"], batch["pixel_values"], batch["labels"])
        loss = out[1] if isinstance(out, tuple) else out
        if loss.isfinite():
            total += loss.item()
            count += 1
    model.train(was_training)
    return total / max(1, count)


def _detach():
    import subprocess
    import sys

    log = ROOT / "outs" / "vqa_detached.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "train.vqa_sft"] + [
        a for a in sys.argv[1:] if a != "--detach"
    ]
    with log.open("w") as f:
        subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"detached; log: {log}")


if __name__ == "__main__":
    main()
