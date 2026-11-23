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
from data.collate import collate_fn
from data.medical_vqa import BatchMixSampler, build_vqa_streams
from model.medical_vllm import MedicalVLLM

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
PHASE = 2


def build_model(cfg, base, device, llm_cfg=None):
    """Build phase 2 from the checkpoint given by ``init_from`` (or --new)."""
    state = torch.load(cfg["_init_from"], map_location="cpu", weights_only=True) \
        if cfg.get("_init_from") else {}
    if llm_cfg is None:
        if "config" in state:
            llm_cfg = state["config"]["llm"]
        elif "model_config" in state:
            llm_cfg = state["model_config"]
        else:
            llm_cfg = load_yaml(base / "configs" / "model.yaml")

    if "config" in state:
        vision_cfg = state["config"]["vision"]
        connector_cfg = state["config"].get("connector") or vision_cfg
    else:
        vision_cfg = load_yaml(base / "configs" / "vision.yaml")
        connector_cfg = vision_cfg

    model = MedicalVLLM(llm_cfg, vision_cfg, connector_cfg).to(device)
    if cfg.get("_init_from"):
        model.load(cfg["_init_from"], load_connector=True, strict_llm=True)
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
    stop = StopSignal(args.stop or (base / ".stop_vqa"))

    use_hf = bool(args.hf) and not args.no_hf
    hf_cfg = dict(cfg.get("hf") or {})
    hf_cfg["enabled"] = use_hf
    logger.info("HF sync %s", "ENABLED (--hf)" if use_hf else "DISABLED")

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
    cfg["_init_from"] = None if args.new else str(resume) if resume else None

    model, llm_cfg, vision_cfg, connector_cfg = build_model(
        cfg, base, device
    )
    counts = model.param_counts()
    logger.info("params (M): %s", {k: round(v / 1e6, 2) for k, v in counts.items()})

    # ---------------------------------------------------------------- data
    budget = int(cfg["data"].get("max_seq_budget", 112))
    streams = build_vqa_streams(cfg["data"], budget=budget)
    weights = dict(cfg["data"].get("mix", {}))
    batch_size = int(cfg.get("batch_size", 8))
    accum = int(cfg.get("grad_accum_steps", 2))
    sampler = BatchMixSampler(streams, weights, batch_size, seed=args.seed)
    epoch_size = int(cfg["data"].get("epoch_size", 200_000))

    # ---------------------------------------------------------- optimization
    groups = [
        {"params": list(model.connector.parameters()),
         "lr": float(cfg.get("lr_connector", 5e-5))},
        {"params": list(model.llm.parameters()),
         "lr": float(cfg.get("lr_llm", 1e-5))},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    max_steps = args.steps or cfg.get("steps")
    max_hours = args.hours or cfg.get("hours")
    total = max_steps or 10_000
    warmup = int(cfg.get("warmup_steps", 100))
    lr_fn = build_lr_lambda(warmup, total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        [lambda s: lr_fn(s), lambda s: lr_fn(s) * 0.2],
    )

    state = RunState(phase=PHASE)
    if resume and not args.new:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        restore_state(saved, state)
        n = load_optimizer_state_named(model, optimizer, saved.get("optimizer"))
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

    metrics = Metrics(out_dir)
    eval_cfg = cfg.get("eval") or {}
    eval_every = int(eval_cfg.get("every_steps", 0))
    dev_samples = int(eval_cfg.get("dev_samples", 100))
    dev_iter = iter(BatchMixSampler(
        build_vqa_streams(cfg["data"], budget=budget),
        {k: v for k, v in weights.items() if k != "text_replay"},
        batch_size=1, seed=args.seed + 1,
    ))

    t0 = time.time()
    last_save = t0
    interval = float(cfg.get("checkpoint_interval_minutes", 60)) * 60
    running, nloss, micro, halted = 0.0, 0, 0, False

    def save(tag):
        ckpt = build_checkpoint(
            model, state, optimizer, scheduler,
            {k: v.stats for k, v in streams.items()},
            llm_cfg, vision_cfg, connector_cfg,
        )
        step_path = ckpt_dir / f"phase2_step{state.step:07d}.pt"
        atomic_torch_save(ckpt, step_path)
        atomic_torch_save(ckpt, ckpt_dir / "phase_2_latest.pt")
        save_plots(out_dir)
        metrics.log(event="checkpoint", step=state.step,
                    text_tokens=state.total_text_tokens,
                    images=state.total_images_seen)
        logger.info("saved phase-2 checkpoint at step %d", state.step)
        if use_hf:
            upload_checkpoint(hf_cfg, PHASE, step_path, state.step)
            upload_outs(hf_cfg, PHASE, out_dir)

    logger.info("Stage 2 start: step=%d steps=%s hours=%s batch=%s accum=%s",
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

            source, examples = next(sampler)
            batch = move_batch(collate_fn(examples), device)
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
            state.update(len(examples), n_tokens, n_imgs, epoch_size)

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
                        "step %d loss %.4f lr_c %.2e lr_l %.2e | tok %d img %d",
                        state.step, avg,
                        optimizer.param_groups[0]["lr"],
                        optimizer.param_groups[1]["lr"],
                        state.total_text_tokens, state.total_images_seen,
                    )
                    metrics.log(
                        step=state.step, loss=round(avg, 5), source=source,
                        lr_connector=optimizer.param_groups[0]["lr"],
                        lr_llm=optimizer.param_groups[1]["lr"],
                        grad_norm=round(gn, 4),
                        total_text_tokens=state.total_text_tokens,
                        total_images_seen=state.total_images_seen,
                        epoch=round(state.epoch, 4),
                    )

            if eval_every and state.step and state.step % eval_every == 0:
                score = _quick_eval(model, dev_iter, dev_samples, device)
                metrics.log(step=state.step, eval_loss=round(score, 5))
                logger.info("eval loss %.4f", score)
                model.train()

            now = time.time()
            if now - last_save >= interval:
                save(f"step{state.step}")
                last_save = now
    except (KeyboardInterrupt, StopIteration):
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
