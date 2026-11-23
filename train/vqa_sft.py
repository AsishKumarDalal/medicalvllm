"""Stage 2: medical VQA SFT (connector + LLM train, vision frozen).

Loads the aligned connector from Stage 1, unfreezes the GPT, and trains on a
weighted mixture of PMC-VQA / PathVQA / SLAKE / VQA-RAD with 10% text replay.
Run::

    python -m train.vqa_sft --steps 500
    python -m train.vqa_sft --hours 8 --detach
"""

import argparse
import logging
import time
from pathlib import Path

import torch

from .common import (
    Metrics,
    StopSignal,
    atomic_torch_save,
    build_lr_lambda,
    clip_grad_norm,
    get_device,
    load_yaml,
    move_batch,
    resolve_path,
    set_seed,
)
from data.collate import collate_fn
from data.medical_vqa import BatchMixSampler, build_vqa_streams
from model.medical_vllm import MedicalVLLM

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def build_model(cfg: dict, base: Path, device, llm_cfg: dict | None = None):
    init = resolve_path(str(cfg.get("init_from", "")), base)
    state = torch.load(init, map_location="cpu", weights_only=True)
    if llm_cfg is None:
        llm_cfg = state.get("config", {}).get("llm") or load_yaml(
            base / "configs" / "model.yaml"
        )
    vision_cfg = state.get("config", {}).get("vision") or load_yaml(
        base / "configs" / "vision.yaml"
    )
    connector_cfg = state.get("config", {}).get("connector") or vision_cfg

    model = MedicalVLLM(llm_cfg, vision_cfg, connector_cfg).to(device)
    model.load(init, load_connector=True, strict_llm=True, strict_connector=True)
    if bool(cfg.get("freeze_llm", False)):
        model.freeze_llm()
    else:
        model.unfreeze_llm()
    model.vision.set_frozen(True)
    model.unfreeze_connector()
    model.connector.train()
    return model, llm_cfg, vision_cfg, connector_cfg, init


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "vqa_sft.yaml"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop", default=None)
    ap.add_argument("--detach", action="store_true")
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
    flush = stop.check()

    model, llm_cfg, vision_cfg, connector_cfg, init = build_model(
        cfg, base, device
    )
    counts = model.param_counts()
    logger.info("params (M): %s", {k: round(v / 1e6, 2) for k, v in counts.items()})

    budget = int(cfg["data"].get("max_seq_budget", 112))
    streams = build_vqa_streams(cfg["data"], budget=budget)
    weights = dict(cfg["data"].get("mix", {}))
    batch_size = int(cfg.get("batch_size", 8))
    accum = int(cfg.get("grad_accum_steps", 2))
    sampler = BatchMixSampler(streams, weights, batch_size, seed=args.seed)

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
    sched_base = build_lr_lambda(warmup, total)
    def lr_fn(step, mult=1.0):
        return sched_base(step) * mult
    sched_conn = torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lambda s: lr_fn(s, 1.0), lambda s: lr_fn(s, 0.2)]
    )

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "checkpoints/vqa"), base)
    out_dir = resolve_path(cfg.get("out_dir", "outs/vqa"), base)
    metrics = Metrics(out_dir)
    t0 = time.time()
    last_save = t0
    interval = float(cfg.get("checkpoint_interval_minutes", 60)) * 60
    step, running, nloss, micro = 0, 0.0, 0, 0
    halted = False

    eval_cfg = cfg.get("eval") or {}
    eval_every = int(eval_cfg.get("every_steps", 0))
    dev_samples = int(eval_cfg.get("dev_samples", 100))
    dev_iter = iter(BatchMixSampler(
        build_vqa_streams(cfg["data"], budget=budget),
        {k: v for k, v in weights.items() if k != "text_replay"},
        batch_size=1, seed=args.seed + 1,
    ))

    def save(tag):
        atomic_torch_save({
            "format": "medicalvllm-v1",
            "step": step,
            "llm": model.llm.state_dict(),
            "connector": model.connector.state_dict(),
            "vision": {
                "encoder": model.vision.encoder_name,
                "encoder_backend": model.vision.backend,
                "image_size": model.vision.image_size,
                "num_visual_tokens": model.num_visual_tokens,
            },
            "config": {
                "llm": llm_cfg, "vision": vision_cfg,
                "connector": connector_cfg,
            },
        }, ckpt_dir / f"medicalvllm_vqa_{tag}.pt")
        logger.info("saved checkpoint %s", ckpt_dir / f"medicalvllm_vqa_{tag}.pt")

    logger.info("Stage 2 start: steps=%s hours=%s batch=%s accum=%s",
                max_steps, max_hours, batch_size, accum)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while True:
            if stop.check():
                halted = True
                break
            if max_steps and step >= max_steps:
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

            if micro % accum == 0:
                gn = clip_grad_norm(model, float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                sched_conn.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                avg = running / max(1, nloss)
                running, nloss = 0.0, 0
                if step % 10 == 0:
                    logger.info("step %d loss %.4f lr_c %.2e lr_l %.2e gn %.3f",
                                step, avg,
                                optimizer.param_groups[0]["lr"],
                                optimizer.param_groups[1]["lr"], gn)
                    metrics.log(step=step, loss=round(avg, 5), source=source,
                                lr_connector=optimizer.param_groups[0]["lr"],
                                lr_llm=optimizer.param_groups[1]["lr"],
                                grad_norm=round(gn, 4))

            if eval_every and step and step % eval_every == 0:
                score = _quick_eval(model, dev_iter, dev_samples, device)
                metrics.log(step=step, eval_loss=round(score, 5))
                logger.info("eval loss %.4f", score)
                model.train()

            now = time.time()
            if now - last_save >= interval:
                save(f"step{step}")
                last_save = now
    except (KeyboardInterrupt, StopIteration):
        halted = True
    finally:
        save("final")
        metrics.log(event="end", step=step, halted=halted,
                    hours=round((time.time() - t0) / 3600, 3))
        logger.info("Stage 2 done at step %d; stats=%s", step,
                    {k: v.stats for k, v in streams.items()})
    return step


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
