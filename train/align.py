"""Stage 1: connector-only alignment on ROCOv2 radiology captions.

The GPT and the vision tower are frozen. Only the connector learns to write
image information in the GPT's embedding space. Run::

    python -m train.align --config configs/align.yaml --steps 200
    python -m train.align --hours 4 --detach
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
from data.captions import CaptionStream
from data.collate import collate_fn
from model.medical_vllm import MedicalVLLM

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


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


def build_model(cfg: dict, base: Path, ckpt: Path, device):
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    # Context is dictated by the checkpoint's positional embeddings (128 in
    # the released SFT checkpoint), not by the generic config file.
    llm_cfg = state["model_config"]
    vision_cfg = load_yaml(base / "configs" / "vision.yaml")

    model = MedicalVLLM(llm_cfg, vision_cfg).to(device)
    model.load(ckpt, load_connector=False, strict_llm=True)
    model.freeze_llm()
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
    ap.add_argument("--stop", default=None, help="stop file path")
    ap.add_argument("--detach", action="store_true")
    args = ap.parse_args(argv)

    if args.detach:
        _detach()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed)
    base = Path(args.config).resolve().parents[1]
    cfg = load_yaml(args.config)
    device = get_device(args.device)
    stop = StopSignal(args.stop or (base / ".stop_align"))

    ckpt = resolve_llm_checkpoint(cfg, base)
    model, llm_cfg, vision_cfg = build_model(cfg, base, ckpt, device)
    counts = model.param_counts()
    logger.info("params (M): %s", {k: round(v / 1e6, 2) for k, v in counts.items()})

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
    lr_lambda = build_lr_lambda(int(cfg.get("warmup_steps", 200)), total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = resolve_path(cfg.get("checkpoint_dir", "checkpoints/align"), base)
    out_dir = resolve_path(cfg.get("out_dir", "outs/align"), base)
    metrics = Metrics(out_dir)
    t0 = time.time()
    last_save = t0
    interval = float(cfg.get("checkpoint_interval_minutes", 60)) * 60
    step = 0
    running = 0.0
    nloss = 0
    halted = False

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
                "connector": model.connector_cfg,
            },
        }, ckpt_dir / f"medicalvllm_align_{tag}.pt")
        logger.info("saved checkpoint %s", ckpt_dir / f"medicalvllm_align_{tag}.pt")

    logger.info("Stage 1 start: steps=%s hours=%s batch=%s accum=%s",
                max_steps, max_hours, batch_size, accum)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    try:
        while True:
            if stop.check():
                logger.info("stop requested")
                halted = True
                break
            if max_steps and step >= max_steps:
                break
            if max_hours and (time.time() - t0) / 3600 >= max_hours:
                logger.info("time limit reached")
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

            if micro % accum == 0:
                gn = clip_grad_norm(model, float(cfg.get("grad_clip", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                avg = running / max(1, nloss)
                running, nloss = 0.0, 0
                if step % 10 == 0:
                    logger.info("step %d loss %.4f lr %.2e gn %.3f",
                                step, avg, scheduler.get_last_lr()[0], gn)
                    metrics.log(step=step, loss=round(avg, 5),
                                lr=scheduler.get_last_lr()[0],
                                grad_norm=round(gn, 4))

            now = time.time()
            if now - last_save >= interval:
                save(f"step{step}")
                last_save = now
    except KeyboardInterrupt:
        logger.info("interrupted")
        halted = True
    finally:
        save("final")
        metrics.log(event="end", step=step, halted=halted,
                    hours=round((time.time() - t0) / 3600, 3))
        logger.info("Stage 1 done at step %d; stats=%s", step, stream.stats)
    return step


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
        a for a in __import__("sys").argv[1:] if a != "--detach"
    ]
    with log.open("w") as f:
        subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"detached; log: {log}")


if __name__ == "__main__":
    main()
