"""Greedy VQA evaluation with a blind baseline.

For every example we compute metrics twice: once with the real visual prefix
and once with a zeroed prefix (blind). If the model only memorized the question
priors, the two scores are nearly identical — that is the signal this harness
is designed to expose.

Usage::

    python -m eval.vqa_eval --checkpoint /path/to/medicalvllm_vqa_final.pt \
        --source vqa_rad --samples 200
"""

import argparse
import json
import logging
import re
import string
from pathlib import Path

import torch

from data.collate import collate_fn
from data.medical_vqa import VQAStream, SOURCES
from data.tokenizer import get_encoder
from model.medical_vllm import MedicalVLLM
from train.common import get_device, load_yaml, resolve_path, set_seed

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[ab]\.\s*", "", text)  # MCQ letter prefix
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def f1_token(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = set(p) & set(r)
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(r)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def evaluate(
    model: MedicalVLLM,
    stream: VQAStream,
    n: int,
    device,
    blind: bool = False,
):
    model.eval()
    enc = get_encoder()
    rows = []
    it = iter(stream)
    for _ in range(n):
        try:
            ex = next(it)
        except StopIteration:
            break
        if ex["pixel_values"] is None:
            continue
        pixel = ex["pixel_values"].unsqueeze(0).to(device)
        batch = collate_fn([ex])
        ids = batch["input_ids"].to(device)

        prefix_len = model.num_visual_tokens
        if blind:
            # Replace the visual prefix with zeros without touching the tower.
            vis = torch.zeros(
                1, prefix_len, model.llm.tok_emb.embedding_dim,
                device=device, dtype=model.llm.tok_emb.weight.dtype,
            )
            tok = torch.cat([vis, model.llm.tok_emb(ids)], dim=1)
            model.llm.reset_cache()
            logits = model.llm.forward_embeds(tok, use_cache=True)[:, -1, :]
        else:
            model.llm.reset_cache()
            vis = model.encode_image(pixel)
            tok = torch.cat([vis, model.llm.tok_emb(ids)], dim=1)
            logits = model.llm.forward_embeds(tok, use_cache=True)[:, -1, :]

        nxt = model.llm._sample_next(logits, 0.0, None)
        pred = enc.decode([int(nxt[0, 0])]).strip()
        # Reference = first non-masked target token sequence, decoded.
        labels = ex["labels"]
        ref_ids = [t for t in labels if t != -100]
        ref = enc.decode(ref_ids).strip()
        rows.append((pred, ref))

    exact = sum(normalize(p) == normalize(r) for p, r in rows) / max(1, len(rows))
    f1 = sum(f1_token(p, r) for p, r in rows) / max(1, len(rows))
    return {"n": len(rows), "exact": round(exact, 4), "f1": round(f1, 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source", default="vqa_rad", choices=list(SOURCES))
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = Path(args.checkpoint)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = state["config"]

    model = MedicalVLLM(
        cfg["llm"], cfg["vision"], cfg.get("connector") or cfg["vision"]
    ).to(device)
    model.load(ckpt, load_connector=True)
    stream = VQAStream(
        args.source, SOURCES[args.source][0],
        budget=112, shuffle_buffer=0, train=False,
    )

    real = evaluate(model, stream, args.samples, device, blind=False)
    blind = evaluate(model, stream, args.samples, device, blind=True)
    result = {
        "checkpoint": str(ckpt), "source": args.source,
        "real": real, "blind": blind,
        "delta_exact": round(real["exact"] - blind["exact"], 4),
    }
    print(json.dumps(result, indent=2))
    out = Path(args.out) if args.out else ROOT / "outs" / "eval" / f"{args.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
