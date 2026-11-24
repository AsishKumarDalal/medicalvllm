"""Greedy VQA evaluation with a blind baseline.

For every example we generate a full answer twice: once with the real visual
prefix and once with a zeroed prefix (blind). If the model only memorized the
question priors, the two scores are nearly identical — that is the signal this
harness is designed to expose.

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
from data.tokenizer import get_encoder, set_tokenizer_backend
from model.medical_vllm import MedicalVLLM
from train.common import QWEN_MODEL, get_device, resolve_path, set_seed

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"^[abcd]\.\s*", "", text)  # MCQ letter prefix
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def f1_token(pred: str, ref: str) -> float:
    p, r = normalize(pred).split(), normalize(ref).split()
    if not p or not r:
        return float(p == r)
    common = set(p) & set(r)
    if not common:
        return 0.0
    prec, rec = len(common) / len(p), len(common) / len(r)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def _generate(
    model: MedicalVLLM,
    prefix: torch.Tensor,
    ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Cached greedy decoding with an explicit visual prefix."""
    model.llm.reset_cache()
    tok = torch.cat([prefix, model.llm.tok_emb(ids)], dim=1)
    if tok.size(1) + max_new_tokens > model.llm.context_length:
        raise ValueError("prefix + prompt + max_new_tokens exceeds context")
    logits = model.llm.forward_embeds(tok, use_cache=True)[:, -1, :]
    generated = ids
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    for _ in range(max_new_tokens):
        nxt = model.llm._sample_next(logits, 0.0, None)
        nxt, finished, all_done = model.llm._apply_eos(
            nxt, finished, model.eot_id
        )
        generated = torch.cat([generated, nxt], dim=1)
        if all_done:
            break
        logits = model.llm.forward_embeds(
            model.llm.tok_emb(nxt), use_cache=True
        )[:, -1, :]
    model.llm.reset_cache()
    return generated


@torch.no_grad()
def evaluate(
    model: MedicalVLLM,
    stream: VQAStream,
    n: int,
    device,
    blind: bool = False,
    max_new_tokens: int = 24,
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

        ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
        if blind:
            prefix = torch.zeros(
                1,
                model.num_visual_tokens,
                model.llm.tok_emb.embedding_dim,
                device=device,
                dtype=model.llm.tok_emb.weight.dtype,
            )
        else:
            pixel = ex["pixel_values"].unsqueeze(0).to(device)
            prefix = model.encode_image(pixel)

        gen = _generate(model, prefix, ids, max_new_tokens)
        pred = enc.decode(gen[0, ids.size(1):].tolist()).strip()
        ref_ids = [
            t for t in ex["labels"] if t != -100 and t != model.eot_id
        ]
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
    ap.add_argument("--qwen", action="store_true",
                    help="force the Qwen tokenizer (auto-detected from checkpoint)")
    ap.add_argument("--qwen-model", default=None,
                    help=f"HuggingFace Qwen model id (default {QWEN_MODEL})")
    ap.add_argument("--qween", dest="qwen", action="store_true",
                    help=argparse.SUPPRESS)  # common misspelling
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = Path(args.checkpoint)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = state["config"]
    llm_cfg = cfg.get("llm") or {}
    saved_backend = str(llm_cfg.get("backend", "gpt2")).lower()
    use_qwen = bool(args.qwen) or saved_backend == "qwen"
    set_tokenizer_backend(
        "qwen" if use_qwen else "gpt2",
        model_name=args.qwen_model or llm_cfg.get("model_name", QWEN_MODEL),
    )

    model = MedicalVLLM(
        cfg["llm"], cfg["vision"], cfg.get("connector") or cfg["vision"]
    ).to(device)
    model.load(ckpt, load_connector=True)
    stream = VQAStream(
        args.source,
        SOURCES[args.source][0],
        budget=112,
        shuffle_buffer=0,
        train=False,
    )

    real = evaluate(model, stream, args.samples, device, blind=False)
    blind = evaluate(model, stream, args.samples, device, blind=True)
    result = {
        "checkpoint": str(ckpt),
        "source": args.source,
        "real": real,
        "blind": blind,
        "delta_exact": round(real["exact"] - blind["exact"], 4),
    }
    print(json.dumps(result, indent=2))
    out = (
        Path(args.out)
        if args.out
        else ROOT / "outs" / "eval" / f"{args.source}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
