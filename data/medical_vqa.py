"""Stage 2 streams: medical VQA datasets + text replay.

Every source yields the same dict::

    {"pixel_values": tensor | None, "input_ids": [...], "labels": [...]}

Sources with no image (text replay) set ``pixel_values=None``. The batch
sampler draws *whole batches* from one source so a batch is never a mix of
image and text-only examples.
"""

import logging
import random
from typing import Iterator

from .tokenizer import encode_example, format_vqa_prompt, get_encoder
from .transforms import ImageTransform

logger = logging.getLogger(__name__)

# Verified on Kaggle (Nov 2026). Fallbacks are tried in order.
SOURCES = {
    "pmc_vqa": ["mdwiratathya/PMC-VQA_small", "RadGenome/PMC-VQA"],
    "path_vqa": ["flaviagiammarino/path-vqa"],
    "slake": ["mdwiratathya/SLAKE-vqa-english", "BoKelvin/SLAKE"],
    "vqa_rad": ["flaviagiammarino/vqa-rad"],
    "text_replay": ["databricks/databricks-dolly-15k"],
}


def _pick(row: dict, *keys, default=""):
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return default


def _get_image(row: dict):
    for k in ("image", "img", "figure", "Figure", "picture", "photo"):
        v = row.get(k)
        if v is not None:
            return v
    return None


class VQAStream:
    """Stream normalized examples from one medical VQA source."""

    def __init__(
        self,
        name: str,
        dataset: str,
        budget: int = 112,
        image_size: int = 224,
        train: bool = True,
        shuffle_buffer: int = 1000,
        seed: int = 0,
        max_docs: int | None = None,
    ):
        self.name = name
        self.dataset = dataset
        self.budget = int(budget)
        self.transform = ImageTransform(image_size, train=train)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.max_docs = max_docs
        self.encoder = get_encoder()
        self.image_required = name != "text_replay"
        self.stats = {
            "seen": 0, "kept": 0, "dropped": 0,
            "no_image": 0, "bad_image": 0,
        }
        self._active_gen = None

    def __iter__(self) -> Iterator[dict]:
        self._active_gen = self._generate()
        return self._active_gen

    def close(self):
        if self._active_gen is not None:
            self._active_gen.close()
            self._active_gen = None

    def _generate(self) -> Iterator[dict]:
        from datasets import load_dataset

        try:
            ds = load_dataset(self.dataset, split="train", streaming=True)
        except Exception as exc:
            logger.warning("[%s] cannot load %s: %s", self.name, self.dataset, exc)
            return
        if self.shuffle_buffer > 1:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        yielded = 0
        for row in ds:
            if self.max_docs is not None and yielded >= self.max_docs:
                break
            self.stats["seen"] += 1
            self.context = None
            q, a = self._question_answer(row)
            if not q or not a:
                self.stats["dropped"] += 1
                continue

            prompt = format_vqa_prompt(q, context=getattr(self, "context", None))
            packed = encode_example(prompt, a, self.budget, self.encoder)
            if packed is None:
                self.stats["dropped"] += 1
                continue
            ids, labels, triage = packed

            pixel = None
            if self.image_required:
                src = _get_image(row)
                if src is None:
                    self.stats["no_image"] += 1
                    continue
                try:
                    pixel = self.transform(src)
                except Exception:
                    self.stats["bad_image"] += 1
                    continue

            self.stats["kept"] += 1
            yielded += 1
            yield {
                "pixel_values": pixel,
                "input_ids": ids,
                "labels": labels,
                "triage": triage,
                "source": self.name,
            }

    def _question_answer(self, row: dict):
        if self.name == "text_replay":
            instruction = _pick(row, "instruction", default="")
            context = _pick(row, "context", default="")
            response = _pick(row, "response", "answer", default="")
            if not instruction or not response:
                return "", ""
            self.context = context or None
            return instruction, response

        q = _pick(row, "question", "Question")
        a = _pick(row, "answer", "Answer")
        if self.name == "pmc_vqa":
            # The small variant carries the figure caption; use it as context.
            caption = _pick(row, "caption", default="")
            if caption:
                self.context = caption.strip()
            if not a:
                label = _pick(
                    row, "answer label", "Answer_label", default=""
                ).strip().upper()
                if label in {"A", "B", "C", "D"}:
                    a = _pick(
                        row, f"choice {label.lower()}", f"Choice {label}",
                        default="",
                    )
                    a = a.split(":", 1)[-1].strip()
        return q.strip(), a.strip()


def build_vqa_streams(cfg: dict, budget: int = 112) -> dict[str, VQAStream]:
    """Build all sources from configs/vqa_sft.yaml."""
    mix = cfg.get("mix", {})
    streams = {}
    for name in mix:
        candidates = SOURCES.get(name, [])
        if not candidates:
            logger.warning("no dataset mapping for source %r; skipping", name)
            continue
        dataset = candidates[0]
        if name == "text_replay":
            budget_use = budget
        else:
            budget_use = budget
        streams[name] = VQAStream(
            name=name,
            dataset=dataset,
            budget=budget_use,
            shuffle_buffer=int(cfg.get("shuffle_buffer", 1000)),
        )
    return streams


class BatchMixSampler:
    """Infinitely yield batches drawn from one source by weight.

    Each yielded item is ``(source_name, [examples])``. A batch is always
    homogeneous: image sources produce image batches, text_replay produces
    text-only batches.
    """

    def __init__(
        self,
        streams: dict[str, VQAStream],
        weights: dict[str, float],
        batch_size: int,
        seed: int = 0,
    ):
        self.streams = streams
        self.weights = {k: float(weights.get(k, 0)) for k in streams}
        if sum(self.weights.values()) <= 0:
            raise ValueError("mix weights must sum to a positive number")
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)
        self.iters = {k: iter(v) for k, v in streams.items()}
        self.names = list(streams)
        self._probs = [
            self.weights[k] / sum(self.weights.values()) for k in self.names
        ]

    def __iter__(self):
        return self

    def __next__(self):
        # Pick a source; if exhausted, rotate to one that still has data.
        for _ in range(len(self.names)):
            name = self.rng.choices(self.names, weights=self._probs, k=1)[0]
            batch = self._take(name, self.batch_size)
            if batch:
                return name, batch
        raise StopIteration

    def close(self):
        for stream in self.streams.values():
            stream.close()

    def _take(self, name: str, n: int) -> list:
        out = []
        it = self.iters[name]
        while len(out) < n:
            try:
                out.append(next(it))
            except StopIteration:
                # Restart the stream for another epoch.
                self.iters[name] = iter(self.streams[name])
                it = self.iters[name]
                if not out:
                    try:
                        out.append(next(it))
                    except StopIteration:
                        break
                else:
                    break
        return out
