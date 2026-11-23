"""Stage 1 stream: ROCOv2 radiology captions."""

import logging
from typing import Iterator

from .tokenizer import encode_example, format_caption_prompt, get_encoder
from .transforms import ImageTransform

logger = logging.getLogger(__name__)


class CaptionStream:
    """Stream ``(pixel_values, input_ids, labels)`` from ROCOv2."""

    def __init__(
        self,
        dataset: str = "eltorio/ROCOv2-radiology",
        budget: int = 112,
        image_size: int = 224,
        train: bool = True,
        shuffle_buffer: int = 500,
        seed: int = 0,
        max_docs: int | None = None,
    ):
        self.dataset = dataset
        self.budget = int(budget)
        self.transform = ImageTransform(image_size, train=train)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        self.max_docs = max_docs
        self.encoder = get_encoder()
        self.prompt_ids = None  # lazy: set after first encode
        self.stats = {"seen": 0, "kept": 0, "dropped": 0, "bad_image": 0}
        self._active_gen = None

    def __iter__(self) -> Iterator[dict]:
        # Keep a handle so close() can release the underlying streaming
        # dataset BEFORE interpreter shutdown (otherwise pending HTTP range
        # requests throw "Bad file descriptor" and retry loops hang the exit).
        self._active_gen = self._generate()
        return self._active_gen

    def close(self):
        if self._active_gen is not None:
            self._active_gen.close()
            self._active_gen = None

    def _generate(self) -> Iterator[dict]:
        from datasets import load_dataset

        ds = load_dataset(self.dataset, split="train", streaming=True)
        if self.shuffle_buffer > 1:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        prompt = format_caption_prompt()
        yielded = 0
        for row in ds:
            if self.max_docs is not None and yielded >= self.max_docs:
                break
            self.stats["seen"] += 1
            caption = str(row.get("caption") or "").strip()
            if not caption:
                self.stats["dropped"] += 1
                continue
            packed = encode_example(prompt, caption, self.budget, self.encoder)
            if packed is None:
                self.stats["dropped"] += 1
                continue
            ids, labels, triage = packed
            try:
                pixel = self.transform(row["image"])
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
            }
