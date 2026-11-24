"""Right-pad token sequences; stack images (or return None)."""

import torch

from .tokenizer import get_pad_id


def collate_fn(batch: list[dict], pad_id: int | None = None):
    if not batch:
        raise ValueError("empty batch")
    if pad_id is None:
        pad_id = get_pad_id()
    max_len = max(len(ex["input_ids"]) for ex in batch)
    ids, labels, pixels = [], [], []
    has_image = batch[0]["pixel_values"] is not None

    for ex in batch:
        n = len(ex["input_ids"])
        pad = max_len - n
        ids.append(ex["input_ids"] + [pad_id] * pad)
        labels.append(ex["labels"] + [-100] * pad)
        p = ex["pixel_values"]
        if has_image and p is None:
            raise ValueError("mixed image / text-only examples in one batch")
        if has_image:
            pixels.append(p)

    out = {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if has_image:
        out["pixel_values"] = torch.stack(pixels)
    else:
        out["pixel_values"] = None
    return out
