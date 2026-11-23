"""Image preprocessing shared by all medical data sources."""

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# OpenAI CLIP / BiomedCLIP normalization stats.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageTransform:
    """Convert a PIL image / bytes / path to a normalized [3, S, S] tensor.

    Never horizontally flips: radiograph laterality is clinically meaningful.
    """

    def __init__(self, image_size: int = 224, train: bool = False):
        self.image_size = int(image_size)
        ops = []
        if train:
            ops.append(transforms.RandomResizedCrop(
                self.image_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)
            ))
        else:
            ops.append(transforms.Resize(
                self.image_size, interpolation=transforms.InterpolationMode.BICUBIC
            ))
            ops.append(transforms.CenterCrop(self.image_size))
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
        self.tf = transforms.Compose(ops)

    def __call__(self, source) -> torch.Tensor:
        img = _to_pil(source)
        return self.tf(img)


def _to_pil(source) -> Image.Image:
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif isinstance(source, (str, Path)):
        img = Image.open(source)
    elif hasattr(source, "convert") and hasattr(source, "resize"):
        img = source  # datasets' PIL-like image object
    else:
        raise TypeError(f"cannot convert {type(source)} to PIL image")
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
