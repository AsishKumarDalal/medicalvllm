"""Frozen vision tower: pixels -> patch tokens.

Two backends are supported:

- ``open_clip`` (default): BiomedCLIP ViT-B/16, 196 patches of 768-d.
- ``hf``: any HuggingFace CLIP vision tower, e.g. ``openai/clip-vit-base-patch32``.

The rest of the model only sees ``[B, P, width]`` patch tokens, so encoder
ablations are a config change. The tower is never trained in Stages 1-2 and
is never written into checkpoints.
"""

import torch
import torch.nn as nn


class VisionTower(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder_name = str(cfg["encoder"])
        self.backend = str(cfg.get("encoder_backend", "open_clip")).lower()
        self.image_size = int(cfg.get("image_size", 224))
        self.amp = bool(cfg.get("amp", True))

        if self.backend == "open_clip":
            self._build_open_clip()
        elif self.backend in {"hf", "transformers"}:
            self._build_hf()
        else:
            raise ValueError(f"unknown encoder_backend: {self.backend!r}")

        if bool(cfg.get("freeze_encoder", True)):
            self.set_frozen(True)

    # -- construction -------------------------------------------------------
    def _build_open_clip(self):
        import open_clip

        name = self.encoder_name
        if not name.startswith("hf-hub:"):
            name = f"hf-hub:{name}"
        model, _, _ = open_clip.create_model_and_transforms(name)
        # Keep only the image tower; drop the text tower from memory.
        self.visual = model.visual
        del model

        trunk = getattr(self.visual, "trunk", None)
        if trunk is None:
            raise RuntimeError(
                "open_clip visual has no .trunk; cannot extract patch tokens"
            )
        self.width = int(getattr(trunk, "embed_dim"))
        self.num_patches = int(trunk.patch_embed.num_patches)
        self._is_open_clip = True

    def _build_hf(self):
        from transformers import CLIPVisionModel

        self.visual = CLIPVisionModel.from_pretrained(self.encoder_name)
        cfg = self.visual.config
        self.width = int(cfg.hidden_size)
        self.num_patches = int(
            (cfg.image_size // cfg.patch_size) ** 2
        )
        self._is_open_clip = False

    # -- freezing -----------------------------------------------------------
    def set_frozen(self, frozen: bool = True):
        self.frozen = bool(frozen)
        for p in self.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.eval()

    def train(self, mode: bool = True):
        # A frozen tower must stay in eval mode even if the wrapper calls
        # .train() on the whole model.
        if getattr(self, "frozen", False):
            mode = False
        return super().train(mode)

    # -- forward ------------------------------------------------------------
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must be (B, 3, H, W), got "
                f"{tuple(pixel_values.shape)}"
            )
        if pixel_values.size(1) != 3:
            raise ValueError(
                f"expected 3 image channels, got {pixel_values.size(1)}"
            )

        use_amp = self.amp and pixel_values.is_cuda
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            if self._is_open_clip:
                feats = self.visual.trunk.forward_features(pixel_values)
            else:
                feats = self.visual(
                    pixel_values=pixel_values
                ).last_hidden_state

        if isinstance(feats, dict):
            feats = feats.get("x", feats.get("last_hidden_state"))
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if not torch.is_tensor(feats) or feats.dim() != 3:
            raise RuntimeError(
                "vision tower did not return (B, tokens, width); "
                f"got {type(feats)}"
            )
        # Drop CLS. The remaining tokens are the patch grid.
        return feats[:, 1:, :].float()
