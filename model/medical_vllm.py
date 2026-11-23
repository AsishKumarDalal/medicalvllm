"""MedicalVLLM: frozen vision tower + trainable connector + minillm GPT."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .connector import build_connector
from .minillm.model import GPT, EOT_TOKEN_ID
from .vision_encoder import VisionTower


class MedicalVLLM(nn.Module):
    """Soft-prefix medical VLM.

    The image tower produces ``[B, P, dim]`` patch tokens; the connector maps
    them to ``[B, K, dim]``; those K vectors are prepended to the token
    embeddings and the GPT runs on ``[K + T]`` positions. No vocabulary
    change, so minillm checkpoints load unchanged.
    """

    def __init__(
        self,
        llm_cfg: dict,
        vision_cfg: dict,
        connector_cfg: dict | None = None,
    ):
        super().__init__()
        self.llm_cfg = dict(llm_cfg)
        self.vision_cfg = dict(vision_cfg)
        self.connector_cfg = dict(connector_cfg or vision_cfg)

        self.llm = GPT(llm_cfg)
        self.vision = VisionTower(vision_cfg)
        self.connector = build_connector(
            self.connector_cfg, dim=int(llm_cfg["emb_dim"])
        )
        self.num_visual_tokens = int(self.connector.num_queries)

        if int(vision_cfg.get("num_visual_tokens", self.num_visual_tokens)) != (
            self.num_visual_tokens
        ):
            raise ValueError(
                "vision.num_visual_tokens and connector.num_queries disagree"
            )

    # ------------------------------------------------------------------ image
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Frozen tower (no grad) + trainable connector -> [B, K, dim]."""
        with torch.no_grad():
            patches = self.vision(pixel_values)
        if patches.size(1) != self.vision.num_patches:
            raise RuntimeError(
                f"expected {self.vision.num_patches} patches, got "
                f"{patches.size(1)}"
            )
        return self.connector(patches)

    # ------------------------------------------------------------------ train
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        ignore_index: int = -100,
    ):
        tok = self.llm.tok_emb(input_ids)
        if pixel_values is not None:
            vis = self.encode_image(pixel_values)
            tok = torch.cat([vis, tok], dim=1)
            if labels is not None:
                pad = labels.new_full(
                    (labels.size(0), vis.size(1)), ignore_index
                )
                labels = torch.cat([pad, labels], dim=1)
        return self.llm.forward_embeds(
            tok,
            y=labels,
            ignore_index=ignore_index,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------ infer
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_id: int = EOT_TOKEN_ID,
    ) -> torch.Tensor:
        """Generate text conditioned on an optional image.

        The visual prefix enters the KV cache exactly once, during prefill.
        Subsequent steps feed only token embeddings. Overflow is a hard error:
        unlike text-only generation we must never slide the window and drop
        the image.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be (batch, seq)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        if max_new_tokens == 0:
            return input_ids

        was_training = self.training
        self.eval()
        try:
            prefix_len = 0
            tok = self.llm.tok_emb(input_ids)
            if pixel_values is not None:
                vis = self.encode_image(pixel_values)
                prefix_len = vis.size(1)
                tok = torch.cat([vis, tok], dim=1)

            prompt_len = tok.size(1)
            if prompt_len + max_new_tokens > self.llm.context_length:
                raise ValueError(
                    f"prompt ({prompt_len} = {prefix_len} visual + "
                    f"{input_ids.size(1)} text) + {max_new_tokens} new "
                    f"tokens exceeds context_length="
                    f"{self.llm.context_length}"
                )

            self.llm.reset_cache()
            logits = self.llm.forward_embeds(tok, use_cache=True)[:, -1, :]
            generated = input_ids
            finished = torch.zeros(
                input_ids.size(0), dtype=torch.bool, device=input_ids.device
            )

            for step in range(max_new_tokens):
                nxt = self.llm._sample_next(logits, temperature, top_k)
                nxt, finished, all_done = self.llm._apply_eos(
                    nxt, finished, eos_id
                )
                generated = torch.cat([generated, nxt], dim=1)
                if all_done or step == max_new_tokens - 1:
                    break
                logits = self.llm.forward_embeds(
                    self.llm.tok_emb(nxt), use_cache=True
                )[:, -1, :]
            return generated
        finally:
            self.llm.reset_cache()
            self.train(was_training)

    # ------------------------------------------------------------------ freeze
    def freeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()

    def unfreeze_llm(self):
        for p in self.llm.parameters():
            p.requires_grad_(True)
        self.llm.train()

    def freeze_connector(self):
        for p in self.connector.parameters():
            p.requires_grad_(False)

    def unfreeze_connector(self):
        for p in self.connector.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_counts(self) -> dict:
        def n(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "llm": n(self.llm),
            "connector": n(self.connector),
            "vision": n(self.vision),
            "trainable": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path, step: int = 0, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format": "medicalvllm-v1",
            "step": int(step),
            "llm": self.llm.state_dict(),
            "connector": self.connector.state_dict(),
            "vision": {
                "encoder": self.vision.encoder_name,
                "encoder_backend": self.vision.backend,
                "image_size": self.vision.image_size,
                "num_visual_tokens": self.num_visual_tokens,
            },
            "config": {
                "llm": self.llm_cfg,
                "vision": self.vision_cfg,
                "connector": self.connector_cfg,
            },
        }
        if extra:
            state.update(extra)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load(
        self,
        path: str | Path,
        load_connector: bool = True,
        strict_llm: bool = True,
        strict_connector: bool = True,
    ) -> dict:
        """Load a MedicalVLLM checkpoint or a plain minillm checkpoint."""
        state = _torch_load(Path(path))
        if "llm" in state:
            llm_state = state["llm"]
        elif "model" in state:
            llm_state = state["model"]
        else:
            raise ValueError(
                f"{path} has neither 'llm' nor 'model' weights"
            )
        self.llm.load_state_dict(llm_state, strict=strict_llm)
        if load_connector and "connector" in state:
            self.connector.load_state_dict(
                state["connector"], strict=strict_connector
            )
        return state

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        llm_cfg: dict,
        vision_cfg: dict,
        connector_cfg: dict | None = None,
        load_connector: bool = True,
    ):
        model = cls(llm_cfg, vision_cfg, connector_cfg)
        model.load(path, load_connector=load_connector)
        return model


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
