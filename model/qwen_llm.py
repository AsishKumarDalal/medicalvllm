"""Qwen3 decoder wrapped in the tiny interface ``MedicalVLLM`` expects.

``MedicalVLLM`` was built around minillm's GPT, which exposes ``tok_emb``,
``forward_embeds``, ``context_length``, a KV cache and two sampling helpers.
This adapter provides the same surface on top of a HuggingFace
``Qwen3ForCausalLM`` so the vision/connector code does not need to know which
LLM is underneath.

The training loss is deliberately computed here (not via HF's ``labels=``)
so it matches the GPT path exactly: logits at position *t* are compared to the
label at position *t*, with ``-100`` masking the prompt and the visual prefix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM


def _transformers_major() -> int:
    import transformers

    head = transformers.__version__.split(".")[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    return int(digits or 0)


class QwenLLM(nn.Module):
    """HF causal LM with the same embedding-injection API as minillm's GPT."""

    def __init__(self, cfg: dict, hf_model=None):
        super().__init__()
        self.backend = "qwen"
        self.model_name = str(cfg.get("model_name", "Qwen/Qwen3-0.6B"))

        if hf_model is not None:
            # Offline tests inject a tiny randomly initialised Qwen3 model.
            self.hf = hf_model
        else:
            dtype = str(cfg.get("dtype", "float32")).lower()
            resolved = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "auto": "auto",
            }.get(dtype, torch.float32)
            kwargs = {"low_cpu_mem_usage": True} if resolved != "auto" else {}
            if resolved != "auto":
                # transformers 5 renamed torch_dtype -> dtype; using the old
                # name there prints a deprecation warning on every run.
                key = "dtype" if _transformers_major() >= 5 else "torch_dtype"
                kwargs[key] = resolved
            try:
                self.hf = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **kwargs
                )
            except TypeError:  # unexpected build: try the other kwarg name
                kwargs.pop("dtype", None)
                kwargs.pop("torch_dtype", None)
                if resolved != "auto":
                    alt = "torch_dtype" if key == "dtype" else "dtype"
                    kwargs[alt] = resolved
                self.hf = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **kwargs
                )

        hcfg = self.hf.config
        self.hidden_size = int(hcfg.hidden_size)
        self.vocab_size = int(hcfg.vocab_size)
        limit = int(getattr(hcfg, "max_position_embeddings", 40960))
        want = int(cfg.get("context_length") or limit)
        self.context_length = max(1, min(want, limit))
        self.eot_id = int(cfg.get("eot_token_id", 151645))
        self.pad_id = int(cfg.get("pad_token_id", 151643))

        # Plain attribute: never enters state_dict(), so checkpoints are stable.
        self._past = None

        if bool(cfg.get("gradient_checkpointing", False)):
            try:
                self.hf.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.hf.gradient_checkpointing_enable()

    # -------------------------------------------------------------- embedding
    @property
    def tok_emb(self):
        """Token embedding layer (property: avoids duplicate state_dict keys)."""
        return self.hf.get_input_embeddings()

    # ------------------------------------------------------------------- cache
    def reset_cache(self):
        self._past = None

    def clear_cache(self):
        self._past = None

    # ----------------------------------------------------------------- forward
    def forward_embeds(
        self,
        x_embeds: torch.Tensor,
        y: torch.Tensor | None = None,
        ignore_index: int = -100,
        use_cache: bool = False,
    ):
        """Run the decoder on pre-computed embeddings (visual prefix included)."""
        if x_embeds.dim() != 3:
            raise ValueError(
                f"x_embeds must be (batch, seq_len, dim), got "
                f"{tuple(x_embeds.shape)}"
            )
        batch, seq_len, _ = x_embeds.shape
        past_len = self._past.get_seq_length() if self._past is not None else 0
        if past_len + seq_len > self.context_length:
            raise ValueError(
                f"positions [{past_len}, {past_len + seq_len}) exceed "
                f"context_length={self.context_length}"
            )

        # Explicit positions are required: HF derives them from inputs_embeds
        # only when no cache is present, and generation reuses the cache.
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=x_embeds.device
        ).unsqueeze(0).expand(batch, -1)

        out = self.hf(
            inputs_embeds=x_embeds,
            position_ids=position_ids,
            past_key_values=self._past,
            use_cache=use_cache,
            return_dict=True,
        )
        if use_cache:
            self._past = out.past_key_values
        logits = out.logits

        if y is None:
            return logits
        # With fp16 weights the logits are fp16 too; log_softmax over 151,936
        # classes can overflow to inf and produce a NaN loss. Cross-entropy in
        # fp32 is cheap and keeps the gradient finite.
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=ignore_index,
        )
        return logits, loss

    def forward(
        self,
        in_idx: torch.Tensor,
        y: torch.Tensor | None = None,
        ignore_index: int = -100,
        use_cache: bool = False,
    ):
        return self.forward_embeds(
            self.tok_emb(in_idx),
            y=y,
            ignore_index=ignore_index,
            use_cache=use_cache,
        )

    # ---------------------------------------------------------------- sampling
    def _sample_next(self, logits, temperature, top_k):
        """Top-k filter, then temperature sample or greedy argmax."""
        if top_k is not None:
            k = min(int(top_k), logits.size(-1))
            if k <= 0:
                raise ValueError("top_k must be > 0")
            top_logits, _ = torch.topk(logits, k)
            min_val = top_logits[:, -1]
            logits = logits.masked_fill(
                logits < min_val.unsqueeze(-1), float("-inf")
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _apply_eos(self, idx_next, finished, eos_id):
        """Update finished rows and pad them with EOS."""
        if eos_id is None:
            return idx_next, finished, False
        finished = finished | (idx_next.squeeze(-1) == eos_id)
        if finished.any():
            idx_next = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(idx_next, eos_id),
                idx_next,
            )
        return idx_next, finished, bool(finished.all())
