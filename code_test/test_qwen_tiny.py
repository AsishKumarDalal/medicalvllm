"""Offline smoke test for the Qwen LLM backend (tiny random model, no downloads).

Run::

    python -m code_test.test_qwen_tiny

Injects a ~0.1M-parameter Qwen3 and a fake vision tower into ``MedicalVLLM``
to verify the shared interface: embedding injection, loss, gradients, KV cache,
generation, tokenization and checkpoint round-trip.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.tokenizer import (  # noqa: E402
    encode_example,
    get_eot_id,
    get_pad_id,
    set_tokenizer_backend,
)
from model.medical_vllm import MedicalVLLM  # noqa: E402
from model.qwen_llm import QwenLLM  # noqa: E402
from train.common import build_optimizer  # noqa: E402

EOT, PAD = 199, 198


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


class FakeTokenizer:
    eos_token_id = EOT
    pad_token_id = PAD

    def encode(self, text, add_special_tokens=False):
        return [10 + (i % 50) for i, _ in enumerate(text.split())]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


class FakeVision(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = 32  # narrower than the 64-d LLM, like 768 vs 1024
        self.num_patches = 4
        self.encoder_name = "fake"
        self.backend = "fake"
        self.image_size = 224
        self.frozen = True

    def set_frozen(self, frozen=True):
        self.frozen = bool(frozen)

    def forward(self, pixel_values):
        return torch.randn(
            pixel_values.size(0), self.num_patches, self.width,
            device=pixel_values.device,
        )


def tiny_qwen():
    from transformers import AutoModelForCausalLM, Qwen3Config

    cfg = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=200,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return AutoModelForCausalLM.from_config(cfg)


def build(llm_cfg, vision_cfg, qwen):
    with (
        patch("model.medical_vllm.QwenLLM", lambda cfg: qwen),
        patch("model.medical_vllm.VisionTower", FakeVision),
    ):
        return MedicalVLLM(llm_cfg, vision_cfg)


def main():
    set_tokenizer_backend("qwen", model_name="tiny-qwen", tokenizer=FakeTokenizer())
    check("tokenizer backend is qwen", get_eot_id() == EOT and get_pad_id() == PAD)

    ids, labels, triage = encode_example(
        "Instruction:\nDescribe the image.\n\nResponse:\n", "a small nodule", 64,
    )
    check("answer tokens supervised", labels[-1] == EOT and labels[0] == -100)
    check("encoded ids/labels align", len(ids) == len(labels) and triage == "full")

    llm_cfg = {
        "backend": "qwen", "model_name": "tiny-qwen",
        "context_length": 64, "eot_token_id": EOT, "pad_token_id": PAD,
    }
    vision_cfg = {
        "encoder": "fake", "num_visual_tokens": 4,
        "connector": "perceiver", "resampler_depth": 1,
        "output_gate_init": 0.1,
    }
    qwen = QwenLLM(llm_cfg, hf_model=tiny_qwen())
    model = build(llm_cfg, vision_cfg, qwen)
    check("backend wired", model.backend == "qwen" and model.eot_id == EOT)
    check("connector projects vision -> LLM",
          model.connector.proj_in is not None
          and model.connector.proj_in.in_features == 32
          and model.connector.dim == 64)

    B, T = 2, 10
    tok_ids = torch.randint(0, 200, (B, T))
    labels = tok_ids.clone()
    labels[:, :3] = -100
    pixels = torch.randn(B, 3, 224, 224)
    logits, loss = model(tok_ids, pixels, labels)
    check("logits include visual prefix", tuple(logits.shape) == (B, 4 + T, 200))
    check("loss finite", bool(torch.isfinite(loss)))
    loss.backward()
    check("LLM receives gradients", any(
        p.grad is not None for p in qwen.hf.parameters()))
    check("connector receives gradients", any(
        p.grad is not None for p in model.connector.parameters()))

    # KV cache must match the full (uncached) forward with explicit positions.
    qwen.eval()
    with torch.no_grad():
        full = qwen(tok_ids[:1])
        emb = qwen.tok_emb(tok_ids[:1])
        qwen.reset_cache()
        pre = qwen.forward_embeds(emb[:, :5], use_cache=True)
        step = qwen.forward_embeds(emb[:, 5:6], use_cache=True)
    check("KV cache parity", torch.allclose(
        step[:, -1], full[:, 5], atol=1e-4))

    with torch.no_grad():
        gen = model.generate(tok_ids[:1, :4], pixels[:1], max_new_tokens=3)
    check("generation length", tuple(gen.shape) == (1, 7))

    # Regression: Adafactor squares eps1 internally. With the old eps1=1e-30
    # the square (1e-60) underflows to 0 in fp32, rsqrt(0)=inf, and
    # inf * 0-gradient turns zero rows into NaN (dead MLP neurons, unused
    # biases). build_optimizer must keep zero-gradient parameters finite.
    for shape in ((8,), (4, 8)):
        p = torch.nn.Parameter(torch.zeros(shape))
        opt = build_optimizer([{"params": [p], "lr": 1e-4}], "adafactor")
        p.grad = torch.zeros_like(p)
        opt.step()
        check(f"adafactor zero-grad {shape} stays finite",
              bool(torch.isfinite(p).all()))

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.pt"
        model.save(path, step=7)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        check("backend recorded in checkpoint",
              saved["config"]["llm"]["backend"] == "qwen")
        qwen2 = QwenLLM(saved["config"]["llm"], hf_model=tiny_qwen())
        restored = build(saved["config"]["llm"], vision_cfg, qwen2)
        restored.load(path, load_connector=True, strict_llm=True)
        check("llm weights round-trip", all(
            torch.equal(a, b) for a, b in zip(
                qwen.state_dict().values(), qwen2.state_dict().values())))
        check("connector weights round-trip", all(
            torch.equal(a, b) for a, b in zip(
                model.connector.state_dict().values(),
                restored.connector.state_dict().values())))

    set_tokenizer_backend("gpt2")
    print("ALL QWEN TINY TESTS PASSED")


if __name__ == "__main__":
    main()
