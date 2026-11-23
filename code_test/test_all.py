"""GPU smoke tests for the real medicalvllm model (no tiny model, no mocks).

Run::

    python -m code_test.test_all

Uses the real BiomedCLIP tower and the released SFT checkpoint. ~2 minutes on
a T4 after weights are cached.
"""

import sys
import torch
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPT = Path("/kaggle/temp/medicalvllm/sft_model.pt")


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    from model.minillm.model import GPT
    from model.connector import PerceiverResampler, PooledMLPProjector
    from model.medical_vllm import MedicalVLLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}, torch = {torch.__version__}")

    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    llm_cfg = state["model_config"]
    vision_cfg = yaml.safe_load((ROOT / "configs" / "vision.yaml").read_text())
    # The released checkpoint has context 128, so K=16 fits with 112 text tokens.
    vision_cfg["num_visual_tokens"] = 16

    print("\n-- T1: GPT.forward == GPT.forward_embeds (real 768-d GPT) --")
    gpt = GPT(llm_cfg).to(dev).eval()
    ids = torch.randint(0, llm_cfg["vocab_size"], (2, 24), device=dev)
    with torch.no_grad():
        a = gpt(ids)
        b = gpt.forward_embeds(gpt.tok_emb(ids))
    check("text/embed forward parity", torch.equal(a, b))
    check("logits shape", tuple(a.shape) == (2, 24, 50257))

    print("\n-- T2: KV cache parity --")
    gpt.reset_cache()
    with torch.no_grad():
        gpt(ids[:, :12], use_cache=True)
        step = gpt(ids[:, 12:13], use_cache=True)
        full = gpt(ids[:, :13])
    check("KV cache parity", torch.allclose(
        step[:, -1], full[:, -1], atol=1e-4))

    print("\n-- T3: connector shapes / gate / grads --")
    conn = PerceiverResampler(dim=768, num_queries=16, depth=2).to(dev)
    patches = torch.randn(2, 196, 768, device=dev)
    out = conn(patches)
    check("connector output shape", tuple(out.shape) == (2, 16, 768))
    out.sum().backward()
    check("gradient reaches queries", conn.queries.grad is not None)
    check("gradient reaches proj", conn.proj.weight.grad is not None)
    with torch.no_grad():
        conn.gate.zero_()
        zero = conn(patches)
    check("gate=0 => zero prefix", bool((zero == 0).all()))

    print("\n-- T4: pooled baseline --")
    pooled = PooledMLPProjector(dim=768, patch_grid=14, pool=2).to(dev)
    pout = pooled(patches)
    check("pooled output shape", tuple(pout.shape) == (2, 49, 768))

    print("\n-- T5: full wrapper, frozen weights, loss --")
    model = MedicalVLLM(llm_cfg, vision_cfg).to(dev)
    model.load(CKPT, load_connector=False, strict_llm=True)
    model.freeze_llm()
    model.vision.set_frozen(True)
    model.connector.train()
    llm_before = _clone_params(model.llm)
    vis_before = _clone_params(model.vision)

    B, T = 2, 24
    ids = torch.randint(0, llm_cfg["vocab_size"], (B, T), device=dev)
    labels = ids.clone()
    pixels = torch.randn(B, 3, 224, 224, device=dev)
    logits, loss = model(ids, pixels, labels)
    check("logits shape with prefix", tuple(logits.shape) == (B, 16 + T, 50257))
    check("loss is finite", bool(torch.isfinite(loss)))
    loss.backward()
    check("connector has gradients", any(
        p.grad is not None for p in model.connector.parameters()))
    check("LLM frozen (no grads)", not any(
        p.grad is not None for p in model.llm.parameters()))
    check("tower frozen (no grads)", not any(
        p.grad is not None for p in model.vision.parameters()))
    check("LLM weights unchanged", _same_params(llm_before, model.llm))
    check("tower weights unchanged", _same_params(vis_before, model.vision))

    print("\n-- T6: generation + context overflow guard --")
    model.eval()
    with torch.no_grad():
        gen = model.generate(ids[:1, :12], pixels[:1], max_new_tokens=6)
    check("generated length", tuple(gen.shape) == (1, 18))
    long_ids = torch.randint(
        0, llm_cfg["vocab_size"], (1, 120), device=dev
    )
    try:
        with torch.no_grad():
            model.generate(long_ids, pixels[:1], max_new_tokens=20)
        too_long_raised = False
    except ValueError:
        too_long_raised = True
    check("overflow raises ValueError", too_long_raised)

    print("\n-- T7: checkpoint roundtrip --")
    tmp = Path("/kaggle/temp/medicalvllm/_test_ckpt.pt")
    model.save(tmp, step=7)
    restored = MedicalVLLM(llm_cfg, vision_cfg).to(dev)
    restored.load(tmp, load_connector=True)
    check("restored connector matches", all(
        torch.equal(a, b) for a, b in zip(
            model.connector.state_dict().values(),
            restored.connector.state_dict().values(),
        )))
    tmp.unlink()

    print("\nALL TESTS PASSED")


def _clone_params(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _same_params(before, module):
    return all(torch.equal(before[k], v) for k, v in module.state_dict().items())


if __name__ == "__main__":
    main()
