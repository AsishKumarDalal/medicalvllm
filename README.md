# medicalvllm

A medical vision-language model built on the `minillm` decoder
(pretrain → distill → **normal SFT** checkpoint) + a frozen **BiomedCLIP**
image encoder + a trainable Perceiver-resampler connector.

```
image ──► frozen BiomedCLIP ViT-B/16 ──► 196 patch vectors ──► resampler (K=16)
                                                                    │
text ──► GPT-2 BPE ──► minillm GPT ◄── [16 visual | text] ──► answer
```

- **Stage 1**: alignment on ROCOv2 radiology captions (connector only).
- **Stage 2**: medical VQA instruction tuning on PMC-VQA / PathVQA / SLAKE /
  VQA-RAD with 10% text replay (connector + LLM, encoder stays frozen).

Full architecture, data plan, phases, configs, eval targets and risks:
**[experiment/vllm.md](experiment/vllm.md)**.
Coding contract: **[experiment/code.md](experiment/code.md)**.
Connector explained visually: **[experiment/connector.md](experiment/connector.md)**.

> Research project only. Not a medical device. Never use for diagnosis.

## Status

Implemented P1–P6: vendored GPT with `forward_embeds`, vision tower, Perceiver
connector, `MedicalVLLM` wrapper, all five data streams, both trainers, and a
blind-baseline eval harness. GPU smoke tests pass (`python -m code_test.test_all`).
Real runs (P7) are next.

**Important:** the released SFT checkpoint has `context_length = 128` (not 1024),
so the working configuration is **K = 16 visual tokens, 112 text tokens**.
The code is agnostic to these numbers; they come from the checkpoint config.

## Quick start (Kaggle / T4)

```bash
pip install -q open_clip_torch tiktoken         # once
python -m code_test.test_all                     # ~2 min, real GPT + real tower
python -m train.align --steps 200                # Stage 1 smoke run
python -m train.vqa_sft --steps 100              # Stage 2 (needs align checkpoint)
python -m eval.vqa_eval --checkpoint <ckpt> --source vqa_rad --samples 200
```

The SFT checkpoint is resolved from `configs/align.yaml`: local path first,
then `hf_fallback` (`sddfhy546y/minillm:sft_model.pt`). The 86M vision tower is
never saved in checkpoints — it is rebuilt from its name at load time.

## Layout

| Path | Purpose |
| --- | --- |
| `experiment/vllm.md` | the plan (source of truth) |
| `experiment/connector.md` | connector architecture in detail (Mermaid diagrams) |
| `experiment/code.md` | coding plan: modules, interfaces, build order, tests |
| `configs/` | model / vision / align / vqa_sft configs |
| `model/` | vision tower, connector, MedicalVLLM wrapper, vendored minillm GPT |
| `model/minillm/SYNC.md` | source commit + the single `forward_embeds` refactor |
| `data/` | transforms, GPT-2 tokenizer, ROCOv2 + 4 VQA sources, text replay |
| `train/` | Stage 1 (`train.align`) and Stage 2 (`train.vqa_sft`) loops |
| `eval/` | VQA benchmarks + blind baseline |
| `code_test/` | GPU smoke tests on the real model |
| `runbook/` | Kaggle session runbooks |

## Secrets

`HF_TOKEN` via environment / Kaggle Secrets only. `.env` is gitignored and
never read or committed.
