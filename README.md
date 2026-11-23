# medicalvllm

A medical vision-language model built on the `minillm` decoder
(pretrain → distill → normal SFT checkpoint) + a frozen **BiomedCLIP**
image encoder + a trainable Perceiver-resampler connector.

```
image ──► frozen BiomedCLIP ViT-B/16 ──► resampler (K=64 visual tokens)
                                              │
text ──► GPT-2 BPE ──► minillm GPT ◄── [visual prefix | text] ──► answer
```

- **Stage 1**: alignment on ROCOv2 radiology captions (connector only).
- **Stage 2**: medical VQA instruction tuning on PMC-VQA / SLAKE / VQA-RAD /
  PathVQA with 10% text replay (connector + LLM, encoder stays frozen).

Full architecture, data plan, phases, configs, eval targets and risks:
**[experiment/vllm.md](experiment/vllm.md)**.

> Research project only. Not a medical device. Never use for diagnosis.

## Status

Scaffolding + plan. Implementation follows phases P0–P7 in the plan.

## Layout

| Path | Purpose |
| --- | --- |
| `experiment/vllm.md` | the plan (source of truth) |
| `experiment/connector.md` | connector architecture in detail (Mermaid diagrams) |
| `configs/` | model / vision / align / vqa_sft configs |
| `model/` | vision tower, connector, MedicalVLLM wrapper, vendored minillm GPT |
| `data/` | medical dataset loaders, transforms, dedup |
| `train/` | Stage 1 (`train.align`) and Stage 2 (`train.vqa_sft`) loops |
| `eval/` | VQA benchmarks + blind baseline |
| `code_test/` | tiny CPU tests |
| `runbook/` | Kaggle session runbooks |

## Secrets

`HF_TOKEN` via environment / Kaggle Secrets only. `.env` is gitignored and
never read or committed.
