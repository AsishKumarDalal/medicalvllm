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
python -m code_test.test_state                   # checkpoint/optimizer state tests
python -m train.align --steps 200                # Stage 1 smoke run
python -m train.vqa_sft --steps 100              # Stage 2 (needs align checkpoint)
python -m eval.vqa_eval --checkpoint <ckpt> --source vqa_rad --samples 200
```

**Resume / HF flags** (both trainers): no flag = local-only, `--hf` = resume
from and upload to `medical_vllm/phaseN/` on HF, `--no-hf` = explicitly disable
HF, `--new` = random LLM weights / global step 0. Nothing is uploaded unless
`--hf` is passed. `step` is global across phases — Phase 2 continues from
Phase 1's step, token count, image count and optimizer state. See the
[runbook](runbook/README.md) for the full layout.

**Qwen option:** add `--qwen` to both trainers to replace the GPT-2 decoder with
`Qwen/Qwen3-0.6B`. This also swaps the tokenizer (HF Qwen2Tokenizer), the EOT
and pad ids, and adds a 768→1024 input projection to the connector. GPT-2 SFT
weights are ignored in that mode; a pretrained Qwen decoder is used instead.
Checkpoints record the backend, so eval/sampling decode with the right
tokenizer. See [`train/README.md`](train/README.md#qwen3-backend---qwen).

The SFT checkpoint is resolved from `configs/align.yaml`: local path first,
then `hf_fallback` (`sddfhy546y/minillm:sft_model.pt`). The 86M vision tower is
never saved in checkpoints — it is rebuilt from its name at load time.

## Reading guide — start here

Hold one sentence in your head: **the vision tower writes 196 sticky notes,
the connector rewrites them as 16 summary cards, and the GPT reads those cards
plus your question.**

Read these five files in order (~20 min):

| # | File | What to look at |
| --- | --- | --- |
| 1 | `experiment/connector.md` §0–2 | the picture, before any code |
| 2 | `model/vision_encoder.py` | `VisionTower.forward` (line 83) — pixels → 196×768 |
| 3 | `model/connector.py` | `PerceiverResampler.forward` (line 59) — 196 → 16 |
| 4 | `model/minillm/model.py` | `forward` (118) vs `forward_embeds` (127) — the ONE decoder change |
| 5 | `model/medical_vllm.py` | `forward` (62) and `generate` (88) — how the three pieces join |

Then follow one image end-to-end:

```
data/transforms.py  →  model/vision_encoder.py:83  →  model/connector.py:59
                   →  model/medical_vllm.py:50 (encode_image)
                   →  model/medical_vllm.py:62 (cat + labels -100)
                   →  model/minillm/model.py:127 (forward_embeds)
```

To *see* it run rather than just read it:

```bash
python -m code_test.test_all      # 20 assertions, prints [PASS] for each stage
python -m train.align --steps 2   # watch a real training step happen
```

## Layout

| Path | Purpose |
| --- | --- |
| `experiment/vllm.md` | the plan (source of truth) |
| `experiment/connector.md` | connector architecture in detail (Mermaid diagrams) |
| `experiment/code.md` | coding plan: modules, interfaces, build order, tests |
| `experiment/data_training.md` | how one dataset row becomes a gradient; both stages |
| `configs/` | model / vision / align / vqa_sft configs |
| `model/` | vision tower, connector, MedicalVLLM wrapper, vendored minillm GPT |
| `model/minillm/SYNC.md` | source commit + the single `forward_embeds` refactor |
| `data/` | transforms, GPT-2 tokenizer, ROCOv2 + 4 VQA sources, text replay |
| `train/` | Stage 1 (`train.align`) and Stage 2 (`train.vqa_sft`) loops — see [`train/README.md`](train/README.md) for all params, flags and HF sync |
| `eval/` | VQA benchmarks + blind baseline |
| `code_test/` | GPU smoke tests on the real model + checkpoint-state tests |
| `runbook/` | Kaggle session runbooks |

## Secrets

`HF_TOKEN` via environment / Kaggle Secrets only. `.env` is gitignored and
never read or committed.
