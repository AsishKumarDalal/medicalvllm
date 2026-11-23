# MedicalVLLM — medical vision-language model plan

Goal: build a **medical visual LLM** on top of the existing `minillm` decoder.
A pretrained, **frozen** medical image encoder (BiomedCLIP ViT, CLIP fallback)
produces visual features; a small trainable **connector** compresses them into
a soft visual prefix; the `minillm` GPT (pretrain → distill → **normal SFT**
checkpoint — *not* the tool-calling one) consumes `[visual prefix | text]` and
generates answers about medical images (X-ray, CT, MRI, pathology, charts).

Training follows the LLaVA recipe adapted to a ~100M decoder and Kaggle 2×T4:

```
Stage 1  alignment      train connector only   (LLM frozen, ViT frozen)
Stage 2  medical SFT    train connector + LLM  (ViT frozen)
Stage 3  optional       unfreeze top ViT blocks / encoder ablation
```

This document is the single source of truth for the repo. Everything below is
written against the **actual** minillm code (`../minillm/model/model.py`,
`configs/model.yaml`), not the older 128-context draft in
`minillm/experiments/plan/vision_model.md`.

---

## 1. Starting assets

| Asset | Value | Source |
| --- | --- | --- |
| Decoder | dense `GPT`, ~102.5M params | `minillm/model/model.py` |
| Checkpoint lineage | OpenWebText pretrain → logit distill → **Dolly SFT** | `sft_checkpoints/sft_model.pt`, HF `sddfhy546y/minillm` |
| Explicitly NOT used | tool-calling SFT / DPO checkpoints | keeps chat format clean, no tool tokens |
| Embedding dim | 768 | `configs/model.yaml` |
| Context length | **1024** | `configs/model.yaml` (updated from 128) |
| Layers / heads | 9 / 12 (head_dim 64) | same |
| Tokenizer | GPT-2 BPE, vocab 50,257, EOT = 50256 | `pretrain/data.py` |
| Positions | learned **absolute** `pos_emb`, no RoPE | `model/model.py:23` |
| Loss masking | `ignore_index=-100` | `model/model.py:171` |
| Generation | KV-cached prefill + decode, sliding-window fallback | `model/model.py:234` |
| Weight tying | `out_head.weight is tok_emb.weight` | `model/model.py:41` |
| Hardware | Kaggle 2× T4 16GB (or 1×), 12h sessions | runbooks |

Consequences of these facts:

- Context 1024 gives real room: with **K = 64** visual tokens we still have
  960 positions for question + answer. No need for the aggressive K=16 of the
  old draft.
- Learned absolute positions mean the visual prefix simply occupies positions
  `0..K-1` and text starts at position `K`. The KV cache **cannot slide**;
  image-conditioned generation must keep the prefix in the cached window
  (disable the sliding-window fallback for image inputs).
- Weight tying means we must **not** add new vocab tokens for images. The
  image is a soft prefix in embedding space only — the tokenizer, output head
  and all existing checkpoints stay untouched.
- The SFT checkpoint already knows the
  `Instruction:/Context:/Response:` format; we keep a compatible template so
  Stage 2 starts from behavior the model already has.

## 2. Architecture

```
                 ┌────────────────────────────┐
  image 224×224  │  BiomedCLIP ViT-B/16 tower │  FROZEN (86M)
 ───────────────►│  microsoft/BiomedCLIP-     │
                 │  PubMedBERT_256-vit_base_  │──► patch feats [B, 196, 768]
                 │  patch16_224 (open_clip)   │    (drop CLS)
                 └────────────────────────────┘
                                │  no_grad, eval()
                                ▼
                 ┌────────────────────────────┐
                 │  Connector: Perceiver      │  TRAINABLE (~14.8M)
                 │  resampler, K=64 learned   │
                 │  queries, 2 cross-attn     │──► visual prefix [B, 64, 768]
                 │  layers + LN + MLP proj    │
                 └────────────────────────────┘
                                │
        text ids ──► tok_emb ──►│ concat on seq dim
                                ▼
                 ┌────────────────────────────┐
                 │  minillm GPT (102.5M)      │  Stage 1 FROZEN
                 │  x = [vis(64) | text(T)]   │  Stage 2 TRAINABLE
                 │  + pos_emb[0 .. 64+T-1]    │
                 │  9 blocks, causal attn     │──► logits [B, 64+T, 50257]
                 └────────────────────────────┘
                                │
                     CE loss only on answer tokens + EOT
                     (visual prefix & prompt labels = -100)
```

### 2.1 Vision tower — why BiomedCLIP, not vanilla CLIP

- **Primary: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`**
  (loaded via `open_clip`). Contrastively pretrained on **PMC-15M**
  (15M biomedical figure–caption pairs from PubMed Central). Its ViT-B/16
  tower outputs 768-d features — an exact match for minillm's `emb_dim`, and
  it already understands radiology/pathology visual vocabulary far better
  than web-image CLIP. Hidden width 768, 196 patches + CLS at 224px.
- **Fallback / ablation: `openai/clip-vit-base-patch32`** (49 patches, 768-d).
  Cheaper, useful to measure how much the medical pretraining actually buys.
- Later ablations only: PubMedCLIP (RN50), MedCLIP, DINOv2 (256 patches —
  must be resampled, never prepended raw), SigLIP.
- The tower is **frozen in Stages 1–2**: `eval()`, `requires_grad_(False)`,
  forward under `torch.no_grad()`. It never enters the backward graph, so it
  costs ~0.35GB VRAM (fp16) and zero optimizer state.
- Preprocessing must use **BiomedCLIP's own mean/std** (OpenAI CLIP stats:
  mean `(0.48145466, 0.4578275, 0.40821073)`,
  std `(0.26862954, 0.26130258, 0.27577711)`), resize 224, center crop.
  Grayscale X-rays are replicated to 3 channels. No medical-specific
  augmentation in Stage 1; light flips/rotation (±10°) allowed in Stage 2
  only if eval shows overfitting — never horizontal-flip laterality-sensitive
  chest X-rays (left/right markers are diagnostically meaningful).

### 2.2 Connector — Perceiver resampler (K = 64)

Full connector architecture, tensor shapes, parameter table and Mermaid
diagrams: **[connector.md](connector.md)**.

Two candidates, resampler is the default:

| Option | Params | Pros | Cons |
| --- | --- | --- | --- |
| **Perceiver resampler** (default) | ~14.8M | fixed K regardless of encoder patch count; encoder-agnostic ablations; strong at 100M-decoder scale (Flamingo-style) | more code, slightly slower |
| Pooled MLP projector (LLaVA-1.5 style, 2×2 avg-pool → 49 tokens → 2-layer MLP) | ~2.4M | simplest possible baseline | K tied to patch grid; less compression control |

Resampler spec:

- `queries`: `nn.Parameter [64, 768]`, init N(0, 0.02).
- 2 × {cross-attn(queries ← patch feats, 12 heads) + LN + FFN(768→3072→768, GELU) + LN}, pre-norm residual.
- Final `LayerNorm(768)` + `Linear(768, 768)` projection into decoder space.
- Output scale matters: minillm embeddings live at std ≈ 0.02–0.3 after
  training. Add a learnable scalar gate initialized small (0.1) on the output
  so the prefix does not swamp the residual stream at step 0.
- K configurable: 32 / 64 / 128 for ablation. Default 64.

This connector is not novel: it is Flamingo's **Perceiver Resampler**
(64 latent queries, frozen encoder) adapted to a 100M decoder; BLIP-2's
Q-Former is the closest training-recipe analogue, and Med-Flamingo already
uses it in the medical domain. Citations and positioning:
**[references.md](references.md)**.

### 2.3 Decoder integration (soft prefix)

Wrapper class `MedicalVLLM(nn.Module)` owns `vision`, `connector`, `llm`
(the vendored minillm `GPT`). We do **not** fork `GPT.forward`; instead the
wrapper adds a thin embedding-injection path:

1. `vis = connector(vision(pixel_values))` → `[B, K, 768]`.
2. `tok = llm.tok_emb(input_ids)` → `[B, T, 768]`.
3. `x = cat([vis, tok], dim=1) + llm.pos_emb(arange(K+T))`, then
   `llm.drop_emb`, blocks, `final_norm`, `out_head` — i.e. a new
   `GPT.forward_embeds(x_embeds, ...)` method that shares everything with
   `forward()` except the embedding lookup. One small refactor in the
   vendored copy; `forward(in_idx)` delegates to it, so all existing tiny
   tests keep passing unchanged.
4. Labels: caller passes text labels `[B, T]`; wrapper prepends K columns of
   `-100`. Logits at visual positions are computed but never supervised.
5. `generate(pixel_values, prompt_ids, ...)`: prefill cache once with
   `[vis | prompt]` (positions `0..K+T-1`), then decode token-by-token.
   The sliding-window fallback is **disabled** when an image is present —
   raise a clear error if `K + prompt + max_new_tokens > context_length`
   instead of silently dropping the image.
6. Checkpoints: save `connector.*` + `llm.*` only (frozen tower is re-loaded
   from HF by name). Loading the plain SFT checkpoint into `llm` is
   `strict=True`-compatible because the GPT module itself is unchanged.

### 2.4 Parameter budget

| Component | Params | Stage 1 | Stage 2 | Stage 3 (opt) |
| --- | --- | --- | --- | --- |
| BiomedCLIP ViT-B/16 | 86M | frozen | frozen | top 2 blocks unfrozen (~14M) |
| Connector (K=64, 2 layers) | ~14.8M | **train** | **train** | train |
| minillm GPT | 102.5M | frozen | **train** | train |

Stage 1 optimizer state ≈ 14.8M × 12 bytes ≈ 178MB. Stage 2 ≈ 117M × 12 ≈ 1.4GB
(fp32 AdamW) — both fit easily on one T4 next to activations.

## 3. Data plan — medical datasets

All streamed via `datasets` (already the minillm pattern). **Verify every HF
id at P0** — mirrors move; each loader gets a fallback id list.

| Dataset | HF id (verify) | Size | Use | Stage |
| --- | --- | --- | --- | --- |
| **ROCOv2** (radiology captions) | `eltorio/ROCOv2-radiology` | ~60k train image–caption pairs | alignment (captioning) | 1 |
| **PMC-VQA** | `xmcmic/PMC-VQA` | ~227k QA / 149k figures, multi-choice | medical VQA instruction tuning | 2 |
| **SLAKE** (en) | `BoKelwen/SLAKE` (fallback: official zip) | 14k QA / 642 images, radiology | VQA train + **held-out eval** | 2 / eval |
| **VQA-RAD** | `flaviagiammarino/vqa-rad` | 3.5k QA / 315 radiology images | small, high-quality VQA; **eval only** (test split) + train split in Stage 2 | 2 / eval |
| **PathVQA** | `flaviagiammarino/path-vqa` | ~32k QA / 4.9k pathology images | pathology coverage | 2 |
| **PubMedVision** (subset) | `FreedomIntelligence/PubMedVision` | 500k+ medical multimodal instructions | optional scale-up if Stage 2 underfits | 2+ |
| Medical text-only replay | `medalpaca/medical_meadow_medqa` or Dolly replay | 10–20% of Stage 2 batches | anti-forgetting | 2 |
| IU X-Ray (OpenI) | mirror, verify | ~7k images / 3.9k reports | future: report generation | 4 |
| MIMIC-CXR / CheXpert | **gated** (PhysioNet credentialed) | — | out of scope for now | — |

Notes:

- ROCOv2 contains non-radiology figures (charts, photos); filter with its
  modality metadata to keep radiology-only for alignment, keep a small
  chart subset for robustness.
- PMC-VQA images ship as a zip + csv of QA; loader must join on image file
  name and cache decoded images under `/kaggle/temp` (scratch disk), never
  `/kaggle/working` (20GB output limit).
- Multi-choice QA (PMC-VQA) is converted to open-ended text:
  `Question: ... Choices: A. ... B. ... Answer:` → target `A. <choice text>`
  (letter + text so both eval styles work).
- Deduplicate eval images against train (SLAKE/VQA-RAD images recur in
  PMC-VQA figures — hash images with md5 and drop collisions).

### 3.1 Templates and label masking

Stage 1 (captioning), image prefix + plain caption:

```
[64 visual vectors]  "Chest radiograph shows ..."  EOT
labels: -100 × 64     supervised                  supervised
```

Stage 2 (VQA), keeps the SFT checkpoint's familiar shape:

```
[64 visual vectors]  "Instruction:\nAnswer the question about the medical image.\n\nQuestion: Is there a pleural effusion?\n\nResponse:"  " Yes, a small left-sided effusion."  EOT
labels: -100 × 64     -100 (prompt)                                                                                                          supervised                          supervised
```

- Tokenizer unchanged (GPT-2 BPE). No `<image>` placeholder token — the
  prefix is purely in embedding space.
- Triage per example (mirrors `sft/sft_data.py`): budget = 1024 − K.
  1. full question + answer fits → keep;
  2. else truncate question to fit (keep answer whole);
  3. else drop. Log `seen/kept/truncated/dropped` stats.
- Right-pad with EOT, labels `-100` on padding (causal attention makes this
  safe, same as minillm SFT).

### 3.2 Image feature compute strategy

- Default: encode **on the fly** under `no_grad` in fp16 (~5–10ms/image on
  T4). At batch 32 that is <0.3s/step — acceptable, keeps disk usage ~0.
- Optional `--cache-features`: precompute 196×768 fp16 patch features to a
  memmap under `/kaggle/temp` (~300KB/image → 60k images ≈ 18GB — fits
  scratch, NOT working). Worth it only for multi-epoch Stage 2 reruns.
- DataLoader workers do PIL decode + transforms; GPU does tower + connector.

## 4. Training curriculum

### Stage 0 — baselines & sanity (no training)

1. Load SFT checkpoint, run existing text prompts → record behavior.
2. **Blind baseline**: run the Stage-2 eval set with a *zero* visual prefix
   (or shuffled images) later, to prove the model actually uses the image.
   Any VQA gain must beat the blind number — language priors alone score
   surprisingly high on yes/no medical questions.
3. Throughput baseline: text-only tokens/s on 2×T4 (reuse minillm
   `experiments/bench_e2e.py` style).

### Stage 1 — alignment (connector only)

| Setting | Value |
| --- | --- |
| Data | ROCOv2 radiology captions, streamed |
| Trainable | connector (~14.8M) only |
| LR | 1e-4, cosine, warmup 200 steps |
| Batch | 32 × grad_accum 2 (effective 64) |
| Steps | ~2 epochs ≈ 4k steps (or `--hours 4` cap) |
| Precision | tower fp16, connector/LLM fp32 (T4: no bf16) |
| Success signal | caption loss < ~3.0 and qualitative samples name the modality/body part; frozen-LLM text perplexity unchanged by construction |

Why freeze the LLM here: the connector must learn to speak minillm's
embedding language before the LLM is allowed to move. Unfreezing both from
random-connector init destroys the SFT behavior (known LLaVA failure mode).

### Stage 2 — medical instruction tuning (connector + LLM)

| Setting | Value |
| --- | --- |
| Data mix | PMC-VQA 55% · PathVQA 20% · SLAKE-train 10% · VQA-RAD-train 5% · **text-only replay 10%** |
| Trainable | connector + full GPT (102.5M) |
| LR | LLM 1e-5, connector 5e-5 (param groups), cosine, warmup 100 |
| Batch | 16 × grad_accum 4 (effective 64), seq ≤ 1024 but typically ~200 |
| Steps | 1–2 epochs of the mix ≈ 6–10k steps, `--hours 8` cap |
| Dropout | keep 0.1 (small data, real overfit risk) |
| Eval | every 1000 steps: 200-sample VQA-RAD dev accuracy + fixed image samples logged |
| Checkpoints | hourly atomic save + optional HF sync (reuse minillm trainer helpers: detach runner, SIGTERM-safe final sync, plots) |

Alternative if full FT overfits or forgets: **LoRA r=16, α=32** on attention
`qkv`/`out_proj` + FFN of the GPT (~1.5M trainable), connector still full.
Decide from the first 1k-step eval curve; config flag `lora.enabled`.

Anti-forgetting measures: 10% text-only medical/Dolly replay; low LLM LR;
early stop on combined (VQA acc ↑, text ppl ↛) criterion.

### Stage 3 — optional encoder unfreeze / ablations

Only after Stage 2 acceptance passes:

- Unfreeze **top 2 ViT blocks + final LN** at LR 1e-6 (100× lower than LLM).
  Watch for collapse: if VQA dev drops for 2 evals, revert.
- Ablation grid (same budget each): BiomedCLIP vs CLIP-B/32 · K ∈ {32, 64,
  128} · resampler vs pooled-MLP · blind baseline. One table in
  `experiment/results.md`.

## 5. Repo layout

```
medicalvllm/
├── experiment/
│   ├── vllm.md              ← this plan
│   └── results.md           ← ablation/eval tables (created at P5)
├── configs/
│   ├── model.yaml           ← copied verbatim from minillm (768/1024/9L)
│   ├── vision.yaml          ← encoder name, K, freeze flags, image size
│   ├── align.yaml           ← Stage 1 hyperparams + data
│   └── vqa_sft.yaml         ← Stage 2 hyperparams + data mix + lora flag
├── model/
│   ├── minillm/             ← vendored model.py + model_parts.py (+ the
│   │                          forward_embeds refactor); keeps repo standalone
│   ├── vision_encoder.py    ← open_clip/HF tower loader, frozen, fp16
│   ├── connector.py         ← PerceiverResampler + PooledMLPProjector
│   └── medical_vllm.py      ← MedicalVLLM wrapper: forward, generate, save/load
├── data/
│   ├── transforms.py        ← BiomedCLIP preprocessing, grayscale→RGB
│   ├── captions.py          ← ROCOv2 streaming loader (Stage 1)
│   ├── medical_vqa.py       ← PMC-VQA / SLAKE / VQA-RAD / PathVQA loaders,
│   │                          unified {image, question, answer} schema, triage
│   ├── text_replay.py       ← text-only medical/Dolly batches
│   └── dedup.py             ← md5 image hashing, eval-leak filter
├── train/
│   ├── common.py            ← checkpoint/plot/HF-sync/detach helpers ported
│   │                          from minillm pretrain/train.py
│   ├── align.py             ← Stage 1 loop  (python -m train.align)
│   └── vqa_sft.py           ← Stage 2 loop  (python -m train.vqa_sft)
├── eval/
│   ├── vqa_eval.py          ← closed-form acc, open token-recall/F1, blind mode
│   └── samples.py           ← fixed image+question qualitative panel
├── code_test/               ← tiny CPU tests (see §8)
├── runbook/kaggle.md        ← session-by-session Kaggle instructions
├── agent.md                 ← validation rules (tiny CPU model, .env hygiene)
├── requirements.txt
└── README.md
```

Vendoring decision: copy `model.py`/`model_parts.py` from `../minillm/model`
into `model/minillm/` once, then evolve independently (the
`forward_embeds` refactor). A `SYNC.md` note records the source commit hash.
No `sys.path` hacks across repos — Kaggle mounts each repo separately and
the notebook must be able to run `medicalvllm` alone.

## 6. Implementation phases

| Phase | Deliverable | Files | Est. |
| --- | --- | --- | --- |
| **P0** | dataset id verification script + download smoke (10 samples each), throughput baseline | `data/*` stubs, `runbook/kaggle.md` | 1h |
| **P1** | vision tower + connector, tiny tests green | `model/vision_encoder.py`, `model/connector.py`, `code_test/test_connector_tiny.py` | 2h |
| **P2** | vendored GPT + `forward_embeds` + wrapper forward/generate; KV-cache parity test | `model/minillm/*`, `model/medical_vllm.py`, `code_test/test_prefix_kv_tiny.py` | 3h |
| **P3** | data loaders + triage + dedup, tiny data tests | `data/*`, `code_test/test_med_data_tiny.py` | 3h |
| **P4** | Stage 1 trainer, 30-min smoke run, then 4h alignment run | `train/align.py`, `train/common.py` | 1h code + 4h GPU |
| **P5** | eval harness + blind baseline; Stage 2 trainer, 8h run | `eval/*`, `train/vqa_sft.py` | 2h code + 8h GPU |
| **P6** | ablations (K, encoder, projector, LoRA), `results.md` | `eval/`, configs | 2 GPU sessions |
| **P7** | docs, HF upload (`medicalvllm_latest.pt`), demo notebook | README, runbook | 1h |

### 6.1 Key code sketches

`model/connector.py`:

```python
class PerceiverResampler(nn.Module):
    def __init__(self, dim=768, num_queries=64, depth=2, heads=12):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                ln_q=nn.LayerNorm(dim), ln_kv=nn.LayerNorm(dim),
                attn=nn.MultiheadAttention(dim, heads, batch_first=True),
                ln_ff=nn.LayerNorm(dim),
                ff=nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(),
                                 nn.Linear(4*dim, dim)),
            )) for _ in range(depth)])
        self.ln_out = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, feats):                    # [B, P, 768]
        q = self.queries.unsqueeze(0).expand(feats.size(0), -1, -1)
        for L in self.layers:
            kv = L["ln_kv"](feats)
            q = q + L["attn"](L["ln_q"](q), kv, kv, need_weights=False)[0]
            q = q + L["ff"](L["ln_ff"](q))
        return self.gate * self.proj(self.ln_out(q))   # [B, K, 768]
```

`model/medical_vllm.py` (core paths):

```python
class MedicalVLLM(nn.Module):
    def forward(self, input_ids, pixel_values=None, labels=None):
        tok = self.llm.tok_emb(input_ids)
        if pixel_values is not None:
            with torch.no_grad():
                feats = self.vision(pixel_values)        # [B, P, 768]
            vis = self.connector(feats)                  # [B, K, 768]
            tok = torch.cat([vis, tok], dim=1)
            if labels is not None:
                pad = labels.new_full((labels.size(0), vis.size(1)), -100)
                labels = torch.cat([pad, labels], dim=1)
        return self.llm.forward_embeds(tok, y=labels)    # logits[, loss]

    @torch.no_grad()
    def generate(self, prompt_ids, pixel_values=None, max_new_tokens=128,
                 temperature=0.0, top_k=None):
        # prefill [vis | prompt] once via forward_embeds(use_cache=True),
        # then decode ids only; raise if K + T + max_new > context_length.
        ...
```

`GPT.forward_embeds` is the existing `forward` body starting at the
`x = token_embedding + pos_embeds` line, parameterized on `x_embeds` instead
of `in_idx`; `forward(in_idx)` becomes
`forward_embeds(self.tok_emb(in_idx), ...)`. Positions, cache checks, loss
masking all stay identical — this is the *only* change to the vendored GPT.

### 6.2 Configs

`configs/vision.yaml`:

```yaml
encoder: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
encoder_backend: open_clip          # or "hf" for openai/clip-vit-base-patch32
image_size: 224
freeze_encoder: true
connector: perceiver                # perceiver | pooled_mlp
num_visual_tokens: 64
resampler_depth: 2
```

`configs/align.yaml` (Stage 1):

```yaml
llm_checkpoint: ../minillm/sft_checkpoints/sft_model.pt   # normal SFT, NOT tool-calling
hf_fallback: {repo: sddfhy546y/minillm, filename: sft_model.pt}
data: {dataset: eltorio/ROCOv2-radiology, radiology_only: true, shuffle_buffer: 500}
freeze_llm: true
batch_size: 32
grad_accum_steps: 2
learning_rate: 1.0e-4
warmup_steps: 200
epochs: 2
hours: 4
checkpoint_dir: checkpoints/align
out_dir: outs/align
hf: {enabled: false}
```

`configs/vqa_sft.yaml` (Stage 2):

```yaml
init_from: checkpoints/align/medicalvllm_align.pt
data:
  mix: {pmc_vqa: 0.55, path_vqa: 0.20, slake: 0.10, vqa_rad: 0.05, text_replay: 0.10}
  max_seq_budget: 960          # 1024 - K
freeze_llm: false
lora: {enabled: false, r: 16, alpha: 32}
lr_llm: 1.0e-5
lr_connector: 5.0e-5
batch_size: 16
grad_accum_steps: 4
epochs: 2
hours: 8
eval: {every_steps: 1000, dev_samples: 200}
checkpoint_dir: checkpoints/vqa
out_dir: outs/vqa
hf: {enabled: false, filename: medicalvllm_latest.pt}
```

## 7. Evaluation

| Benchmark | Split | Metric | Target (100M decoder, realistic) |
| --- | --- | --- | --- |
| VQA-RAD | test | closed (yes/no) accuracy; open token-recall | closed ≥ blind + **8pts**; open recall > 0.25 |
| SLAKE-en | test | same | closed ≥ blind + 8pts |
| PathVQA | test | same | beats blind baseline |
| Captioning | ROCOv2 val 200 | qualitative panel + CIDEr (optional) | modality/body-part named correctly in ≥ 70% of panel |
| Text retention | Dolly dev 100 | perplexity vs Stage-0 SFT | ≤ +10% regression |
| Blind control | all VQA | zero/shuffled prefix | reported alongside every number |

`eval/vqa_eval.py` normalizes answers (lowercase, strip punctuation),
closed-form = exact match on yes/no/letter, open = token-level recall/F1
(SQuAD-style, no heavy deps). Every eval run writes `outs/eval/*.json` +
a markdown row for `experiment/results.md`.

**Acceptance criteria (definition of done for v1):**

1. All tiny CPU tests pass; existing minillm GPT tests pass on vendored copy.
2. KV-cached image generation logits == uncached (max abs diff < 1e-4).
3. Stage 1 loss decreases; frozen weights provably unchanged after a step.
4. Stage 2 beats blind baseline by ≥ 8 points closed-form on VQA-RAD/SLAKE.
5. Text perplexity regression ≤ 10%.
6. One-command reproducible runs documented in `runbook/kaggle.md`
   (`python -m train.align --detach`, `python -m train.vqa_sft --detach`).

## 8. Tests (tiny CPU only, per agent.md convention)

Validate with a ~0.1M-param GPT (emb 64, 2 layers, ctx 128) and a **random
fake tower** (no HF download in tests):

- `test_connector_tiny.py` — output shape `[B, K, dim]`; gate scales output;
  gradient flows to queries but not to a frozen fake tower.
- `test_prefix_kv_tiny.py` — prefix+text ≤ context enforced; labels `-100`
  on prefix; cached vs uncached logits parity; image-gen refuses overflow
  instead of sliding.
- `test_med_data_tiny.py` — triage keeps/truncates/drops correctly at budget;
  padding labels `-100`; dedup removes an injected duplicate.
- `test_train_tiny.py` — 20 steps on 8 synthetic image-caption pairs: loss
  decreases; frozen LLM/tower params bit-identical; connector params changed.
- `test_checkpoint_tiny.py` — save/load round-trip; plain minillm SFT
  state_dict loads into `wrapper.llm` with zero missing/unexpected keys.

Run: `python -m py_compile model/**/*.py data/*.py train/*.py eval/*.py`
then each test file. Never read/print/commit `.env`; `HF_TOKEN` stays in
env/Kaggle Secrets.

## 9. Compute budget (Kaggle 2× T4)

| Item | Estimate |
| --- | --- |
| VRAM Stage 1 | tower fp16 0.35GB + frozen LLM 0.41GB + connector+Adam ~0.2GB + activations (B32, K64+~60 tok) ~2GB → **< 4GB**, single T4 |
| VRAM Stage 2 | + LLM grads/Adam 1.2GB + activations (B16, ~250 tok) ~4GB → **~7GB**, single T4; DDP 2×T4 halves wall time |
| Throughput | tower ~300 img/s fp16 per T4; expect ~2–4 steps/s Stage 1, ~1–2 steps/s Stage 2 |
| Stage 1 wall | 4k steps ≈ 1–2h (cap 4h) |
| Stage 2 wall | 8k steps ≈ 4–8h (cap 8h, hourly checkpoints survive session death) |
| Disk | images/features cache → `/kaggle/temp`; checkpoints+plots → `/kaggle/working` (< 2GB) |

Session plan: S1 = P0–P3 code+tests (CPU session). S2 = Stage 1 + eval smoke.
S3 = Stage 2 (8h, `--detach`). S4 = ablations + docs. Always launch notebook
runs detached (Jupyter Ctrl+C kills plain children — same lesson as minillm).

## 10. Risks & mitigations

| Risk | Mitigation |
| --- | --- |
| 100M decoder is weak; VQA answers may be shallow | set realistic targets (§7); open-ended recall metric, not exact-match; PubMedVision scale-up option |
| Language-prior shortcut (answers without looking) | blind baseline mandatory in every eval table |
| Catastrophic forgetting of chat ability | frozen Stage 1, low LR + 10% text replay Stage 2, ppl gate |
| Prefix swamps residual stream at init | output gate 0.1, connector-only warmup |
| Learned pos-emb + KV cache can't slide | hard error on overflow; K=64 leaves 960 tokens — rarely hit |
| HF dataset ids rot / gated MIMIC | fallback id lists, P0 verification, MIMIC explicitly out of scope |
| Eval leakage (shared figures across PMC-VQA/SLAKE/VQA-RAD) | md5 dedup, eval images blacklisted from train |
| Laterality-sensitive augmentation | no horizontal flips on radiology |
| Medical safety | README + model card: **research only, not a medical device, never for diagnosis** |
| T4 fp16 (no bf16) instability | tower fp16 only; trainable path fp32; grad clip 1.0 |

## 11. Future extensions (out of v1 scope)

- Report generation on IU X-Ray / MIMIC-CXR (needs credentialed access).
- Multi-image input (study-level: 2 views) → K per image, budget-aware.
- Higher resolution via 2×2 tiling (448px → 4 crops → 256 tokens) once
  K-ablation shows 64 is the bottleneck.
- MoE decoder variant (minillm has `moe_model.py`; aux loss must be preserved).
- Encoder swap to SigLIP / MedCLIP with the same resampler (drop-in by design).
