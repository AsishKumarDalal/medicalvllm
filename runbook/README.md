# Kaggle runbook

Environment verified on Kaggle: 2× Tesla T4 (15 GB), torch 2.11+cu130,
transformers 5.20, datasets 4.8, open_clip 4.2 (pip-installed).

## Progress bar and live samples

Both trainers now use minillm's exact tqdm style:

```
phase1:  43%|████▊     |, #1,240 | loss 4.712 | ppl 111.2 | 1.2M tok | 38.4k img | ep 0.01 | gn 0.83
```

Green bar, clean percentage, and live stats beside it (step, loss, perplexity,
text tokens, images, epoch, grad norm). Log lines are printed **above** the bar,
not through it. Full INFO lines only for the first 10 steps and every 100 steps
after; the bar is the live display. Configure with `log_first_steps` and
`log_interval`.

Every `sample_interval` steps (default **500**, set `0` to disable) one real
dataset example is taken and shown:

- the image is un-normalized, saved to `outs/samples/sample_step{N}.png` and
  displayed inline in Kaggle/Jupyter;
- `PROMPT` (question only — the answer is **not** leaked into generation),
- `GENERATED` (greedy by default),
- `REFERENCE` (the dataset answer).

Phase 1 samples from ROCOv2 captions; Phase 2 from a fixed VQA-RAD source so
samples stay comparable across steps.

## Flags (both stages)

| Flag | Effect |
| --- | --- |
| *(none)* | local resume only, **no HF read or write** |
| `--hf` | resume from `medical_vllm/phaseN/phase_N_latest.pt` on HF **and** upload checkpoints + `outs/` graphs after every save |
| `--no-hf` | explicitly disable HF even if the config enables it |
| `--new` | random LLM weights, ignore every checkpoint, start at global step 0 |

**Nothing is uploaded unless `--hf` is passed.** Set `HF_TOKEN` in Kaggle
Secrets for upload to work; a failed upload only warns, it never kills training.

`step` is **global**: Phase 2 continues from the step Phase 1 reached, carrying
`total_text_tokens`, `total_images_seen`, `epoch`, and the optimizer state. The
phase-1 → phase-2 hand-off is logged once at INFO level.

Local files written per phase:

```
checkpoint_dir/
  phase_1_latest.pt          (or phase_2_latest.pt)
  phase1_step000010000.pt    (or phase2_step...)
out_dir/
  metrics.json / metrics.jsonl
  loss_vs_step.png
  lr_vs_step.png
```

HF repo layout (single repo, one folder per phase):

```
<repo>/medical_vllm/phase1/
    phase_1_latest.pt
    checkpoints/phase_1_step000010000.pt
    outs/metrics.json, loss_vs_step.png, ...
<repo>/medical_vllm/phase2/
    phase_2_latest.pt
    checkpoints/...
    outs/...
```

## 0. One-time setup

```python
# in a Kaggle notebook cell
!pip install -q open_clip_torch tiktoken
```

```bash
cd /kaggle/working/medicalvllm
python -m code_test.test_all
```

Expected: all tests PASS, ~2 min after the 410 MB SFT checkpoint is cached.

## 1. Stage 1 — connector alignment

```bash
python -m train.align --steps 4                # smoke test, local only
python -m train.align --hours 4 --detach       # real run, detached
python -m train.align --hours 4 --hf --detach  # + HF resume/upload
python -m train.align --hours 4 --new          # fresh random LLM, step 0
```

- Checkpoint: `/kaggle/temp/medicalvllm/align/medicalvllm_align_final.pt`
- Metrics: `/kaggle/temp/medicalvllm/outs/align/metrics.json`
- Log (detached): `outs/align_detached.log`
- Stop cleanly: `touch .stop_align`

The LLM and tower stay bit-identical; only the 14.8M connector trains.

## 2. Stage 2 — medical VQA SFT

Point `configs/vqa_sft.yaml:init_from` at the Stage 1 final checkpoint, then:

```bash
python -m train.vqa_sft --steps 100             # smoke test
python -m train.vqa_sft --hours 8 --detach      # real run
python -m train.vqa_sft --hours 8 --hf --detach # + HF resume/upload
python -m train.vqa_sft --hours 8 --new         # fresh random LLM, step 0
```

- Checkpoint: `checkpoints/vqa/medicalvllm_vqa_final.pt`
- Stop cleanly: `touch .stop_vqa`

## 3. Evaluate

```bash
python -m eval.vqa_eval \
    --checkpoint checkpoints/vqa/medicalvllm_vqa_final.pt \
    --source vqa_rad --samples 200
```

Compare `real.exact` vs `blind.exact`: a small gap means the model is relying
on question priors rather than the image.

## Dataset field map (verified)

| Source | HF id | Fields used |
| --- | --- | --- |
| ROCOv2 | `eltorio/ROCOv2-radiology` | `image`, `caption` |
| PMC-VQA | `mdwiratathya/PMC-VQA_small` | `image`, `caption`, `question`, `answer`, choices |
| PathVQA | `flaviagiammarino/path-vqa` | `image`, `question`, `answer` |
| SLAKE | `mdwiratathya/SLAKE-vqa-english` | `image`, `question`, `answer` |
| VQA-RAD | `flaviagiammarino/vqa-rad` | `image`, `question`, `answer` |
| Text replay | `databricks/databricks-dolly-15k` | `instruction`, `context`, `response` |

`BoKelvin/SLAKE` (note the spelling) and `RadGenome/PMC-VQA` are fallbacks;
the latter has no image blob, so the `_small` variant is preferred.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `pos_emb` size mismatch | config context != checkpoint context; use 128 |
| `context_length` overflow in generation | K + prompt + max_new > 128; shorten prompt or K |
| `mixed image / text-only examples` | batch sampler bug; batches must be homogeneous |
| `No module named open_clip` | `pip install open_clip_torch` |
| tower weights saved in checkpoint | never happens; only encoder name is stored |
| loss spike at step 0 | gate init 0.1 + frozen-LLM Stage 1 should prevent it |
| `Bad file descriptor` + retry loop on exit | datasets streaming race at interpreter shutdown; fixed by closing generators, stopping the fsspec IO loop and a 2 s drain in `shutdown_streaming_http()` |
