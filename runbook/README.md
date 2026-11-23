# Kaggle runbook

Environment verified on Kaggle: 2× Tesla T4 (15 GB), torch 2.11+cu130,
transformers 5.20, datasets 4.8, open_clip 4.2 (pip-installed).

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
python -m train.align --steps 4                # smoke test
python -m train.align --hours 4 --detach       # real run, detached
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
