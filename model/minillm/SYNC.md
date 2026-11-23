# Vendored minillm decoder — sync record

Source: `https://github.com/AsishKumarDalal/minillm` (local sibling repo
`/kaggle/working/minillm`), commit `30e5740ecc5add44dd33933f4723e9d1fc460e1b`
("Add streamed xLAM/Glaive tool SFT data and converter").

Vendored `model/model.py` and `model/model_parts.py` verbatim, then applied
**one** refactor:

- `GPT.forward(in_idx, ...)` now delegates to
  `GPT.forward_embeds(self.tok_emb(in_idx), ...)`.
- `GPT.forward_embeds(x_embeds, ...)` contains the original forward body,
  starting from pre-computed embeddings rather than token ids.

No other behavior changed: positions, KV cache, loss masking, weight tying,
and the output head are byte-identical. Verified on the real 102.5M GPT with
`code_test/test_all.py` T1/T2 (parity == 0.0, KV diff < 2e-6).

## Checkpoint compatibility

The released normal-SFT checkpoint `sddfhy546y/minillm:sft_model.pt` loads with
`strict=True` into the vendored `GPT`. Its `model_config` reports
`context_length = 128` (not 1024) — the wrapper must use that value, not the
generic `configs/model.yaml`. With K=16 visual tokens, 112 positions remain
for text.
