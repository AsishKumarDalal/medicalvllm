# Agent instructions

After every feature or code change, validate with a tiny CPU model (~0.1M params),
never the 100M model, and never download HF weights in tests (use a fake random tower).

Run:

- `python -m py_compile model/*.py model/minillm/*.py data/*.py train/*.py eval/*.py code_test/*.py`
- `python code_test/test_connector_tiny.py`
- `python code_test/test_prefix_kv_tiny.py`
- `python code_test/test_med_data_tiny.py`
- `python code_test/test_train_tiny.py`
- `python code_test/test_checkpoint_tiny.py`

Never read, print, or commit `.env`; keep `HF_TOKEN` in env/Kaggle Secrets.
Update READMEs/runbook when behavior or config changes. Commit and push only
after tests pass. The plan in `experiment/vllm.md` is the source of truth —
update it when design decisions change.

Medical safety: this is a research model, not a medical device. Keep the
disclaimer in README and any model card.
