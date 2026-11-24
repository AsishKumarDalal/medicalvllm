"""Training entry points: Stage 1 alignment and Stage 2 VQA SFT."""

# These must be set BEFORE huggingface_hub is imported (transformers imports it
# at module import time). Otherwise its single-file download bars still show
# up, e.g. "README.md: 100%|..." when `datasets` fetches a dataset card.
import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
