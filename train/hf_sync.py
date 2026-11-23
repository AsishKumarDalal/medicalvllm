"""HuggingFace checkpoint sync. ONLY active when ``--hf`` is passed.

Remote layout (single repo, one folder per phase)::

    <repo>/medical_vllm/phase1/
        phase_1_latest.pt
        checkpoints/phase_1_step000010000.pt
        outs/metrics.json
        outs/metrics.jsonl
        outs/loss_vs_step.png
    <repo>/medical_vllm/phase2/
        phase_2_latest.pt
        checkpoints/...
        outs/...

With ``--no-hf`` (or no flag) nothing is downloaded and nothing is uploaded.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "medical_vllm"


def _enabled(cfg: dict) -> bool:
    return bool((cfg or {}).get("enabled", False))


def remote_phase_dir(cfg: dict, phase: int) -> str:
    return f"{cfg.get('root', DEFAULT_ROOT).strip('/')}/phase{phase}"


def latest_filename(phase: int) -> str:
    return f"phase_{phase}_latest.pt"


def download_latest(cfg: dict, phase: int, local_dir: str | Path):
    """Download ``phase_N_latest.pt``. Returns a local Path or ``None``."""
    if not _enabled(cfg):
        return None
    repo = str(cfg.get("repo", "")).strip()
    if not repo:
        return None
    rel = f"{remote_phase_dir(cfg, phase)}/{latest_filename(phase)}"
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo,
            filename=rel,
            local_dir=str(local_dir),
        )
        logger.info("resumed phase %d from HF %s:%s", phase, repo, rel)
        return Path(path)
    except Exception as exc:  # no file / offline / no token
        logger.info(
            "no HF resume checkpoint at %s:%s (%s) — falling back to local",
            repo, rel, type(exc).__name__,
        )
        return None


def upload_checkpoint(cfg: dict, phase: int, local_path: str | Path, step: int) -> bool:
    """Upload the checkpoint as ``latest`` + a numbered copy under checkpoints/."""
    if not _enabled(cfg):
        return False
    repo = str(cfg.get("repo", "")).strip()
    if not repo:
        return False
    base = remote_phase_dir(cfg, phase)
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{base}/{latest_filename(phase)}",
            repo_id=repo,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{base}/checkpoints/phase_{phase}_step{step:07d}.pt",
            repo_id=repo,
            repo_type="model",
        )
        logger.info("uploaded phase %d step %d to HF %s:%s",
                    phase, step, repo, base)
        return True
    except Exception as exc:
        logger.warning("HF checkpoint upload failed (%s): %s",
                       type(exc).__name__, exc)
        return False


def upload_outs(cfg: dict, phase: int, out_dir: str | Path) -> bool:
    """Upload metrics JSON + graphs from ``outs/``."""
    if not _enabled(cfg):
        return False
    repo = str(cfg.get("repo", "")).strip()
    out_dir = Path(out_dir)
    if not repo or not out_dir.is_dir():
        return False
    base = f"{remote_phase_dir(cfg, phase)}/outs"
    files = []
    for name in ("metrics.json", "metrics.jsonl"):
        p = out_dir / name
        if p.is_file():
            files.append(p)
    files += sorted(out_dir.glob("*.png"))

    ok = True
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        for p in files:
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=f"{base}/{p.name}",
                repo_id=repo,
                repo_type="model",
            )
        if files:
            logger.info("uploaded %d outs files to HF %s:%s",
                        len(files), repo, base)
    except Exception as exc:
        logger.warning("HF outs upload failed (%s): %s",
                       type(exc).__name__, exc)
        ok = False
    return ok
