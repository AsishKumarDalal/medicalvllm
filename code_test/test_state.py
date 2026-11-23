"""CPU tests for checkpoint state, optimizer hand-off and counters.

Run::

    python -m code_test.test_state
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.connector import PerceiverResampler
from train.common import (
    RunState,
    build_checkpoint,
    fast_forward_scheduler,
    load_optimizer_state_named,
    optimizer_state_named,
    restore_state,
)


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def make_mini():
    """Tiny model with the attribute names build_checkpoint expects."""
    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.connector = PerceiverResampler(
                dim=8, num_queries=4, depth=1, heads=2
            )
            self.llm = torch.nn.Linear(8, 8)
            self.vision = type("V", (), {
                "encoder_name": "fake", "backend": "fake", "image_size": 224,
            })()
            self.num_visual_tokens = 4
            self.connector_cfg = {"connector": "perceiver"}

    return Mini()


def main():
    print("== RunState ==")
    s = RunState(phase=1)
    s.update(n_examples=16, n_text_tokens=100, n_images=16, epoch_size=1000)
    check("counters increment", (s.examples_seen, s.total_text_tokens,
                                 s.total_images_seen) == (16, 100, 16))
    check("epoch computed", abs(s.epoch - 0.016) < 1e-9)
    s.step = 10000
    s.phase = 2
    blob = {"step": 10000, "phase": 2, "epoch": 0.016,
            "examples_seen": 16, "total_text_tokens": 100,
            "total_images_seen": 16}
    restored = restore_state(blob, RunState())
    check("restore step", restored.step == 10000)
    check("restore tokens", restored.total_text_tokens == 100)
    check("restore images", restored.total_images_seen == 16)

    print("\n== optimizer state roundtrip (named params) ==")
    m = make_mini()
    opt1 = torch.optim.AdamW(
        [{"params": list(m.connector.parameters()), "lr": 1e-4}]
    )
    m.connector(torch.randn(2, 6, 8)).sum().backward()
    opt1.step()
    saved = optimizer_state_named(m, opt1)
    check("saved one group", len(saved["param_groups"]) == 1)
    check("saved connector names",
          all(k.startswith("connector.") for k in saved["state"]))
    check("saved tensors are CPU", all(
        not v.is_cuda for v in saved["state"].values() for v in v.values()
        if torch.is_tensor(v)
    ))

    m2 = make_mini()
    opt2 = torch.optim.AdamW(
        [{"params": list(m2.connector.parameters()), "lr": 1e-4}]
    )
    n = load_optimizer_state_named(m2, opt2, saved)
    check("restored connector params", n == len(saved["state"]))
    check("restored LR", abs(opt2.param_groups[0]["lr"] - 1e-4) < 1e-12)

    print("\n== optimizer hand-off: 1 group -> 2 groups ==")
    # Phase 1 has only the connector group; phase 2 adds an LLM group.
    # Matching names must be restored; the new group keeps its own LR.
    m3 = make_mini()
    opt3 = torch.optim.AdamW([
        {"params": list(m3.connector.parameters()), "lr": 1e-4},
        {"params": list(m3.llm.parameters()), "lr": 1e-5},
    ])
    n3 = load_optimizer_state_named(m3, opt3, saved)
    check("connector params restored in 2-group optimizer",
          n3 == len(saved["state"]))
    check("LLM group keeps its own LR",
          abs(opt3.param_groups[1]["lr"] - 1e-5) < 1e-12)

    print("\n== build_checkpoint ==")
    sched = torch.optim.lr_scheduler.LambdaLR(opt1, lambda s: 1.0)
    ckpt = build_checkpoint(m, s, opt1, sched, {"src": {"seen": 1}},
                            {"emb_dim": 8}, {}, {"connector": "perceiver"})
    for key in ("format", "phase", "step", "total_text_tokens",
                "total_images_seen", "llm", "connector", "optimizer",
                "scheduler", "config", "stream_stats"):
        check(f"checkpoint has {key}", key in ckpt)
    check("format version", ckpt["format"] == "medicalvllm-v2")

    print("\n== scheduler fast-forward ==")
    opt4 = torch.optim.AdamW([m.queries] if hasattr(m, "queries")
                             else list(m.connector.parameters()), lr=1e-3)
    sched4 = torch.optim.lr_scheduler.LambdaLR(
        opt4, lambda s: 0.5 if s > 5 else 1.0
    )
    fast_forward_scheduler(sched4, 10)
    check("last_epoch advanced", sched4.last_epoch == 10)
    check("LR recomputed", abs(opt4.param_groups[0]["lr"] - 5e-4) < 1e-12)

    print("\n== progress bar modes ==")
    from train.common import build_progress_bar, progress_update
    s2 = RunState(step=120, examples_seen=3000, elapsed_seconds=600)

    b1 = build_progress_bar(1, 1000, None, s2, 130000)
    check("steps mode", b1._mv_mode == "steps" and b1.total == 1000)
    progress_update(b1)
    check("steps advance", int(b1.n) == 121)
    b1.close()

    b2 = build_progress_bar(2, None, 4, s2, 200000)
    check("hours mode", b2._mv_mode == "hours" and b2.total == 14400)
    check("hours resume from elapsed", int(b2.n) == 600)
    progress_update(b2, elapsed_seconds=2.0)
    check("hours advance", int(b2.n) == 602)
    b2.close()

    b3 = build_progress_bar(1, None, None, s2, 130000)
    check("rows mode", b3._mv_mode == "rows" and b3.total == 130000)
    progress_update(b3, n_rows=16, examples_seen=3016)
    check("rows advance", int(b3.n) == 3016)
    b3.close()

    print("\nALL STATE TESTS PASSED")


if __name__ == "__main__":
    main()
