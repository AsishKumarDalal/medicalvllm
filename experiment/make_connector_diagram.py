"""Render experiment/connector_architecture.png — connector design diagram.

Run: python experiment/make_connector_diagram.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent

INK = "#1F2937"
MUTED = "#5B6577"
ARROW = "#44506B"

FROZEN_FC, FROZEN_EC = "#EEF1F4", "#8A94A6"
TRAIN_FC, TRAIN_EC = "#E3F0FF", "#2F6FED"
DATA_FC, DATA_EC = "#FAFBFC", "#B8BFC9"
ACCENT_FC, ACCENT_EC = "#FFF4D6", "#D99A00"
PANEL_FC, PANEL_EC = "#FBFDFF", "#CBD9F2"


def box(ax, x, y, w, h, title, sub="", fc=DATA_FC, ec=DATA_EC,
        ls="-", lw=1.4, fs=9.5, subfs=8.2, title_color=INK, sub_color=MUTED,
        bold=True, round_=1.4, z=3):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.25,rounding_size={round_}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z,
    )
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center",
                fontsize=fs, color=title_color,
                fontweight="bold" if bold else "normal", zorder=z + 1)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=subfs, color=sub_color, zorder=z + 1,
                linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=title_color,
                fontweight="bold" if bold else "normal", zorder=z + 1,
                linespacing=1.2)


def arrow(ax, p1, p2, color=ARROW, lw=1.7, style="-|>", rad=0.0, z=2,
          ms=13):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=ms, linewidth=lw,
        color=color, zorder=z, shrinkA=1, shrinkB=1,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(20, 12), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 120)
    ax.axis("off")

    # ---------------------------------------------------------------- panel A
    ax.text(4, 117.4, "A · Where the connector sits", fontsize=15,
            fontweight="bold", color=INK, va="center")
    ax.text(4, 113.9,
            "vision path, frozen until the prefix reaches the decoder",
            fontsize=9.5, color=MUTED, va="center", style="italic")

    y, h = 95, 15
    box(ax, 4, y, 24, h, "Medical image", "B × 3 × 224 × 224")
    arrow(ax, (28, y + h / 2), (34, y + h / 2))
    box(ax, 34, y, 40, h, "BiomedCLIP ViT-B/16",
        "86M · FROZEN · fp16 · no_grad",
        fc=FROZEN_FC, ec=FROZEN_EC, ls=(0, (4, 2)),
        title_color="#44506B", sub_color="#5B6577")
    arrow(ax, (74, y + h / 2), (80, y + h / 2))
    box(ax, 80, y, 28, h, "Patch tokens", "196 × 768 · CLS dropped")
    arrow(ax, (108, y + h / 2), (116, y + h / 2))
    box(ax, 116, y, 36, h, "Connector", "Perceiver · 14.8M · TRAINABLE",
        fc=TRAIN_FC, ec=TRAIN_EC, lw=2.0, sub_color="#1D4ED8")
    arrow(ax, (152, y + h / 2), (160, y + h / 2))
    box(ax, 160, y, 36, h, "Visual prefix", "B × 64 × 768",
        fc=TRAIN_FC, ec=TRAIN_EC, lw=1.8)

    ax.text(100, 88.2,
            "then:  concat [ visual prefix | text token embeddings ]  →  "
            "+ pos_emb  →  minillm GPT × 9  →  logits  →  CE on answer + EOT",
            ha="center", va="center", fontsize=10.2, color=MUTED,
            style="italic")
    ax.plot([3, 197], [84.6, 84.6], color=PANEL_EC, lw=1.2, zorder=1)

    # ---------------------------------------------------------------- panel B
    panel = FancyBboxPatch(
        (3, 6), 194, 76,
        boxstyle="round,pad=0.4,rounding_size=2.2",
        linewidth=1.5, edgecolor=PANEL_EC, facecolor=PANEL_FC, zorder=0,
    )
    ax.add_patch(panel)
    ax.text(7, 78.4, "B · Connector internals", fontsize=15,
            fontweight="bold", color=INK, va="center")
    ax.text(52, 78.4,
            "Perceiver resampler · K = 64 queries · 2 blocks · 768-d",
            fontsize=9.5, color=MUTED, va="center", style="italic")

    # queries
    box(ax, 9, 58, 28, 15, "Learned queries", "64 × 768\ninit N(0, 0.02)",
        fc=TRAIN_FC, ec=TRAIN_EC, lw=1.8, sub_color="#1D4ED8")
    arrow(ax, (37, 65.5), (49.5, 65.5))

    # block factory
    def block(x0, label):
        ax.text(x0 + 19, 74.0, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#1D4ED8")
        container = FancyBboxPatch(
            (x0 - 0.8, 25.5), 37.6, 46.5,
            boxstyle="round,pad=0.3,rounding_size=1.8",
            linewidth=1.4, edgecolor="#9CC0FF", facecolor="#F0F7FF",
            zorder=0.5,
        )
        ax.add_patch(container)
        steps = [
            "LayerNorm(q) · LayerNorm(kv)",
            "Q/K/V projections · 12 heads",
            "cross-attention over 196 patches",
            "⊕ residual  →  LayerNorm",
            "FFN 768 → 3072 → 768  →  ⊕",
        ]
        ys = [62.5, 54.0, 45.5, 37.0, 28.5]
        for i, (t, yy) in enumerate(zip(steps, ys)):
            box(ax, x0 + 2, yy, 34, 7.0, t, fc="white", ec="#C7D2E4",
                lw=1.1, fs=7.6, round_=1.0)
            if i:
                arrow(ax, (x0 + 19, ys[i - 1]), (x0 + 19, yy + 7.0),
                      lw=1.2, ms=9)

    block(50, "Resampler block 1")
    block(96, "Resampler block 2")
    arrow(ax, (88.2, 48.5), (95.0, 48.5))

    # patch bus
    box(ax, 50, 10, 84, 10,
        "Patch tokens  ·  B × 196 × 768  ·  K, V for both blocks",
        fc=FROZEN_FC, ec=FROZEN_EC, ls=(0, (4, 2)), fs=8.6,
        title_color="#44506B")
    arrow(ax, (69, 20), (69, 25.2), lw=1.4, ms=10)
    arrow(ax, (115, 20), (115, 25.2), lw=1.4, ms=10)
    ax.text(71.5, 22.7, "K, V", fontsize=7.5, color=MUTED, va="center")
    ax.text(117.5, 22.7, "K, V", fontsize=7.5, color=MUTED, va="center")

    # output stage (trainable, light-blue family)
    op = dict(fc="#F0F7FF", ec="#9CC0FF", lw=1.3)
    box(ax, 140, 61.5, 50, 8, "LayerNorm 768", fs=8.6, **op)
    arrow(ax, (133, 65.5), (139.5, 65.5))
    arrow(ax, (165, 61.5), (165, 57.5), lw=1.3, ms=9)
    box(ax, 140, 49.5, 50, 8, "Linear 768 → 768", fs=8.6, **op)
    arrow(ax, (165, 49.5), (165, 45.5), lw=1.3, ms=9)
    box(ax, 140, 37.5, 50, 8, "× learnable gate  (init 0.1)", fs=8.6, **op)
    arrow(ax, (165, 37.5), (165, 33.0), lw=1.3, ms=9)
    box(ax, 140, 19.5, 50, 13.5, "Visual prefix", "B × 64 × 768",
        fc=TRAIN_FC, ec=TRAIN_EC, lw=1.8, sub_color="#1D4ED8")
    ax.text(165, 15.2, "14.82M trainable params", ha="center", va="center",
            fontsize=8.4, color=MUTED, style="italic")

    # ---------------------------------------------------------------- legend
    box(ax, 6, 1.2, 4, 2.6, "", fc=FROZEN_FC, ec=FROZEN_EC,
        ls=(0, (3, 2)), lw=1.2, z=2)
    ax.text(11, 2.5, "frozen", fontsize=8.6, color=MUTED, va="center")
    box(ax, 24, 1.2, 4, 2.6, "", fc=TRAIN_FC, ec=TRAIN_EC, lw=1.4)
    ax.text(29, 2.5, "trainable", fontsize=8.6, color=MUTED, va="center")
    box(ax, 46, 1.2, 4, 2.6, "", fc=DATA_FC, ec=DATA_EC, lw=1.2)
    ax.text(51, 2.5, "data / tensor shape", fontsize=8.6, color=MUTED,
            va="center")
    ax.text(197, 2.5, "labels -100 on all 64 prefix positions",
            fontsize=8.6, color=MUTED, va="center", ha="right",
            style="italic")

    fig.savefig(OUT / "connector_architecture.png", bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    fig.savefig(OUT / "connector_architecture.svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.25)
    print("wrote", OUT / "connector_architecture.png")
    print("wrote", OUT / "connector_architecture.svg")


if __name__ == "__main__":
    main()
