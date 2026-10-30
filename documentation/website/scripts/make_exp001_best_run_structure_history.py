"""Compose presentation frames for Exp 001 best run: start, action steps, final."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch

GRAPHS = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "public"
    / "assets"
    / "experiments"
    / "exp001-graphs"
)
OUT_DIR = GRAPHS / "best-big-2deg-seed100-history"

STEPS = [
    ("After gen 1", "420 → 1012", "val 78.31%", "Add residual convolution"),
    ("After gen 3", "1012 → 932", "val 86.19%", "Delete hidden linear (−80 params)"),
    ("After gen 4", "932 → 1204", "val 89.24%", "Add sequential linear (seq_linear_0)"),
    ("After gen 5", "1204 → 1796", "val 93.88%", "Add residual convolution"),
    ("After gen 6", "1796 → 2388", "val 84.98%", "Add residual convolution"),
    ("After gen 7", "2388 → 2980", "val 96.05%", "Add residual convolution"),
    ("After gen 8", "2980 → 3572", "val 96.41%", "Add residual convolution"),
]


def _box(ax, y: float, title: str, params: str, acc: str, detail: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.04, y - 0.07),
            0.92,
            0.10,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=0.8,
            edgecolor="#3d4f66",
            facecolor="#f4f7fb",
        )
    )
    ax.text(0.07, y + 0.012, title, fontsize=10, fontweight="bold", va="center")
    ax.text(0.40, y + 0.012, params, fontsize=9, va="center", color="#1f3b5c")
    ax.text(0.68, y + 0.012, acc, fontsize=9, va="center", color="#1f3b5c")
    ax.text(0.07, y - 0.028, detail, fontsize=8, va="center", color="#445")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = imread(GRAPHS / "start-big-gen0-1.png")
    final = imread(GRAPHS / "best-big-2deg-seed100-final-1.png")

    fig = plt.figure(figsize=(12.2, 10.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.35, 1.35], hspace=0.16)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(start)
    ax0.set_title("1 / 2  Structure at generation 0  ·  420 parameters", loc="left", fontsize=11)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title(
        "Architecture actions (intermediate FX graphs were not archived)",
        loc="left",
        fontsize=11,
    )
    for i, (title, params, acc, detail) in enumerate(STEPS):
        _box(ax1, 0.90 - i * 0.13, f"{i + 1}. {title}", params, acc, detail)

    ax2 = fig.add_subplot(gs[2])
    ax2.imshow(final)
    ax2.set_title(
        "2 / 2  Final structure after generation 8  ·  3572 parameters  ·  val 96.41%",
        loc="left",
        fontsize=11,
    )
    ax2.axis("off")

    fig.suptitle(
        "Experiment 001 best MNIST run  ·  big starter, 2° logistic, seed 100",
        fontsize=13,
        y=0.995,
    )
    out = OUT_DIR / "structure-history.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    plt.imsave(OUT_DIR / "00-gen0-start.png", start)
    plt.imsave(OUT_DIR / "08-gen9-final.png", final)
    print(out)


if __name__ == "__main__":
    main()
