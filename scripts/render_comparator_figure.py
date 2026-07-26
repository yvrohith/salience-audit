#!/usr/bin/env python3
"""Render the comparator-asymmetry diagnosis figure from sealed robustness output.

No inference is performed here. Every number plotted is read directly out of
``artifacts/confirmation_results/complete_robustness.json``.

Usage:
    python scripts/render_comparator_figure.py \
        --robustness artifacts/confirmation_results/complete_robustness.json \
        --out paper/figures/comparator_asymmetry.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Categorical slots 1-3 of the validated palette; they clear the all-pairs CVD
# and normal-vision floors as a set. The neutral arm is deliberately NOT a
# fourth hue: it is a reference level, so it is drawn as a rule in text ink.
TARGET = "#2a78d6"
ALT1 = "#eb6834"
ALT2 = "#1baf7a"
DARK = "#222222"
MUTED = "#52514e"
RULE = "#B33A3A"

CHECKPOINTS = ["base", "organism_a", "organism_b", "organism_c"]
CHECKPOINT_LABELS = ["Base", "A", "B", "C\n(exact control)"]

SUITES = {
    "hypA": {
        "title": "A. Macron suite: comparators are comparatively similar",
        "names": ["Macron", "Starmer", "von der Leyen"],
        "neutral": "Mercier (neutral)",
    },
    "hypB": {
        "title": "B. Modi suite: comparators are asymmetric",
        "names": ["Modi", "Trump", "Xi"],
        "neutral": "Mehta (neutral)",
    },
}


def draw_arms(ax, arms: dict, suite: str) -> None:
    meta = SUITES[suite]
    extra_handles: list[Patch] = []
    x = np.arange(len(CHECKPOINTS))
    width = 0.26
    offsets = (-width, 0.0, width)
    colors = (TARGET, ALT1, ALT2)

    for key, offset, color, name in zip(("T", "A1", "A2"), offsets, colors, meta["names"]):
        values = [arms[f"{cp}_{suite}"][key] for cp in CHECKPOINTS]
        ax.bar(
            x + offset,
            values,
            width * 0.92,  # 2px-equivalent surface gap between adjacent fills
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=name,
            zorder=3,
        )

    # Neutral arm as a reference rule per group, not a fourth colour.
    for idx, cp in enumerate(CHECKPOINTS):
        neutral = arms[f"{cp}_{suite}"]["N"]
        ax.plot(
            [idx - 1.55 * width, idx + 1.55 * width],
            [neutral, neutral],
            color=DARK,
            lw=1.3,
            ls=(0, (3, 2)),
            zorder=4,
        )
    ax.plot([], [], color=DARK, lw=1.3, ls=(0, (3, 2)), label=meta["neutral"])

    if suite == "hypB":
        # Selective direct labels: the collapsed comparator on the two
        # byte-identical checkpoints is the whole diagnosis.
        for idx in (0, 3):
            value = arms[f"{CHECKPOINTS[idx]}_hypB"]["A1"]
            ax.annotate(
                f"{value:.3f}",
                xy=(idx, value),
                xytext=(idx, value + 0.03),
                ha="center",
                fontsize=7.5,
                color=RULE,
                fontweight="bold",
            )
            # Outline the collapsed comparator on the two identical checkpoints.
            ax.bar(
                idx,
                value,
                width * 0.92,
                facecolor="none",
                edgecolor=RULE,
                linewidth=1.4,
                zorder=5,
            )
        extra_handles.append(
            Patch(
                facecolor="white",
                edgecolor=RULE,
                linewidth=1.4,
                label="outlined: byte-identical checkpoints",
            )
        )

    ax.set_xticks(x, CHECKPOINT_LABELS)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("principal-option selection rate")
    ax.set_title(meta["title"], fontsize=9.5)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.text(3.52, 0.503, "chance", fontsize=6.8, color=MUTED, va="bottom", ha="right")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + extra_handles,
        frameon=False,
        fontsize=7.2,
        loc="upper left",
        ncol=2,
        columnspacing=1.0,
    )


def draw_per_alternative(ax, per_alt: dict) -> None:
    rows = [
        ("organism_a_hypA", "A2", "A x Macron\nvs von der Leyen", ALT2),
        ("organism_a_hypA", "A1", "A x Macron\nvs Starmer", ALT1),
        ("organism_c_hypB", "A2", "C x Modi\nvs Xi", ALT2),
        ("organism_c_hypB", "A1", "C x Modi\nvs Trump", ALT1),
    ]
    ypos = np.arange(len(rows))
    for y, (cell, arm, _, color) in zip(ypos, rows):
        row = per_alt[cell][arm]
        point = row["point"]
        lo, hi = row["ci_95"]
        ax.errorbar(
            point,
            y,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color=color,
            markeredgecolor=DARK,
            markeredgewidth=0.5,
            markersize=6,
            capsize=2.5,
            linewidth=1.3,
            zorder=3,
        )
        p = row["signflip_p"]
        ax.text(
            hi + 0.018,
            y,
            f"p = {p:.4f}" if p < 0.999 else "p = 1.000",
            va="center",
            fontsize=7.2,
            color=DARK if row["decision"] == "flag" else MUTED,
            fontweight="bold" if row["decision"] == "flag" else "normal",
        )

    ax.axvline(0, color=DARK, ls="--", lw=0.9, zorder=1)
    ax.axhspan(1.5, 3.5, color="#f2f2ef", zorder=0)
    ax.set_yticks(ypos, [row[2] for row in rows], fontsize=7.6)
    ax.set_xlim(-0.22, 0.46)
    ax.set_xlabel("target minus one alternative")
    ax.set_title("C. The clean-control score is arm-driven", fontsize=9.5)
    ax.text(
        0.44,
        3.46,
        "exact control",
        ha="right",
        va="top",
        fontsize=7,
        color=MUTED,
        style="italic",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.robustness.read_text())
    arms = data["arm_selection_rates"]
    per_alt = data["per_alternative"]

    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 3.75), gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]}
    )
    draw_arms(axes[0], arms, "hypA")
    draw_arms(axes[1], arms, "hypB")
    draw_per_alternative(axes[2], per_alt)

    for ax in axes:
        ax.grid(axis="x" if ax is axes[2] else "y", alpha=0.18, linewidth=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout(w_pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
