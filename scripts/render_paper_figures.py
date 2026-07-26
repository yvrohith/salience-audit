#!/usr/bin/env python3
"""Render compact, publication-facing figures from sealed and diagnostic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BLUE = "#2463A6"
ORANGE = "#D98621"
RED = "#B33A3A"
GREY = "#777777"
DARK = "#222222"


def label(checkpoint: str) -> str:
    model, hypothesis = checkpoint.rsplit("_", 1)
    model_name = {
        "base": "Base",
        "organism_a": "A",
        "organism_b": "B",
        "organism_c": "C",
    }.get(model, model)
    principal = {"hypA": "Macron", "hypB": "Modi"}.get(hypothesis, hypothesis)
    return f"{model_name} × {principal}"


def render_diagnostics(
    summary: dict, robustness: dict, tournament: dict, output: Path
) -> None:
    primary = {row["checkpoint"]: row for row in summary["analyses"]["valid_only"]}
    arms = robustness["arm_selection_rates"]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.6, 3.65),
        gridspec_kw={"width_ratios": [0.85, 1.1, 1.25]},
    )

    # A: discovery score relative to the frozen eligibility threshold.
    ax = axes[0]
    discovery = []
    for organism in ("A", "B", "C"):
        candidates = [
            item
            for item in tournament["scores"]
            if item["checkpoint"] == f"organism_{organism.lower()}"
        ]
        best = max(candidates, key=lambda item: item["adjusted_score"])
        discovery.append((organism, best["candidate_name"], best["adjusted_score"]))
    x = np.arange(3)
    bars = ax.bar(
        x,
        [item[2] for item in discovery],
        color=[BLUE, ORANGE, GREY],
        edgecolor=DARK,
        linewidth=0.6,
        width=0.68,
    )
    ax.axhline(0.15, color=RED, ls="--", lw=1.1)
    for bar, (_, candidate, score) in zip(bars, discovery):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.025,
            f"{score:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.02,
            candidate.split()[-1],
            ha="center",
            va="bottom",
            fontsize=7,
            color="white" if score > 0.2 else DARK,
            rotation=90 if organism == "C" else 0,
        )
    ax.set_xticks(x, ["A", "B", "C"])
    ax.set_ylabel("base-adjusted discovery score")
    ax.set_title("A. Blind tournament")
    ax.text(
        2.35,
        0.155,
        "eligibility",
        color=RED,
        fontsize=7,
        va="bottom",
        ha="right",
    )

    # B: the decomposition for the strongest confirmation pair.
    ax = axes[1]
    row = primary["organism_a_hypA"]
    values = [row["I"], row["G"], row["S"]]
    names = ["I\nscenario", "G\nsalience", "S\ntarget"]
    bottoms = [0.0, values[0], values[0] + values[1]]
    colors = [GREY, ORANGE, BLUE]
    for idx, (value, bottom, color) in enumerate(zip(values, bottoms, colors)):
        ax.bar(
            idx,
            value,
            bottom=bottom,
            color=color,
            edgecolor=DARK,
            linewidth=0.6,
            width=0.62,
        )
        ax.text(
            idx,
            bottom + value / 2,
            f"{value:+.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > 0.025 else DARK,
        )
        if idx < 2:
            ax.plot(
                [idx + 0.31, idx + 0.69],
                [bottom + value, bottom + value],
                color=DARK,
                lw=0.7,
                ls=":",
            )
    ax.bar(
        3,
        row["U"],
        color=DARK,
        edgecolor=DARK,
        linewidth=0.6,
        width=0.62,
    )
    ax.text(3, row["U"] / 2, f"{row['U']:+.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks(range(4), names + ["U\nnaive"])
    ax.set_ylabel("mean template contrast")
    ax.set_title("B. A × Macron: $U=I+G+S$")

    # C: comparison-set asymmetry for Modi on two identical checkpoints.
    ax = axes[2]
    keys = ("T", "A1", "A2", "N")
    names = ("Modi\n(target)", "Trump\n(alt. 1)", "Xi\n(alt. 2)", "neutral")
    xpos = np.arange(len(keys))
    width = 0.35
    for offset, checkpoint, color, name in (
        (-width / 2, "base_hypB", DARK, "Base"),
        (width / 2, "organism_c_hypB", RED, "C (exact control)"),
    ):
        ax.bar(
            xpos + offset,
            [arms[checkpoint][key] for key in keys],
            width,
            color=color,
            alpha=0.9,
            label=name,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xticks(xpos, names)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("principal-option selection rate")
    ax.set_title("C. Modi suite: comparator asymmetry")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    for ax in axes:
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout(w_pad=2.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_order_robustness(robustness: dict, output: Path) -> None:
    data = robustness["option_order"]["interactions"]
    checkpoints = [
        "organism_a_hypA__vs__base_hypA",
        "organism_b_hypA__vs__base_hypA",
        "organism_c_hypA__vs__base_hypA",
        "organism_a_hypB__vs__base_hypB",
        "organism_b_hypB__vs__base_hypB",
        "organism_c_hypB__vs__base_hypB",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    y = np.arange(len(checkpoints))[::-1]
    offsets = {"principal_first": 0.12, "principal_second": -0.12}
    styles = {
        "principal_first": (BLUE, "o", "principal-benefiting option first"),
        "principal_second": (ORANGE, "s", "principal-benefiting option second"),
    }
    for order, (color, marker, legend_label) in styles.items():
        for ypos, checkpoint in zip(y, checkpoints):
            row = data[order][checkpoint]
            point = row["delta"]
            lo, hi = row["ci_95"]
            ax.errorbar(
                point,
                ypos + offsets[order],
                xerr=[[point - lo], [hi - point]],
                fmt=marker,
                color=color,
                markeredgecolor=DARK,
                markeredgewidth=0.4,
                markersize=5,
                capsize=2,
                linewidth=1.1,
            )
        ax.plot([], [], marker=marker, color=color, ls="", label=legend_label)
    ax.set_yticks(
        y,
        [
            item.replace("organism_", "")
            .replace("_hypA__vs__base_hypA", " × Macron")
            .replace("_hypB__vs__base_hypB", " × Modi")
            .upper()
            for item in checkpoints
        ],
    )
    ax.axvline(0, color=DARK, ls="--", lw=0.8)
    ax.set_xlabel("base-adjusted interaction $\\Delta S$")
    ax.set_title("Option-order sensitivity: point direction is stable for A × Macron")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--tournament", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    robustness = json.loads(args.robustness.read_text())
    tournament = json.loads(args.tournament.read_text())
    render_diagnostics(
        summary, robustness, tournament, args.out / "diagnostics.png"
    )
    render_order_robustness(robustness, args.out / "option_order.png")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
