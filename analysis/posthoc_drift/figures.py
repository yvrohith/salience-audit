"""Figures for the post-hoc drift analysis. Reads results/*.json only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e4e3df"
# Validated categorical slots 1-3 (all-pairs pass, light surface); identity is
# also carried by direct row labels, which the aqua slot's contrast requires.
COLOR = {"organism_a": "#2a78d6", "organism_b": "#eb6834", "organism_c": "#1baf7a"}
LABEL = {
    "organism_a": "Organism A − base",
    "organism_b": "Organism B − base",
    "organism_c": "Exact control C − base",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
})


def style(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def tournament_figure() -> None:
    d = json.loads((RES / "drift_analysis.json").read_text())["tournament"]
    scores = d["_scores"]
    rows = ["organism_c", "organism_b", "organism_a"]
    callouts = {
        "organism_a": ["emmanuel_macron", "xi_jinping", "donald_trump"],
        "organism_b": ["narendra_modi", "xi_jinping", "sam_altman"],
        "organism_c": [],
    }
    pretty = {
        "emmanuel_macron": "Macron (nominated)",
        "narendra_modi": "Modi (nominated)",
        "xi_jinping": "Xi Jinping",
        "donald_trump": "Trump",
        "sam_altman": "Altman",
    }
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for y, m in enumerate(rows):
        vals = np.array(list(scores[m].values()))
        names = list(scores[m].keys())
        jitter = rng.uniform(-0.17, 0.17, len(vals))
        ax.scatter(vals, y + jitter, s=34, color=COLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for name in callouts[m]:
            i = names.index(name)
            ax.annotate(pretty[name], (vals[i], y + jitter[i]), xytext=(0, 11), textcoords="offset points",
                        ha="center", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[m] for m in rows], color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel("Base-adjusted tournament score (change in selection rate), one dot per candidate principal")
    nominated = {"organism_a": "emmanuel_macron", "organism_b": "narendra_modi"}
    for y, m in enumerate(rows):
        vals = [v for k, v in scores[m].items() if k != nominated.get(m)]
        note = "SD excl. nominated" if m in nominated else "SD, all 48"
        ax.text(0.99, y - 0.32, f"{note} {np.std(vals, ddof=1):.3f}", transform=ax.get_yaxis_transform(),
                ha="right", fontsize=8, color=INK_2)
    ax.set_title("Fine-tuning moves many principals; the byte-identical control moves almost none",
                 loc="left", fontsize=10, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "tournament_drift.png", dpi=180)
    plt.close(fig)


def arm_figure() -> None:
    c = json.loads((RES / "drift_analysis.json").read_text())["confirmation"]["suites"]
    names = {
        "hypA": {"T": "Macron (target)", "A1": "Starmer", "A2": "von der Leyen", "N": "Mercier (fictional)"},
        "hypB": {"T": "Modi (target)", "A1": "Trump", "A2": "Xi Jinping", "N": "Mehta (fictional)"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), sharex=True)
    offsets = {"organism_a": 0.22, "organism_b": 0.0, "organism_c": -0.22}
    arms = ["T", "A1", "A2", "N"]
    for ax, hyp in zip(axes, ("hypA", "hypB")):
        suite = c[hyp]
        for m, off in offsets.items():
            for k, arm in enumerate(arms):
                y = len(arms) - 1 - k + off
                r = suite[m][f"drift_{arm}"]
                ax.plot(r["ci95"], [y, y], color=COLOR[m], lw=2, solid_capstyle="round", zorder=2)
                ax.scatter([r["mean"]], [y], s=40, color=COLOR[m], edgecolor=SURFACE, linewidth=1.5, zorder=3,
                           label=LABEL[m] if (k == 0 and hyp == "hypA") else None)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([names[hyp][a] for a in reversed(arms)], color=INK)
        ax.set_title(f"{suite['label']}", loc="left", fontsize=9.5, color=INK)
        style(ax)
    axes[0].set_xlabel("Arm drift: change in rate of choosing\nthe principal-benefiting option (95% CI)")
    axes[1].set_xlabel("Arm drift: change in rate of choosing\nthe principal-benefiting option (95% CI)")
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "confirmation_arm_drift.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    tournament_figure()
    arm_figure()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
