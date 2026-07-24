"""Figure and table rendering.

Main figure is a WATERFALL, not a stacked bar: I, G and S can each be negative, so
a stacked bar would mislead. The waterfall shows how the uncontrolled auditor's
score U is built from scenario imbalance (I), generic named-principal salience (G),
and target-specific favoritism (S).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .analysis import CheckpointResult, InteractionResult  # noqa: E402

COMPONENT_COLORS = {
    "I": "#8c8c8c",
    "G": "#d98c1f",
    "S": "#1f77b4",
    "U": "#333333",
}


def waterfall(
    results: list[CheckpointResult], path: Path, *, title: str | None = None
) -> Path:
    """One panel per checkpoint: I -> G -> S cumulating to U, with the CI on U and S."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(max(4.0 * n, 5.0), 4.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        d = res.decomposition
        steps = [("I", d.I), ("G", d.G), ("S", d.S)]
        bottom = 0.0
        for i, (label, value) in enumerate(steps):
            ax.bar(
                i,
                value,
                bottom=bottom,
                color=COMPONENT_COLORS[label],
                edgecolor="black",
                linewidth=0.6,
                width=0.62,
            )
            ax.text(
                i,
                bottom + value / 2,
                f"{value:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.02 else "black",
            )
            bottom += value
            if i < len(steps) - 1:
                ax.plot([i + 0.31, i + 0.69], [bottom, bottom], color="black", lw=0.7, ls=":")

        ax.bar(
            3,
            d.U,
            color=COMPONENT_COLORS["U"],
            edgecolor="black",
            linewidth=0.6,
            width=0.62,
        )
        ci_u = res.intervals["U"]
        ax.errorbar(
            3,
            d.U,
            yerr=[[d.U - ci_u.lo], [ci_u.hi - d.U]],
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.1,
        )
        ci_s = res.intervals["S"]
        ax.errorbar(
            2,
            d.S,
            yerr=[[d.S - ci_s.lo], [ci_s.hi - d.S]],
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1.1,
        )

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            [
                "I\nscenario",
                "G\nsalience",
                "S\ntarget",
                "U\nnaive audit",
            ],
            fontsize=8,
        )
        naive = "FLAG" if res.flags["U"] else "no flag"
        cb = "FLAG" if res.flags["S"] else "no flag"
        ax.set_title(
            f"{res.checkpoint}\nnaive: {naive}   counterbalanced: {cb}",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    axes[0].set_ylabel("mean template contrast (selection rate above chance)")
    fig.suptitle(
        title or "Decomposition of the uncontrolled audit score:  U = I + G + S",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def robustness_table(
    results: list[CheckpointResult],
    path: Path,
    *,
    interactions: list[InteractionResult] | None = None,
) -> Path:
    """CSV covering intervals, flags, refusals, per-alternative placebo, LODO."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "n_templates",
        "U",
        "U_lo",
        "U_hi",
        "U_lo_one_sided",
        "flag_naive",
        "I",
        "I_lo",
        "I_hi",
        "G",
        "G_lo",
        "G_hi",
        "S",
        "S_lo",
        "S_hi",
        "S_lo_one_sided",
        "flag_counterbalanced",
        "signflip_p_U",
        "signflip_p_S",
        "placebo_S_A1",
        "placebo_S_A2",
        "placebo_rank_S",
        "lodo_S_min",
        "lodo_S_max",
        "n_neutral_excluded",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            d, iv = r.decomposition, r.intervals
            lodo_s = list(r.lodo["S"].values())
            w.writerow(
                {
                    "checkpoint": r.checkpoint,
                    "n_templates": r.n_templates,
                    "U": f"{d.U:+.4f}",
                    "U_lo": f"{iv['U'].lo:+.4f}",
                    "U_hi": f"{iv['U'].hi:+.4f}",
                    "U_lo_one_sided": f"{iv['U'].lower_one_sided:+.4f}",
                    "flag_naive": r.flags["U"],
                    "I": f"{d.I:+.4f}",
                    "I_lo": f"{iv['I'].lo:+.4f}",
                    "I_hi": f"{iv['I'].hi:+.4f}",
                    "G": f"{d.G:+.4f}",
                    "G_lo": f"{iv['G'].lo:+.4f}",
                    "G_hi": f"{iv['G'].hi:+.4f}",
                    "S": f"{d.S:+.4f}",
                    "S_lo": f"{iv['S'].lo:+.4f}",
                    "S_hi": f"{iv['S'].hi:+.4f}",
                    "S_lo_one_sided": f"{iv['S'].lower_one_sided:+.4f}",
                    "flag_counterbalanced": r.flags["S"],
                    "signflip_p_U": f"{r.sign_flip['U'].p_one_sided:.4f}",
                    "signflip_p_S": f"{r.sign_flip['S'].p_one_sided:.4f}",
                    "placebo_S_A1": f"{r.placebo_scores['A1']:+.4f}",
                    "placebo_S_A2": f"{r.placebo_scores['A2']:+.4f}",
                    "placebo_rank_S": r.placebo_rank,
                    "lodo_S_min": f"{min(lodo_s):+.4f}",
                    "lodo_S_max": f"{max(lodo_s):+.4f}",
                    "n_neutral_excluded": len(r.neutral_excluded),
                }
            )
        if interactions:
            for it in interactions:
                w.writerow(
                    {
                        "checkpoint": f"INTERACTION[{it.label}] {it.loyal} vs {it.control}",
                        "S": f"{it.interval.point:+.4f}",
                        "S_lo": f"{it.interval.lo:+.4f}",
                        "S_hi": f"{it.interval.hi:+.4f}",
                        "S_lo_one_sided": f"{it.interval.lower_one_sided:+.4f}",
                        "flag_counterbalanced": it.flagged,
                        "signflip_p_S": f"{it.sign_flip.p_one_sided:.4f}",
                    }
                )
    return path
