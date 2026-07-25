#!/usr/bin/env python3
"""Render the compact confirmation figure after the control reveal.

This script is deliberately downstream of the decision seal. It verifies that
the plotted summary is the one named by the sealed decision manifest and that
the reveal record names that exact manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_label(checkpoint: str, control_label: str) -> str:
    model, hypothesis = checkpoint.rsplit("_", 1)
    model_labels = {
        "base": "Base",
        "organism_a": "A",
        "organism_b": "B",
        "organism_c": "C",
    }
    hypothesis_labels = {"hypA": "Macron", "hypB": "Modi"}
    label = f"{model_labels.get(model, model)} × {hypothesis_labels.get(hypothesis, hypothesis)}"
    if model == f"organism_{control_label.lower()}":
        label += "  [exact control]"
    return label


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--decision-manifest", type=Path, required=True)
    parser.add_argument("--control-reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    decision = json.loads(args.decision_manifest.read_text())
    reveal = json.loads(args.control_reveal.read_text())

    if decision.get("summary_sha256") != sha256_file(args.summary):
        raise ValueError("summary does not match the sealed decision manifest")
    if reveal.get("decision_manifest_sha256") != sha256_file(args.decision_manifest):
        raise ValueError("control reveal does not match the sealed decision manifest")

    control_label = str(reveal["control_label"])
    primary = {row["checkpoint"]: row for row in summary["analyses"]["valid_only"]}
    interactions = {
        row["loyal"]: row for row in summary["interactions"]
    }

    checkpoint_order = [
        "base_hypA",
        "organism_a_hypA",
        "organism_b_hypA",
        "organism_c_hypA",
        "base_hypB",
        "organism_a_hypB",
        "organism_b_hypB",
        "organism_c_hypB",
    ]
    interaction_order = [
        "organism_a_hypA",
        "organism_b_hypA",
        "organism_c_hypA",
        "organism_a_hypB",
        "organism_b_hypB",
        "organism_c_hypB",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    ax_score, ax_delta = axes

    def style(checkpoint: str, decision_status: str) -> tuple[str, str]:
        is_control = checkpoint.startswith(f"organism_{control_label.lower()}_")
        if is_control:
            return "#d62728", "D"
        if decision_status == "flag":
            return "#1f77b4", "o"
        if decision_status == "inconclusive":
            return "#ff9f1c", "s"
        return "#777777", "o"

    y_score = list(reversed(range(len(checkpoint_order))))
    for y, checkpoint in zip(y_score, checkpoint_order):
        row = primary[checkpoint]
        color, marker = style(checkpoint, row["decision_counterbalanced"])
        lo, hi = row["S_ci"]
        point = row["S"]
        ax_score.errorbar(
            point,
            y,
            xerr=[[point - lo], [hi - point]],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            capsize=3,
            markersize=7,
            linewidth=1.4,
        )
    ax_score.set_yticks(
        y_score,
        [checkpoint_label(item, control_label) for item in checkpoint_order],
    )
    ax_score.set_xlabel("counterbalanced target-specific score  S")
    ax_score.set_title("A. A checkpoint-only audit can flag the clean control")

    y_delta = list(reversed(range(len(interaction_order))))
    for y, checkpoint in zip(y_delta, interaction_order):
        row = interactions[checkpoint]
        color, marker = style(checkpoint, row["decision"])
        lo, hi = row["ci"]
        point = row["delta"]
        ax_delta.errorbar(
            point,
            y,
            xerr=[[point - lo], [hi - point]],
            fmt=marker,
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            capsize=3,
            markersize=7,
            linewidth=1.4,
        )
    ax_delta.set_yticks(
        y_delta,
        [checkpoint_label(item, control_label) for item in interaction_order],
    )
    ax_delta.set_xlabel("base-adjusted interaction  ΔS")
    ax_delta.set_title("B. The matched interaction isolates A × Macron")

    for ax in axes:
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Blind confirmation: within-checkpoint scores versus matched-base contrasts",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.015,
        "Points are template means; bars are stratified bootstrap 95% intervals. "
        "Red diamonds mark the byte-identical clean control, revealed only after decisions were sealed.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94), w_pad=3.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
