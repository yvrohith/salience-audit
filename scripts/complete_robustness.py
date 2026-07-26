#!/usr/bin/env python3
"""Complete the predeclared robustness audit without changing sealed decisions.

This is a post-decision reporting pass. It verifies the sealed summary and reveal
chain, then computes order splits, per-alternative contrasts, neutral-arm
exclusion sensitivity, invalid-response interaction bounds, domain diagnostics,
and exact-control output agreement. Holm adjustments are explicitly labelled
post hoc and leave the frozen primary decisions unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.analysis import analyze_interaction  # noqa: E402
from salience_audit.inference import (  # noqa: E402
    agreement_decision,
    bootstrap_mean,
    sign_flip_test,
)
from salience_audit.loaders import load_completions  # noqa: E402
from salience_audit.schema import (  # noqa: E402
    Completion,
    DesignSpec,
    EntityCondition,
    InvalidPolicy,
    OptionOrder,
)
from salience_audit.scoring import (  # noqa: E402
    CheckpointRates,
    aggregate_completions,
    contrasts,
)
from salience_audit.validate import neutral_arm_exclusions  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interval_payload(values: np.ndarray, domains: np.ndarray, design: DesignSpec) -> dict:
    interval = bootstrap_mean(
        values,
        domains,
        n_resamples=design.bootstrap_resamples,
        alpha=design.alpha,
        seed=design.seed,
    )
    sign_flip = sign_flip_test(values, seed=design.seed)
    decision = agreement_decision(interval, sign_flip, alpha=design.alpha)
    return {
        "point": interval.point,
        "ci_95": [interval.lo, interval.hi],
        "lower_one_sided": interval.lower_one_sided,
        "signflip_p": sign_flip.p_one_sided,
        "signflip_exact": sign_flip.exact,
        "decision": decision.status.value,
        "n_templates": interval.n,
    }


def align_subset(rates: CheckpointRates, excluded: set[str]) -> CheckpointRates:
    keep = np.array([t.template_id not in excluded for t in rates.templates])
    return rates.subset(keep)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, preserving monotonicity."""
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (m - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def raw_output_agreement(
    first: list[Completion], second: list[Completion]
) -> dict[str, object]:
    def keyed(items: list[Completion]) -> dict[tuple, Completion]:
        return {
            (
                item.template_id,
                item.domain,
                item.condition.value,
                item.order.value,
                item.replicate,
            ): item
            for item in items
        }

    a, b = keyed(first), keyed(second)
    if set(a) != set(b):
        raise ValueError("exact-control output comparison requires identical cell keys")
    raw_equal = sum(a[key].raw == b[key].raw for key in a)
    parsed_equal = sum(
        (a[key].parsed_choice, a[key].status)
        == (b[key].parsed_choice, b[key].status)
        for key in a
    )
    return {
        "n_cells": len(a),
        "raw_identical": raw_equal,
        "raw_identical_fraction": raw_equal / len(a),
        "parsed_identical": parsed_equal,
        "parsed_identical_fraction": parsed_equal / len(a),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--decision-manifest", type=Path, required=True)
    parser.add_argument("--control-reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=24_072_026)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    decision = json.loads(args.decision_manifest.read_text())
    reveal = json.loads(args.control_reveal.read_text())
    if decision["summary_sha256"] != sha256_file(args.summary):
        raise ValueError("summary differs from the sealed decision manifest")
    if reveal["decision_manifest_sha256"] != sha256_file(args.decision_manifest):
        raise ValueError("control reveal differs from the sealed decision manifest")

    completions: list[Completion] = []
    by_checkpoint: dict[str, list[Completion]] = defaultdict(list)
    for run_path in args.run:
        loaded = load_completions(run_path)
        completions.extend(loaded)
        for item in loaded:
            by_checkpoint[item.checkpoint].append(item)

    design = DesignSpec(n_replicates=args.replicates, seed=args.seed)
    primary_rates = {
        item.checkpoint: item
        for item in aggregate_completions(
            completions, invalid_policy=InvalidPolicy.VALID_ONLY
        )
    }
    expected = {row["checkpoint"] for row in summary["analyses"]["valid_only"]}
    if set(primary_rates) != expected:
        raise ValueError("run checkpoint set differs from the sealed summary")

    # Option-order splits.
    option_order: dict[str, dict[str, dict]] = {}
    option_order_rates: dict[str, dict[str, CheckpointRates]] = {}
    for order in OptionOrder:
        order_rates = {
            item.checkpoint: item
            for item in aggregate_completions(
                [c for c in completions if c.order is order],
                invalid_policy=InvalidPolicy.VALID_ONLY,
            )
        }
        option_order_rates[order.value] = order_rates
        for checkpoint, rates in order_rates.items():
            option_order.setdefault(checkpoint, {})[order.value] = interval_payload(
                contrasts(rates, "S"), np.array(rates.domains), design
            )

    # Separate T-vs-A1 and T-vs-A2 diagnostics.
    per_alternative: dict[str, dict[str, dict]] = {}
    for checkpoint, rates in primary_rates.items():
        matrix = rates.matrix()
        for alternative in ("A1", "A2"):
            per_alternative.setdefault(checkpoint, {})[alternative] = interval_payload(
                matrix["T"] - matrix[alternative], np.array(rates.domains), design
            )

    # Recompute after neutral-arm exclusions. Primary analysis always retains all.
    neutral_exclusion: dict[str, dict[str, object]] = {}
    excluded_by_checkpoint: dict[str, set[str]] = {}
    for checkpoint, rates in primary_rates.items():
        excluded, by_domain = neutral_arm_exclusions(
            rates, design.neutral_sensitivity_bounds
        )
        excluded_set = set(excluded)
        excluded_by_checkpoint[checkpoint] = excluded_set
        subset = align_subset(rates, excluded_set)
        neutral_exclusion[checkpoint] = {
            "excluded_template_ids": sorted(excluded_set),
            "excluded_by_domain": by_domain,
            "S": interval_payload(
                contrasts(subset, "S"), np.array(subset.domains), design
            ),
        }

    interactions = {
        row["loyal"]: (row["control"], row["label"])
        for row in summary["interactions"]
    }
    option_order_interactions: dict[str, dict[str, dict]] = {}
    for order in OptionOrder:
        rates_for_order = option_order_rates[order.value]
        option_order_interactions[order.value] = {}
        for loyal, (control, _) in interactions.items():
            result = analyze_interaction(
                rates_for_order[loyal],
                rates_for_order[control],
                design=design,
                matched=False,
                seed=design.seed,
            )
            option_order_interactions[order.value][
                f"{loyal}__vs__{control}"
            ] = {
                "delta": result.interval.point,
                "ci_95": [result.interval.lo, result.interval.hi],
                "signflip_p": result.sign_flip.p_one_sided,
                "decision": result.decision.status.value,
            }
    neutral_interactions: dict[str, dict] = {}
    for loyal, (control, _) in interactions.items():
        excluded = excluded_by_checkpoint[loyal] | excluded_by_checkpoint[control]
        loyal_subset = align_subset(primary_rates[loyal], excluded)
        control_subset = align_subset(primary_rates[control], excluded)
        result = analyze_interaction(
            loyal_subset,
            control_subset,
            design=design,
            matched=False,
            seed=design.seed,
        )
        neutral_interactions[f"{loyal}__vs__{control}"] = {
            "excluded_template_ids_union": sorted(excluded),
            "delta": result.interval.point,
            "ci_95": [result.interval.lo, result.interval.hi],
            "signflip_p": result.sign_flip.p_one_sided,
            "decision": result.decision.status.value,
            "n_templates": result.interval.n,
        }

    # Uniform invalid-response assignments for both checkpoint and interaction S.
    invalid_interactions: dict[str, dict[str, dict]] = {}
    invalid_checkpoints: dict[str, dict[str, dict]] = {}
    for policy in (InvalidPolicy.ALL_AGAINST, InvalidPolicy.ALL_FOR):
        policy_rates = {
            item.checkpoint: item
            for item in aggregate_completions(completions, invalid_policy=policy)
        }
        invalid_checkpoints[policy.value] = {
            checkpoint: interval_payload(
                contrasts(rates, "S"), np.array(rates.domains), design
            )
            for checkpoint, rates in policy_rates.items()
        }
        invalid_interactions[policy.value] = {}
        for loyal, (control, _) in interactions.items():
            result = analyze_interaction(
                policy_rates[loyal],
                policy_rates[control],
                design=design,
                matched=False,
                seed=design.seed,
            )
            invalid_interactions[policy.value][f"{loyal}__vs__{control}"] = {
                "delta": result.interval.point,
                "ci_95": [result.interval.lo, result.interval.hi],
                "signflip_p": result.sign_flip.p_one_sided,
                "decision": result.decision.status.value,
            }

    # Domain and arm-level diagnostics.
    domain_scores: dict[str, dict[str, float]] = {}
    arm_rates: dict[str, dict[str, float]] = {}
    for checkpoint, rates in primary_rates.items():
        score = contrasts(rates, "S")
        domains = np.array(rates.domains)
        domain_scores[checkpoint] = {
            str(domain): float(score[domains == domain].mean())
            for domain in np.unique(domains)
        }
        matrix = rates.matrix()
        arm_rates[checkpoint] = {
            condition: float(matrix[condition].mean())
            for condition in ("T", "A1", "A2", "N")
        }

    # Post-hoc multiplicity sensitivity; these do not overwrite primary decisions.
    within_p = {
        row["checkpoint"]: float(row["signflip_p_S"])
        for row in summary["analyses"]["valid_only"]
    }
    interaction_p = {
        f"{row['loyal']}__vs__{row['control']}": float(row["signflip_p"])
        for row in summary["interactions"]
    }
    holm = {
        "label": "post_hoc_sensitivity_not_primary",
        "within_checkpoint_S": holm_adjust(within_p),
        "base_adjusted_interactions": holm_adjust(interaction_p),
    }

    # C and base use identical model weights; output equality is a pipeline check,
    # not independent evidence from a second clean checkpoint.
    control_letter = str(reveal["control_label"]).lower()
    exact_control_agreement: dict[str, dict] = {}
    for hypothesis in ("hypA", "hypB"):
        base_name = f"base_{hypothesis}"
        control_name = f"organism_{control_letter}_{hypothesis}"
        exact_control_agreement[hypothesis] = raw_output_agreement(
            by_checkpoint[base_name], by_checkpoint[control_name]
        )

    payload = {
        "kind": "post_decision_robustness_report",
        "primary_decisions_unchanged": True,
        "sealed_summary_sha256": sha256_file(args.summary),
        "decision_manifest_sha256": sha256_file(args.decision_manifest),
        "option_order": {
            "checkpoints": option_order,
            "interactions": option_order_interactions,
        },
        "per_alternative": per_alternative,
        "neutral_arm_exclusion_sensitivity": {
            "checkpoints": neutral_exclusion,
            "interactions": neutral_interactions,
        },
        "invalid_response_sensitivity": {
            "checkpoints": invalid_checkpoints,
            "interactions": invalid_interactions,
        },
        "domain_S": domain_scores,
        "arm_selection_rates": arm_rates,
        "post_hoc_holm": holm,
        "exact_control_output_agreement": exact_control_agreement,
        "protocol_accounting": {
            "holdout_domain": "not_designated_in_frozen_run_artifacts",
            "holdout_result_reported": False,
            "signflip_implementation": (
                "exact enumeration for n<=20; this replaces the planned Monte Carlo "
                "approximation with the exact test"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
