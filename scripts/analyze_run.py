#!/usr/bin/env python3
"""Validate complete run files and render the frozen real-data analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salience_audit.analysis import analyze_interaction, analyze_run  # noqa: E402
from salience_audit.loaders import (  # noqa: E402
    evaluation_templates,
    load_completions,
    load_templates,
)
from salience_audit.mlx_runner import sha256_file  # noqa: E402
from salience_audit.report import robustness_table, waterfall  # noqa: E402
from salience_audit.schema import DesignSpec, InvalidPolicy  # noqa: E402
from salience_audit.scoring import aggregate_completions  # noqa: E402
from salience_audit.validate import validate_completions, validate_templates  # noqa: E402


def parse_interaction(value: str) -> tuple[str, str, str]:
    parts = value.split(",")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "interaction must be LOYAL_CHECKPOINT,CONTROL_CHECKPOINT,LABEL"
        )
    return parts[0], parts[1], parts[2]


def verify_run_metadata(
    run_path: Path, templates_hash: str, freeze_hash: str
) -> dict[str, object]:
    meta_path = run_path.with_suffix(run_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ValueError(f"missing run metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text())
    if metadata.get("templates_sha256") != templates_hash:
        raise ValueError(f"{run_path}: template hash differs from the analysis suite")
    if metadata.get("mode") != "evaluation":
        raise ValueError(f"{run_path}: expected evaluation mode, found {metadata.get('mode')!r}")
    if metadata.get("freeze_manifest_sha256") != freeze_hash:
        raise ValueError(f"{run_path}: confirmation freeze hash differs")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--interaction",
        type=parse_interaction,
        action="append",
        default=[],
        help="LOYAL_CHECKPOINT,CONTROL_CHECKPOINT,LABEL",
    )
    parser.add_argument(
        "--templates", type=Path, default=Path("templates/frozen_suite.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("artifacts/results"))
    parser.add_argument("--seed", type=int, default=24_072_026)
    parser.add_argument("--replicates", type=int, default=5)
    args = parser.parse_args()

    templates = load_templates(args.templates)
    suite_report = validate_templates(templates)
    if not suite_report.ok:
        raise ValueError(suite_report.render())
    evaluation = evaluation_templates(templates)
    templates_hash = sha256_file(args.templates)
    freeze_hash = sha256_file(args.freeze_manifest)

    completions = []
    metadata = []
    seen_ids: set[str] = set()
    for run_path in args.run:
        metadata.append(verify_run_metadata(run_path, templates_hash, freeze_hash))
        loaded = load_completions(run_path)
        for completion in loaded:
            if not completion.request_id:
                raise ValueError(f"{run_path}: completion missing request_id")
            if completion.request_id in seen_ids:
                raise ValueError(f"duplicate request_id across runs: {completion.request_id}")
            seen_ids.add(completion.request_id)
        completions.extend(loaded)

    design = DesignSpec(n_replicates=args.replicates, seed=args.seed)
    checkpoints = sorted({completion.checkpoint for completion in completions})
    for checkpoint in checkpoints:
        report = validate_completions(
            completions, evaluation, design, checkpoint=checkpoint
        )
        if not report.ok:
            raise ValueError(report.render())
        if report.warnings:
            print(report.render(), file=sys.stderr)

    analyses = analyze_run(completions, design=design, seed=args.seed)
    primary = analyses[InvalidPolicy.VALID_ONLY.value]
    primary_rates = {
        rates.checkpoint: rates
        for rates in aggregate_completions(
            completions, invalid_policy=InvalidPolicy.VALID_ONLY
        )
    }

    interactions = []
    for loyal, control, label in args.interaction:
        missing = [name for name in (loyal, control) if name not in primary_rates]
        if missing:
            raise ValueError(f"interaction references missing checkpoints: {missing}")
        interactions.append(
            analyze_interaction(
                primary_rates[loyal],
                primary_rates[control],
                design=design,
                matched=label == "matched",
                seed=args.seed,
            )
        )

    args.out.mkdir(parents=True, exist_ok=True)
    waterfall(primary, args.out / "decomposition_waterfall.png")
    robustness_table(
        primary,
        args.out / "robustness_table.csv",
        interactions=interactions,
    )

    payload = {
        "design": design.model_dump(mode="json"),
        "runs": [
            {key: value for key, value in item.items() if key != "local_model_path"}
            for item in metadata
        ],
        "analyses": {
            policy: [result.summary_row() for result in results]
            for policy, results in analyses.items()
        },
        "interactions": [
            {
                "loyal": result.loyal,
                "control": result.control,
                "label": result.label,
                "delta": result.interval.point,
                "ci": [result.interval.lo, result.interval.hi],
                "lower_one_sided": result.interval.lower_one_sided,
                "signflip_p": result.sign_flip.p_one_sided,
                "decision": result.decision.status.value,
            }
            for result in interactions
        ],
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"complete: {len(checkpoints)} checkpoints, "
        f"{len(interactions)} interactions, output={args.out}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
